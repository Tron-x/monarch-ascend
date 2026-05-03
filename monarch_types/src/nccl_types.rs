/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! GPU-independent types for collective communication bootstrap.
//!
//! These types are used in the message protocol between tensor workers and must
//! be available even in CPU-only builds where `nccl-sys` / `hccl-sys` are not
//! compiled.  They follow the principle that **NCCL and HCCL are two independent
//! backends** -- the wire type is a portable enum carrying either backend's
//! opaque bootstrap identifier; each backend impl pattern-matches the variant
//! it expects and rejects the other (which can never happen in practice because
//! NCCL/HCCL communicators cannot interoperate at the driver level anyway).

use std::fmt;
use std::fmt::Write;

use serde::Deserialize;
use serde::Deserializer;
use serde::Serialize;
use serde::Serializer;
use serde::ser::SerializeSeq;

/// Rust version of `ncclRedOp_t` / `HcclReduceOp` (they share the same shape).
#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq, Eq)]
pub enum ReduceOp {
    Sum = 0,
    Prod = 1,
    Max = 2,
    Min = 3,
    Avg = 4,
}

// ---------------------------------------------------------------------------
// NcclUniqueId (128-byte) -- bit-for-bit ncclUniqueId from nccl-sys.
// ---------------------------------------------------------------------------

/// Wire-compatible representation of `ncclUniqueId`.
///
/// 128-byte opaque identifier used to bootstrap NCCL communicators.  The struct
/// layout and serialization format match `ncclUniqueId` from `nccl-sys` exactly,
/// so that messages are wire-compatible regardless of whether the sender or
/// receiver was built with NCCL support.
#[repr(C)]
#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
pub struct NcclUniqueId {
    #[serde(
        serialize_with = "serialize_array_128",
        deserialize_with = "deserialize_array_128"
    )]
    pub internal: [::std::os::raw::c_char; 128usize],
}

fn deserialize_array_128<'de, D>(
    deserializer: D,
) -> Result<[::std::os::raw::c_char; 128], D::Error>
where
    D: Deserializer<'de>,
{
    let vec: Vec<::std::os::raw::c_char> = Deserialize::deserialize(deserializer)?;
    vec.try_into().map_err(|v: Vec<::std::os::raw::c_char>| {
        serde::de::Error::invalid_length(v.len(), &"expected an array of length 128")
    })
}

fn serialize_array_128<S>(
    array: &[::std::os::raw::c_char; 128],
    serializer: S,
) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    let mut seq = serializer.serialize_seq(Some(128))?;
    for element in array {
        seq.serialize_element(element)?;
    }
    seq.end()
}

// ---------------------------------------------------------------------------
// HcclRootInfo (4108-byte) -- bit-for-bit HcclRootInfo from hccl-sys.
//
// Defined here (mirroring NcclUniqueId) so monarch_types stays free of any
// optional dependency on hccl-sys.  See ``HCCL_ROOT_INFO_BYTES`` in
// ``hccl-sys/src/bridge.h``.
// ---------------------------------------------------------------------------

/// Wire-compatible representation of `HcclRootInfo`.
///
/// 4108-byte opaque identifier used to bootstrap HCCL communicators on Ascend
/// NPUs.  Mirrors ``HcclRootInfo::internal`` in ``hccl-sys`` byte-for-byte.
#[repr(C)]
#[derive(Copy, Clone, Serialize, Deserialize)]
pub struct HcclRootInfo {
    #[serde(
        serialize_with = "serialize_array_4108",
        deserialize_with = "deserialize_array_4108"
    )]
    pub internal: [::std::os::raw::c_char; 4108usize],
}

impl fmt::Debug for HcclRootInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Only print first 32 bytes for brevity (matches torch_sys_ascend's
        // RootInfo Debug impl).
        let preview = self.internal.iter().take(32).fold(String::new(), |mut o, b| {
            let _ = write!(o, "{:02x}", b);
            o
        });
        f.debug_struct("HcclRootInfo")
            .field("internal", &format_args!("{}...", preview))
            .finish()
    }
}

fn deserialize_array_4108<'de, D>(
    deserializer: D,
) -> Result<[::std::os::raw::c_char; 4108], D::Error>
where
    D: Deserializer<'de>,
{
    let vec: Vec<::std::os::raw::c_char> = Deserialize::deserialize(deserializer)?;
    vec.try_into().map_err(|v: Vec<::std::os::raw::c_char>| {
        serde::de::Error::invalid_length(v.len(), &"expected an array of length 4108")
    })
}

fn serialize_array_4108<S>(
    array: &[::std::os::raw::c_char; 4108],
    serializer: S,
) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    let mut seq = serializer.serialize_seq(Some(4108))?;
    for element in array {
        seq.serialize_element(element)?;
    }
    seq.end()
}

// ---------------------------------------------------------------------------
// UniqueId -- portable enum, the wire type carried in WorkerMessage.
// ---------------------------------------------------------------------------

/// Backend-agnostic collective-communicator unique ID.
///
/// Carries either an NCCL ``ncclUniqueId`` (128 bytes) or an HCCL
/// ``HcclRootInfo`` (4108 bytes).  Each backend impl pattern-matches the
/// variant it expects and rejects the other.
///
/// Wire compatibility note: a NCCL-built sender and a HCCL-built receiver
/// (or vice versa) cannot meaningfully share a communicator anyway -- the
/// driver stacks are mutually exclusive.  So the cross-variant case is
/// nominally a logic error, not a wire-format concern.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum UniqueId {
    /// NCCL-shaped 128-byte unique identifier.
    Nccl(NcclUniqueId),
    /// HCCL-shaped 4108-byte root info.
    Hccl(HcclRootInfo),
}

impl UniqueId {
    // --- NCCL helpers (kept for backward compat with existing CUDA call sites) ---

    /// Construct an NCCL-variant ``UniqueId`` from raw 128-byte buffer.
    pub fn from_internal(internal: [::std::os::raw::c_char; 128]) -> Self {
        Self::Nccl(NcclUniqueId { internal })
    }

    /// Access the inner 128-byte NCCL bytes; panics if this is an HCCL variant.
    pub fn internal(&self) -> &[::std::os::raw::c_char; 128] {
        match self {
            Self::Nccl(inner) => &inner.internal,
            Self::Hccl(_) => panic!(
                "UniqueId::internal() called on HCCL variant; use as_hccl_root_info() or pattern-match"
            ),
        }
    }

    /// Borrow the inner ``NcclUniqueId``; panics if this is an HCCL variant.
    pub fn as_nccl_unique_id(&self) -> &NcclUniqueId {
        match self {
            Self::Nccl(inner) => inner,
            Self::Hccl(_) => {
                panic!("UniqueId::as_nccl_unique_id() called on HCCL variant")
            }
        }
    }

    /// Consume and return the inner ``NcclUniqueId``; panics on HCCL.
    pub fn into_nccl_unique_id(self) -> NcclUniqueId {
        match self {
            Self::Nccl(inner) => inner,
            Self::Hccl(_) => {
                panic!("UniqueId::into_nccl_unique_id() called on HCCL variant")
            }
        }
    }

    // --- HCCL helpers ---

    /// Construct an HCCL-variant ``UniqueId`` from raw 4108-byte buffer.
    pub fn from_hccl_internal(internal: [::std::os::raw::c_char; 4108]) -> Self {
        Self::Hccl(HcclRootInfo { internal })
    }

    /// Borrow the inner ``HcclRootInfo``; panics if this is an NCCL variant.
    pub fn as_hccl_root_info(&self) -> &HcclRootInfo {
        match self {
            Self::Hccl(inner) => inner,
            Self::Nccl(_) => panic!("UniqueId::as_hccl_root_info() called on NCCL variant"),
        }
    }

    /// Consume and return the inner ``HcclRootInfo``; panics on NCCL.
    pub fn into_hccl_root_info(self) -> HcclRootInfo {
        match self {
            Self::Hccl(inner) => inner,
            Self::Nccl(_) => panic!("UniqueId::into_hccl_root_info() called on NCCL variant"),
        }
    }

    // --- Discriminator helpers ---

    pub fn is_nccl(&self) -> bool {
        matches!(self, Self::Nccl(_))
    }

    pub fn is_hccl(&self) -> bool {
        matches!(self, Self::Hccl(_))
    }
}
