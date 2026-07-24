/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// RDMA requires frequent unsafe code blocks
#![allow(clippy::undocumented_unsafe_blocks)]

use local_memory::KeepaliveLocalMemory;
use serde::Deserialize;
use serde::Serialize;

#[macro_use]
mod macros;

mod action;
pub mod backend;
#[cfg(not(feature = "hixl"))]
pub mod config;
#[cfg(not(feature = "hixl"))]
pub mod device_selection;
#[cfg(not(feature = "hixl"))]
pub mod efa;
mod errors;
pub mod local_memory;
mod rdma_components;
mod rdma_manager_actor;
mod rdma_manager_owner;
mod rdma_runtime;

#[cfg(not(feature = "hixl"))]
pub use backend::ibverbs::primitives::*;

/// Whether any RDMA backend is available on this system.
///
/// Returns true if ibverbs hardware is present, or if TCP fallback
/// is enabled via [`config::RDMA_ALLOW_TCP_FALLBACK`].
#[cfg(not(feature = "hixl"))]
pub fn rdma_supported() -> bool {
    ibverbs_supported() || hyperactor_config::global::get(config::RDMA_ALLOW_TCP_FALLBACK)
}
pub use action::RdmaAction;
// Re-export the CUDA segment scanner API for the extension/test crates to
// install a process-wide scanner (see `backend::ibverbs::mlx_domain`).
#[cfg(not(feature = "hixl"))]
pub use backend::ibverbs::mlx_domain::CudaSegmentScanner;
#[cfg(not(feature = "hixl"))]
pub use backend::ibverbs::mlx_domain::ScannedSegment;
#[cfg(not(feature = "hixl"))]
pub use backend::ibverbs::mlx_domain::register_cuda_segment_scanner;
pub use errors::RdmaInitError;
pub use rdma_components::RdmaRemoteBuffer;
pub use rdma_components::*;
pub use rdma_manager_actor::*;
pub use rdma_manager_owner::*;
// Re-export rdmaxcel_sys for extension crate to access types
#[cfg(not(feature = "hixl"))]
pub use rdmaxcel_sys;
#[cfg(not(feature = "hixl"))]
pub use test_utils::is_cuda_available;

/// Type of RDMA operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RdmaOpType {
    ReadIntoLocal,
    WriteFromLocal,
}

/// A single RDMA operation to be submitted to a backend.
#[derive(Debug)]
pub struct RdmaOp {
    pub op_type: RdmaOpType,
    pub local: KeepaliveLocalMemory,
    pub remote: RdmaRemoteBuffer,
}

/// Transport level for single-sided communication, ordered slowest to fastest.
///
/// HiXL reports `Nic` for RoCE and `Hccs` for the Ascend intra-supernode
/// interconnect.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RdmaTransportLevel {
    /// TCP/IP sockets (fallback transport).
    Tcp,
    /// RDMA NIC (RoCE, InfiniBand, EFA) — inter-supernode on NPU.
    Nic,
    /// HCCS interconnect (Ascend NPU intra-supernode). Higher bandwidth and
    /// lower latency than NIC, supports both collective and single-sided ops.
    Hccs,
    /// Direct memory access (NVLink, shared memory).
    Memory,
}

#[cfg(not(feature = "hixl"))]
pub fn print_device_info_if_debug_enabled(context: *mut rdmaxcel_sys::ibv_context) {
    if std::env::var("MONARCH_DEBUG_RDMA").is_ok() {
        unsafe {
            rdmaxcel_sys::rdmaxcel_print_device_info(context);
        }
    }
}

#[cfg(not(feature = "hixl"))]
pub fn print_device_info(context: *mut rdmaxcel_sys::ibv_context) {
    unsafe {
        rdmaxcel_sys::rdmaxcel_print_device_info(context);
    }
}

#[cfg(not(feature = "hixl"))]
mod test_utils;
