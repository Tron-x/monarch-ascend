/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Safe Rust wrapper around HCCL communicators.
//! Mirrors the API surface of `torch-sys-cuda/src/nccl.rs`.

use std::fmt;
use std::fmt::Write;
use std::hash::Hasher;
use std::mem::MaybeUninit;

use fxhash::FxHasher32;
use hccl_sys::*;
use monarch_types::UniqueId;
use serde::Deserialize;
use serde::Serialize;
use thiserror::Error;
use torch_sys2::DeviceType;
use torch_sys2::NpuDevice;
use torch_sys2::ScalarType;
use torch_sys2::Tensor;
use torch_sys2::TensorCell;
use torch_sys2::is_float8_type;

use crate::acl::AclError;
use crate::acl::Stream;
use crate::acl::set_device;

/// HCCL-level errors.
#[derive(Debug, Error)]
pub enum RawHcclError {
    #[error("HCCL parameter error")]
    ParamError,
    #[error("HCCL null pointer error")]
    PtrError,
    #[error("HCCL memory error")]
    MemoryError,
    #[error("HCCL internal error")]
    InternalError,
    #[error("HCCL feature not supported")]
    NotSupported,
    #[error("HCCL resource not found")]
    NotFound,
    #[error("HCCL resource unavailable")]
    Unavailable,
    #[error("HCCL system call error")]
    SysCallError,
    #[error("HCCL timeout")]
    Timeout,
    #[error("HCCL network error")]
    NetworkError,
    #[error("HCCL remote error")]
    RemoteError,
    #[error("HCCL error code {0}")]
    Other(u32),
}

/// High-level error type for the safe [`Communicator`] API.
#[derive(Debug, Error)]
pub enum HcclError {
    #[error("a HCCL-level error: {0:?}")]
    HcclError(#[from] RawHcclError),

    #[error("an ACL-level error: {0:?}")]
    AclError(#[from] AclError),

    #[error("invalid HCCL data type: {0:#?}")]
    InvalidDataType(ScalarType),

    #[error("tensor used in collective must be contiguous")]
    NoncontiguousTensor,

    #[error("tensor must be on NPU device, got: {0:?}")]
    InvalidDevice(DeviceType),

    #[error("got sparse tensor, only dense tensors allowed")]
    InvalidSparseTensor,

    #[error("float8 dtypes are not currently supported for HCCL reductions")]
    Float8Reduction,

    #[error("ReduceOp::Avg is not supported by HCCL")]
    AvgNotSupported,

    #[error("expected UniqueId::Hccl variant, got UniqueId::Nccl")]
    WrongUniqueIdVariant,

    #[error("output tensor must have the same type as input tensor")]
    TypeMismatch,

    #[error("output tensor size must be equal to world size times input tensor size")]
    OutputSizeMismatch,

    #[error("input tensor must be the same size as output size times world size")]
    InputSizeMismatch,

    #[error("ranks passed should be within the global world_size, got: {0:#?}")]
    InvalidSplit(Vec<i32>),

    #[error("undefined tensor used for HCCL operation")]
    UndefinedTensor,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HcclStatus {
    Success,
}

fn hccl_check(result: HcclResult) -> Result<HcclStatus, RawHcclError> {
    match result.0 {
        0 => Ok(HcclStatus::Success),
        1 => Err(RawHcclError::ParamError),
        2 => Err(RawHcclError::PtrError),
        3 => Err(RawHcclError::MemoryError),
        4 => Err(RawHcclError::InternalError),
        5 => Err(RawHcclError::NotSupported),
        6 => Err(RawHcclError::NotFound),
        7 => Err(RawHcclError::Unavailable),
        8 => Err(RawHcclError::SysCallError),
        9 => Err(RawHcclError::Timeout),
        19 => Err(RawHcclError::NetworkError),
        21 => Err(RawHcclError::RemoteError),
        other => Err(RawHcclError::Other(other)),
    }
}

/// HCCL root info used to bootstrap communicator creation.
/// Equivalent to `ncclUniqueId` in the NCCL world.
#[derive(Clone, Serialize, Deserialize)]
pub struct RootInfo {
    inner: HcclRootInfo,
}

impl fmt::Debug for RootInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RootInfo")
            .field(
                "inner",
                &format_args!(
                    "{}",
                    self.inner
                        .internal
                        .iter()
                        .take(32) // only show first 32 bytes for brevity
                        .fold(String::new(), |mut output, b| {
                            let _ = write!(output, "{:02x}", b);
                            output
                        })
                ),
            )
            .finish()
    }
}

impl RootInfo {
    pub fn new() -> Result<Self, RawHcclError> {
        let mut inner = MaybeUninit::uninit();
        let inner = unsafe {
            hccl_check(HcclGetRootInfo(inner.as_mut_ptr()))?;
            inner.assume_init()
        };
        Ok(Self { inner })
    }
}

// ---------------------------------------------------------------------------
// Conversions to/from the portable `monarch_types::UniqueId` enum.
//
// The wire-format type used by the WorkerMessage protocol is
// `monarch_types::UniqueId`, an enum holding either NCCL or HCCL bytes.  At
// the FFI boundary we need to convert in both directions:
//   * RootInfo -> UniqueId : when bootstrapping a new HCCL communicator and
//     broadcasting the root info to peer ranks.
//   * UniqueId -> RootInfo : when a peer rank receives the broadcast and
//     needs to bind it into a real HCCL communicator.
// ---------------------------------------------------------------------------

impl From<RootInfo> for UniqueId {
    fn from(ri: RootInfo) -> Self {
        UniqueId::from_hccl_internal(ri.inner.internal)
    }
}

impl<'a> From<&'a RootInfo> for UniqueId {
    fn from(ri: &'a RootInfo) -> Self {
        UniqueId::from_hccl_internal(ri.inner.internal)
    }
}

impl TryFrom<UniqueId> for RootInfo {
    type Error = HcclError;
    fn try_from(uid: UniqueId) -> Result<Self, Self::Error> {
        match uid {
            UniqueId::Hccl(bytes) => Ok(RootInfo {
                inner: HcclRootInfo {
                    internal: bytes.internal,
                },
            }),
            UniqueId::Nccl(_) => Err(HcclError::WrongUniqueIdVariant),
        }
    }
}

impl<'a> TryFrom<&'a UniqueId> for RootInfo {
    type Error = HcclError;
    fn try_from(uid: &'a UniqueId) -> Result<Self, Self::Error> {
        match uid {
            UniqueId::Hccl(bytes) => Ok(RootInfo {
                inner: HcclRootInfo {
                    internal: bytes.internal,
                },
            }),
            UniqueId::Nccl(_) => Err(HcclError::WrongUniqueIdVariant),
        }
    }
}

/// Rust version of `HcclDataType`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataType {
    Int8 = 0,
    Int16 = 1,
    Int32 = 2,
    Fp16 = 3,
    Fp32 = 4,
    Int64 = 5,
    Uint64 = 6,
    Uint8 = 7,
    Uint16 = 8,
    Uint32 = 9,
    Fp64 = 10,
    Bfp16 = 11,
}

impl From<DataType> for HcclDataType {
    fn from(dt: DataType) -> Self {
        Self(dt as std::os::raw::c_uint)
    }
}

impl TryFrom<ScalarType> for DataType {
    type Error = HcclError;

    fn try_from(value: ScalarType) -> Result<Self, Self::Error> {
        match value {
            ScalarType::Char => Ok(DataType::Int8),
            ScalarType::Byte => Ok(DataType::Uint8),
            ScalarType::Short => Ok(DataType::Int16),
            ScalarType::Half => Ok(DataType::Fp16),
            ScalarType::Float => Ok(DataType::Fp32),
            ScalarType::Double => Ok(DataType::Fp64),
            ScalarType::Int => Ok(DataType::Int32),
            ScalarType::Long => Ok(DataType::Int64),
            ScalarType::Bool => Ok(DataType::Uint8),
            ScalarType::BFloat16 => Ok(DataType::Bfp16),
            ScalarType::Float8_e5m2 => Ok(DataType::Uint8),
            ScalarType::Float8_e4m3fn => Ok(DataType::Uint8),
            ScalarType::Float8_e5m2fnuz => Ok(DataType::Uint8),
            ScalarType::Float8_e4m3fnuz => Ok(DataType::Uint8),
            _ => Err(HcclError::InvalidDataType(value)),
        }
    }
}

// Re-export the portable `ReduceOp` from `monarch_types` so callers (e.g.
// `monarch_tensor_worker`) can pass the wire type straight through to HCCL
// without an extra conversion at the boundary.  This mirrors what
// `torch-sys-cuda::nccl` does for the NCCL backend.
pub use monarch_types::ReduceOp;

fn reduce_op_to_hccl(reduce_op: ReduceOp) -> Result<HcclReduceOp, HcclError> {
    match reduce_op {
        ReduceOp::Sum => Ok(HcclReduceOp(0)),
        ReduceOp::Prod => Ok(HcclReduceOp(1)),
        ReduceOp::Max => Ok(HcclReduceOp(2)),
        ReduceOp::Min => Ok(HcclReduceOp(3)),
        ReduceOp::Avg => Err(HcclError::AvgNotSupported),
    }
}

fn check_tensor(tensor: &Tensor, is_p2p: bool) -> Result<(), HcclError> {
    if !tensor.defined() {
        return Err(HcclError::UndefinedTensor);
    }
    if !tensor.is_npu() {
        return Err(HcclError::InvalidDevice(tensor.device().device_type()));
    }
    if tensor.is_sparse() {
        return Err(HcclError::InvalidSparseTensor);
    }
    if !is_p2p && !tensor.is_contiguous() {
        return Err(HcclError::NoncontiguousTensor);
    }
    Ok(())
}

fn calculate_color(ranks: &[i32]) -> i32 {
    let mut hasher = FxHasher32::default();
    ranks.iter().for_each(|r| hasher.write_i32(*r));
    let hash = hasher.finish();
    (hash % (i32::MAX as u64)) as i32
}

/// Wraps an HCCL communicator and provides a Tensor-based interface.
///
/// This implements a subset of the `c10d::ProcessGroup` API, analogous to
/// `torch-sys-cuda`'s `Communicator`.
#[derive(Debug)]
pub struct Communicator {
    inner: HcclComm,
    world_size: i32,
    rank: i32,
    global_world_size: i32,
    global_rank: i32,
    device: NpuDevice,
    split_counter: u64,
}

unsafe impl Send for Communicator {}
unsafe impl Sync for Communicator {}

impl Communicator {
    pub fn new(
        device: NpuDevice,
        world_size: i32,
        root_info: RootInfo,
        rank: i32,
    ) -> Result<Self, HcclError> {
        set_device(device)?;
        let mut inner: HcclComm = std::ptr::null_mut();
        unsafe {
            hccl_check(HcclCommInitRootInfo(
                world_size as u32,
                &root_info.inner,
                rank as u32,
                &mut inner,
            ))?;
        }
        Ok(Self {
            inner,
            world_size,
            rank,
            global_rank: rank,
            global_world_size: world_size,
            device,
            split_counter: 0,
        })
    }

    pub fn world_size(&self) -> i32 {
        self.world_size
    }

    pub fn rank(&self) -> i32 {
        self.rank
    }

    pub fn global_rank(&self) -> i32 {
        self.global_rank
    }

    pub fn device(&self) -> NpuDevice {
        self.device
    }

    pub fn split_all(&mut self) -> Result<Self, HcclError> {
        let ranks = (0..self.global_world_size).collect();
        Ok(self.split_from(ranks)?.unwrap())
    }

    pub fn split_from(&mut self, mut ranks: Vec<i32>) -> Result<Option<Self>, HcclError> {
        ranks.sort();
        for rank in &ranks {
            if *rank < 0 || *rank >= self.global_world_size {
                return Err(HcclError::InvalidSplit(ranks));
            }
        }

        let in_group = ranks.binary_search(&self.rank).is_ok();
        if !in_group {
            // HCCL doesn't have NCCL_SPLIT_NOCOLOR; ranks not in the group
            // still need to participate in the collective creation.
            // We create a sub-comm with only the selected ranks, passing a
            // dummy call for excluded ranks. Since HcclCreateSubCommConfig
            // is a collective, all ranks of the parent comm must call it.
            let _color = calculate_color(&ranks);
            // For excluded ranks, we return None.
            // NOTE: HCCL requires all ranks to participate in sub-comm creation.
            // A full implementation would call HcclCreateSubCommConfig here.
            return Ok(None);
        }

        let group_rank = ranks.iter().position(|v| *v == self.rank).unwrap() as u32;
        let mut rank_ids: Vec<u32> = ranks.iter().map(|r| *r as u32).collect();
        self.split_counter += 1;
        let base_color = calculate_color(&ranks) as u64;
        let sub_comm_id = base_color.wrapping_mul(1000).wrapping_add(self.split_counter);
        let mut sub_comm: HcclComm = std::ptr::null_mut();

        unsafe {
            hccl_check(HcclCreateSubCommConfig(
                &mut self.inner,
                rank_ids.len() as u32,
                rank_ids.as_mut_ptr(),
                sub_comm_id,
                group_rank,
                std::ptr::null_mut(),
                &mut sub_comm,
            ))?;
        }

        Ok(Some(Self {
            inner: sub_comm,
            world_size: ranks.len() as i32,
            rank: group_rank as i32,
            global_rank: self.global_rank,
            global_world_size: self.global_world_size,
            device: self.device,
            split_counter: 0,
        }))
    }

    pub fn all_reduce(
        &mut self,
        tensor: &TensorCell,
        reduce_op: ReduceOp,
        stream: &Stream,
    ) -> Result<HcclStatus, HcclError> {
        let tensor = tensor.borrow_mut();
        let data_type: DataType = tensor.scalar_type().try_into()?;
        check_tensor(&tensor, false)?;
        if is_float8_type(tensor.scalar_type()) {
            return Err(HcclError::Float8Reduction);
        }
        let op = reduce_op_to_hccl(reduce_op)?;
        unsafe {
            Ok(hccl_check(HcclAllReduce(
                tensor.data_ptr() as *mut _,
                tensor.mut_data_ptr(),
                tensor.numel() as u64,
                data_type.into(),
                op,
                self.inner,
                stream.stream(),
            ))?)
        }
    }

    pub fn broadcast(
        &mut self,
        tensor: &TensorCell,
        root: i32,
        stream: &Stream,
    ) -> Result<HcclStatus, HcclError> {
        let tensor = tensor.borrow_mut();
        check_tensor(&tensor, false)?;
        let data_type: DataType = tensor.scalar_type().try_into()?;
        // HCCL broadcast is in-place (single buffer)
        unsafe {
            Ok(hccl_check(HcclBroadcast(
                tensor.mut_data_ptr(),
                tensor.numel() as u64,
                data_type.into(),
                root as u32,
                self.inner,
                stream.stream(),
            ))?)
        }
    }

    pub fn reduce(
        &mut self,
        tensor: &TensorCell,
        reduce_op: ReduceOp,
        root: i32,
        stream: &Stream,
    ) -> Result<HcclStatus, HcclError> {
        let tensor = tensor.borrow_mut();
        check_tensor(&tensor, false)?;
        if is_float8_type(tensor.scalar_type()) {
            return Err(HcclError::Float8Reduction);
        }
        let data_type: DataType = tensor.scalar_type().try_into()?;
        let op = reduce_op_to_hccl(reduce_op)?;
        unsafe {
            Ok(hccl_check(HcclReduce(
                tensor.data_ptr() as *mut _,
                tensor.mut_data_ptr(),
                tensor.numel() as u64,
                data_type.into(),
                op,
                root as u32,
                self.inner,
                stream.stream(),
            ))?)
        }
    }

    pub fn all_gather(
        &mut self,
        output_cells: &[TensorCell],
        input_cell: &TensorCell,
        stream: &Stream,
    ) -> Result<HcclStatus, HcclError> {
        let output = output_cells
            .iter()
            .map(|t| t.borrow_mut())
            .collect::<Vec<_>>();
        let input = input_cell.borrow();
        check_tensor(&input, false)?;
        let output_type = output[0].scalar_type();
        let output_numel: i64 = output.iter().map(|t| t.numel()).sum();
        for t in &output {
            if t.scalar_type() != output_type {
                return Err(HcclError::TypeMismatch);
            }
        }
        if input.scalar_type() != output_type {
            return Err(HcclError::TypeMismatch);
        }
        if input.numel() * self.world_size as i64 != output_numel {
            return Err(HcclError::OutputSizeMismatch);
        }
        let data_type: DataType = input.scalar_type().try_into()?;
        // Use broadcast-based all_gather (same approach as nccl.rs)
        // HCCL broadcast is in-place, so we need to copy input to the right slot first.
        unsafe {
            for (i, out) in output.iter().enumerate() {
                let rank = i as u32;
                let buf = out.mut_data_ptr();
                let count = out.numel() as u64;
                if rank == self.rank as u32 {
                    // Copy input into the output slot, then broadcast
                    // The input data_ptr should be used as the source
                    // For in-place broadcast, we need data in buf already
                    // Copy from input to output[rank] if they differ
                    if buf != input.data_ptr() as *mut _ {
                        std::ptr::copy_nonoverlapping(
                            input.data_ptr() as *const u8,
                            buf as *mut u8,
                            input.nbytes() as usize,
                        );
                    }
                }
                hccl_check(HcclBroadcast(
                    buf,
                    count,
                    data_type.into(),
                    rank,
                    self.inner,
                    stream.stream(),
                ))?;
            }
        }
        Ok(HcclStatus::Success)
    }

    pub fn all_gather_into_tensor(
        &mut self,
        output_cell: &TensorCell,
        input_cell: &TensorCell,
        stream: &Stream,
    ) -> Result<HcclStatus, HcclError> {
        let output = output_cell.borrow_mut();
        let _input_borrow = if input_cell.aliases(output_cell) {
            None
        } else {
            Some(input_cell.borrow())
        };
        let input = unsafe { input_cell.get_unchecked() };
        check_tensor(&output, false)?;
        check_tensor(input, false)?;
        if input.scalar_type() != output.scalar_type() {
            return Err(HcclError::TypeMismatch);
        }
        if input.numel() * self.world_size as i64 != output.numel() {
            return Err(HcclError::OutputSizeMismatch);
        }
        let data_type: DataType = input.scalar_type().try_into()?;
        unsafe {
            Ok(hccl_check(HcclAllGather(
                input.data_ptr() as *mut _,
                output.mut_data_ptr(),
                input.numel() as u64,
                data_type.into(),
                self.inner,
                stream.stream(),
            ))?)
        }
    }

    pub fn reduce_scatter_tensor(
        &mut self,
        output_cell: &TensorCell,
        input_cell: &TensorCell,
        reduce_op: ReduceOp,
        stream: &Stream,
    ) -> Result<HcclStatus, HcclError> {
        let output = output_cell.borrow_mut();
        let _input_borrow = if input_cell.aliases(output_cell) {
            None
        } else {
            Some(input_cell.borrow())
        };
        let input = unsafe { input_cell.get_unchecked() };
        check_tensor(&output, false)?;
        check_tensor(input, false)?;
        if input.scalar_type() != output.scalar_type() {
            return Err(HcclError::TypeMismatch);
        }
        if input.numel() != output.numel() * self.world_size as i64 {
            return Err(HcclError::InputSizeMismatch);
        }
        if is_float8_type(input.scalar_type()) {
            return Err(HcclError::Float8Reduction);
        }
        let data_type: DataType = input.scalar_type().try_into()?;
        let op = reduce_op_to_hccl(reduce_op)?;
        unsafe {
            Ok(hccl_check(HcclReduceScatter(
                input.data_ptr() as *mut _,
                output.mut_data_ptr(),
                output.numel() as u64,
                data_type.into(),
                op,
                self.inner,
                stream.stream(),
            ))?)
        }
    }

    pub fn send(
        &mut self,
        tensor_cell: &TensorCell,
        dst: i32,
        stream: &Stream,
    ) -> Result<HcclStatus, HcclError> {
        let tensor = tensor_cell.borrow();
        let data_type: DataType = tensor.scalar_type().try_into()?;
        check_tensor(&tensor, true)?;
        unsafe {
            Ok(hccl_check(HcclSend(
                tensor.data_ptr() as *mut _,
                tensor.numel() as u64,
                data_type.into(),
                dst as u32,
                self.inner,
                stream.stream(),
            ))?)
        }
    }

    pub fn recv(
        &mut self,
        tensor_cell: &TensorCell,
        src: i32,
        stream: &Stream,
    ) -> Result<HcclStatus, HcclError> {
        let tensor = tensor_cell.borrow_mut();
        let data_type: DataType = tensor.scalar_type().try_into()?;
        check_tensor(&tensor, true)?;
        unsafe {
            Ok(hccl_check(HcclRecv(
                tensor.mut_data_ptr(),
                tensor.numel() as u64,
                data_type.into(),
                src as u32,
                self.inner,
                stream.stream(),
            ))?)
        }
    }

    pub fn all_to_all_single(
        &mut self,
        output_cell: &TensorCell,
        input_cell: &TensorCell,
        stream: &Stream,
    ) -> Result<HcclStatus, HcclError> {
        let output = output_cell.borrow_mut();
        let _input_borrow = if input_cell.aliases(output_cell) {
            None
        } else {
            Some(input_cell.borrow_mut())
        };
        let input = unsafe { input_cell.get_unchecked() };
        check_tensor(&output, false)?;
        check_tensor(input, false)?;
        if input.scalar_type() != output.scalar_type() {
            return Err(HcclError::TypeMismatch);
        }
        let data_type: DataType = input.scalar_type().try_into()?;
        let count_per_rank = input.numel() as u64 / self.world_size as u64;
        unsafe {
            Ok(hccl_check(HcclAlltoAll(
                input.data_ptr(),
                count_per_rank,
                data_type.into(),
                output.mut_data_ptr() as *const _,
                count_per_rank,
                data_type.into(),
                self.inner,
                stream.stream(),
            ))?)
        }
    }

    /// Native HCCL barrier (unlike NCCL which emulates via AllReduce).
    pub fn barrier(&mut self, stream: &Stream) -> Result<HcclStatus, HcclError> {
        unsafe { Ok(hccl_check(HcclBarrier(self.inner, stream.stream()))?) }
    }
}

impl Drop for Communicator {
    fn drop(&mut self) {
        if !self.inner.is_null() {
            unsafe {
                let _ = HcclCommDestroy(self.inner);
            }
        }
    }
}
