/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! # RDMA Components
//!
//! Core RDMA building blocks for establishing and managing RDMA connections.
//! Supports ibverbs (GPU) and HIXL (NPU) backends via feature flags.

use std::result::Result;
use std::time::Duration;

use hyperactor::ActorRef;
use hyperactor::context;
use serde::Deserialize;
use serde::Serialize;
use typeuri::Named;

use crate::RdmaAction;
use crate::RdmaManagerActor;
use crate::ReleaseBufferClient;
use crate::backend::RdmaRemoteBackends;
use crate::local_memory::KeepaliveLocalMemory;

/// Lightweight handle representing a registered RDMA buffer.
#[derive(Debug, Named, Clone, Serialize, Deserialize)]
pub struct RdmaRemoteBuffer {
    pub id: usize,
    pub size: usize,
    pub owner: ActorRef<RdmaManagerActor>,
    pub(crate) backends: RdmaRemoteBackends,
}
wirevalue::register_type!(RdmaRemoteBuffer);

impl RdmaRemoteBuffer {
    /// Push data from local memory into this remote buffer (local->remote).
    pub async fn write_from_local(
        &self,
        client: &(impl context::Actor + Send + Sync),
        local: KeepaliveLocalMemory,
        timeout: u64,
    ) -> Result<bool, anyhow::Error> {
        let mut action = RdmaAction::new();
        action.add_write_from_local(self.clone(), local)?;
        action.submit(client, Duration::from_secs(timeout)).await?;
        Ok(true)
    }

    /// Pull data from this remote buffer into local memory (remote->local).
    pub async fn read_into_local(
        &self,
        client: &(impl context::Actor + Send + Sync),
        local: KeepaliveLocalMemory,
        timeout: u64,
    ) -> Result<bool, anyhow::Error> {
        let mut action = RdmaAction::new();
        action.add_read_into_local(self.clone(), local)?;
        action.submit(client, Duration::from_secs(timeout)).await?;
        Ok(true)
    }

    /// Drop the buffer and release remote handles.
    pub async fn drop_buffer(&self, client: &impl context::Actor) -> Result<(), anyhow::Error> {
        tracing::debug!("[buffer] dropping buffer id={}", self.id);
        self.owner.release_buffer(client, self.id).await?;
        Ok(())
    }
}

/// Utility to validate CUDA execution context (GPU only).
#[cfg(not(feature = "hixl"))]
pub async fn validate_execution_context() -> Result<(), anyhow::Error> {
    use std::fs;
    match fs::read_to_string("/proc/modules") {
        Ok(contents) => {
            if !contents.contains("nvidia_peermem") {
                return Err(anyhow::anyhow!(
                    "nvidia_peermem module not found in /proc/modules"
                ));
            }
        }
        Err(e) => {
            return Err(anyhow::anyhow!(e));
        }
    }
    match fs::read_to_string("/proc/driver/nvidia/params") {
        Ok(contents) => {
            if !contents.contains("PeerMappingOverride=1") {
                return Err(anyhow::anyhow!(
                    "PeerMappingOverride=1 not found in /proc/driver/nvidia/params"
                ));
            }
        }
        Err(e) => {
            return Err(anyhow::anyhow!(e));
        }
    }
    Ok(())
}

/// Get all segments that have been registered with MRs for the given PD.
///
/// Each protection domain maintains independent segment registrations, so
/// callers must pass the PD whose lkeys they intend to use.
#[cfg(not(feature = "hixl"))]
pub fn get_registered_cuda_segments(
    pd: *mut rdmaxcel_sys::ibv_pd,
) -> Vec<rdmaxcel_sys::rdma_segment_info_t> {
    unsafe {
        let segment_count = rdmaxcel_sys::rdma_get_active_segment_count(pd);
        if segment_count <= 0 {
            return Vec::new();
        }

        let mut segments = vec![
            std::mem::MaybeUninit::<rdmaxcel_sys::rdma_segment_info_t>::zeroed()
                .assume_init();
            segment_count as usize
        ];
        let actual_count = rdmaxcel_sys::rdma_get_all_registered_segment_info(
            pd,
            segments.as_mut_ptr(),
            segment_count,
        );

        if actual_count > 0 {
            segments.truncate(actual_count as usize);
            segments
        } else {
            Vec::new()
        }
    }
}

/// Segment scanner callback type alias.
#[cfg(not(feature = "hixl"))]
pub type SegmentScannerFn = rdmaxcel_sys::RdmaxcelSegmentScannerFn;
#[cfg(feature = "hixl")]
pub type SegmentScannerFn = Option<unsafe extern "C" fn(*mut std::ffi::c_void, i32) -> i32>;

/// Register a segment scanner callback.
#[cfg(not(feature = "hixl"))]
pub fn register_segment_scanner(scanner: SegmentScannerFn) {
    unsafe { rdmaxcel_sys::rdmaxcel_register_segment_scanner(scanner) }
}
#[cfg(feature = "hixl")]
pub fn register_segment_scanner(_scanner: SegmentScannerFn) {}
