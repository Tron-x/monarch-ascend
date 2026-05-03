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

use std::sync::Arc;
use std::time::Duration;

use hyperactor as reference;
use hyperactor::ActorRef;
use hyperactor::context;
use serde::Deserialize;
use serde::Serialize;
use typeuri::Named;

use crate::RdmaManagerActor;
use crate::ReleaseBufferClient;
use crate::backend::RdmaRemoteBackendContext;

#[cfg(not(feature = "hixl"))]
use crate::RdmaOp;
#[cfg(not(feature = "hixl"))]
use crate::RdmaOpType;

#[cfg(not(feature = "hixl"))]
use crate::backend::RdmaBackend;
#[cfg(not(feature = "hixl"))]
use crate::backend::ibverbs::IbvBuffer;
#[cfg(not(feature = "hixl"))]
use crate::backend::ibverbs::manager_actor::IbvBackend;
#[cfg(not(feature = "hixl"))]
use crate::backend::ibverbs::manager_actor::IbvManagerActor;
#[cfg(not(feature = "hixl"))]
use crate::backend::ibverbs::manager_actor::IbvManagerMessageClient;
#[cfg(not(feature = "hixl"))]
use crate::backend::tcp::manager_actor::TcpBackend;
#[cfg(not(feature = "hixl"))]
use crate::backend::tcp::manager_actor::TcpManagerActor;
use crate::local_memory::RdmaLocalMemory;

#[cfg(feature = "hixl")]
use crate::backend::hixl::HixlBuffer;

/// Lightweight handle representing a registered RDMA buffer.
#[derive(Debug, Named, Clone, Serialize, Deserialize)]
pub struct RdmaRemoteBuffer {
    pub id: usize,
    pub size: usize,
    pub owner: reference::ActorRef<RdmaManagerActor>,
    pub backends: Vec<RdmaRemoteBackendContext>,
}
wirevalue::register_type!(RdmaRemoteBuffer);

/// Backend handle returned by [`RdmaRemoteBuffer::choose_backend`].
///
/// `RdmaBackend` is not object-safe (associated type + generic parameter
/// on `submit`), so we use an enum that delegates to the concrete handle.
#[cfg(not(feature = "hixl"))]
#[derive(Debug)]
pub enum RdmaLocalBackend {
    Ibv(IbvBackend),
    Tcp(TcpBackend),
}

#[cfg(not(feature = "hixl"))]
impl RdmaLocalBackend {
    async fn submit(
        &mut self,
        cx: &(impl context::Actor + Send + Sync),
        ops: Vec<RdmaOp>,
        timeout: Duration,
    ) -> Result<(), anyhow::Error> {
        match self {
            RdmaLocalBackend::Ibv(h) => h.submit(cx, ops, timeout).await,
            RdmaLocalBackend::Tcp(h) => h.submit(cx, ops, timeout).await,
        }
    }
}

impl RdmaRemoteBuffer {
    // ----------------------------------------------------------------
    // GPU: ibverbs/tcp path
    // ----------------------------------------------------------------

    /// Choose the best available backend for this buffer.
    ///
    /// Prefers ibverbs when both the local and remote sides support it.
    /// Falls back to TCP when ibverbs is unavailable and
    /// [`RDMA_ALLOW_TCP_FALLBACK`](crate::config::RDMA_ALLOW_TCP_FALLBACK)
    /// is enabled.
    #[cfg(not(feature = "hixl"))]
    pub async fn choose_backend(
        &self,
        client: &(impl context::Actor + Send + Sync),
    ) -> Result<RdmaLocalBackend, anyhow::Error> {
        if self.has_ibverbs_backend() {
            if let Ok(ibv_handle) = IbvManagerActor::local_handle(client).await {
                return Ok(RdmaLocalBackend::Ibv(IbvBackend(ibv_handle)));
            }

            return self
                .tcp_fallback_or_bail("no ibverbs backend on the local side", client)
                .await;
        }

        self.tcp_fallback_or_bail(
            &format!(
                "no ibverbs backend on the remote side (owner={})",
                self.owner.actor_id()
            ),
            client,
        )
        .await
    }

    /// Push data from local memory into this remote buffer (local->remote).
    #[cfg(not(feature = "hixl"))]
    pub async fn write_from_local(
        &self,
        client: &(impl context::Actor + Send + Sync),
        local: Arc<dyn RdmaLocalMemory>,
        timeout: u64,
    ) -> Result<bool, anyhow::Error> {
        let mut backend = self.choose_backend(client).await?;
        backend
            .submit(
                client,
                vec![RdmaOp {
                    op_type: RdmaOpType::WriteFromLocal,
                    local,
                    remote: self.clone(),
                }],
                Duration::from_secs(timeout),
            )
            .await?;
        Ok(true)
    }

    /// Pull data from this remote buffer into local memory (remote->local).
    #[cfg(not(feature = "hixl"))]
    pub async fn read_into_local(
        &self,
        client: &(impl context::Actor + Send + Sync),
        local: Arc<dyn RdmaLocalMemory>,
        timeout: u64,
    ) -> Result<bool, anyhow::Error> {
        let mut backend = self.choose_backend(client).await?;
        backend
            .submit(
                client,
                vec![RdmaOp {
                    op_type: RdmaOpType::ReadIntoLocal,
                    local,
                    remote: self.clone(),
                }],
                Duration::from_secs(timeout),
            )
            .await?;
        Ok(true)
    }

    /// Get a TCP backend handle, or bail if TCP fallback is disabled.
    #[cfg(not(feature = "hixl"))]
    async fn tcp_fallback_or_bail(
        &self,
        reason: &str,
        client: &(impl context::Actor + Send + Sync),
    ) -> Result<RdmaLocalBackend, anyhow::Error> {
        if !hyperactor_config::global::get(crate::config::RDMA_ALLOW_TCP_FALLBACK) {
            anyhow::bail!(
                "{reason}, and TCP fallback is disabled; \
                 enable it with monarch.configure(rdma_allow_tcp_fallback=True)"
            );
        }

        tracing::warn!("falling back to TCP transport ({reason})");

        let tcp_handle = TcpManagerActor::local_handle(client).await?;
        Ok(RdmaLocalBackend::Tcp(TcpBackend(tcp_handle)))
    }

    // ----------------------------------------------------------------
    // NPU: HIXL path (transfers via hixl-sys)
    // ----------------------------------------------------------------

    #[cfg(feature = "hixl")]
    pub async fn write_from_local(
        &self,
        client: &(impl context::Actor + Send + Sync),
        local: Arc<dyn RdmaLocalMemory>,
        timeout: u64,
    ) -> Result<bool, anyhow::Error> {
        let hixl_buf = self.resolve_hixl()?;
        let timeout_ms = timeout.min(i32::MAX as u64) as i32;

        crate::backend::hixl::manager_actor::register_mem_if_needed(
            local.addr(),
            local.size(),
        )?;

        crate::backend::hixl::manager_actor::ensure_connected(
            client,
            &self.owner,
            &hixl_buf.engine_id,
        )
        .await?;

        crate::backend::hixl::manager_actor::with_state(|state| {
            state
                .engine
                .transfer_write(
                    &hixl_buf.engine_id,
                    local.addr(),
                    hixl_buf.addr,
                    local.size(),
                    timeout_ms,
                )
                .map_err(|ret| {
                    anyhow::anyhow!(
                        "hixl_transfer_write failed: local={:#x} remote={:#x}@{} len={} ret={}",
                        local.addr(),
                        hixl_buf.addr,
                        hixl_buf.engine_id,
                        local.size(),
                        ret,
                    )
                })
        })?;

        Ok(true)
    }

    #[cfg(feature = "hixl")]
    pub async fn read_into_local(
        &self,
        client: &(impl context::Actor + Send + Sync),
        local: Arc<dyn RdmaLocalMemory>,
        timeout: u64,
    ) -> Result<bool, anyhow::Error> {
        let hixl_buf = self.resolve_hixl()?;
        let timeout_ms = timeout.min(i32::MAX as u64) as i32;

        crate::backend::hixl::manager_actor::register_mem_if_needed(
            local.addr(),
            local.size(),
        )?;

        crate::backend::hixl::manager_actor::ensure_connected(
            client,
            &self.owner,
            &hixl_buf.engine_id,
        )
        .await?;

        for attempt in 0..3u32 {
            let result = crate::backend::hixl::manager_actor::with_state(|state| {
                state.engine.transfer_read(
                    &hixl_buf.engine_id,
                    local.addr(),
                    hixl_buf.addr,
                    local.size(),
                    timeout_ms,
                ).map_err(|ret| anyhow::anyhow!(
                    "hixl_transfer_read failed: local={:#x} remote={:#x}@{} len={} ret={}",
                    local.addr(), hixl_buf.addr, hixl_buf.engine_id, local.size(), ret,
                ))
            });
            match result {
                Ok(()) => return Ok(true),
                Err(_) if attempt < 2 => {
                    tracing::warn!(
                        "[hixl] transfer_read attempt {} failed, retrying in 1s…",
                        attempt,
                    );
                    tokio::time::sleep(std::time::Duration::from_secs(1)).await;
                }
                Err(e) => return Err(e),
            }
        }
        unreachable!()
    }

    // ----------------------------------------------------------------
    // Common
    // ----------------------------------------------------------------

    /// Drop the buffer and release remote handles.
    pub async fn drop_buffer(&self, client: &impl context::Actor) -> Result<(), anyhow::Error> {
        tracing::debug!("[buffer] dropping buffer id={}", self.id);
        self.owner.release_buffer(client, self.id).await?;
        Ok(())
    }

    /// Whether this buffer has an ibverbs backend context.
    #[cfg(not(feature = "hixl"))]
    fn has_ibverbs_backend(&self) -> bool {
        self.backends
            .iter()
            .any(|b| matches!(b, RdmaRemoteBackendContext::Ibverbs(..)))
    }

    /// Resolve the ibverbs backend context for this buffer.
    ///
    /// Returns `None` if the buffer has no ibverbs backend context (i.e.,
    /// the remote side was created without ibverbs). Returns `Some(Err(...))`
    /// if the context exists but lazy MR resolution fails. Returns
    /// `Some(Ok(...))` on success.
    #[cfg(not(feature = "hixl"))]
    pub async fn resolve_ibv(
        &self,
        client: &impl context::Actor,
    ) -> Option<Result<(reference::ActorRef<IbvManagerActor>, IbvBuffer), anyhow::Error>> {
        let (remote_ibv_mgr, remote_ibv_buf): (
            &reference::ActorRef<IbvManagerActor>,
            &Arc<tokio::sync::OnceCell<IbvBuffer>>,
        ) = self.backends.iter().find_map(|b| match b {
            RdmaRemoteBackendContext::Ibverbs(mgr, buf) => Some((mgr, buf)),
            _ => None,
        })?;

        Some(
            remote_ibv_buf
                .get_or_try_init(async {
                    remote_ibv_mgr
                        .request_buffer(client, self.id)
                        .await?
                        .ok_or_else(|| anyhow::anyhow!("buffer {} not found", self.id))
                })
                .await
                .cloned()
                .map(|buf| (remote_ibv_mgr.clone(), buf)),
        )
    }

    /// Extract the TCP backend context from this buffer.
    ///
    /// Unlike [`resolve_ibv`], no lazy initialization is needed -- the
    /// TCP backend only needs the remote actor ref and the buffer id.
    #[cfg(not(feature = "hixl"))]
    pub fn resolve_tcp(&self) -> Result<(ActorRef<TcpManagerActor>, usize), anyhow::Error> {
        self.backends
            .iter()
            .find_map(|b| match b {
                RdmaRemoteBackendContext::Tcp(tcp_ref) => Some((tcp_ref.clone(), self.id)),
                _ => None,
            })
            .ok_or_else(|| anyhow::anyhow!("tcp backend not found for buffer: {:?}", self))
    }

    /// Extract the [`HixlBuffer`] from the backend context (NPU path).
    #[cfg(feature = "hixl")]
    pub fn resolve_hixl(&self) -> Result<HixlBuffer, anyhow::Error> {
        self.backends
            .first()
            .map(|ctx| {
                let RdmaRemoteBackendContext::Hixl(buf) = ctx;
                buf.clone()
            })
            .ok_or_else(|| {
                anyhow::anyhow!("HIXL backend not found for buffer: {:?}", self)
            })
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
