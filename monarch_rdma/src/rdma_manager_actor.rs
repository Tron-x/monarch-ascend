/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! # RDMA Manager Actor
//!
//! Per-proc singleton service actor that owns RDMA buffer registrations and
//! delegates transport-specific work to the configured backend registry.
//!
//! ## Responsibilities
//!
//! - Assigns a unique `remote_buf_id` to each registered local memory handle
//!   and stores the [`KeepaliveLocalMemory`] for later retrieval.
//! - Produces [`RdmaRemoteBuffer`] tokens that can be sent to remote peers so
//!   they can address this buffer over RDMA.
//! - Delegates MR registration, QP management, and data movement to a NIC
//!   backend when available, or falls back to the TCP backend
//!   ([`TcpManagerActor`]). HiXL builds expose only the HiXL backend.
//! - Handles remote [`ReleaseBuffer`] requests to clean up registrations.
//!
//! ## Service topology and readiness
//!
//! `spawn_service("rdma_manager", ...)` creates or reuses one
//! `RdmaManagerActor` on each target proc and materializes an `ActorMesh` view
//! over them. Overlapping proc-mesh views reuse the same actor on shared procs;
//! each view still has its own `ActorMeshController`.
//!
//! The owner casts [`RdmaManagerReady`] to every actor in a view. Actor
//! messages are dispatched only after `init()` succeeds, so the resulting
//! [`ReadyAck`](crate::ReadyAck) certifies that this proc's manager backends
//! are initialized. If initialization fails, no acknowledgement is sent and
//! every controller monitoring that actor reports the failure to the owner.

use std::collections::HashMap;
use std::sync::OnceLock;

use async_trait::async_trait;
use hyperactor::Actor;
use hyperactor::ActorHandle;
use hyperactor::ActorRef;
use hyperactor::Context;
use hyperactor::HandleClient;
use hyperactor::Handler;
use hyperactor::Instance;
use hyperactor::OncePortHandle;
use hyperactor::OncePortRef;
use hyperactor::RefClient;
use hyperactor::RemoteSpawn;
use hyperactor::context;
use hyperactor_config::Flattrs;
use serde::Deserialize;
use serde::Serialize;
use typeuri::Named;

use crate::backend::RdmaBackendHandle;
use crate::backend::RdmaBackends;
use crate::backend::RdmaConfig;
#[cfg(not(feature = "hixl"))]
use crate::backend::ibverbs::primitives::IbvConfig;
#[cfg(not(feature = "hixl"))]
use crate::backend::tcp::manager_actor::TcpManagerActor;
use crate::local_memory::KeepaliveLocalMemory;
use crate::rdma_components::RdmaRemoteBuffer;
use crate::rdma_manager_owner::EntryId;
use crate::rdma_manager_owner::RdmaManagerOwnerActor;
use crate::rdma_manager_owner::RdmaManagerReady;
use crate::rdma_manager_owner::RdmaManagerReadyHandler;
use crate::rdma_manager_owner::ReadyAckClient;

/// Helper function to get detailed error messages from RDMAXCEL error codes.
#[cfg(not(feature = "hixl"))]
pub fn get_rdmaxcel_error_message(error_code: i32) -> String {
    unsafe {
        let c_str = rdmaxcel_sys::rdmaxcel_error_string(error_code);
        std::ffi::CStr::from_ptr(c_str)
            .to_string_lossy()
            .into_owned()
    }
}

/// Local-only messages for the [`RdmaManagerActor`].
///
/// These messages carry [`KeepaliveLocalMemory`] and are therefore not
/// serializable -- they can only be sent within the same process.
#[derive(Handler, HandleClient, Debug)]
pub enum RdmaManagerMessage {
    /// Register a local memory handle and return a [`RdmaRemoteBuffer`] that
    /// remote peers can use to address this buffer over RDMA.
    RequestBuffer {
        local: KeepaliveLocalMemory,
        #[reply]
        reply: OncePortHandle<RdmaRemoteBuffer>,
    },
    /// Look up the local memory handle for a given `remote_buf_id`. Returns
    /// `None` if the id does not correspond to a registered buffer.
    RequestLocalMemory {
        remote_buf_id: usize,
        #[reply]
        reply: OncePortHandle<Option<KeepaliveLocalMemory>>,
    },
    /// Return in-process handles to all spawned backends, in priority order.
    GetBackendHandles {
        #[reply]
        reply: OncePortHandle<Vec<RdmaBackendHandle>>,
    },
}

/// Serializable release message for wire transport.
///
/// Used by [`RdmaRemoteBuffer::drop_buffer`] to release a buffer
/// from a remote process.
#[derive(Handler, HandleClient, RefClient, Debug, Serialize, Deserialize, Named)]
pub struct ReleaseBuffer {
    pub id: usize,
}
wirevalue::register_type!(ReleaseBuffer);

/// Serializable cross-process message asking the receiver to establish
/// an HIXL connection to the given `peer_engine_id`. HIXL requires
/// bidirectional Connect() before TransferSync can succeed.
#[cfg(feature = "hixl")]
#[derive(Handler, HandleClient, RefClient, Debug, Serialize, Deserialize, Named)]
pub struct EnsurePeerConnected {
    pub peer_engine_id: String,
    #[reply]
    pub reply: OncePortRef<()>,
}
#[cfg(feature = "hixl")]
wirevalue::register_type!(EnsurePeerConnected);
/// Serializable query for resolving the [`TcpManagerActor`] ref
/// from a remote [`RdmaManagerActor`].
#[cfg(not(feature = "hixl"))]
#[derive(Handler, HandleClient, RefClient, Debug, Serialize, Deserialize, Named)]
pub struct GetTcpActorRef {
    #[reply]
    pub reply: OncePortRef<ActorRef<TcpManagerActor>>,
}
#[cfg(not(feature = "hixl"))]
wirevalue::register_type!(GetTcpActorRef);

#[derive(Debug)]
#[cfg_attr(not(feature = "hixl"), hyperactor::export(
    handlers = [
        GetTcpActorRef,
        ReleaseBuffer,
        RdmaManagerReady,
    ],
))]
#[cfg_attr(feature = "hixl", hyperactor::export(
    handlers = [
        EnsurePeerConnected,
        ReleaseBuffer,
        RdmaManagerReady,
    ],
))]
#[hyperactor::spawnable]
pub struct RdmaManagerActor {
    next_remote_buf_id: usize,
    buffers: HashMap<usize, KeepaliveLocalMemory>,
    #[cfg(not(feature = "hixl"))]
    params: Option<IbvConfig>,
    #[cfg(feature = "hixl")]
    params: Option<HixlConfig>,
    backends: OnceLock<RdmaBackends>,
}

impl RdmaManagerActor {
    pub fn local_handle(client: &impl context::Actor) -> ActorHandle<Self> {
        let actor_ref = ActorRef::attest(
            client
                .mailbox()
                .actor_addr()
                .proc_addr()
                .actor_addr("rdma_manager"),
        );
        actor_ref
            .downcast_handle(client)
            .expect("RdmaManagerActor is not in the local process")
    }
}

#[cfg(not(feature = "hixl"))]
#[async_trait]
impl RemoteSpawn for RdmaManagerActor {
    type Params = Option<IbvConfig>;

    async fn new(params: Self::Params, _environment: Flattrs) -> Result<Self, anyhow::Error> {
        Ok(Self {
            next_remote_buf_id: 0,
            buffers: HashMap::new(),
            params,
            backends: OnceLock::new(),
        })
    }
}

#[cfg(not(feature = "hixl"))]
#[async_trait]
impl Actor for RdmaManagerActor {
    async fn init(&mut self, this: &Instance<Self>) -> Result<(), anyhow::Error> {
        // An explicit per-manager target takes precedence over
        // `RDMA_IBVERBS_TARGET`. Otherwise validate the process setting here:
        // `spawn_available` may ignore an ibverbs initialization failure when
        // TCP starts, silently turning a malformed pin into TCP fallback.
        if self
            .params
            .as_ref()
            .and_then(|config| config.target.as_ref())
            .is_none()
        {
            let _ = crate::backend::ibverbs::device_selection::configured_ibverbs_target()?;
        }

        // Spawn every available backend. `spawn_available` bails when none
        // is available (e.g. no NIC and TCP fallback disabled).
        let backends = RdmaBackends::spawn_available(
            this,
            &RdmaConfig {
                ibv: self.params.clone(),
            },
        )
        .await?;
        self.backends.set(backends).expect("backends set once");
        Ok(())
    }

    // This actor is implemented in Rust, but the RDMA registration path may enter
    // Python and take the GIL. Run its loop on the dedicated rdma runtime rather
    // than the shared control-plane runtime; see `crate::rdma_runtime`.
    fn spawn_server_task<F>(future: F) -> tokio::task::JoinHandle<F::Output>
    where
        F: std::future::Future + Send + 'static,
        F::Output: Send + 'static,
    {
        crate::rdma_runtime::spawn_on_rdma_runtime(future)
    }
}

#[cfg(not(feature = "hixl"))]
#[async_trait]
#[hyperactor::handle(GetTcpActorRef)]
impl GetTcpActorRefHandler for RdmaManagerActor {
    async fn get_tcp_actor_ref(
        &mut self,
        _cx: &Context<Self>,
    ) -> Result<ActorRef<TcpManagerActor>, anyhow::Error> {
        self.backends
            .get()
            .expect("backends set in init")
            .handles()
            .into_iter()
            .find_map(|h| match h {
                RdmaBackendHandle::Tcp(backend) => Some(backend.bind()),
                _ => None,
            })
            .ok_or_else(|| anyhow::anyhow!("TCP backend not available"))
    }
}

#[async_trait]
#[hyperactor::handle(ReleaseBuffer)]
impl ReleaseBufferHandler for RdmaManagerActor {
    async fn release_buffer(&mut self, cx: &Context<Self>, id: usize) -> Result<(), anyhow::Error> {
        self.buffers.remove(&id);
        self.backends
            .get()
            .expect("backends set in init")
            .release_all(cx, id)
            .await?;
        Ok(())
    }
}

#[async_trait]
#[hyperactor::handle(RdmaManagerMessage)]
impl RdmaManagerMessageHandler for RdmaManagerActor {
    async fn request_buffer(
        &mut self,
        cx: &Context<Self>,
        local: KeepaliveLocalMemory,
    ) -> Result<RdmaRemoteBuffer, anyhow::Error> {
        let remote_buf_id = self.next_remote_buf_id;
        self.next_remote_buf_id += 1;
        let size = local.size();

        let backends = self
            .backends
            .get()
            .expect("backends set in init")
            .register_all(cx, remote_buf_id, local.clone())
            .await?;
        self.buffers.insert(remote_buf_id, local);

        Ok(RdmaRemoteBuffer {
            id: remote_buf_id,
            size,
            owner: cx.bind().clone(),
            backends,
        })
    }

    async fn request_local_memory(
        &mut self,
        _cx: &Context<Self>,
        remote_buf_id: usize,
    ) -> Result<Option<KeepaliveLocalMemory>, anyhow::Error> {
        Ok(self.buffers.get(&remote_buf_id).cloned())
    }

    async fn get_backend_handles(
        &mut self,
        _cx: &Context<Self>,
    ) -> Result<Vec<RdmaBackendHandle>, anyhow::Error> {
        Ok(self.backends.get().expect("backends set in init").handles())
    }
}

#[async_trait]
#[hyperactor::handle(RdmaManagerReady)]
impl RdmaManagerReadyHandler for RdmaManagerActor {
    async fn rdma_manager_ready(
        &mut self,
        cx: &Context<Self>,
        owner: ActorRef<RdmaManagerOwnerActor>,
        entry: EntryId,
    ) -> Result<(), anyhow::Error> {
        // Runs strictly after `init()` (RMR-1): ack the owner so it can count
        // this rank toward readiness.
        owner.ready_ack(cx, entry).await?;
        Ok(())
    }
}

// ============================================================================
// HIXL backend (Ascend NPU)
// ============================================================================

/// HIXL configuration for the RdmaManagerActor.
#[cfg(feature = "hixl")]
#[derive(Debug, Named, Clone, Serialize, Deserialize, Default)]
pub struct HixlConfig {
    pub engine_id: Option<String>,
    pub device_id: Option<i32>,
}

#[cfg(feature = "hixl")]
pub(crate) fn local_ip_for_hixl() -> String {
    if std::env::var("MONARCH_HIXL_USE_LOOPBACK").is_ok() {
        return "127.0.0.1".to_string();
    }

    if let Ok(hostname) = hostname::get() {
        if let Ok(addrs) =
            std::net::ToSocketAddrs::to_socket_addrs(&format!("{}:0", hostname.to_string_lossy()))
        {
            for addr in addrs {
                if addr.is_ipv4() && !addr.ip().is_loopback() {
                    return addr.ip().to_string();
                }
            }
        }
    }

    if let Ok(output) = std::process::Command::new("hostname").arg("-I").output() {
        if let Ok(ips) = std::str::from_utf8(&output.stdout) {
            if let Some(ip) = ips.split_whitespace().next() {
                if !ip.starts_with("127.") {
                    return ip.to_string();
                }
            }
        }
    }

    tracing::warn!(
        "[hixl] could not resolve real machine IP, falling back to 127.0.0.1 — \
         HCCS may not work"
    );
    "127.0.0.1".to_string()
}
#[cfg(feature = "hixl")]
wirevalue::register_type!(HixlConfig);

#[cfg(feature = "hixl")]
#[async_trait]
impl RemoteSpawn for RdmaManagerActor {
    type Params = Option<HixlConfig>;

    async fn new(params: Self::Params, _environment: Flattrs) -> Result<Self, anyhow::Error> {
        Ok(Self {
            next_remote_buf_id: 0,
            buffers: HashMap::new(),
            params,
            backends: OnceLock::new(),
        })
    }
}

#[cfg(feature = "hixl")]
#[async_trait]
impl Actor for RdmaManagerActor {
    async fn init(&mut self, this: &Instance<Self>) -> Result<(), anyhow::Error> {
        let backends = RdmaBackends::spawn_available(
            this,
            &RdmaConfig {
                hixl: self.params.clone(),
            },
        )
        .await?;
        self.backends.set(backends).expect("backends set once");
        tracing::debug!("RdmaManagerActor initialized with HIXL backend");
        Ok(())
    }

    // HiXL registration can enter Python/ACL runtime code as well, so use
    // the same dedicated RDMA runtime as the upstream NIC manager.
    fn spawn_server_task<F>(future: F) -> tokio::task::JoinHandle<F::Output>
    where
        F: std::future::Future + Send + 'static,
        F::Output: Send + 'static,
    {
        crate::rdma_runtime::spawn_on_rdma_runtime(future)
    }
}

#[cfg(feature = "hixl")]
#[async_trait]
#[hyperactor::handle(EnsurePeerConnected)]
impl EnsurePeerConnectedHandler for RdmaManagerActor {
    async fn ensure_peer_connected(
        &mut self,
        _cx: &Context<Self>,
        peer_engine_id: String,
    ) -> Result<(), anyhow::Error> {
        tracing::info!("RdmaManager: ensure_peer_connected to {}", peer_engine_id);
        let backend = self
            .backends
            .get()
            .expect("backends set in init")
            .handles()
            .into_iter()
            .find_map(|backend| match backend {
                RdmaBackendHandle::Hixl(backend) => Some(backend),
            })
            .ok_or_else(|| anyhow::anyhow!("HiXL backend not available"))?;
        backend.connect(&peer_engine_id)
    }
}
