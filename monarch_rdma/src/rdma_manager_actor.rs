/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! # RDMA Manager Actor
//!
//! Per-process actor that owns RDMA buffer registrations and delegates
//! transport-specific work to backend actors ([`IbvManagerActor`],
//! [`TcpManagerActor`], and optionally [`HixlManagerActor`]).
//!
//! ## Backend Selection
//!
//! - **ibverbs** (default): Uses [`IbvManagerActor`] for RDMA over InfiniBand/RoCE.
//! - **tcp**: Uses [`TcpManagerActor`] as TCP fallback transport.
//! - **hixl** (feature `hixl`): Uses [`HixlManagerActor`] for HIXL over HCCS/RDMA
//!   on Ascend NPUs.
//!
//! ## Responsibilities
//!
//! - Assigns a unique `remote_buf_id` to each registered local memory handle
//!   and stores the `Arc<dyn RdmaLocalMemory>` for later retrieval.
//! - Produces [`RdmaRemoteBuffer`] tokens that can be sent to remote peers so
//!   they can address this buffer over RDMA.
//! - Delegates MR registration, QP management, and data movement to the
//!   ibverbs backend ([`IbvManagerActor`]) when available, or falls back to
//!   the TCP backend ([`TcpManagerActor`]). When the `hixl` feature is enabled,
//!   delegates to the HIXL backend instead.
//! - Handles remote [`ReleaseBuffer`] requests to clean up registrations.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use hyperactor as reference;
use hyperactor::Actor;
use hyperactor::ActorHandle;
use hyperactor::Context;
use hyperactor::HandleClient;
use hyperactor::Handler;
use hyperactor::Instance;
use hyperactor::OncePortHandle;
use hyperactor::RefClient;
use hyperactor::RemoteSpawn;
use hyperactor::context;
use hyperactor::supervision::ActorSupervisionEvent;
use hyperactor_config::Flattrs;
use serde::Deserialize;
use serde::Serialize;
use typeuri::Named;

use crate::backend::RdmaRemoteBackendContext;
use crate::local_memory::RdmaLocalMemory;
use crate::rdma_components::RdmaRemoteBuffer;

// ---- ibverbs backend imports ----
#[cfg(not(feature = "hixl"))]
use tokio::sync::OnceCell;
#[cfg(not(feature = "hixl"))]
use crate::backend::ibverbs::manager_actor::IbvManagerActor;
#[cfg(not(feature = "hixl"))]
use crate::backend::ibverbs::manager_actor::IbvManagerMessageClient;
#[cfg(not(feature = "hixl"))]
use crate::backend::ibverbs::primitives::IbvConfig;
#[cfg(not(feature = "hixl"))]
use crate::backend::tcp::manager_actor::TcpManagerActor;

// ---- HIXL backend imports ----
#[cfg(feature = "hixl")]
use crate::backend::hixl::manager_actor::HixlManagerActor;
#[cfg(feature = "hixl")]
use crate::backend::hixl::manager_actor::HixlManagerMessageClient;

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
/// These messages carry `Arc<dyn RdmaLocalMemory>` and are therefore
/// not serializable -- they can only be sent within the same process.
#[derive(Handler, HandleClient, Debug)]
pub enum RdmaManagerMessage {
    /// Register a local memory handle and return a [`RdmaRemoteBuffer`] that
    /// remote peers can use to address this buffer over RDMA.
    RequestBuffer {
        local: Arc<dyn RdmaLocalMemory>,
        #[reply]
        reply: OncePortHandle<RdmaRemoteBuffer>,
    },
    /// Look up the local memory handle for a given `remote_buf_id`. Returns
    /// `None` if the id does not correspond to a registered buffer.
    RequestLocalMemory {
        remote_buf_id: usize,
        #[reply]
        reply: OncePortHandle<Option<Arc<dyn RdmaLocalMemory>>>,
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
    pub reply: reference::OncePortRef<()>,
}
#[cfg(feature = "hixl")]
wirevalue::register_type!(EnsurePeerConnected);

/// Serializable query for resolving the [`IbvManagerActor`] ref
/// from a remote [`RdmaManagerActor`]. Only used in testing.
#[cfg(not(feature = "hixl"))]
#[derive(Handler, HandleClient, RefClient, Debug, Serialize, Deserialize, Named)]
pub struct GetIbvActorRef {
    #[reply]
    pub reply: reference::OncePortRef<Option<reference::ActorRef<IbvManagerActor>>>,
}
#[cfg(not(feature = "hixl"))]
wirevalue::register_type!(GetIbvActorRef);

/// Serializable query for resolving the [`TcpManagerActor`] ref
/// from a remote [`RdmaManagerActor`].
#[cfg(not(feature = "hixl"))]
#[derive(Handler, HandleClient, RefClient, Debug, Serialize, Deserialize, Named)]
pub struct GetTcpActorRef {
    #[reply]
    pub reply: reference::OncePortRef<reference::ActorRef<TcpManagerActor>>,
}
#[cfg(not(feature = "hixl"))]
wirevalue::register_type!(GetTcpActorRef);

#[derive(Debug)]
enum RdmaBackendActor<A: Actor> {
    Uninit,
    Created(A),
    Spawned(ActorHandle<A>),
}

impl<A: Actor> RdmaBackendActor<A> {
    fn spawn(&mut self, rdma_manager: &Instance<RdmaManagerActor>) -> anyhow::Result<()> {
        let created = std::mem::replace(self, RdmaBackendActor::Uninit);
        let actor = if let RdmaBackendActor::Created(actor) = created {
            actor
        } else {
            panic!("rdma backend actor already spawned");
        };
        let handle = rdma_manager.spawn(actor)?;
        *self = RdmaBackendActor::Spawned(handle);
        Ok(())
    }

    fn handle(&self) -> &ActorHandle<A> {
        if let RdmaBackendActor::Spawned(handle) = self {
            handle
        } else {
            panic!("cannot get handle")
        }
    }
}

// ============================================================================
// ibverbs backend (default)
// ============================================================================

#[cfg(not(feature = "hixl"))]
#[derive(Debug)]
#[hyperactor::export(
    handlers = [
        GetIbvActorRef,
        GetTcpActorRef,
        ReleaseBuffer,
    ],
)]
#[hyperactor::spawnable]
pub struct RdmaManagerActor {
    next_remote_buf_id: usize,
    buffers: HashMap<usize, Arc<dyn RdmaLocalMemory>>,
    ibverbs: Option<RdmaBackendActor<IbvManagerActor>>,
    tcp: RdmaBackendActor<TcpManagerActor>,
}

#[cfg(not(feature = "hixl"))]
impl RdmaManagerActor {
    pub fn local_handle(client: &impl context::Actor) -> ActorHandle<Self> {
        let actor_ref = reference::ActorRef::attest(
            client
                .mailbox()
                .actor_id()
                .proc_ref()
                .actor_id("rdma_manager"),
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
        let ibv = if hyperactor_config::global::get(crate::config::RDMA_DISABLE_IBVERBS) {
            if hyperactor_config::global::get(crate::config::RDMA_ALLOW_TCP_FALLBACK) {
                tracing::info!("ibverbs disabled by configuration, using TCP transport");
                None
            } else {
                anyhow::bail!(
                    "ibverbs is disabled (rdma_disable_ibverbs=true) \
                     but TCP fallback is also disabled"
                );
            }
        } else {
            match IbvManagerActor::new(params).await {
                Ok(actor) => Some(RdmaBackendActor::Created(actor)),
                Err(e) => {
                    if hyperactor_config::global::get(crate::config::RDMA_ALLOW_TCP_FALLBACK) {
                        tracing::warn!(
                            "ibverbs initialization failed, TCP fallback enabled: {}",
                            e
                        );
                        None
                    } else {
                        return Err(e);
                    }
                }
            }
        };

        let tcp = RdmaBackendActor::Created(TcpManagerActor::new());

        Ok(Self {
            next_remote_buf_id: 0,
            buffers: HashMap::new(),
            ibverbs: ibv,
            tcp,
        })
    }
}

#[cfg(not(feature = "hixl"))]
#[async_trait]
impl Actor for RdmaManagerActor {
    async fn init(&mut self, this: &Instance<Self>) -> Result<(), anyhow::Error> {
        if let Some(ibv) = &mut self.ibverbs {
            ibv.spawn(this)?;
        }
        self.tcp.spawn(this)?;
        tracing::debug!("RdmaManagerActor initialized with lazy domain/QP creation");
        Ok(())
    }

    async fn handle_supervision_event(
        &mut self,
        _cx: &Instance<Self>,
        event: &ActorSupervisionEvent,
    ) -> Result<bool, anyhow::Error> {
        if !event.is_error() {
            return Ok(true);
        }
        tracing::error!("rdmaManagerActor supervision event: {:?}", event);
        tracing::error!("rdmaManagerActor error occurred, stop the worker process, exit code: 1");
        std::process::exit(1);
    }
}

#[cfg(not(feature = "hixl"))]
#[async_trait]
#[hyperactor::handle(GetIbvActorRef)]
impl GetIbvActorRefHandler for RdmaManagerActor {
    async fn get_ibv_actor_ref(
        &mut self,
        _cx: &Context<Self>,
    ) -> Result<Option<reference::ActorRef<IbvManagerActor>>, anyhow::Error> {
        Ok(self.ibverbs.as_ref().map(|ibv| ibv.handle().bind()))
    }
}

#[cfg(not(feature = "hixl"))]
#[async_trait]
#[hyperactor::handle(GetTcpActorRef)]
impl GetTcpActorRefHandler for RdmaManagerActor {
    async fn get_tcp_actor_ref(
        &mut self,
        _cx: &Context<Self>,
    ) -> Result<reference::ActorRef<TcpManagerActor>, anyhow::Error> {
        Ok(self.tcp.handle().bind())
    }
}

#[cfg(not(feature = "hixl"))]
#[async_trait]
#[hyperactor::handle(ReleaseBuffer)]
impl ReleaseBufferHandler for RdmaManagerActor {
    async fn release_buffer(&mut self, cx: &Context<Self>, id: usize) -> Result<(), anyhow::Error> {
        self.buffers.remove(&id);
        if let Some(ibv) = &self.ibverbs {
            ibv.handle().release_buffer(cx, id).await?;
        }
        Ok(())
    }
}

#[cfg(not(feature = "hixl"))]
#[async_trait]
#[hyperactor::handle(RdmaManagerMessage)]
impl RdmaManagerMessageHandler for RdmaManagerActor {
    async fn request_buffer(
        &mut self,
        cx: &Context<Self>,
        local: Arc<dyn RdmaLocalMemory>,
    ) -> Result<RdmaRemoteBuffer, anyhow::Error> {
        let remote_buf_id = self.next_remote_buf_id;
        self.next_remote_buf_id += 1;
        let size = local.size();

        self.buffers.insert(remote_buf_id, local);

        let mut backends = Vec::new();

        if let Some(ibv) = &self.ibverbs {
            backends.push(RdmaRemoteBackendContext::Ibverbs(
                ibv.handle().bind(),
                Arc::new(OnceCell::new()),
            ));
        }

        backends.push(RdmaRemoteBackendContext::Tcp(self.tcp.handle().bind()));

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
    ) -> Result<Option<Arc<dyn RdmaLocalMemory>>, anyhow::Error> {
        Ok(self.buffers.get(&remote_buf_id).cloned())
    }
}

// ============================================================================
// HIXL backend (Ascend NPU)
// ============================================================================

#[cfg(feature = "hixl")]
#[derive(Debug)]
#[hyperactor::export(
    handlers = [
        ReleaseBuffer,
        EnsurePeerConnected,
    ],
)]
#[hyperactor::spawnable]
pub struct RdmaManagerActor {
    next_remote_buf_id: usize,
    buffers: HashMap<usize, Arc<dyn RdmaLocalMemory>>,
    hixl: RdmaBackendActor<HixlManagerActor>,
}

#[cfg(feature = "hixl")]
impl RdmaManagerActor {
    pub fn local_handle(client: &impl context::Actor) -> ActorHandle<Self> {
        let actor_ref = reference::ActorRef::attest(
            client
                .mailbox()
                .actor_id()
                .proc_ref()
                .actor_id("rdma_manager"),
        );
        actor_ref
            .downcast_handle(client)
            .expect("RdmaManagerActor is not in the local process")
    }
}

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
        if let Ok(addrs) = std::net::ToSocketAddrs::to_socket_addrs(
            &format!("{}:0", hostname.to_string_lossy()),
        ) {
            for addr in addrs {
                if addr.is_ipv4() && !addr.ip().is_loopback() {
                    return addr.ip().to_string();
                }
            }
        }
    }

    if let Ok(output) = std::process::Command::new("hostname")
        .arg("-I")
        .output()
    {
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
        let config = params.unwrap_or_default();
        let engine_id = config.engine_id.unwrap_or_default();
        let device_id = config.device_id.unwrap_or(-1);
        let hixl_actor = HixlManagerActor::new(engine_id, device_id);
        let hixl = RdmaBackendActor::Created(hixl_actor);
        Ok(Self {
            next_remote_buf_id: 0,
            buffers: HashMap::new(),
            hixl,
        })
    }
}

#[cfg(feature = "hixl")]
#[async_trait]
impl Actor for RdmaManagerActor {
    async fn init(&mut self, this: &Instance<Self>) -> Result<(), anyhow::Error> {
        self.hixl.spawn(this)?;
        tracing::debug!("RdmaManagerActor initialized with HIXL backend");
        Ok(())
    }

    async fn handle_supervision_event(
        &mut self,
        _cx: &Instance<Self>,
        _event: &ActorSupervisionEvent,
    ) -> Result<bool, anyhow::Error> {
        tracing::error!("rdmaManagerActor supervision event: {:?}", _event);
        tracing::error!("rdmaManagerActor error occurred, stop the worker process, exit code: 1");
        std::process::exit(1);
    }
}

#[cfg(feature = "hixl")]
#[async_trait]
#[hyperactor::handle(ReleaseBuffer)]
impl ReleaseBufferHandler for RdmaManagerActor {
    async fn release_buffer(&mut self, cx: &Context<Self>, id: usize) -> Result<(), anyhow::Error> {
        self.buffers.remove(&id);
        self.hixl.handle().release_buffer(cx, id).await
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
        crate::backend::hixl::manager_actor::do_connect(&peer_engine_id)?;
        Ok(())
    }
}

#[cfg(feature = "hixl")]
#[async_trait]
#[hyperactor::handle(RdmaManagerMessage)]
impl RdmaManagerMessageHandler for RdmaManagerActor {
    async fn request_buffer(
        &mut self,
        cx: &Context<Self>,
        local: Arc<dyn RdmaLocalMemory>,
    ) -> Result<RdmaRemoteBuffer, anyhow::Error> {
        let remote_buf_id = self.next_remote_buf_id;
        self.next_remote_buf_id += 1;
        let size = local.size();

        let addr = local.addr();
        tracing::warn!("RdmaManager: request_buffer id={} addr={:#x} size={}", remote_buf_id, addr, size);
        self.buffers.insert(remote_buf_id, local);

        let hixl_buf = self
            .hixl
            .handle()
            .request_buffer(cx, remote_buf_id, addr, size)
            .await?
            .ok_or_else(|| {
                anyhow::anyhow!("HIXL request_buffer returned None for id {}", remote_buf_id)
            })?;
        tracing::debug!("RdmaManager: got HixlBuffer: {:?}", hixl_buf);

        Ok(RdmaRemoteBuffer {
            id: remote_buf_id,
            size,
            owner: cx.bind().clone(),
            backends: vec![RdmaRemoteBackendContext::Hixl(hixl_buf)],
        })
    }

    async fn request_local_memory(
        &mut self,
        _cx: &Context<Self>,
        remote_buf_id: usize,
    ) -> Result<Option<Arc<dyn RdmaLocalMemory>>, anyhow::Error> {
        Ok(self.buffers.get(&remote_buf_id).cloned())
    }
}
