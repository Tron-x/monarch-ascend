/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! # HIXL Manager Actor
//!
//! The manager actor and its backend handles share one [`HixlEngineState`].
//! Registration metadata remains actor-owned, while backend handles execute
//! data-plane transfers directly against the shared state.
//!
//! The C shim restores ACL context via `aclrtSetCurrentContext` before each
//! HiXL call, so operations can run from any thread without serialisation.

use std::collections::HashMap;
use std::collections::HashSet;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::OnceLock;

use anyhow::Result;
use async_trait::async_trait;
use hyperactor::Actor;
use hyperactor::ActorHandle;
use hyperactor::ActorRef;
use hyperactor::Context;
use hyperactor::HandleClient;
use hyperactor::Handler;
use hyperactor::Instance;
use hyperactor::OncePortHandle;

use super::HixlBuffer;
use crate::RdmaOp;
use crate::RdmaOpType;
use crate::RdmaTransportLevel;
use crate::backend::RdmaBackend;
use crate::backend::RdmaConfig;
use crate::local_memory::KeepaliveLocalMemory;
use crate::rdma_manager_actor::EnsurePeerConnectedClient;
use crate::rdma_manager_actor::RdmaManagerActor;

trait HixlEngineOps: Send + Sync {
    fn connect(&self, remote_engine_id: &str, timeout_ms: i32) -> Result<(), i32>;
    fn register_mem(&self, addr: usize, size: usize) -> Result<(), i32>;
    fn deregister_mem(&self, addr: usize) -> Result<(), i32>;
    fn transfer_read(
        &self,
        remote_engine_id: &str,
        local_addr: usize,
        remote_addr: usize,
        len: usize,
        timeout_ms: i32,
    ) -> Result<(), i32>;
    fn transfer_write(
        &self,
        remote_engine_id: &str,
        local_addr: usize,
        remote_addr: usize,
        len: usize,
        timeout_ms: i32,
    ) -> Result<(), i32>;
}

impl HixlEngineOps for hixl_sys::HixlEngine {
    fn connect(&self, remote_engine_id: &str, timeout_ms: i32) -> Result<(), i32> {
        self.connect(remote_engine_id, timeout_ms)
    }

    fn register_mem(&self, addr: usize, size: usize) -> Result<(), i32> {
        self.register_mem(addr, size)
    }

    fn deregister_mem(&self, addr: usize) -> Result<(), i32> {
        self.deregister_mem(addr)
    }

    fn transfer_read(
        &self,
        remote_engine_id: &str,
        local_addr: usize,
        remote_addr: usize,
        len: usize,
        timeout_ms: i32,
    ) -> Result<(), i32> {
        self.transfer_read(remote_engine_id, local_addr, remote_addr, len, timeout_ms)
    }

    fn transfer_write(
        &self,
        remote_engine_id: &str,
        local_addr: usize,
        remote_addr: usize,
        len: usize,
        timeout_ms: i32,
    ) -> Result<(), i32> {
        self.transfer_write(remote_engine_id, local_addr, remote_addr, len, timeout_ms)
    }
}

struct HixlEngineState {
    engine: Box<dyn HixlEngineOps>,
    engine_id: String,
    connected_peers: Mutex<HashSet<String>>,
    /// Real HiXL registrations: `addr → (size, refcount)`.  The first
    /// `register_mem_if_needed` call for a given address performs the
    /// actual `hixl_register_mem` and adds an entry here.  The entry is
    /// removed (and `hixl_deregister_mem` called) once refcount drops
    /// back to 0.
    registered_addrs: Mutex<HashMap<usize, (usize, usize)>>,
    /// Sub-range aliases: `aliased_addr → (owner_addr, refcount)`.
    ///
    /// **Historical workaround for CANN 9.0.0-beta.1**.  That version of
    /// HiXL rejected more than one registered memory region per engine
    /// pair: a second `hixl_register_mem` would succeed but every
    /// subsequent `TransferSync` returned `503900` on that pair.  The
    /// workaround was to detect when a new `register_mem_if_needed(addr,
    /// size)` call fell fully inside an already-registered range and
    /// silently alias it back to the owner — keeping HiXL's view of the
    /// world at "one region per pair".
    ///
    /// **CANN 9.0.0 release fixes this on the old P2P API**, so the
    /// aliasing path is disabled by default starting with this build.
    /// The field, populator branch, and `deregister_mem` cleanup are
    /// retained for backward compatibility: set
    /// `MONARCH_HIXL_ENABLE_ALIAS=1` (see [`HixlEngineState::alias_enabled`])
    /// to re-enable the old behaviour when running against
    /// CANN 9.0.0-beta or earlier.
    aliased_addrs: Mutex<HashMap<usize, (usize, usize)>>,
    /// `true` when running over RoCE instead of HCCS.
    force_roce: bool,
    /// `true` if the range-containment alias workaround should be
    /// active.  Default `false` (CANN 9.0.0 release no longer needs it);
    /// set the env var `MONARCH_HIXL_ENABLE_ALIAS=1` to force it on for
    /// older CANN builds.
    alias_enabled: bool,
}

impl std::fmt::Debug for HixlEngineState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HixlEngineState")
            .field("engine_id", &self.engine_id)
            .field("force_roce", &self.force_roce)
            .finish_non_exhaustive()
    }
}

/// Default connect timeout in milliseconds.
pub const DEFAULT_CONNECT_TIMEOUT_MS: i32 = 10_000;

const HCCS_CONNECT_ERROR: i32 = 503900;

const HCCS_ALIGNMENT: usize = 2 * 1024 * 1024; // 2 MB

impl HixlEngineState {
    fn new(
        engine: Box<dyn HixlEngineOps>,
        engine_id: String,
        force_roce: bool,
        alias_enabled: bool,
    ) -> Self {
        Self {
            engine,
            engine_id,
            connected_peers: Mutex::new(HashSet::new()),
            registered_addrs: Mutex::new(HashMap::new()),
            aliased_addrs: Mutex::new(HashMap::new()),
            force_roce,
            alias_enabled,
        }
    }

    fn connect(&self, peer_eid: &str) -> Result<()> {
        self.connect_with_timeout(peer_eid, DEFAULT_CONNECT_TIMEOUT_MS)
    }

    fn connect_with_timeout(&self, peer_eid: &str, timeout_ms: i32) -> Result<()> {
        // Keep the set locked through Connect so concurrent first-use calls
        // cannot race and issue duplicate HiXL connections.
        let mut connected = self.connected_peers.lock().unwrap();
        if connected.contains(peer_eid) {
            return Ok(());
        }

        tracing::info!(
            "[hixl] connecting to peer {} (timeout={}ms)",
            peer_eid,
            timeout_ms
        );
        self.engine.connect(peer_eid, timeout_ms).map_err(|ret| {
            if ret == HCCS_CONNECT_ERROR && !self.force_roce {
                anyhow::anyhow!(
                    "hixl_connect({}) failed: ret={} — HCCS channel creation failed. \
                     Ensure all registered memory is 2MB-aligned (use alloc_aligned_tensor()). \
                     Or set MONARCH_HIXL_TRANSPORT=roce to use RoCE instead.",
                    peer_eid,
                    ret,
                )
            } else {
                anyhow::anyhow!("hixl_connect({}) failed: ret={}", peer_eid, ret)
            }
        })?;
        connected.insert(peer_eid.to_string());
        tracing::info!("[hixl] connected to peer {}", peer_eid);
        Ok(())
    }

    /// Register a device memory region.  Reference-counted: the first call
    /// for a given address performs the HiXL RegisterMem, subsequent calls
    /// just bump the count.
    ///
    /// In HCCS mode (the default), the address **must** be 2 MB aligned.
    /// If it is not, a warning is logged: the RegisterMem call will still
    /// be attempted, but the subsequent `TransferSync` may fail at runtime.
    fn register_mem_if_needed(&self, addr: usize, size: usize) -> Result<()> {
        let mut addrs = self.registered_addrs.lock().unwrap();
        let mut aliases = self.aliased_addrs.lock().unwrap();

        // Case 1: exact match on an existing real registration -> bump.
        if let Some((_sz, refcnt)) = addrs.get_mut(&addr) {
            *refcnt += 1;
            return Ok(());
        }

        // Case 2: exact match on a known alias -> bump both alias and
        // the real owner's refcount (so the real HiXL registration
        // can't be torn down while any alias is in flight).
        //
        // Reachable only when `alias_enabled` (legacy CANN 9.0.0-beta).
        // On CANN 9.0.0 release the alias table is never populated, so
        // this branch is effectively dead code but kept for backward
        // compatibility with older CANN builds.
        if let Some((owner, arefcnt)) = aliases.get_mut(&addr) {
            *arefcnt += 1;
            let owner_key = *owner;
            if let Some((_sz, orefcnt)) = addrs.get_mut(&owner_key) {
                *orefcnt += 1;
            }
            return Ok(());
        }

        // Case 3 (legacy): new addr but fully contained in an existing
        // real registration's range -> record as alias, skip HiXL call.
        //
        // **Disabled by default.**  This workaround was required for
        // CANN 9.0.0-beta.1 where HiXL rejected more than one
        // registered memory region per engine pair (every subsequent
        // `TransferSync` returned 503900).  CANN 9.0.0 release fixes
        // this on the old P2P API — confirmed by
        // `tests/hixl/e2e/test_multi_region_per_pair.py` PASS on
        // both HCCS and RoCE transports — so we now fall through to
        // Case 4 and let HiXL register each region individually.
        //
        // Set `MONARCH_HIXL_ENABLE_ALIAS=1` to re-enable the
        // containment check when running against CANN 9.0.0-beta or
        // earlier.
        if self.alias_enabled {
            let containing_owner = addrs.iter().find_map(|(&owner_addr, &(owner_size, _))| {
                if addr >= owner_addr
                    && size <= owner_size
                    && addr + size <= owner_addr + owner_size
                {
                    Some((owner_addr, owner_size))
                } else {
                    None
                }
            });
            if let Some((owner_addr, _owner_size)) = containing_owner {
                aliases.insert(addr, (owner_addr, 1));
                if let Some((_sz, orefcnt)) = addrs.get_mut(&owner_addr) {
                    *orefcnt += 1;
                }
                tracing::debug!(
                    "[hixl] aliasing addr={:#x} size={} to owner={:#x} (skipping hixl_register_mem) [legacy CANN workaround]",
                    addr,
                    size,
                    owner_addr,
                );
                return Ok(());
            }
        }

        // Case 4: fresh registration.  Fall through to the original
        // path: warn on alignment, call HiXL, record as real.
        // This is the default path on CANN 9.0.0 release and later.
        if !self.force_roce && (addr % HCCS_ALIGNMENT != 0) {
            tracing::warn!(
                "[hixl] memory addr={:#x} is NOT 2 MB aligned (offset={:#x}). \
                 HCCS transfers may fail — consider using alloc_aligned_tensor() \
                 or set MONARCH_HIXL_USE_ROCE=1 to fall back to RoCE.",
                addr,
                addr % HCCS_ALIGNMENT,
            );
        }

        self.engine.register_mem(addr, size).map_err(|ret| {
            anyhow::anyhow!(
                "hixl_register_mem(addr={:#x}, size={}) failed: ret={}",
                addr,
                size,
                ret
            )
        })?;
        addrs.insert(addr, (size, 1));
        Ok(())
    }

    /// Decrement the reference count for a registered memory region.
    /// When the count reaches 0, the region is deregistered from HiXL.
    fn deregister_mem(&self, addr: usize) -> Result<()> {
        let mut addrs = self.registered_addrs.lock().unwrap();
        let mut aliases = self.aliased_addrs.lock().unwrap();

        // Case 1: alias entry.  Decrement its refcount and also the
        // real owner's refcount.  When either hits 0, propagate the
        // teardown: the alias is removed from the map, and the owner
        // is released (via hixl_deregister_mem) only if *all* of its
        // aliases and its direct refs have been released.
        if let Some((owner, arefcnt)) = aliases.get_mut(&addr) {
            *arefcnt -= 1;
            let owner_key = *owner;
            let alias_now_zero = *arefcnt == 0;
            if alias_now_zero {
                aliases.remove(&addr);
            }
            let mut owner_now_zero = false;
            if let Some((_sz, orefcnt)) = addrs.get_mut(&owner_key) {
                if *orefcnt > 1 {
                    *orefcnt -= 1;
                } else {
                    addrs.remove(&owner_key);
                    owner_now_zero = true;
                }
            }
            drop(aliases);
            drop(addrs);
            if owner_now_zero {
                match self.engine.deregister_mem(owner_key) {
                    Ok(()) => tracing::debug!(
                        "[hixl] deregistered owner={:#x} via alias={:#x}",
                        owner_key,
                        addr,
                    ),
                    Err(ret) => tracing::warn!(
                        "[hixl] deregister_mem owner={:#x} (via alias={:#x}) ret={} (treating as success)",
                        owner_key,
                        addr,
                        ret,
                    ),
                }
            }
            return Ok(());
        }

        // Case 2: direct (real) entry.  Standard refcount path.
        match addrs.get_mut(&addr) {
            Some((_sz, refcnt)) if *refcnt > 1 => {
                *refcnt -= 1;
                return Ok(());
            }
            Some(_) => {
                addrs.remove(&addr);
            }
            None => {
                tracing::warn!(
                    "[hixl] deregister_mem: addr={:#x} not registered, ignoring",
                    addr
                );
                return Ok(());
            }
        }
        drop(aliases);
        drop(addrs);
        match self.engine.deregister_mem(addr) {
            Ok(()) => {
                tracing::debug!("[hixl] deregistered mem addr={:#x}", addr);
            }
            Err(ret) => {
                // ret=103900 means the HiXL driver reports the address is no
                // longer registered (e.g. already deregistered by another path,
                // or re-used after an earlier deregister+register cycle).  We
                // have already removed the addr from registered_addrs above, so
                // no further deregistration will be attempted — treat as success.
                tracing::warn!(
                    "[hixl] deregister_mem: addr={:#x} hixl_deregister_mem ret={} (treating as success)",
                    addr,
                    ret
                );
            }
        }
        Ok(())
    }

    fn transfer_write(
        &self,
        remote_engine_id: &str,
        local_addr: usize,
        remote_addr: usize,
        len: usize,
        timeout_ms: i32,
    ) -> Result<()> {
        self.engine
            .transfer_write(remote_engine_id, local_addr, remote_addr, len, timeout_ms)
            .map_err(|ret| {
                anyhow::anyhow!(
                    "hixl_transfer_write failed: local={:#x} remote={:#x}@{} len={} ret={}",
                    local_addr,
                    remote_addr,
                    remote_engine_id,
                    len,
                    ret,
                )
            })
    }

    fn transfer_read(
        &self,
        remote_engine_id: &str,
        local_addr: usize,
        remote_addr: usize,
        len: usize,
        timeout_ms: i32,
    ) -> Result<()> {
        self.engine
            .transfer_read(remote_engine_id, local_addr, remote_addr, len, timeout_ms)
            .map_err(|ret| {
                anyhow::anyhow!(
                    "hixl_transfer_read failed: local={:#x} remote={:#x}@{} len={} ret={}",
                    local_addr,
                    remote_addr,
                    remote_engine_id,
                    len,
                    ret,
                )
            })
    }
}

impl Drop for HixlEngineState {
    fn drop(&mut self) {
        let addrs = std::mem::take(self.registered_addrs.get_mut().unwrap());
        self.aliased_addrs.get_mut().unwrap().clear();
        for addr in addrs.into_keys() {
            if let Err(ret) = self.engine.deregister_mem(addr) {
                tracing::warn!(
                    "[hixl] cleanup deregister_mem addr={:#x} ret={} (continuing)",
                    addr,
                    ret
                );
            }
        }
        // The boxed engine is dropped immediately after this method, which
        // performs hixl_cleanup once the last actor/backend Arc is gone.
    }
}

// ============================================================================
// Ensure connected (async — used from data-plane code)
// ============================================================================

/// Ensure that the local engine is connected to `remote_eid`.
///
/// 1. Sends `EnsurePeerConnected(my_eid)` to the remote `RdmaManagerActor`
///    so the remote side calls `Connect(my_eid)`.
/// 2. Then does the local `Connect(remote_eid)`.
///
/// Both directions must be connected before TransferSync can succeed.
async fn ensure_connected(
    state: &HixlEngineState,
    client: &(impl hyperactor::context::Actor + Send + Sync),
    remote_rdma_mgr: &ActorRef<RdmaManagerActor>,
    remote_eid: &str,
) -> Result<()> {
    if state.connected_peers.lock().unwrap().contains(remote_eid) {
        return Ok(());
    }

    let my_eid = state.engine_id.clone();
    tracing::info!(
        "[hixl] ensure_connected: asking remote {} to connect to us ({})",
        remote_eid,
        my_eid,
    );

    remote_rdma_mgr
        .ensure_peer_connected(client, my_eid)
        .await
        .map_err(|e| anyhow::anyhow!("EnsurePeerConnected to {} failed: {}", remote_eid, e))?;

    tracing::info!(
        "[hixl] ensure_connected: remote acked, now local connect to {}",
        remote_eid,
    );
    state.connect(remote_eid)?;

    Ok(())
}

// ============================================================================
// HixlManagerMessage
// ============================================================================

/// Local-only messages for the child HiXL actor.
///
/// Registration carries process-local memory metadata and must reply through a
/// local handle. Using a serializable `OncePortRef` here can strand the reply
/// when registration is initiated from the RDMA owner actor.
#[derive(Handler, HandleClient, Debug)]
pub enum HixlManagerMessage {
    RequestBuffer {
        remote_buf_id: usize,
        addr: usize,
        size: usize,
        #[reply]
        reply: OncePortHandle<Option<HixlBuffer>>,
    },
    ReleaseBuffer {
        remote_buf_id: usize,
        #[reply]
        reply: OncePortHandle<()>,
    },
    GetEngineId {
        #[reply]
        reply: OncePortHandle<String>,
    },
}

// ============================================================================
// HixlManagerActor
// ============================================================================

#[derive(Debug)]
pub struct HixlManagerActor {
    state: Arc<HixlEngineState>,
    owner: OnceLock<ActorHandle<RdmaManagerActor>>,
    /// buf_id → registered device address, for deregistration on release.
    buf_addrs: std::collections::HashMap<usize, usize>,
}

impl HixlManagerActor {
    fn new(state: Arc<HixlEngineState>) -> Self {
        Self {
            state,
            owner: OnceLock::new(),
            buf_addrs: std::collections::HashMap::new(),
        }
    }
}

impl Drop for HixlManagerActor {
    fn drop(&mut self) {
        for (_, addr) in self.buf_addrs.drain() {
            if let Err(error) = self.state.deregister_mem(addr) {
                tracing::warn!(
                    "[hixl] failed to release actor-owned registration addr={:#x}: {}",
                    addr,
                    error
                );
            }
        }
    }
}

fn resolve_engine_id(hint: &str) -> String {
    if !hint.is_empty() {
        return hint.to_string();
    }
    if let Ok(eid) = std::env::var("MONARCH_PYTHON_HIXL_ENGINE_ID") {
        return eid;
    }
    let ip = crate::rdma_manager_actor::local_ip_for_hixl();
    let port = 20000 + (std::process::id() % 40000);
    format!("{}:{}", ip, port)
}

fn resolve_device_id(hint: i32) -> i32 {
    if hint >= 0 {
        return hint;
    }
    std::env::var("MONARCH_NPU_DEVICE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0)
}

/// Initialise the HiXL engine on a **dedicated OS thread** so that
/// `aclrtSetDevice` + `Hixl::Initialize` run on a clean, non-tokio
/// thread.  The C shim captures the ACL context during init and
/// restores it via `aclrtSetCurrentContext` before every subsequent
/// operation, so post-init calls may run on any thread.
///
/// **Transport selection**:
/// - Default: HCCS (intra-supernode high-speed interconnect).
/// - Set `MONARCH_HIXL_USE_ROCE=1` to force RoCE (for cross-supernode or
///   when device memory is not 2 MB aligned).
fn init_engine_on_dedicated_thread(dev: i32, eid: String) -> Result<Arc<HixlEngineState>> {
    // Transport selection via MONARCH_HIXL_TRANSPORT env var:
    //   "hccs"  → HCCS (intra-supernode, default — requires 2MB-aligned memory)
    //   "roce"  → RoCE (inter-/intra-node, no alignment requirement)
    //   unset   → defaults to HCCS
    let transport = std::env::var("MONARCH_HIXL_TRANSPORT")
        .unwrap_or_else(|_| "hccs".to_string())
        .to_lowercase();

    let force_roce = match transport.as_str() {
        "roce" => {
            unsafe { std::env::set_var("HCCL_INTRA_ROCE_ENABLE", "1") };
            tracing::info!("[hixl] MONARCH_HIXL_TRANSPORT=roce → using RoCE");
            true
        }
        _ => {
            // Default: HCCS for intra-supernode high-speed transfers.
            // Requires all registered memory to be 2MB-aligned.
            unsafe { std::env::remove_var("HCCL_INTRA_ROCE_ENABLE") };
            if std::env::var("HCCL_NPU_SOCKET_PORT_RANGE").is_err() {
                unsafe { std::env::set_var("HCCL_NPU_SOCKET_PORT_RANGE", "auto") };
            }
            tracing::info!("[hixl] using HCCS (default) — all RDMA buffers must be 2MB-aligned",);
            false
        }
    };

    do_init_engine(dev, eid, force_roce)
}

fn do_init_engine(dev: i32, eid: String, force_roce: bool) -> Result<Arc<HixlEngineState>> {
    let (tx, rx) = std::sync::mpsc::channel::<Result<Arc<HixlEngineState>, String>>();
    let eid_clone = eid.clone();

    std::thread::Builder::new()
        .name(format!("hixl-init-dev{}", dev))
        .spawn(move || {
            tracing::info!(
                "[hixl] dedicated init thread started: dev={} eid={} force_roce={} tid={:?}",
                dev,
                eid_clone,
                force_roce,
                std::thread::current().id(),
            );
            let result = hixl_sys::HixlEngine::new(dev, &eid_clone);
            match result {
                Ok(engine) => {
                    let alias_enabled = std::env::var("MONARCH_HIXL_ENABLE_ALIAS")
                        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                        .unwrap_or(false);
                    if alias_enabled {
                        tracing::warn!(
                            "[hixl] MONARCH_HIXL_ENABLE_ALIAS=1 — using legacy \
                             range-containment alias workaround (only needed on \
                             CANN 9.0.0-beta and earlier)"
                        );
                    }
                    let new_state = Arc::new(HixlEngineState::new(
                        Box::new(engine),
                        eid_clone.clone(),
                        force_roce,
                        alias_enabled,
                    ));
                    unsafe { std::env::set_var("MONARCH_PYTHON_HIXL_ENGINE_ID", &eid_clone) };
                    let _ = tx.send(Ok(new_state));
                }
                Err(e) => {
                    let _ = tx.send(Err(e));
                }
            }
        })
        .map_err(|e| anyhow::anyhow!("failed to spawn hixl-init thread: {e}"))?;

    let state = rx
        .recv()
        .map_err(|e| anyhow::anyhow!("hixl-init thread channel closed: {e}"))?
        .map_err(|e| anyhow::anyhow!("hixl_init_engine failed on dedicated thread: {e}"))?;

    tracing::info!(
        "[hixl] engine initialised on dedicated thread: dev={} eid={} force_roce={}",
        dev,
        eid,
        force_roce,
    );
    Ok(state)
}

#[async_trait]
impl Actor for HixlManagerActor {
    async fn init(&mut self, this: &Instance<Self>) -> Result<(), anyhow::Error> {
        let owner = this
            .parent_handle()
            .ok_or_else(|| anyhow::anyhow!("RdmaManagerActor not found as parent"))?;
        self.owner
            .set(owner)
            .expect("owner should only be set once during init");

        tracing::info!(
            "[hixl] HixlManagerActor ready: engine_id={}",
            self.state.engine_id
        );
        Ok(())
    }
}

#[async_trait]
#[hyperactor::handle(HixlManagerMessage)]
impl HixlManagerMessageHandler for HixlManagerActor {
    async fn request_buffer(
        &mut self,
        _cx: &Context<Self>,
        remote_buf_id: usize,
        addr: usize,
        size: usize,
    ) -> Result<Option<HixlBuffer>, anyhow::Error> {
        tracing::debug!(
            "[hixl] request_buffer: id={} addr={:#x} size={}",
            remote_buf_id,
            addr,
            size,
        );

        self.state.register_mem_if_needed(addr, size)?;
        self.buf_addrs.insert(remote_buf_id, addr);

        Ok(Some(HixlBuffer {
            engine_id: self.state.engine_id.clone(),
            addr,
            size,
        }))
    }

    async fn release_buffer(
        &mut self,
        _cx: &Context<Self>,
        remote_buf_id: usize,
    ) -> Result<(), anyhow::Error> {
        if let Some(addr) = self.buf_addrs.remove(&remote_buf_id) {
            tracing::debug!(
                "[hixl] release_buffer: id={} addr={:#x}",
                remote_buf_id,
                addr
            );
            self.state.deregister_mem(addr)?;
        } else {
            tracing::debug!("[hixl] release_buffer: id={} (not tracked)", remote_buf_id);
        }
        Ok(())
    }

    async fn get_engine_id(&mut self, _cx: &Context<Self>) -> Result<String, anyhow::Error> {
        Ok(self.state.engine_id.clone())
    }
}

/// Handle used by the upstream backend registry.
///
/// The actor owns per-buffer registration bookkeeping. Backend clones share the
/// same state and keep the off-actor data plane alive until the last clone exits.
#[derive(Debug, Clone)]
pub struct HixlBackend {
    actor: ActorHandle<HixlManagerActor>,
    state: Arc<HixlEngineState>,
}

impl HixlBackend {
    pub(crate) fn connect(&self, peer_engine_id: &str) -> Result<()> {
        self.state.connect(peer_engine_id)
    }
}

#[async_trait]
impl RdmaBackend for HixlBackend {
    type RemoteBackendContext = HixlBuffer;
    type TransportInfo = ();

    fn available() -> bool {
        true
    }

    async fn spawn(
        cx: &(impl hyperactor::context::Actor + Send + Sync),
        config: &RdmaConfig,
    ) -> Result<Self> {
        let config = config.hixl.clone().unwrap_or_default();
        let engine_id = resolve_engine_id(&config.engine_id.unwrap_or_default());
        let device_id = resolve_device_id(config.device_id.unwrap_or(-1));
        tracing::info!(
            "[hixl] initialising HiXL engine on dedicated OS thread: dev={} eid={}",
            device_id,
            engine_id,
        );
        let state = init_engine_on_dedicated_thread(device_id, engine_id)?;
        let actor = cx.spawn(HixlManagerActor::new(state.clone()));
        Ok(Self { actor, state })
    }

    async fn register_remote_buffer(
        &self,
        cx: &(impl hyperactor::context::Actor + Send + Sync),
        remote_buf_id: usize,
        local: KeepaliveLocalMemory,
    ) -> Result<HixlBuffer> {
        self.actor
            .request_buffer(cx, remote_buf_id, local.addr(), local.size())
            .await?
            .ok_or_else(|| anyhow::anyhow!("HiXL failed to register buffer {remote_buf_id}"))
    }

    async fn release_buffer(
        &self,
        cx: &(impl hyperactor::context::Actor + Send + Sync),
        remote_buf_id: usize,
    ) -> Result<()> {
        self.actor.release_buffer(cx, remote_buf_id).await
    }

    async fn submit(
        &self,
        cx: &(impl hyperactor::context::Actor + Send + Sync),
        ops: Vec<RdmaOp>,
        timeout: std::time::Duration,
    ) -> Result<()> {
        let timeout_ms = timeout.as_millis().min(i32::MAX as u128) as i32;

        for op in ops {
            let remote = op
                .remote
                .resolve_hixl()
                .ok_or_else(|| anyhow::anyhow!("op routed to incompatible HiXL backend"))?;

            // HiXL requires the local region to be registered before either
            // side connects, and both connections before transfer.
            self.state
                .register_mem_if_needed(op.local.addr(), op.local.size())?;
            ensure_connected(&self.state, cx, &op.remote.owner, &remote.engine_id).await?;

            match op.op_type {
                RdmaOpType::WriteFromLocal => {
                    self.state.transfer_write(
                        &remote.engine_id,
                        op.local.addr(),
                        remote.addr,
                        op.local.size(),
                        timeout_ms,
                    )?;
                }
                RdmaOpType::ReadIntoLocal => {
                    for attempt in 0..3u32 {
                        let result = self.state.transfer_read(
                            &remote.engine_id,
                            op.local.addr(),
                            remote.addr,
                            op.remote.size,
                            timeout_ms,
                        );
                        match result {
                            Ok(()) => break,
                            Err(_) if attempt < 2 => {
                                tracing::warn!(
                                    "[hixl] transfer_read attempt {} failed, retrying in 1s",
                                    attempt + 1,
                                );
                                tokio::time::sleep(std::time::Duration::from_secs(1)).await;
                            }
                            Err(error) => return Err(error),
                        }
                    }
                }
            }
        }

        Ok(())
    }

    fn transport_level(&self) -> RdmaTransportLevel {
        if self.state.force_roce {
            RdmaTransportLevel::Nic
        } else {
            RdmaTransportLevel::Hccs
        }
    }

    fn transport_info(&self) -> Option<Self::TransportInfo> {
        None
    }
}

#[cfg(all(test, feature = "hixl-mock"))]
mod tests {
    use hixl_sys::MockCall;

    use super::*;

    type CallLog = Arc<Mutex<Vec<MockCall>>>;

    fn mock_state() -> (Arc<HixlEngineState>, CallLog) {
        let engine = hixl_sys::HixlEngine::new(0, "local").unwrap();
        let calls = engine.call_log();
        let state = Arc::new(HixlEngineState::new(
            Box::new(engine),
            "local".to_string(),
            true,
            false,
        ));
        (state, calls)
    }

    fn calls(log: &CallLog) -> Vec<MockCall> {
        log.lock().unwrap().clone()
    }

    #[test]
    fn memory_registration_is_reference_counted() {
        let (state, log) = mock_state();
        state.register_mem_if_needed(0x1000, 4096).unwrap();
        state.register_mem_if_needed(0x1000, 4096).unwrap();
        state.deregister_mem(0x1000).unwrap();

        assert_eq!(
            calls(&log),
            vec![MockCall::RegisterMem {
                addr: 0x1000,
                size: 4096,
            }]
        );

        state.deregister_mem(0x1000).unwrap();
        assert_eq!(
            calls(&log),
            vec![
                MockCall::RegisterMem {
                    addr: 0x1000,
                    size: 4096,
                },
                MockCall::DeregisterMem { addr: 0x1000 },
            ]
        );
    }

    #[test]
    fn concurrent_connect_is_deduplicated() {
        let (state, log) = mock_state();
        let threads: Vec<_> = (0..8)
            .map(|_| {
                let state = state.clone();
                std::thread::spawn(move || state.connect("remote").unwrap())
            })
            .collect();
        for thread in threads {
            thread.join().unwrap();
        }

        assert_eq!(
            calls(&log),
            vec![MockCall::Connect {
                remote_engine_id: "remote".to_string(),
                timeout_ms: DEFAULT_CONNECT_TIMEOUT_MS,
            }]
        );
    }

    #[test]
    fn read_and_write_route_to_matching_engine_operations() {
        let (state, log) = mock_state();
        state
            .transfer_read("remote", 0x1000, 0x2000, 64, 100)
            .unwrap();
        state
            .transfer_write("remote", 0x3000, 0x4000, 128, 200)
            .unwrap();

        assert_eq!(
            calls(&log),
            vec![
                MockCall::TransferRead {
                    remote_engine_id: "remote".to_string(),
                    local_addr: 0x1000,
                    remote_addr: 0x2000,
                    len: 64,
                    timeout_ms: 100,
                },
                MockCall::TransferWrite {
                    remote_engine_id: "remote".to_string(),
                    local_addr: 0x3000,
                    remote_addr: 0x4000,
                    len: 128,
                    timeout_ms: 200,
                },
            ]
        );
    }

    #[test]
    fn register_connect_transfer_order_and_final_cleanup() {
        let (state, log) = mock_state();
        state.register_mem_if_needed(0x1000, 4096).unwrap();
        state.connect("remote").unwrap();
        state
            .transfer_write("remote", 0x1000, 0x2000, 4096, 500)
            .unwrap();
        drop(state);

        assert_eq!(
            calls(&log),
            vec![
                MockCall::RegisterMem {
                    addr: 0x1000,
                    size: 4096,
                },
                MockCall::Connect {
                    remote_engine_id: "remote".to_string(),
                    timeout_ms: DEFAULT_CONNECT_TIMEOUT_MS,
                },
                MockCall::TransferWrite {
                    remote_engine_id: "remote".to_string(),
                    local_addr: 0x1000,
                    remote_addr: 0x2000,
                    len: 4096,
                    timeout_ms: 500,
                },
                MockCall::DeregisterMem { addr: 0x1000 },
                MockCall::Cleanup,
            ]
        );
    }

    #[test]
    fn actor_drop_releases_tracked_registrations() {
        let (state, log) = mock_state();
        state.register_mem_if_needed(0x1000, 4096).unwrap();
        state.register_mem_if_needed(0x1000, 4096).unwrap();
        let mut actor = HixlManagerActor::new(state.clone());
        actor.buf_addrs.insert(1, 0x1000);
        actor.buf_addrs.insert(2, 0x1000);
        drop(actor);

        assert_eq!(
            calls(&log),
            vec![
                MockCall::RegisterMem {
                    addr: 0x1000,
                    size: 4096,
                },
                MockCall::DeregisterMem { addr: 0x1000 },
            ]
        );
    }
}
