/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! RDMA backend implementations.
//!
//! On GPU: ibverbs backend (rdmaxcel) for InfiniBand/RoCE.
//! On NPU: HIXL backend for Ascend RDMA/RoCE/HCCS.

#[cfg(all(test, not(feature = "hixl")))]
pub(crate) mod cuda_test_utils;
#[cfg(not(feature = "hixl"))]
pub mod ibverbs;
#[cfg(not(feature = "hixl"))]
pub mod tcp;

#[cfg(feature = "hixl")]
pub mod hixl;

use std::fmt::Debug;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use async_trait::async_trait;
use hyperactor::reference;
use serde::Deserialize;
use serde::Serialize;

use crate::RdmaOp;
use crate::RdmaTransportLevel;

/// Backend-specific context for a remote buffer.
///
/// - **Ibverbs**: native Rust-managed QP/MR transport (GPU).
/// - **Tcp**: TCP fallback transport.
/// - **Hixl**: Rust-managed HIXL transport (Ascend NPU).
#[derive(Debug, Clone)]
pub enum RdmaRemoteBackendContext {
    #[cfg(not(feature = "hixl"))]
    Ibverbs(
        reference::ActorRef<ibverbs::manager_actor::IbvManagerActor>,
        Arc<tokio::sync::OnceCell<ibverbs::IbvBuffer>>,
    ),
    #[cfg(not(feature = "hixl"))]
    Tcp(reference::ActorRef<tcp::manager_actor::TcpManagerActor>),
    #[cfg(feature = "hixl")]
    Hixl(hixl::HixlBuffer),
}

impl Serialize for RdmaRemoteBackendContext {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            #[cfg(not(feature = "hixl"))]
            RdmaRemoteBackendContext::Ibverbs(actor_ref, _) => serializer
                .serialize_newtype_variant("RdmaRemoteBackendContext", 0, "Ibverbs", actor_ref),
            #[cfg(not(feature = "hixl"))]
            RdmaRemoteBackendContext::Tcp(actor_ref) => serializer.serialize_newtype_variant(
                "RdmaRemoteBackendContext",
                1,
                "Tcp",
                actor_ref,
            ),
            #[cfg(feature = "hixl")]
            RdmaRemoteBackendContext::Hixl(buf) => {
                serializer.serialize_newtype_variant("RdmaRemoteBackendContext", 0, "Hixl", buf)
            }
        }
    }
}

impl<'de> Deserialize<'de> for RdmaRemoteBackendContext {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(rename = "RdmaRemoteBackendContext")]
        enum Repr {
            #[cfg(not(feature = "hixl"))]
            Ibverbs(reference::ActorRef<ibverbs::manager_actor::IbvManagerActor>),
            #[cfg(not(feature = "hixl"))]
            Tcp(reference::ActorRef<tcp::manager_actor::TcpManagerActor>),
            #[cfg(feature = "hixl")]
            Hixl(hixl::HixlBuffer),
        }

        match Repr::deserialize(deserializer)? {
            #[cfg(not(feature = "hixl"))]
            Repr::Ibverbs(actor_ref) => Ok(RdmaRemoteBackendContext::Ibverbs(
                actor_ref,
                Arc::new(tokio::sync::OnceCell::new()),
            )),
            #[cfg(not(feature = "hixl"))]
            Repr::Tcp(actor_ref) => Ok(RdmaRemoteBackendContext::Tcp(actor_ref)),
            #[cfg(feature = "hixl")]
            Repr::Hixl(buf) => Ok(RdmaRemoteBackendContext::Hixl(buf)),
        }
    }
}

/// Backend for executing RDMA operations over a specific transport.
///
/// Each backend manages the transport-specific details of connection
/// management and data movement. The backend decides internally how to
/// batch and schedule submitted operations.
///
/// Current implementations:
/// - [`ibverbs::IbvManagerActor`] -- ibverbs NIC transport
/// - [`tcp::TcpManagerActor`] -- TCP fallback transport
/// - `hixl::HixlManagerActor` -- HIXL transport (Ascend NPU, feature `hixl`)
#[async_trait]
pub trait RdmaBackend: Send + Debug {
    type TransportInfo;

    async fn submit(
        &mut self,
        cx: &(impl hyperactor::context::Actor + Send + Sync),
        ops: Vec<RdmaOp>,
        timeout: Duration,
    ) -> Result<()>;

    fn transport_level(&self) -> RdmaTransportLevel;

    fn transport_info(&self) -> Option<Self::TransportInfo>;
}
