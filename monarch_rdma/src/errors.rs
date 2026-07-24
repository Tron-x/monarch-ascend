/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Typed error returned by the RDMA manager owner.

use serde::Deserialize;
use serde::Serialize;
use typeuri::Named;

/// A typed initialization failure returned by the RDMA manager owner.
///
/// This is the wire error carried by the owner's replies
/// (`Result<(), RdmaInitError>`), not a user-facing exception type — a
/// separate PyO3 binding maps it to a catchable Python exception.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Named)]
pub enum RdmaInitError {
    /// The manager's `init()` reported a failure.
    InitFailed(String),
    /// Spawning the manager (or casting to it) failed.
    SpawnFailed(String),
    /// A supervision failure was attributed to the manager.
    Supervision(String),
}
wirevalue::register_type!(RdmaInitError);

#[cfg(test)]
mod tests {
    use wirevalue::Any;

    use super::*;

    #[test]
    fn serde_round_trip_each_variant() {
        for e in [
            RdmaInitError::InitFailed("init failed".to_string()),
            RdmaInitError::SpawnFailed("spawn failed".to_string()),
            RdmaInitError::Supervision("supervision".to_string()),
        ] {
            let serialized: Any = Any::serialize(&e).unwrap();
            let decoded: RdmaInitError = serialized.deserialized().unwrap();
            assert_eq!(decoded, e);
        }
    }

    // Compile-only: `Result<(), RdmaInitError>` must be a `RemoteMessage` so it
    // can cross the owner's reply `OncePort` (the manager is remote).
    fn assert_remote_message<T: hyperactor::RemoteMessage>() {}

    #[test]
    fn result_rdma_init_error_is_remote_message() {
        assert_remote_message::<Result<(), RdmaInitError>>();
    }
}
