# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

"""Entry point for the background mount process.

Launched by :func:`~monarch._src.job.mount_config.Mounts.ensure_open`.
Receives the socket path and lock fd as arguments. Binds the socket
immediately to signal readiness, then serves refresh/shutdown requests.
The first ``refresh`` initialises the mounts; subsequent ones refresh them.

Usage::

    python -m monarch._src.job._mount_worker <socket_path> <lock_fd>
"""

import os
import sys


def main() -> None:
    socket_path = sys.argv[1]
    int(sys.argv[2])  # keep fd open to hold the flock for the process lifetime

    # Re-apply the parent's transport selection. mount_worker is a fresh
    # subprocess, so the in-memory transport configured by the parent's
    # ``enable_transport()`` call is not inherited. Without this, any
    # cross-host operation we issue (notably ``host_mesh.spawn_procs()`` for
    # FUSEActors) advertises the default Unix abstract socket as the client
    # callback address, which remote hosts cannot reach -- causing FUSEActor
    # spawns on remote workers to time out after 30s.
    #
    # ``enable_transport()`` must run before any other monarch API call, so
    # we resolve the transport here, before importing ``mount_config``.
    transport = os.environ.get("MONARCH_DEFAULT_TRANSPORT")
    if transport is None:
        # remote_mount/gather_mount across hosts always require a routable
        # transport; default to TCP when the parent didn't set the hint
        # (e.g. older monarch versions). Single-host LocalJob users can opt
        # into Unix explicitly via ``MONARCH_DEFAULT_TRANSPORT=ipc``.
        transport = "tcp"

    from monarch._src.actor.actor_mesh import enable_transport

    enable_transport(transport)

    from monarch._src.job.mount_config import _run_mount_process

    _run_mount_process(socket_path)


if __name__ == "__main__":
    main()
