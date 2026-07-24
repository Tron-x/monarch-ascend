# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe
import sys
from typing import cast

import pytest

if sys.platform != "linux":
    pytest.skip("linux-only", allow_module_level=True)

from isolate_in_subprocess import isolate_in_subprocess
from monarch._src.actor.actor_mesh import ActorMesh
from monarch._src.actor.proc_mesh import get_or_spawn_controller
from monarch._src.rdma.rdma import RdmaController
from monarch.actor import Actor, endpoint, this_host
from monarch.config import configured
from monarch.rdma import RDMABuffer
from rdma_test_utils import RDMA_BACKENDS, skip_if_ibverbs_unavailable

_PAYLOAD = bytes(range(256))


def _backend_config(rdma_backend: str):
    if rdma_backend == "tcp":
        return configured(rdma_disable_ibverbs=True)
    skip_if_ibverbs_unavailable()
    return configured(rdma_allow_tcp_fallback=False)


class CpuActor(Actor):
    def __init__(self) -> None:
        self.data = bytearray(_PAYLOAD)
        self.buf = None

    @endpoint
    async def ping(self) -> str:
        return "pong"

    @endpoint
    async def create_buffer(self) -> RDMABuffer:
        self.buf = RDMABuffer(memoryview(self.data))
        return self.buf

    @endpoint
    async def read_buffer(self, buf: RDMABuffer) -> bytes:
        dst = bytearray(len(_PAYLOAD))
        await buf.read_into(memoryview(dst))
        return bytes(dst)

    @endpoint
    async def coordinator_name(self) -> str:
        # Internal mesh name of the shared coordinator; differs if it is respawned.
        coord = await get_or_spawn_controller("rdma_controller", RdmaController)
        return await cast(ActorMesh, coord)._name


@pytest.mark.parametrize("rdma_backend", RDMA_BACKENDS)
@pytest.mark.timeout(120)
@isolate_in_subprocess
async def test_shared_rdma_coordinator_survives_consumer_exit(rdma_backend) -> None:
    """A consumer's clean stop() leaves the shared RDMA coordinator usable and unchanged for others."""
    with _backend_config(rdma_backend):
        producer_proc = this_host().spawn_procs(per_host={"processes": 1})
        consumer_a_proc = this_host().spawn_procs(per_host={"processes": 1})

        producer = producer_proc.spawn("producer", CpuActor)
        consumer_a = consumer_a_proc.spawn("consumer_a", CpuActor)

        await producer.ping.call_one()
        buf = await producer.create_buffer.call_one()

        assert await consumer_a.read_buffer.call_one(buf) == _PAYLOAD
        coordinator = await producer.coordinator_name.call_one()

        await consumer_a_proc.stop()

        # Fresh proc = cold cache, so its read re-contacts the coordinator; a warm
        # consumer would reuse its cached manager and never re-contact it.
        consumer_c_proc = this_host().spawn_procs(per_host={"processes": 1})
        consumer_c = consumer_c_proc.spawn("consumer_c", CpuActor)
        assert await consumer_c.read_buffer.call_one(buf) == _PAYLOAD
        assert await consumer_c.coordinator_name.call_one() == coordinator

        await producer_proc.stop()
        await consumer_c_proc.stop()
