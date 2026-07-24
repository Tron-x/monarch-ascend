#!/usr/bin/env python3
"""
Minimal two-mesh NPU RDMA repro for HIXL bridge.

This test only validates:
1) Mesh A (NPU0) creates a device RDMABuffer
2) Mesh B (NPU1) reads from that buffer via read_into()

No GRPO/training logic is involved.
"""

import asyncio
import os
import sys

os.environ.setdefault("PYTHONUNBUFFERED", "1")

import torch

try:
    import torch_npu  # noqa: F401
except ImportError:
    print("ERROR: torch_npu not available")
    sys.exit(1)

from monarch.actor import Actor, endpoint, this_host
from monarch.rdma import RDMABuffer
from monarch._src.actor.host_mesh import default_bootstrap_cmd


def npu_device(dev_id: int):
    """Bootstrap: isolate this process to a single NPU."""
    def _bootstrap():
        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = str(dev_id)
        os.environ["MONARCH_NPU_DEVICE"] = "0"
        import torch
        import torch_npu  # noqa: F401
        torch.npu.set_device(0)
        print(
            f"[bootstrap] visible={os.environ.get('ASCEND_RT_VISIBLE_DEVICES')} "
            f"hixl_dev={os.environ.get('MONARCH_NPU_DEVICE')} "
            f"hccl_ports={os.environ.get('HCCL_NPU_SOCKET_PORT_RANGE')}",
            flush=True,
        )
    return _bootstrap


def npu_bootstrap_command(dev_id: int):
    # Device visibility must be set before the worker imports torch_npu.
    return default_bootstrap_cmd().with_env(
        {
            "ASCEND_RT_VISIBLE_DEVICES": str(dev_id),
            "HCCL_NPU_SOCKET_PORT_RANGE": f"{61000 + dev_id * 100}-{61049 + dev_id * 100}",
            "MONARCH_HIXL_USE_LOCAL_COMM_RES": "1",
            "MONARCH_NPU_DEVICE": "0",
        }
    )


class Producer(Actor):
    def __init__(self):
        self.tensor = torch.arange(16, dtype=torch.float32, device="npu").reshape(4, 4)
        torch.npu.synchronize()
        self.buf = None

    @endpoint
    async def get_handle(self) -> RDMABuffer:
        if self.buf is None:
            byte_view = self.tensor.view(torch.uint8).flatten()
            self.buf = RDMABuffer(byte_view)
        return self.buf

    @endpoint
    async def get_sum(self) -> float:
        return self.tensor.sum().cpu().item()


class Consumer(Actor):
    @endpoint
    async def push(self, remote: RDMABuffer) -> float:
        local = torch.ones(4, 4, dtype=torch.float32, device="npu")
        torch.npu.synchronize()
        await remote.write_from(local.view(torch.uint8).flatten(), timeout=20)
        return local.sum().cpu().item()

    @endpoint
    async def pull(self, remote: RDMABuffer) -> float:
        local = torch.zeros(4, 4, dtype=torch.float32, device="npu")
        torch.npu.synchronize()
        await remote.read_into(local.view(torch.uint8).flatten(), timeout=20)
        torch.npu.synchronize()
        return local.sum().cpu().item()


async def main():
    print("=" * 60)
    print("Minimal HIXL bridge repro (2 meshes, 2 cards)")
    print("  producer_mesh: NPU 0")
    print("  consumer_mesh: NPU 1")
    print("=" * 60)

    host = this_host()
    producer_mesh = host.spawn_procs(
        per_host={"npus": 1},
        bootstrap=npu_device(0),
        bootstrap_command=npu_bootstrap_command(0),
    )
    consumer_mesh = host.spawn_procs(
        per_host={"npus": 1},
        bootstrap=npu_device(1),
        bootstrap_command=npu_bootstrap_command(1),
    )

    producer = producer_mesh.spawn("producer", Producer)
    consumer = consumer_mesh.spawn("consumer", Consumer)

    print("[1/4] Build remote handle...")
    handle = await producer.get_handle.call_one()
    expected = await producer.get_sum.call_one()
    print(f"      producer sum = {expected}")
    await asyncio.sleep(3)

    print("[2/4] Cross-mesh write_from via HIXL...")
    pushed = await consumer.push.call_one(handle)
    remote_after_push = await producer.get_sum.call_one()
    print(f"      pushed sum={pushed}, producer sum after push={remote_after_push}")

    print("[3/4] Cross-mesh read_into via HIXL...")
    actual = await consumer.pull.call_one(handle)
    print(f"      consumer sum from pull = {actual}")

    print("[4/4] Verify...")
    if abs(actual - remote_after_push) > 1e-4:
        raise RuntimeError(
            f"mismatch: expected pulled sum {remote_after_push}, got {actual}"
        )

    print("PASS: minimal HIXL bridge read/write path works")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    finally:
        from monarch._src.actor.actor_mesh import shutdown_context

        shutdown_context().get(timeout=75.0)
