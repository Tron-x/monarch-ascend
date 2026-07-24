#!/usr/bin/env python3
"""
Test HCCS transport: same as test_hixl_bridge_minimal but WITHOUT
HCCL_INTRA_ROCE_ENABLE=1, so HiXL should default to HCCS for intra-node D2D.
"""

import asyncio
import os
import sys

os.environ.setdefault("PYTHONUNBUFFERED", "1")
# Explicitly REMOVE HCCL_INTRA_ROCE_ENABLE if it was inherited
os.environ.pop("HCCL_INTRA_ROCE_ENABLE", None)

import torch

try:
    import torch_npu  # noqa: F401
except ImportError:
    print("ERROR: torch_npu not available")
    sys.exit(1)

from monarch.actor import Actor, endpoint, this_host
from monarch.rdma import RDMABuffer


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
        await remote.read_into(local.view(torch.uint8).flatten(), timeout=20)
        return local.sum().cpu().item()


def use_npu(dev_id: int):
    def _bootstrap():
        os.environ["MONARCH_NPU_DEVICE"] = str(dev_id)
        # Do NOT set HCCL_INTRA_ROCE_ENABLE — let HiXL default to HCCS
        os.environ.pop("HCCL_INTRA_ROCE_ENABLE", None)
        import torch
        import torch_npu  # noqa: F401
        torch.npu.set_device(dev_id)
        print(
            f"[PID={os.getpid()}] NPU {dev_id} ready; "
            "Rust RDMA manager will allocate the HIXL engine_id",
            flush=True,
        )
        print(f"  HCCL_INTRA_ROCE_ENABLE={os.environ.get('HCCL_INTRA_ROCE_ENABLE', 'NOT SET')}", flush=True)

    return _bootstrap


async def main():
    print("=" * 60)
    print("HCCS transport test (NO HCCL_INTRA_ROCE_ENABLE)")
    print(f"  HCCL_INTRA_ROCE_ENABLE={os.environ.get('HCCL_INTRA_ROCE_ENABLE', 'NOT SET')}")
    print("  producer_mesh: NPU 0")
    print("  consumer_mesh: NPU 1")
    print("=" * 60)

    host = this_host()
    producer_mesh = host.spawn_procs(per_host={"gpus": 1}, bootstrap=use_npu(0))
    consumer_mesh = host.spawn_procs(per_host={"gpus": 1}, bootstrap=use_npu(1))

    producer = producer_mesh.spawn("producer", Producer)
    consumer = consumer_mesh.spawn("consumer", Consumer)

    print("[1/4] Build remote handle...")
    handle = await producer.get_handle.call_one()
    expected = await producer.get_sum.call_one()
    print(f"      producer sum = {expected}")
    await asyncio.sleep(3)

    print("[2/4] Cross-mesh write_from via HIXL (should use HCCS)...")
    pushed = await consumer.push.call_one(handle)
    remote_after_push = await producer.get_sum.call_one()
    print(f"      pushed sum={pushed}, producer sum after push={remote_after_push}")

    print("[3/4] Cross-mesh read_into via HIXL (should use HCCS)...")
    actual = await consumer.pull.call_one(handle)
    print(f"      consumer sum from pull = {actual}")

    print("[4/4] Verify...")
    if abs(actual - remote_after_push) > 1e-4:
        raise RuntimeError(
            f"mismatch: expected pulled sum {remote_after_push}, got {actual}"
        )

    print("PASS: HCCS transport works!")


if __name__ == "__main__":
    asyncio.run(main())
