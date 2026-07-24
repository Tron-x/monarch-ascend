#!/usr/bin/env python3
"""
Multi-region per engine-pair test (regression for HiXL old-API 503900).

This test deliberately forces monarch's HiXL backend to issue TWO
`hixl_register_mem` calls on the same engine pair — which historically
on CANN 9.0.0-beta.1 made every subsequent `TransferSync` return 503900.

If CANN 9.0.0 release has fixed the old P2P API too, this test passes
without any monarch source change.  If the limit is still in place,
the second cross-mesh transfer will fail (or the second register_mem
will get a 503900 from inside monarch's manager_actor).

Design:
  Producer (NPU 0) allocates *two independent* 2MB-aligned device
  buffers (via `alloc_aligned_tensor`, which calls aclrtMalloc directly
  and bypasses torch's caching allocator — so the two buffers are NOT
  in a range-containment relationship; monarch's alias workaround
  cannot kick in and the second registration WILL hit HiXL).

  Consumer (NPU 1) does cross-mesh write_from on both handles.

  PASS iff both writes succeed and the producer-side data matches.
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

from monarch._src.rdma.xdma import alloc_aligned_tensor
from monarch.actor import Actor, endpoint, this_host
from monarch.rdma import RDMABuffer

ELEMS = 64 * 1024  # 64K float32 = 256 KB; bumped to 2MB via alignment


def npu_device(dev_id: int):
    def _bootstrap():
        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = str(dev_id)
        os.environ["MONARCH_NPU_DEVICE"] = "0"
        import torch
        import torch_npu  # noqa: F401
        torch.npu.set_device(0)
    return _bootstrap


class Producer(Actor):
    def __init__(self):
        self.t1, self._raw1 = alloc_aligned_tensor(
            (ELEMS,), dtype=torch.float32, device="npu:0"
        )
        self.t2, self._raw2 = alloc_aligned_tensor(
            (ELEMS,), dtype=torch.float32, device="npu:0"
        )
        self.t1.fill_(1.0)
        self.t2.fill_(2.0)
        torch.npu.synchronize()
        addr1 = self.t1.data_ptr()
        addr2 = self.t2.data_ptr()
        print(
            f"[Producer] t1 addr={hex(addr1)} aligned_2mb={addr1 % (2 << 20) == 0}"
        )
        print(
            f"[Producer] t2 addr={hex(addr2)} aligned_2mb={addr2 % (2 << 20) == 0}"
        )
        print(
            f"[Producer] containment? "
            f"{(addr1 <= addr2 < addr1 + self.t1.numel()*4) or (addr2 <= addr1 < addr2 + self.t2.numel()*4)}"
        )
        self.buf1 = None
        self.buf2 = None

    @endpoint
    async def get_handles(self):
        if self.buf1 is None:
            self.buf1 = RDMABuffer(self.t1.view(torch.uint8).flatten())
        if self.buf2 is None:
            self.buf2 = RDMABuffer(self.t2.view(torch.uint8).flatten())
        return self.buf1, self.buf2

    @endpoint
    async def get_sums(self):
        return self.t1.sum().cpu().item(), self.t2.sum().cpu().item()


class Consumer(Actor):
    @endpoint
    async def push(self, remote: RDMABuffer, fill_value: float) -> float:
        local, _raw = alloc_aligned_tensor(
            (ELEMS,), dtype=torch.float32, device="npu:0"
        )
        local.fill_(fill_value)
        torch.npu.synchronize()
        await remote.write_from(local.view(torch.uint8).flatten(), timeout=30)
        return local.sum().cpu().item()


async def main():
    print("=" * 60)
    print("Multi-region per-pair HiXL test")
    print("  producer_mesh: NPU 0 — two INDEPENDENT 2MB-aligned buffers")
    print("  consumer_mesh: NPU 1 — write to both via HiXL")
    print("=" * 60)

    host = this_host()
    producer_mesh = host.spawn_procs(per_host={"npus": 1}, bootstrap=npu_device(0))
    consumer_mesh = host.spawn_procs(per_host={"npus": 1}, bootstrap=npu_device(1))

    producer = producer_mesh.spawn("producer", Producer)
    consumer = consumer_mesh.spawn("consumer", Consumer)

    print("[1/4] Get two independent handles...")
    h1, h2 = await producer.get_handles.call_one()
    s1_before, s2_before = await producer.get_sums.call_one()
    print(f"      before: t1.sum={s1_before:.1f}, t2.sum={s2_before:.1f}")
    await asyncio.sleep(2)

    print("[2/4] First write: consumer → producer.t1 (fill=10.0)")
    pushed_1 = await consumer.push.call_one(h1, 10.0)
    print(f"      consumer side: sum={pushed_1:.1f}")

    print("[3/4] SECOND write on SAME engine pair: consumer → producer.t2 (fill=20.0)")
    print("      (this is the historically-failing case on old API)")
    pushed_2 = await consumer.push.call_one(h2, 20.0)
    print(f"      consumer side: sum={pushed_2:.1f}")

    print("[4/4] Verify both buffers on producer...")
    s1_after, s2_after = await producer.get_sums.call_one()
    expect_1 = ELEMS * 10.0
    expect_2 = ELEMS * 20.0
    print(f"      after : t1.sum={s1_after:.1f} (expected {expect_1:.1f})")
    print(f"              t2.sum={s2_after:.1f} (expected {expect_2:.1f})")

    if abs(s1_after - expect_1) > 1e-4 or abs(s2_after - expect_2) > 1e-4:
        raise RuntimeError("DATA MISMATCH — multi-region transfer corrupted data")

    print()
    print("PASS: multi-region per engine-pair works on new CANN")
    print("  → Old P2P API 503900 limit appears to be FIXED")


if __name__ == "__main__":
    asyncio.run(main())
