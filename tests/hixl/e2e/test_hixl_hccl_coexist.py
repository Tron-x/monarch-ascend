#!/usr/bin/env python3
"""
Coexist test: HiXL + torch.distributed HCCL process_group on the same node.

This targets historical Issue #1 (`HcclCommPrepare ret=0x13`) where HiXL's
2-rank HCCL comm and torch.distributed's HCCL process_group fought over
the same `HcclAdapter` singleton.

Design:
  - Spawn 2 monarch actors on NPU 0 / NPU 1 (mesh_a / mesh_b).
  - Each actor calls `torch.distributed.init_process_group(backend='hccl')`
    against a shared TCP store on master = mesh_a, so the 2-rank HCCL group
    is initialized FIRST.
  - Then mesh_a creates an RDMABuffer (HiXL register_mem + connect) and
    mesh_b does cross-mesh write_from on it.
  - PASS iff the HCCL all-reduce works AND the subsequent HiXL transfer
    works without 0x13 / 503900.
"""

import asyncio
import os
import sys

os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29501")
os.environ.setdefault("HCCL_CONNECT_TIMEOUT", "120")
# LocalCommRes isolates HiXL from torch.distributed's HCCL communicator and
# validates the deployed CANN 9.1 AICPU kernel.
os.environ.setdefault("MONARCH_HIXL_USE_LOCAL_COMM_RES", "1")

import torch

try:
    import torch_npu  # noqa: F401
except ImportError:
    print("ERROR: torch_npu not available")
    sys.exit(1)

from monarch._src.rdma.xdma import alloc_aligned_tensor
from monarch._src.actor.host_mesh import default_bootstrap_cmd
from monarch.actor import Actor, endpoint, this_host
from monarch.rdma import RDMABuffer

ELEMS = 64 * 1024


def npu_device(dev_id: int, rank: int, world_size: int):
    def _bootstrap():
        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = str(dev_id)
        os.environ["MONARCH_NPU_DEVICE"] = "0"
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = "0"
        import torch
        import torch_npu  # noqa: F401
        torch.npu.set_device(0)
    return _bootstrap


def npu_bootstrap_command(dev_id: int, rank: int, world_size: int):
    return default_bootstrap_cmd().with_env(
        {
            "ASCEND_RT_VISIBLE_DEVICES": str(dev_id),
            "HCCL_NPU_SOCKET_PORT_RANGE": f"{62000 + rank * 100}-{62049 + rank * 100}",
            "LOCAL_RANK": "0",
            "MONARCH_HIXL_USE_LOCAL_COMM_RES": "1",
            "MONARCH_NPU_DEVICE": "0",
            "RANK": str(rank),
            "WORLD_SIZE": str(world_size),
        }
    )


class HcclThenHixl(Actor):
    def __init__(self):
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        print(f"[Rank {rank}] init_process_group(hccl) rank={rank}/{world_size}")
        torch.distributed.init_process_group(
            backend="hccl", rank=rank, world_size=world_size
        )
        print(f"[Rank {rank}] HCCL group initialized OK")
        self.t, self._raw = alloc_aligned_tensor(
            (ELEMS,), dtype=torch.float32, device="npu:0"
        )
        self.t.fill_(float(rank + 1))
        torch.npu.synchronize()
        self.buf = None
        self.rank = rank

    @endpoint
    async def allreduce(self) -> float:
        # 1 KB ping over HCCL — proves HCCL group is alive
        x = torch.full((256,), float(self.rank + 1), dtype=torch.float32, device="npu")
        torch.distributed.all_reduce(x)
        torch.npu.synchronize()
        return x.sum().cpu().item()

    @endpoint
    async def get_buf(self):
        if self.buf is None:
            self.buf = RDMABuffer(self.t.view(torch.uint8).flatten())
        return self.buf

    @endpoint
    async def get_sum(self) -> float:
        return self.t.sum().cpu().item()

    @endpoint
    async def write_to(self, remote: RDMABuffer, fill: float) -> float:
        local, _raw = alloc_aligned_tensor(
            (ELEMS,), dtype=torch.float32, device="npu:0"
        )
        local.fill_(fill)
        torch.npu.synchronize()
        await remote.write_from(local.view(torch.uint8).flatten(), timeout=30)
        return local.sum().cpu().item()


async def main():
    print("=" * 60)
    print("HiXL + torch.distributed HCCL coexist test")
    print("  rank 0 (mesh_a, NPU 0) and rank 1 (mesh_b, NPU 1)")
    print("  → 1. HCCL all_reduce works,")
    print("  → 2. HiXL register_mem + cross-mesh write_from after HCCL init")
    print("=" * 60)

    host = this_host()
    mesh_a = host.spawn_procs(
        per_host={"npus": 1},
        bootstrap=npu_device(0, 0, 2),
        bootstrap_command=npu_bootstrap_command(0, 0, 2),
    )
    mesh_b = host.spawn_procs(
        per_host={"npus": 1},
        bootstrap=npu_device(1, 1, 2),
        bootstrap_command=npu_bootstrap_command(1, 1, 2),
    )

    actor_a = mesh_a.spawn("a", HcclThenHixl)
    actor_b = mesh_b.spawn("b", HcclThenHixl)

    print("[1/3] HCCL all_reduce (proves init_process_group works)...")
    # Run both ranks in parallel so all_reduce can complete
    res_a, res_b = await asyncio.gather(
        actor_a.allreduce.call_one(),
        actor_b.allreduce.call_one(),
    )
    print(f"      rank 0 sees: {res_a}  (expect {(1+2)*256} = {3*256})")
    print(f"      rank 1 sees: {res_b}  (expect {3*256})")
    if abs(res_a - 3 * 256) > 1e-4 or abs(res_b - 3 * 256) > 1e-4:
        raise RuntimeError("HCCL all_reduce mismatch — HCCL group broken")

    print()
    print("[2/3] After HCCL init, do HiXL cross-mesh write...")
    handle_a = await actor_a.get_buf.call_one()
    pushed = await actor_b.write_to.call_one(handle_a, 7.0)
    print(f"      consumer side fill_sum = {pushed}")

    print()
    print("[3/3] Verify producer side received the data...")
    s_after = await actor_a.get_sum.call_one()
    expect = ELEMS * 7.0
    print(f"      rank 0 buf sum = {s_after}  (expect {expect})")
    if abs(s_after - expect) > 1e-4:
        raise RuntimeError(
            "HiXL transfer corrupted data when HCCL pg active — Issue #1 still present"
        )

    print()
    print("PASS: HiXL and HCCL process_group coexist on same node")
    print("  → Issue #1 (HcclAdapter contention) appears to be FIXED")


if __name__ == "__main__":
    from monarch._src.actor.actor_mesh import shutdown_context

    try:
        asyncio.run(main())
    finally:
        shutdown_context().get(timeout=90.0)
