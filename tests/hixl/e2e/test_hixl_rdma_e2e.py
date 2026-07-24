#!/usr/bin/env python3
"""
End-to-end test: RDMABuffer with HIXL backend on two NPUs.
Producer creates tensor + RDMABuffer on NPU 0.
Consumer creates tensor on NPU 1 and writes to Producer's RDMABuffer via HIXL.
"""
import asyncio
import os
import sys
import time

os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("HCCL_INTRA_ROCE_ENABLE", "1")
os.environ.setdefault("ASCEND_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/latest")

import torch
try:
    import torch_npu
except ImportError:
    print("ERROR: torch_npu not available"); sys.exit(1)

from monarch.actor import Actor, endpoint, this_host
from monarch.rdma import RDMABuffer

NBYTES = 64


class Producer(Actor):
    def __init__(self):
        dev = int(os.environ.get("MONARCH_NPU_DEVICE", "0"))
        self.tensor = torch.ones(NBYTES // 4, dtype=torch.float32, device=f"npu:{dev}")
        torch.npu.synchronize()

        print(f"[Producer PID={os.getpid()}] Creating RDMABuffer on NPU {dev}...", flush=True)
        t0 = time.time()
        byte_view = self.tensor.view(torch.uint8).flatten()
        self.buf = RDMABuffer(byte_view)
        print(f"[Producer] RDMABuffer created OK ({time.time()-t0:.3f}s)", flush=True)

        hixl_info = self.buf._buffer.hixl_info()
        print(f"[Producer] hixl_info={hixl_info}", flush=True)

    @endpoint
    async def get_buffer(self) -> RDMABuffer:
        return self.buf

    @endpoint
    async def read_sum(self) -> float:
        return self.tensor.sum().cpu().item()


class Consumer(Actor):
    @endpoint
    async def do_write(self, remote_buf: RDMABuffer) -> str:
        dev = int(os.environ.get("MONARCH_NPU_DEVICE", "1"))
        local = torch.full((NBYTES // 4,), 2.0, dtype=torch.float32, device=f"npu:{dev}")
        torch.npu.synchronize()
        local_bytes = local.view(torch.uint8).flatten()

        print(f"[Consumer PID={os.getpid()}] Writing to remote buffer from NPU {dev}...", flush=True)
        t0 = time.time()
        try:
            await remote_buf.write_from(local_bytes)
            dt = time.time() - t0
            print(f"[Consumer] write_from completed in {dt:.3f}s", flush=True)
            return f"OK ({dt:.3f}s)"
        except Exception as e:
            print(f"[Consumer] write_from FAILED: {e}", flush=True)
            return f"FAIL: {e}"


def use_npu(dev_id: int):
    def _bootstrap():
        os.environ["MONARCH_NPU_DEVICE"] = str(dev_id)
        os.environ["HCCL_INTRA_ROCE_ENABLE"] = "1"
        print(
            f"[Bootstrap PID={os.getpid()}] NPU {dev_id}; "
            "Rust RDMA manager will allocate the HIXL engine_id",
            flush=True,
        )
        import torch
        import torch_npu
        torch.npu.set_device(dev_id)
    return _bootstrap


async def main():
    print("=" * 60)
    print("E2E Test: RDMABuffer + HIXL on two NPUs")
    print("=" * 60, flush=True)

    host = this_host()
    pm = host.spawn_procs(per_host={"gpus": 1}, bootstrap=use_npu(0))
    cm = host.spawn_procs(per_host={"gpus": 1}, bootstrap=use_npu(1))

    producer = pm.spawn("producer", Producer)
    consumer = cm.spawn("consumer", Consumer)

    await asyncio.sleep(3)

    print("[Main] Getting buffer from producer...", flush=True)
    buf = await producer.get_buffer.call_one()
    print(f"[Main] Got buffer: size={buf.size()}", flush=True)

    print("[Main] Calling consumer.do_write...", flush=True)
    result = await consumer.do_write.call_one(buf)
    print(f"[Main] Write result: {result}", flush=True)

    await asyncio.sleep(1)
    final_sum = await producer.read_sum.call_one()
    print(f"[Main] Producer sum: {final_sum} (expect 32.0)", flush=True)
    print("PASS" if "OK" in result else "FAIL")


if __name__ == "__main__":
    asyncio.run(main())
