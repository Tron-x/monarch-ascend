#!/usr/bin/env python3
"""
Test: does RDMABuffer creation work alongside Python ctypes HIXL engine?
"""
import asyncio
import os
import sys
import time
import ctypes

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

COORD = "/tmp/hixl_buffer_effect"
NBYTES = 64


def load_hixl():
    lib = ctypes.CDLL("/root/monarch/libtest_hixl.so")
    lib.hixl_init_engine.restype = ctypes.c_void_p
    lib.hixl_init_engine.argtypes = [ctypes.c_int, ctypes.c_char_p]
    lib.hixl_register_mem.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t]
    lib.hixl_connect.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.hixl_connect.restype = ctypes.c_int
    lib.hixl_transfer_write.argtypes = [ctypes.c_void_p, ctypes.c_char_p,
                                         ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t]
    lib.hixl_transfer_write.restype = ctypes.c_int
    return lib


class Producer(Actor):
    def __init__(self):
        dev = int(os.environ.get("MONARCH_NPU_DEVICE", "0"))
        self.dev = dev
        self.tensor = torch.ones(NBYTES // 4, dtype=torch.float32, device=f"npu:{dev}")
        torch.npu.synchronize()
        self.addr = self.tensor.data_ptr()

        port = 60000 + (os.getpid() % 5000)
        self.engine_id = f"127.0.0.1:{port}".encode()
        self.lib = load_hixl()
        t0 = time.time()
        self.ctx = self.lib.hixl_init_engine(dev, self.engine_id)
        self.lib.hixl_register_mem(self.ctx, self.addr, NBYTES)
        print(f"[Producer] HIXL engine ready: {self.engine_id} ({time.time()-t0:.3f}s)", flush=True)

        print(f"[Producer] Creating RDMABuffer...", flush=True)
        t0 = time.time()
        byte_view = self.tensor.view(torch.uint8).flatten()
        self.buf = RDMABuffer(byte_view)
        print(f"[Producer] RDMABuffer created OK ({time.time()-t0:.3f}s)", flush=True)

        with open(COORD, 'w') as f:
            f.write(f"{self.addr}\n{self.engine_id.decode()}\n")
        print(f"[Producer] COORD written", flush=True)

    @endpoint
    async def read_sum(self) -> float:
        return self.tensor.sum().cpu().item()


class Consumer(Actor):
    @endpoint
    async def do_transfer(self) -> str:
        dev = int(os.environ.get("MONARCH_NPU_DEVICE", "1"))
        local = torch.full((NBYTES // 4,), 2.0, dtype=torch.float32, device=f"npu:{dev}")
        torch.npu.synchronize()
        local_addr = local.data_ptr()

        for attempt in range(30):
            if os.path.exists(COORD):
                break
            print(f"[Consumer] Waiting for COORD file (attempt {attempt+1})...", flush=True)
            await asyncio.sleep(1)
        else:
            return "ERROR: COORD file never appeared"

        with open(COORD) as f:
            lines = f.read().strip().split('\n')
            remote_addr = int(lines[0])
            remote_engine = lines[1]

        port = 60000 + (os.getpid() % 5000)
        my_engine = f"127.0.0.1:{port}".encode()
        lib = load_hixl()
        ctx = lib.hixl_init_engine(dev, my_engine)
        lib.hixl_register_mem(ctx, local_addr, NBYTES)

        remote_bytes = remote_engine.encode()
        time.sleep(1)
        for i in range(10):
            ret = lib.hixl_connect(ctx, remote_bytes)
            print(f"[Consumer] connect attempt {i+1}: {ret}", flush=True)
            if ret == 0:
                break
            time.sleep(1)

        if ret == 0:
            ret = lib.hixl_transfer_write(ctx, remote_bytes, local_addr, remote_addr, NBYTES)
            return f"Transfer: {ret} {'OK' if ret==0 else 'FAIL'}"
        else:
            return f"Connect failed: {ret}"


def use_npu(dev_id: int):
    def _bootstrap():
        os.environ["MONARCH_NPU_DEVICE"] = str(dev_id)
        os.environ["HCCL_INTRA_ROCE_ENABLE"] = "1"
        os.environ["MONARCH_PYTHON_HIXL_ENGINE_ID"] = f"127.0.0.1:{60000 + (os.getpid() % 5000)}"
        import torch
        import torch_npu
        torch.npu.set_device(dev_id)
    return _bootstrap


async def main():
    print("=" * 60)
    print("Test: HIXL engine BEFORE RDMABuffer creation")
    print("=" * 60, flush=True)
    try:
        os.unlink(COORD)
    except Exception:
        pass

    host = this_host()
    pm = host.spawn_procs(per_host={"gpus": 1}, bootstrap=use_npu(0))
    cm = host.spawn_procs(per_host={"gpus": 1}, bootstrap=use_npu(1))

    producer = pm.spawn("producer", Producer)
    consumer = cm.spawn("consumer", Consumer)

    result = await consumer.do_transfer.call_one()
    print(f"Transfer result: {result}", flush=True)

    final_sum = await producer.read_sum.call_one()
    print(f"Producer sum: {final_sum} (expect 32.0)", flush=True)
    print("PASS" if "OK" in result else "FAIL")


if __name__ == "__main__":
    asyncio.run(main())
