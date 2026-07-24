#!/usr/bin/env python3
"""
Debug: isolate whether RdmaManagerActor spawning or RDMABuffer creation breaks HIXL.
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

COORD = "/tmp/hixl_rdma_debug_coord"
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
    @endpoint
    async def setup(self) -> str:
        dev = int(os.environ.get("MONARCH_NPU_DEVICE", "0"))
        self.tensor = torch.ones(NBYTES // 4, dtype=torch.float32, device=f"npu:{dev}")
        torch.npu.synchronize()
        addr = self.tensor.data_ptr()

        # Step 1: Create RDMABuffer (this triggers Rust RdmaManagerActor)
        print(f"[Producer] Creating RDMABuffer...", flush=True)
        byte_view = self.tensor.view(torch.uint8).flatten()
        self.buf = RDMABuffer(byte_view)
        print(f"[Producer] RDMABuffer created, hixl_info={self.buf._buffer.hixl_info()}", flush=True)

        # Step 2: AFTER RDMABuffer is created, init our own HIXL engine (for direct ctypes test)
        port = 60000 + (os.getpid() % 5000)
        engine_id = f"127.0.0.1:{port}".encode()
        lib = load_hixl()
        ctx = lib.hixl_init_engine(dev, engine_id)
        ret = lib.hixl_register_mem(ctx, addr, NBYTES)
        print(f"[Producer] ctypes HIXL init: engine={engine_id}, ctx={ctx}, reg={ret}", flush=True)

        with open(COORD, 'w') as f:
            f.write(f"{addr}\n{engine_id.decode()}\n")

        return f"ready engine={engine_id.decode()}"


class Consumer(Actor):
    @endpoint
    async def test_connect(self) -> str:
        dev = int(os.environ.get("MONARCH_NPU_DEVICE", "1"))
        local = torch.full((NBYTES // 4,), 2.0, dtype=torch.float32, device=f"npu:{dev}")
        torch.npu.synchronize()
        local_addr = local.data_ptr()

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
        for i in range(5):
            ret = lib.hixl_connect(ctx, remote_bytes)
            print(f"[Consumer] connect attempt {i+1} to {remote_engine}: {ret}", flush=True)
            if ret == 0:
                break
            time.sleep(1)

        if ret == 0:
            ret = lib.hixl_transfer_write(ctx, remote_bytes, local_addr, remote_addr, NBYTES)
            return f"Transfer: {ret} {'OK' if ret==0 else 'FAIL'}"
        else:
            return f"Connect FAILED: {ret}"


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
    print("Debug: RDMABuffer + direct HIXL ctypes in same process")
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

    await asyncio.sleep(2)

    producer_result = await producer.setup.call_one()
    print(f"Producer: {producer_result}", flush=True)

    await asyncio.sleep(3)

    result = await consumer.test_connect.call_one()
    print(f"Consumer: {result}", flush=True)
    print("PASS" if "OK" in result else "FAIL")


if __name__ == "__main__":
    asyncio.run(main())
