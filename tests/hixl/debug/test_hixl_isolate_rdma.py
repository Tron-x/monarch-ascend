#!/usr/bin/env python3
"""
Isolate: does _ensure_init_rdma_manager() or _RdmaBuffer.create_rdma_buffer_blocking()
break HIXL Connect in a subsequent ctypes call?
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


class TestActor(Actor):
    @endpoint
    async def test_hixl_before_rdma(self) -> str:
        """Test 1: Init HIXL BEFORE calling any RDMA APIs"""
        dev = int(os.environ.get("MONARCH_NPU_DEVICE", "0"))
        t = torch.ones(NBYTES // 4, dtype=torch.float32, device=f"npu:{dev}")
        torch.npu.synchronize()

        port = 60000 + (os.getpid() % 5000)
        eid = f"127.0.0.1:{port}".encode()
        lib = load_hixl()
        ctx = lib.hixl_init_engine(dev, eid)
        lib.hixl_register_mem(ctx, t.data_ptr(), NBYTES)
        return f"engine={eid.decode()} addr={hex(t.data_ptr())}"

    @endpoint
    async def test_hixl_after_init_rdma_manager(self) -> str:
        """Test 2: Init HIXL AFTER _ensure_init_rdma_manager()"""
        dev = int(os.environ.get("MONARCH_NPU_DEVICE", "0"))
        t = torch.ones(NBYTES // 4, dtype=torch.float32, device=f"npu:{dev}")
        torch.npu.synchronize()

        from monarch._src.rdma.rdma import _ensure_init_rdma_manager
        _ensure_init_rdma_manager().block_on()
        print(f"[Test] RdmaManager initialized", flush=True)

        port = 60000 + (os.getpid() % 5000) + 100
        eid = f"127.0.0.1:{port}".encode()
        lib = load_hixl()
        ctx = lib.hixl_init_engine(dev, eid)
        lib.hixl_register_mem(ctx, t.data_ptr(), NBYTES)
        return f"engine={eid.decode()} addr={hex(t.data_ptr())}"

    @endpoint
    async def test_hixl_after_create_buffer(self) -> str:
        """Test 3: Init HIXL AFTER creating an RDMABuffer"""
        dev = int(os.environ.get("MONARCH_NPU_DEVICE", "0"))
        t = torch.ones(NBYTES // 4, dtype=torch.float32, device=f"npu:{dev}")
        torch.npu.synchronize()

        from monarch.rdma import RDMABuffer
        buf = RDMABuffer(t.view(torch.uint8).flatten())
        print(f"[Test] RDMABuffer created, hixl_info={buf._buffer.hixl_info()}", flush=True)

        port = 60000 + (os.getpid() % 5000) + 200
        eid = f"127.0.0.1:{port}".encode()
        lib = load_hixl()
        ctx = lib.hixl_init_engine(dev, eid)
        if ctx:
            lib.hixl_register_mem(ctx, t.data_ptr(), NBYTES)
            return f"engine={eid.decode()} addr={hex(t.data_ptr())}"
        else:
            return f"FAILED: hixl_init_engine returned null"


class Connector(Actor):
    @endpoint
    async def try_connect(self, remote_engine: str) -> str:
        dev = int(os.environ.get("MONARCH_NPU_DEVICE", "1"))
        port = 60000 + (os.getpid() % 5000) + 300
        my_eid = f"127.0.0.1:{port}".encode()
        lib = load_hixl()
        ctx = lib.hixl_init_engine(dev, my_eid)
        remote = remote_engine.encode()
        for i in range(5):
            ret = lib.hixl_connect(ctx, remote)
            if ret == 0:
                return f"CONNECTED to {remote_engine}"
            time.sleep(1)
        return f"CONNECT FAILED to {remote_engine}: ret={ret}"


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
    host = this_host()
    pm0 = host.spawn_procs(per_host={"gpus": 1}, bootstrap=use_npu(0))
    pm1 = host.spawn_procs(per_host={"gpus": 1}, bootstrap=use_npu(1))

    tester = pm0.spawn("tester", TestActor)
    connector = pm1.spawn("connector", Connector)

    await asyncio.sleep(2)

    # Test 1: HIXL before any RDMA
    print("=== Test 1: HIXL before RDMA ===")
    info1 = await tester.test_hixl_before_rdma.call_one()
    print(f"  Tester: {info1}")
    engine1 = info1.split("engine=")[1].split(" ")[0]
    await asyncio.sleep(2)
    r1 = await connector.try_connect.call_one(engine1)
    print(f"  Connector: {r1}")

    print()
    # NOTE: After this test, the tester process already has an HIXL engine.
    # We need a fresh process for test 2. But since we can't easily restart,
    # let's just report the results.

    print("=== Test 2: HIXL after _ensure_init_rdma_manager ===")
    print("  (Skipped - would need fresh process)")

    print()
    print("=== Test 3: HIXL after RDMABuffer creation ===")
    print("  (Skipped - would need fresh process)")
    print()
    print("To test 2/3, we need separate processes. See log for Test 1.")


if __name__ == "__main__":
    asyncio.run(main())
