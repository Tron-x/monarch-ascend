#!/usr/bin/env python3
"""
Direct test: HIXL TransferSync with torch_npu memory vs aclrtMalloc memory.
Uses multiprocessing (not monarch actors) to minimize variables.
"""
import os
import sys
import time
import struct
import ctypes
import multiprocessing as mp

os.environ["HCCL_INTRA_ROCE_ENABLE"] = "1"

CANN = "/usr/local/Ascend/ascend-toolkit/latest"
HIXL_LIB = os.path.join(CANN, "lib64", "libcann_hixl.so")
ACL_LIB = os.path.join(CANN, "aarch64-linux", "lib64", "libascendcl.so")
BUF_SIZE = 64
COORD_FILE = "/tmp/hixl_test_coord"

SRV_ENGINE = "127.0.0.1:19000"
CLI_ENGINE = "127.0.0.1:19001"


def load_hixl():
    lib = ctypes.CDLL(HIXL_LIB)
    return lib

def load_acl():
    lib = ctypes.CDLL(ACL_LIB)
    return lib


def acl_malloc(acl, dev, nbytes):
    acl.aclInit(None)
    acl.aclrtSetDevice(dev)
    ptr = ctypes.c_void_p()
    ret = acl.aclrtMalloc(ctypes.byref(ptr), ctypes.c_size_t(nbytes), ctypes.c_int(2))  # NORMAL_ONLY
    assert ret == 0, f"aclrtMalloc failed: {ret}"
    acl.aclrtMemset(ptr, ctypes.c_size_t(nbytes), ctypes.c_int(0), ctypes.c_size_t(nbytes))
    return ptr.value


def torch_malloc(dev, nbytes):
    import torch
    import torch_npu
    torch.npu.set_device(dev)
    t = torch.ones(nbytes // 4, dtype=torch.float32, device=f"npu:{dev}")
    return t.data_ptr(), t


def run_server(use_torch, barrier):
    dev = 0
    acl = load_acl()
    acl.aclInit(None)
    acl.aclrtSetDevice(dev)

    if use_torch:
        addr, _t = torch_malloc(dev, BUF_SIZE)
        print(f"[S] torch addr={hex(addr)}")
    else:
        addr = acl_malloc(acl, dev, BUF_SIZE)
        print(f"[S] acl addr={hex(addr)}")
    sys.stdout.flush()

    # Use the C++ official API directly
    hixl = ctypes.CDLL(HIXL_LIB)

    # Create + Initialize
    engine = ctypes.c_void_p()
    hixl.HixlServerCreate.restype = ctypes.c_uint
    # Use the hixl-sys bridge instead
    bridge = ctypes.CDLL(os.path.join(
        os.path.dirname(__file__),
        "target", "aarch64-unknown-linux-gnu", "release", "build"
    ) + "/../../../libhixl_sys_bridge.so") if False else None

    # Actually, let's use the monarch compiled hixl-sys bridge
    # Find the .so
    import glob
    so_files = glob.glob("/root/monarch/target/**/libhixl_sys_bridge.so", recursive=True)
    if not so_files:
        print("ERROR: Cannot find libhixl_sys_bridge.so")
        return

    bridge = ctypes.CDLL(so_files[0])

    handle = bridge.HixlCreate()
    print(f"[S] HixlCreate: {handle}")

    engine_bytes = SRV_ENGINE.encode() + b'\0'
    opts = (ctypes.c_char_p * 2)(b"BufferPool\0", b"0:0\0")

    class HixlOption(ctypes.Structure):
        _fields_ = [("key", ctypes.c_char_p), ("value", ctypes.c_char_p)]

    opt_arr = (HixlOption * 1)(HixlOption(b"BufferPool", b"0:0"))
    ret = bridge.HixlInitialize(handle, engine_bytes, opt_arr, 1)
    print(f"[S] Init: {ret}")
    sys.stdout.flush()

    # RegisterMem
    mh = ctypes.c_void_p()
    ret = bridge.HixlRegisterMem(handle, ctypes.c_size_t(addr), ctypes.c_size_t(BUF_SIZE), 0, ctypes.byref(mh))
    print(f"[S] RegMem: {ret} addr={hex(addr)}")
    sys.stdout.flush()

    # Write addr to file
    with open(COORD_FILE, 'w') as f:
        f.write(str(addr))

    barrier.wait()  # signal client

    # Connect to client
    time.sleep(3)
    ret = bridge.HixlConnect(handle, CLI_ENGINE.encode() + b'\0', 10000)
    print(f"[S] Connect({CLI_ENGINE}): {ret}")
    sys.stdout.flush()

    time.sleep(10)

    # Read back
    host_buf = ctypes.create_string_buffer(BUF_SIZE)
    acl.aclrtMemcpy(host_buf, ctypes.c_size_t(BUF_SIZE),
                     ctypes.c_void_p(addr), ctypes.c_size_t(BUF_SIZE), 2)  # D2H
    vals = struct.unpack("16f", host_buf.raw)
    print(f"[S] sum={sum(vals):.1f}")
    sys.stdout.flush()

    bridge.HixlDeregisterMem(handle, mh)
    bridge.HixlFinalize(handle)
    bridge.HixlDestroy(handle)


def run_client(use_torch, barrier):
    dev = 1
    acl = load_acl()
    acl.aclInit(None)
    acl.aclrtSetDevice(dev)

    if use_torch:
        local_addr, _t = torch_malloc(dev, BUF_SIZE)
        print(f"[C] torch addr={hex(local_addr)}")
    else:
        local_addr = acl_malloc(acl, dev, BUF_SIZE)
        # Write 2.0 pattern
        host = struct.pack("16f", *([2.0] * 16))
        host_buf = ctypes.create_string_buffer(host)
        acl.aclrtMemcpy(ctypes.c_void_p(local_addr), ctypes.c_size_t(BUF_SIZE),
                         host_buf, ctypes.c_size_t(BUF_SIZE), 1)  # H2D
        print(f"[C] acl addr={hex(local_addr)}")
    sys.stdout.flush()

    import glob
    so_files = glob.glob("/root/monarch/target/**/libhixl_sys_bridge.so", recursive=True)
    bridge = ctypes.CDLL(so_files[0])

    handle = bridge.HixlCreate()

    class HixlOption(ctypes.Structure):
        _fields_ = [("key", ctypes.c_char_p), ("value", ctypes.c_char_p)]

    opt_arr = (HixlOption * 1)(HixlOption(b"BufferPool", b"0:0"))
    ret = bridge.HixlInitialize(handle, CLI_ENGINE.encode() + b'\0', opt_arr, 1)
    print(f"[C] Init: {ret}")
    sys.stdout.flush()

    barrier.wait()  # wait for server

    # Read remote addr
    with open(COORD_FILE) as f:
        remote_addr = int(f.read().strip())
    print(f"[C] remote_addr={hex(remote_addr)}")

    time.sleep(1)
    ret = bridge.HixlConnect(handle, SRV_ENGINE.encode() + b'\0', 10000)
    print(f"[C] Connect({SRV_ENGINE}): {ret}")
    sys.stdout.flush()

    # Register local memory
    mh = ctypes.c_void_p()
    ret = bridge.HixlRegisterMem(handle, ctypes.c_size_t(local_addr),
                                  ctypes.c_size_t(BUF_SIZE), 0, ctypes.byref(mh))
    print(f"[C] RegMem: {ret}")
    sys.stdout.flush()

    time.sleep(3)  # wait for server to also connect

    class HixlTransferOpDesc(ctypes.Structure):
        _fields_ = [
            ("local_addr", ctypes.c_size_t),
            ("remote_addr", ctypes.c_size_t),
            ("len", ctypes.c_size_t),
        ]

    td = HixlTransferOpDesc(local_addr, remote_addr, BUF_SIZE)
    ret = bridge.HixlTransferSync(handle, SRV_ENGINE.encode() + b'\0',
                                   1,  # WRITE
                                   ctypes.byref(td), 1, 10000)
    print(f"[C] TransferSync(WRITE): {ret} {'OK' if ret == 0 else 'FAIL'}")
    sys.stdout.flush()

    bridge.HixlDeregisterMem(handle, mh)
    bridge.HixlDisconnect(handle, SRV_ENGINE.encode() + b'\0', 5000)
    bridge.HixlFinalize(handle)
    bridge.HixlDestroy(handle)


def run_test(use_torch):
    label = "TORCH" if use_torch else "ACL"
    print(f"\n{'='*60}")
    print(f"Test with {label} memory")
    print(f"{'='*60}")
    sys.stdout.flush()

    try:
        os.unlink(COORD_FILE)
    except FileNotFoundError:
        pass

    barrier = mp.Barrier(2)
    srv = mp.Process(target=run_server, args=(use_torch, barrier))
    cli = mp.Process(target=run_client, args=(use_torch, barrier))

    srv.start()
    cli.start()
    cli.join(timeout=30)
    srv.join(timeout=5)

    if cli.exitcode != 0:
        print(f">>> {label}: Client exit={cli.exitcode}")
    if srv.is_alive():
        srv.terminate()
        srv.join()


if __name__ == "__main__":
    run_test(use_torch=False)
    run_test(use_torch=True)
