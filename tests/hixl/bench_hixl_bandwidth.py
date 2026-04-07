#!/usr/bin/env python3
"""
HiXL single-sided communication bandwidth test.

Tests READ and WRITE throughput between two NPU cards using various buffer sizes.
"""
import argparse
import ctypes
import multiprocessing as mp
import os
import re
import time
import socket

ALIGN_2MB = 2 * 1024 * 1024
MB = 1024 * 1024
GB = 1024 * 1024 * 1024
COORD_FILE = "/tmp/hixl_bw_coord"
WARMUP_ITERS = 5
MEASURE_ITERS = 50


def parse_size(size_text):
    value = size_text.strip().replace(" ", "").upper()
    match = re.fullmatch(r"(\d+)(MB|GB)", value)
    if not match:
        raise argparse.ArgumentTypeError(
            "max_size must use MB or GB, for example 256MB or 2GB"
        )

    amount = int(match.group(1))
    unit = match.group(2)
    size_bytes = amount * (MB if unit == "MB" else GB)
    if size_bytes <= MB:
        raise argparse.ArgumentTypeError("max_size must be greater than 1MB")
    return size_bytes


def format_size(size_bytes):
    if size_bytes % GB == 0:
        return f"{size_bytes // GB} GB"
    if size_bytes % MB == 0:
        return f"{size_bytes // MB} MB"
    return f"{size_bytes} bytes"


def iter_sizes(max_size):
    size = MB
    while size <= max_size:
        yield size
        size *= 2

def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        local_ip = s.getsockname()[0]
    finally:
        s.close()
    return local_ip

def find_lib():
    paths = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "build/libtest_hixl.so"),
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                     "libtest_hixl.so"),
    ]
    env = os.environ.get("MONARCH_HIXL_LIB")
    if env and os.path.isfile(env):
        return env
    for p in paths:
        if os.path.isfile(p):
            return p
    raise FileNotFoundError("libtest_hixl.so not found")


def setup_lib():
    lib = ctypes.CDLL(find_lib())
    lib.hixl_init_engine.argtypes = [ctypes.c_int, ctypes.c_char_p]
    lib.hixl_init_engine.restype = ctypes.c_void_p
    lib.hixl_register_mem.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t]
    lib.hixl_register_mem.restype = ctypes.c_int
    lib.hixl_connect.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.hixl_connect.restype = ctypes.c_int
    lib.hixl_transfer_read.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p,
        ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
    ]
    lib.hixl_transfer_read.restype = ctypes.c_int
    lib.hixl_transfer_write.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p,
        ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
    ]
    lib.hixl_transfer_write.restype = ctypes.c_int
    lib.hixl_cleanup.argtypes = [ctypes.c_void_p]
    return lib


def alloc_aligned(size_bytes, device):
    """Allocate a 2MB-aligned NPU buffer."""
    import torch
    n_elems = (size_bytes + ALIGN_2MB + 3) // 4
    raw = torch.empty(n_elems, dtype=torch.float32, device=device)
    addr = raw.data_ptr()
    offset = (ALIGN_2MB - (addr % ALIGN_2MB)) % ALIGN_2MB
    start = offset // 4
    count = size_bytes // 4
    aligned = raw[start:start + count]
    assert aligned.data_ptr() % ALIGN_2MB == 0
    return aligned, raw


def server(dev_id, barrier, result_queue, max_size):
    """Server side: holds the data buffer, waits for client to finish."""
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = str(dev_id)
    import torch
    import torch_npu  # noqa: F401
    torch.npu.set_device(0)

    lib = setup_lib()
    # 用socket获取本机IP地址，避免环境变量未设置时使用默认IP导致连接失败
    ip = get_local_ip()
    eid = f"{ip}:{40000 + dev_id}"
    ctx = lib.hixl_init_engine(0, eid.encode())
    assert ctx, "server init failed"

    buf, raw = alloc_aligned(max_size, "npu:0")
    buf.view(torch.float32).fill_(3.14)
    torch.npu.synchronize()

    ret = lib.hixl_register_mem(ctx, buf.data_ptr(), max_size)
    assert ret == 0, f"server register_mem failed: {ret}"

    with open(COORD_FILE, "w") as f:
        f.write(f"{eid}\n{buf.data_ptr()}")

    barrier.wait()  # signal ready

    for attempt in range(30):
        peer_eid = f"{ip}:{40000 + int(open(COORD_FILE + '.client').read().strip())}"
        ret = lib.hixl_connect(ctx, peer_eid.encode())
        if ret == 0:
            break
        time.sleep(1)

    barrier.wait()  # connected
    barrier.wait()  # wait for all tests done

    lib.hixl_cleanup(ctx)


def client(dev_id, server_dev_id, barrier, result_queue, max_size):
    """Client side: connects to server and runs bandwidth tests."""
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = str(dev_id)
    import torch
    import torch_npu  # noqa: F401
    torch.npu.set_device(0)

    lib = setup_lib()
    ip = os.environ.get("MONARCH_HIXL_IP", "192.168.0.117")
    eid = f"{ip}:{40000 + dev_id}"
    ctx = lib.hixl_init_engine(0, eid.encode())
    assert ctx, "client init failed"

    buf, raw = alloc_aligned(max_size, "npu:0")
    torch.npu.synchronize()

    ret = lib.hixl_register_mem(ctx, buf.data_ptr(), max_size)
    assert ret == 0, f"client register_mem failed: {ret}"

    with open(COORD_FILE + ".client", "w") as f:
        f.write(str(dev_id))

    barrier.wait()  # wait for server ready

    with open(COORD_FILE) as f:
        lines = f.read().strip().split("\n")
        server_eid = lines[0]
        remote_addr = int(lines[1])

    for attempt in range(30):
        ret = lib.hixl_connect(ctx, server_eid.encode())
        if ret == 0:
            break
        time.sleep(1)
    assert ret == 0, "client connect failed"

    barrier.wait()  # connected

    results = []

    for size in iter_sizes(max_size):
        label = format_size(size)

        for op_name, transfer_fn in [("READ", lib.hixl_transfer_read),
                                      ("WRITE", lib.hixl_transfer_write)]:
            # warmup
            for _ in range(WARMUP_ITERS):
                transfer_fn(ctx, server_eid.encode(),
                            buf.data_ptr(), remote_addr, size)

            # measure
            t0 = time.perf_counter()
            for _ in range(MEASURE_ITERS):
                ret = transfer_fn(ctx, server_eid.encode(),
                                  buf.data_ptr(), remote_addr, size)
                if ret != 0:
                    print(f"  {op_name} {label}: transfer failed ret={ret}", flush=True)
                    break
            t1 = time.perf_counter()

            elapsed = t1 - t0
            total_bytes = size * MEASURE_ITERS
            bw_gbps = (total_bytes / elapsed) / (1024 ** 3)
            lat_ms = (elapsed / MEASURE_ITERS) * 1000

            results.append((op_name, label, size, bw_gbps, lat_ms))
            print(f"  {op_name:5s} {label:>8s}: {bw_gbps:8.2f} GB/s  "
                  f"(latency {lat_ms:.3f} ms, {MEASURE_ITERS} iters)", flush=True)

    result_queue.put(results)
    barrier.wait()  # signal done
    lib.hixl_cleanup(ctx)


def main():
    parser = argparse.ArgumentParser(
        prog="python bench_hixl_bandwidth.py",
        description="HiXL bandwidth benchmark for a sender and a receiver NPU.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Example:\n"
            "  python bench_hixl_bandwidth.py -s 5 -r 6 -e 256MB\n\n"
            "The -e value must be greater than 1MB and support MB or GB units,\n"
            "for example 128MB, 512MB, or 2GB."
        ),
    )
    parser.add_argument(
        "-s",
        "--sender",
        type=int,
        required=True,
        help="sender NPU id",
    )
    parser.add_argument(
        "-r",
        "--receiver",
        type=int,
        required=True,
        help="receiver NPU id",
    )
    parser.add_argument(
        "-e",
        "--max-size",
        dest="max_size",
        type=parse_size,
        required=True,
        help="maximum transfer size, such as 128MB or 2GB",
    )
    args = parser.parse_args()

    dev_a = args.sender
    dev_b = args.receiver
    max_size = args.max_size

    print("=" * 70)
    print(f"HiXL Bandwidth Test: NPU {dev_a} ↔ NPU {dev_b}")
    print(f"  Transport: HCCS (default)")
    print(f"  Buffer sizes: 1 MB → {format_size(max_size)}")
    print(f"  Warmup: {WARMUP_ITERS} iters, Measure: {MEASURE_ITERS} iters")
    print("=" * 70)

    ctx = mp.get_context("spawn")
    barrier = ctx.Barrier(2)
    result_queue = ctx.Queue()

    for f in [COORD_FILE, COORD_FILE + ".client"]:
        try:
            os.unlink(f)
        except OSError:
            pass

    p_server = ctx.Process(target=server,
                           args=(dev_a, barrier, result_queue, max_size))
    p_client = ctx.Process(target=client,
                           args=(dev_b, dev_a, barrier, result_queue, max_size))

    p_server.start()
    time.sleep(1)
    p_client.start()

    p_client.join(timeout=300)
    p_server.join(timeout=30)

    if p_server.is_alive():
        p_server.terminate()
        p_server.join()

    if not result_queue.empty():
        results = result_queue.get()
        print("\n" + "=" * 70)
        print(f"{'Op':>5s}  {'Size':>8s}  {'BW (GB/s)':>10s}  {'Latency (ms)':>12s}")
        print("-" * 45)
        for op, label, size, bw, lat in results:
            print(f"{op:>5s}  {label:>8s}  {bw:>10.2f}  {lat:>12.3f}")
        print("=" * 70)
    else:
        print("No results collected — test may have failed.")


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
