# Monarch RDMA

## Overview

Monarch RDMA is a Rust library that provides high-performance single-sided communication capabilities for the Monarch framework. It supports two backends:

- **GPU (ibverbs/rdmaxcel)**: RDMA over InfiniBand / RoCE for NVIDIA GPUs, using GPUDirect RDMA.
- **NPU (HiXL)**: Single-sided communication for Huawei Ascend NPUs, supporting HCCS (intra-supernode) and RoCE (inter-node) transports.

Both backends share a unified actor-based API (`RdmaManagerActor`) and common Python interface (`RDMABuffer` / `XDMABuffer`), enabling direct memory-to-memory transfers with minimal CPU overhead.

## Features

- **Dual-backend support**: GPU (ibverbs) and NPU (HiXL), selected at compile time via Cargo features
- **Actor-based API**: Clean, actor-based interface (`RdmaManagerActor`) for managing connections and resources
- **HCCS / RoCE transport selection** (NPU): Defaults to HCCS for intra-supernode; RoCE for cross-node, controllable via `MONARCH_HIXL_TRANSPORT`
- **Reference-counted memory registration** (NPU): Efficient per-buffer registration with automatic deregistration on release
- **Bidirectional connection coordination**: Both backends establish connections from both sides for robustness
- **Timeout propagation**: User-specified timeouts are passed through to the underlying transport library

## System Requirements

### GPU Backend

#### Hardware
- RDMA-capable NIC (e.g., Mellanox ConnectX series)
- NVIDIA GPU with CUDA support

#### Software
- **libibverbs**: RDMA verbs library
- **CUDA headers**: For GPU memory integration
- **GPUDirect RDMA**: For direct GPU memory access via RDMA

Install GPUDirect RDMA following:
https://docs.nvidia.com/networking/display/gpudirectrdmav18/installing+gpudirect+rdma

Verify installation:
```bash
lsmod | grep nvidia_peermem
```

Enable peer memory mapping in `/etc/modprobe.d/nvidia.conf`:
```
options nvidia NVreg_RegistryDwords="PeerMappingOverride=1;"
```

### NPU Backend

#### Hardware
- Huawei Ascend 910B NPU (2+ cards recommended)
- HCCS interconnect (intra-supernode) or RoCE NIC (cross-node)

#### Software
- **CANN 9.0+**: Huawei's compute architecture (`source /path/to/cann/set_env.sh`)
- **torch + torch_npu**: Version matching the installed CANN
- **libcann_hixl.so + libascendcl.so**: Provided by CANN SDK

## Building

### GPU (default)

```bash
cargo build -p monarch_extension
```

### NPU

```bash
PYO3_PYTHON=/path/to/python cargo build -p monarch_extension \
  --no-default-features \
  --features "ascend_engine,distributed_sql_telemetry,extension-module"
```

The `hixl-sys` crate's `build.rs` automatically compiles the C shim (`hixl_shim.cpp`) using the `cc` crate and links against CANN libraries.

## Architecture

```
RdmaManagerActor (shared)
├── GPU: IbvManagerActor          NPU: HixlManagerActor
│        ├─ ibv_open_device             ├─ Hixl::Initialize(engine_id)
│        ├─ QP create/connect           ├─ Hixl::Connect(peer_engine_id)
│        ├─ ibv_reg_mr / dereg          ├─ Hixl::RegisterMem / DeregisterMem
│        └─ QP put/get (WRITE/READ)     └─ Hixl::TransferSync (WRITE/READ)
├── rdma_components.rs  (RdmaRemoteBuffer — unified read/write API)
└── rdma_manager_actor.rs (shared message routing, transport_level reporting)
```

### Key Files

| Component | GPU | NPU |
|-----------|-----|-----|
| Manager Actor | `backend/ibverbs/manager_actor.rs` | `backend/hixl/manager_actor.rs` |
| FFI Bindings | `rdmaxcel-sys/src/lib.rs` | `hixl-sys/src/lib.rs` |
| C Shim | rdmaxcel C library | `hixl-sys/cpp/hixl_shim.cpp` |
| Build Script | `rdmaxcel-sys/build.rs` | `hixl-sys/build.rs` |
| Python Buffer | `python/monarch/_src/rdma/rdma.py` | `python/monarch/_src/rdma/xdma.py` |

## Environment Variables

### NPU-specific

| Variable | Description |
|----------|-------------|
| `MONARCH_HIXL_TRANSPORT` | Transport selection: `hccs` (default), `roce`, or `auto` |
| `MONARCH_NPU_DEVICE` | NPU device index for HiXL engine |
| `MONARCH_HIXL_USE_LOOPBACK` | Force engine ID to use 127.0.0.1 instead of real IP |
| `HCCL_NPU_SOCKET_PORT_RANGE` | Set to `auto` automatically in HCCS mode |
| `HCCL_INTRA_ROCE_ENABLE` | Set to `1` to force RoCE at the HCCL/HiXL layer — **must be set before Python starts** (CANN caches the value at library load time). `MONARCH_HIXL_TRANSPORT=roce` sets this from Rust but that is **too late** for the CANN runtime; always export it in the shell that launches the worker. |
| `HCCL_CONNECT_TIMEOUT` | Seconds. CANN 9.0 requires a value in `[120, 7200]`; smaller values are rejected with `EI0001` at `HcclCommInitClusterInfoMemConfig`. |

### GPU-specific

| Variable | Description |
|----------|-------------|
| `CUDA_VISIBLE_DEVICES` | Control visible GPUs |
| `MONARCH_DEBUG_RDMA` | Print device mapping info |

## NPU Memory Alignment

HCCS transport requires **2MB-aligned** device memory addresses. Use the provided helper:

```python
from monarch._src.rdma.xdma import alloc_aligned_tensor

tensor = alloc_aligned_tensor((size,), dtype=torch.float32, device="npu:0")
```

Standard `torch.zeros(..., device="npu:0")` allocations may not be 2MB-aligned. Unaligned memory will fall back to RoCE or fail with error 503900 during connect.

## Cross-Node (RoCE) Setup

### Environment (on every worker)

Export **before Python starts** (e.g., in the worker launcher script):

```bash
export MONARCH_HIXL_TRANSPORT=roce
export HCCL_INTRA_ROCE_ENABLE=1          # CANN caches this at library load
export HCCL_CONNECT_TIMEOUT=120           # must be >= 120 for CANN 9.0
```

Each NPU must have an HCCN RoCE IP assigned (see `hccn_tool -i <dev> -ip -g`).
The NPU RoCE plane is typically the `29.191.0.0/16` or similar out-of-band
network, independent from the host management network.

Validate reachability from the host's shell:

```bash
hccn_tool -i 0 -ping -g address <peer_npu_ip>   # should return low-latency pings
```

### Verification

AReaL's worker manager already exports these variables; you can launch its two-node
test harness with:

```bash
# On the controller
bash AReaL/forge/scripts/worker_manager.sh start --hostfile AReaL/forge/configs/hostfile.txt
python monarch/tests/hixl/e2e/test_multinode_rdma.py --size-mb 256
```

A healthy run achieves ≈20-24 GB/s for a 256 MB cross-host `RDMABuffer.read_into`
(≈200 Gb/s RoCE line-rate).

### Known limitation: one registered region per engine pair

As of CANN 9.0 / HiXL, `TransferSync` fails with error `503900` when a pair of
engines has **more than one simultaneously registered memory region**. The
registration call itself succeeds, but subsequent transfers on any region after
the first return 503900.

To keep callers from having to track this invariant manually,
`register_mem_if_needed` in `monarch_rdma/src/backend/hixl/manager_actor.rs`
maintains a secondary `aliased_addrs` table on top of the real HiXL
registrations. When a new `register_mem_if_needed(addr, size)` request falls
fully inside a range that is *already* registered, we record it as an alias
(bumping the owner's refcount) and skip the second `hixl_register_mem` call
entirely. `deregister_mem` uses refcounts so the underlying HiXL registration
only gets torn down once every alias and every direct reference is released.

In practice, this means upper layers (torchstore, forge, user code) can
register a single large staging buffer up-front and then hand out any number
of sub-slices to different transfers without tripping 503900.

Supported patterns:

- **Staging pool** (recommended, enabled by default in torchstore): a single
  process-global `RDMABuffer` owns a large aligned NPU tensor; per-transfer
  buffers are sub-slices of it. The aliasing logic above keeps HiXL happy.
- **One large buffer per actor**: allocate the max working-set size up front
  and slice it for individual transfers.
- **Serial registration**: create, transfer, drop, then create the next
  (avoid overlapping; overlapping regions also trigger `103900` at register
  time).

## License

This source code is licensed under the BSD-style license found in the LICENSE file in the root directory of this source tree.
