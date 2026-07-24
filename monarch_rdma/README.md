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

### Legacy limitation: one registered region per engine pair (fixed in CANN 9.0.0)

**CANN 9.0.0-beta.1** had a HiXL bug where `TransferSync` returned `503900`
when a pair of engines had more than one simultaneously registered memory
region. The registration call itself succeeded, but subsequent transfers on
any region after the first would fail.

**CANN 9.0.0 release fixes this on the old P2P API.** Validation tests:

- `tests/hixl/e2e/test_multi_region_per_pair.py` registers two *independent*
  (non-overlapping, non-containing) 2MB-aligned NPU buffers on the same
  engine pair, then writes to each via cross-mesh `RDMABuffer.write_from`.
  PASS on both HCCS and RoCE transports.
- `tests/hixl/e2e/test_hixl_hccl_coexist.py` initializes
  `torch.distributed.init_process_group(backend='hccl')` together with HiXL
  in the same process. HCCL `all_reduce` + HiXL `register_mem` + cross-mesh
  `write_from` all succeed without `HcclCommPrepare ret=0x13`. This also
  validates that the historical HCCL-Adapter singleton contention is gone.

In addition, CANN 9.0.0 ships a new **ClientServer one-sided API**
(`HixlCS*` in `include/cs/hixl_cs.h`) that uses `mem_tag` to support an
arbitrary number of registered regions natively (without HCCL comm
dependency). Monarch's HiXL backend continues to use the old P2P API, which
now also supports multi-region registration.

**Backward compatibility**: `register_mem_if_needed` in
`monarch_rdma/src/backend/hixl/manager_actor.rs` retains the
range-containment alias path as opt-in legacy behaviour. Set
`MONARCH_HIXL_ENABLE_ALIAS=1` to re-enable it when running against
CANN 9.0.0-beta.1 or earlier.

Recommended usage patterns (no longer required, but still good practice
for memory locality):

- **Staging pool**: a single process-global aligned NPU buffer split into
  sub-slices for per-transfer use. Reduces NPU allocator pressure.
- **Per-actor large buffer**: allocate the working-set size up front.
- **Truly independent buffers**: now also supported — each one gets its
  own `hixl_register_mem` call.

## License

This source code is licensed under the BSD-style license found in the LICENSE file in the root directory of this source tree.
