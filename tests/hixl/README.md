# HiXL Tests

Ascend NPU 单边通信 (HiXL) 集成测试集。Monarch 的唯一生产数据路径是
`_rust_bindings.rdma → monarch_rdma → hixl-sys`；`native/`、`debug/` 和部分
底层脚本仅用于独立验证 CANN/HiXL，不会被 Python 生产代码加载。

## 前置条件

- CANN 9.0+ (`source /path/to/cann/set_env.sh`)
- torch + torch_npu (版本匹配 CANN)
- conda 环境: `monarch_ascend`
- 至少 2 张 NPU 卡 (910B)
- Rust 编译的 `_rust_bindings.so`（Plan B Rust 后端）

## 通信模式

当前 HiXL 后端支持两种传输模式，通过环境变量 `MONARCH_HIXL_TRANSPORT` 控制：

| 模式 | 说明 | 适用场景 |
|------|------|----------|
| `hccs` (默认) | 走 HCCS 高速互联 | 同一超节点内的 NPU 卡间通信 |
| `roce` | 走 RoCE 以太网 | 跨超节点通信，或 HCCS 不可用时的 fallback |

**HCCS 模式要求设备内存地址 2MB 对齐**。使用 `alloc_aligned_tensor()` 分配对齐的 tensor：

```python
from monarch._src.rdma.xdma import alloc_aligned_tensor
buf = alloc_aligned_tensor((size,), dtype=torch.float32, device="npu:0")
```

## 目录结构

```
tests/hixl/
├── e2e/          # 端到端测试 (Monarch + HIXL 完整链路)
├── unit/         # 单元测试 (聚焦单个组件)
├── native/       # 原生 C/C++ HIXL API 测试
├── debug/        # 调试/排查脚本
├── app/          # 应用级测试 (GRPO 等)
├── util/         # 工具代码 (C shim、trampoline)
├── build/        # 编译产物 (二进制、.so)
├── Makefile      # C/C++ 测试编译
└── README.md
```

## 各目录说明

### e2e/ — 端到端测试

| 文件 | 说明 |
|------|------|
| `test_hixl_bridge_minimal.py` | **核心验收用例**。两卡两 mesh，Producer(NPU0) 创建 RDMABuffer/XDMABuffer，Consumer(NPU1) 通过 write_from/read_into 跨卡读写 |
| `test_hixl_rdma_e2e.py` | HIXL RDMA 端到端：Producer 建 buffer，Consumer 通过 HIXL 写入 |

### unit/ — 单元测试

| 文件 | 说明 |
|------|------|
| `test_npu_backend.py` | NPU 后端基础验证（环境、torch_npu、设备状态）|
| `test_hixl_rdma_minimal.py` | 最小 RDMABuffer 创建测试 |
| `test_hixl_actor_ctypes.py` | 独立 ctypes 厂商诊断，不属于 Monarch 数据路径 |
| `test_hixl_direct.py` | torch_npu vs aclrtMalloc 内存对比传输 |
| `test_hixl_rdma_manager_effect.py` | RdmaManagerActor 对 HIXL 的影响隔离 |
| `test_hixl_buffer_effect.py` | RDMABuffer 创建与 HIXL 引擎共存验证 |
| `test_hixl_acl_mem.py` | ACL 原生内存的 HIXL 传输验证 |
| `test_hixl_torch_vs_acl.py` | torch_npu vs ACL 内存类型对 HIXL 的影响 |
| `test_hixl_thread_isolation.py` | 非主线程调用 HIXL 的线程隔离测试 |
| `test_hixl_dynamic_ports.py` | 动态端口分配场景下的 HIXL 连接测试 |

### native/ — 原生 C/C++ 测试

该目录用于隔离 CANN/HiXL 本身的问题。其二进制和共享库不得被
`python/monarch` 导入，也不代表 Monarch 运行时存在第二个 HiXL engine。

| 文件 | 说明 |
|------|------|
| `test_hixl_connect.cpp` | HIXL Connect 各种配置组合 (HCCS/RoCE, 同卡/异卡, ip/ip:port) |
| `test_hixl_transfer.cpp` | HIXL 双进程 TransferSync (READ/WRITE) |
| `test_hixl_d2d.cpp` | HIXL CS API 的 D2D 传输测试 |
| `test_hixl_single.cpp` | 单进程 HIXL 测试 (类似官方 server_server_d2d) |
| `test_hixl_unidirectional.cpp` | 单向 vs 双向连接对 TransferSync 的影响 |
| `test_hixl_mem_type.cpp` | HUGE_ONLY vs NORMAL_ONLY 内存类型对比 |
| `test_hixl_pthread.c` | dlopen + pthread 隔离调用 HIXL |

### debug/ — 调试脚本

调试脚本允许直接调用测试 shim，但只用于人工诊断，不纳入生产回归入口。

| 文件 | 说明 |
|------|------|
| `test_hixl_isolate_rdma.py` | 隔离 `_ensure_init_rdma_manager` vs `create_rdma_buffer_blocking` |
| `test_hixl_rdma_debug.py` | RDMA Manager 与 HIXL 交互调试 |

### app/ — 应用级测试

| 文件 | 说明 |
|------|------|
| `test_grpo_npu.py` | **主验证用例**。GRPO 训练 (Learner + Generator 双 mesh，跨卡 HIXL 权重同步)。默认走 HCCS，需 2MB 对齐内存 |
| `test_grpo_npu_simple.py` | 简化 GRPO (单 mesh，无 RDMA) |

### util/ — 工具代码

| 文件 | 说明 |
|------|------|
| `test_hixl_from_python.cpp` | 独立 ctypes 诊断 shim；生产路径不加载 |
| `hixl_trampoline.c` | 信号掩码重置 trampoline，解决 HIXL 修改信号处理的问题 |

## 快速运行

```bash
# 环境准备
conda activate monarch_ascend
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# ---- 编译 Rust 后端 (Plan B) ----
PYO3_PYTHON=$(which python) cargo build -p monarch_extension \
  --no-default-features \
  --features "ascend_engine,distributed_sql_telemetry,extension-module"

# ---- 核心验收 — GRPO 双 mesh HIXL 权重同步 ----
python tests/hixl/app/test_grpo_npu.py

# ---- 两卡 bridge 通信 ----
python tests/hixl/e2e/test_hixl_bridge_minimal.py

# ---- NPU 基础环境检查 ----
python tests/hixl/unit/test_npu_backend.py

# ---- 编译并运行 C++ 原生测试 ----
cd tests/hixl && make && make run-connect
```

## 关键环境变量

| 变量 | 说明 |
|------|------|
| `MONARCH_HIXL_TRANSPORT` | 传输模式选择：`hccs`（默认）、`roce`、`auto` |
| `MONARCH_NPU_DEVICE` | HIXL 使用的物理 NPU 设备号 |
| `MONARCH_HIXL_USE_LOOPBACK` | 强制 engine ID 使用 127.0.0.1（调试用） |
| `HCCL_NPU_SOCKET_PORT_RANGE` | HCCS 模式自动设为 `auto`，无需手动配置 |

### 已废弃

| 变量 | 替代方案 |
|------|----------|
| `HCCL_INTRA_ROCE_ENABLE=1` | 不再需要，HCCS 为默认。显式使用 RoCE 请设 `MONARCH_HIXL_TRANSPORT=roce` |
| `MONARCH_PYTHON_HIXL_ENGINE_ID` | Rust 后端自动生成 engine ID |
| `MONARCH_HIXL_LIB` | C shim 已内置于 `hixl-sys` crate，由 `build.rs` 编译 |
| `MONARCH_HIXL_IP` | 已由 `local_ip_for_hixl()` 自动获取真实 IP |
| `MONARCH_HIXL_USE_REAL_IP` | 默认已使用真实 IP，无需手动设置 |
