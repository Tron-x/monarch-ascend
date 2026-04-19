# Monarch NPU (Ascend) 安装指南

基于 Huawei Cloud EulerOS 2.0 + Ascend 910B 实测整理。

## 版本历史

| 日期 | torch | torch_npu | Monarch 分支 | 备注 |
|------|-------|-----------|-------------|------|
| 2026-04-05 | 2.9.0 | 2.8.0.post2 | ascend/actor-plan | 合并 origin/main v0.4.1（89 commits），torchmonarch 0.5.0.dev0 |
| 2026-03-26 | 2.8.0 | 2.8.0.post2 | ascend/actor-plan | 合并 origin/main（98 commits），Rust toolchain → nightly-2026-01-18 |
| 2026-03-xx | 2.9.0 | 2.9.0 | ascend/actor-plan | 初始 Ascend NPU 支持 |

## 当前验证通过的环境

| 组件 | 版本 | 备注 |
|------|------|------|
| OS | Huawei Cloud EulerOS 2.0 (aarch64) | 包管理器: yum/dnf |
| NPU | Ascend 910B1 × 8 | 每卡 64GB HBM |
| CANN | 9.0.0-beta.1 | 自定义路径 `/root/hzz/cann-9.0.0-beta.1/` |
| Python | 3.11.0 | conda 环境 `monarch_ascend` |
| PyTorch | 2.8.0 | torch_npu 提供 NPU 后端 |
| torch_npu | 2.8.0.post2 | 与 CANN 9.0 配套 |
| Rust | 1.94.0-nightly (2026-01-18) | rust-toolchain 指定；需要 nightly（`-Zthreads`、`tracing_unstable`）|
| clang | 12.0.1 | bindgen 需要 |
| protobuf | 3.14.0 | protoc 编译器 |

---

## 安装步骤

### Step 1: CANN 安装与环境变量

CANN 按华为官方文档安装。安装完成后验证：

```bash
npu-smi info
# 应能看到所有 NPU 卡信息
```

如果 CANN 安装在非标准路径，需要 source 环境脚本：

```bash
source /path/to/cann/set_env.sh

# 例如：
source /root/hzz/cann-9.0.0-beta.1/set_env.sh
```

验证 CANN 库文件存在：

```bash
# 以下文件必须存在
ls $ASCEND_HOME/include/hixl/hixl.h       # HiXL 头文件
ls $ASCEND_HOME/lib64/libcann_hixl.so      # HiXL 动态库
ls $ASCEND_HOME/lib64/libascendcl.so       # ACL 运行时
ls $ASCEND_HOME/lib64/libhccl.so           # HCCL 集合通信
```

### Step 2: 创建 conda 环境

```bash
conda create -n monarch_ascend python=3.11 -y
conda activate monarch_ascend
```

> **⚠️ 坑 1: Python 版本必须是 3.11**
>
> PyO3 编译的 `_rust_bindings.so` 会绑定到特定 Python 版本。
> 系统 Python 可能是 3.13，如果 Rust 编译时链接了 3.13，
> 在 3.11 的 conda 环境里就会报 `undefined symbol: PyErr_SetRaisedException`
> （该符号从 Python 3.12 才有）。
>
> **确保编译和运行使用同一个 Python**。

### Step 3: 安装 PyTorch + torch_npu

```bash
pip install torch==2.8.0
pip install torch_npu==2.8.0.post2
```

验证：

```bash
python -c "
import torch
import torch_npu
print('torch:', torch.__version__)
print('torch_npu:', torch_npu.__version__)
print('NPU available:', torch.npu.is_available())
print('NPU count:', torch.npu.device_count())
"
```

> **⚠️ 坑 2: torch 和 torch_npu 版本必须严格匹配**
>
> torch 2.8.0 只能配 torch_npu 2.8.0.post2。版本不匹配会导致
> `RuntimeError: torch_npu is not compatible with the installed torch version`。
> torch_npu 还要与 CANN 版本配套，具体对应关系见华为文档。
>
> 安装时会出现 `forge`、`torchstore` 等包对 `torch==2.9.0` 的依赖冲突警告，
> 这些是环境中其他项目的版本约束，不影响 Monarch 本身功能，可忽略。
>
> **注意**：升级 torch 版本后必须重新编译 Monarch（Rust 扩展链接 libtorch ABI）。

### Step 4: 安装系统依赖

EulerOS 使用 yum/dnf：

```bash
# clang + llvm（hccl-sys 用 bindgen 生成 Rust 绑定，需要 libclang）
yum install clang clang-devel llvm-libs

# protobuf 编译器（Monarch 内部序列化）
yum install protobuf-compiler
```

验证：

```bash
clang --version    # 需要能找到
protoc --version   # 需要能找到
```

> **⚠️ 坑 3: 没有 apt**
>
> EulerOS / openEuler 基于 RPM，用 yum/dnf，不是 apt。

### Step 5: 安装 Rust nightly

项目根目录的 `rust-toolchain` 文件指定了所需的确切版本（当前为 `nightly-2026-01-18`）。
rustup 会在首次编译时自动下载对应工具链。

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
```

验证：

```bash
rustc --version   # 应显示 nightly
cargo --version
```

> **⚠️ 坑（网络受限环境）: rust-toolchain 工具链下载慢**
>
> `rust-toolchain` 会触发 rustup 自动下载指定版本（约 300-500 MB），
> 网络受限时下载速度可能极慢（< 200 KB/s）甚至卡住。
>
> **绕过方法**：若已安装旧版 nightly（如 `nightly-2025-12-05`），
> 可在编译命令前设置 `RUSTUP_TOOLCHAIN` 覆盖 `rust-toolchain`：
>
> ```bash
> RUSTUP_TOOLCHAIN=nightly-2025-12-05 \
>   USE_ASCEND_ENGINE=1 USE_TENSOR_ENGINE=0 \
>   pip install -e . --no-build-isolation
> ```
>
> 新旧 nightly 版本通常兼容，仅 `edition = "2024"` 语法或新 unstable features
> 才强依赖具体版本。若出现编译错误再安装正确版本：
> ```bash
> rustup toolchain install nightly-2026-01-18
> ```

### Step 6: 安装 Python 依赖

```bash
cd /path/to/monarch
pip install -r requirements.txt
pip install setuptools setuptools-rust
```

### Step 7: 设置环境变量并编译

```bash
conda activate monarch_ascend

# 1. source CANN 环境（每个终端都要）
source /root/hzz/cann-9.0.0-beta.1/set_env.sh

# 2. 设置 ASCEND_HOME（指向包含 include/ 和 lib64/ 的架构子目录）
export ASCEND_HOME=/root/hzz/cann-9.0.0-beta.1/aarch64-linux
```

#### 方式一：pip install（推荐）

```bash
export PYO3_PYTHON=$(which python)
USE_ASCEND_ENGINE=1 USE_TENSOR_ENGINE=0 pip install -e . --no-build-isolation
```

> **为什么需要 `--no-build-isolation`**
>
> pip 默认在隔离的临时环境里运行 build backend，会丢失当前激活的 conda 环境的
> `torch`、CANN 路径等信息，导致 setup.py 找不到 libtorch 或 CANN 库。
> `--no-build-isolation` 让 build backend 直接使用当前环境，保留所有环境变量。
>
> 若 `rust-toolchain` 指定的工具链尚未下载，可加 `RUSTUP_TOOLCHAIN` 绕过：
> ```bash
> RUSTUP_TOOLCHAIN=nightly-2025-12-05 \
>   USE_ASCEND_ENGINE=1 USE_TENSOR_ENGINE=0 \
>   pip install -e . --no-build-isolation
> ```

#### 方式二：仅编译 Rust 后端

```bash
PYO3_PYTHON=$(which python) cargo build -p monarch_extension \
  --no-default-features \
  --features "ascend_engine,distributed_sql_telemetry,extension-module"
```

> **⚠️ 坑 4: 必须指定 `PYO3_PYTHON`**
>
> 如果不指定，PyO3 可能找到系统 Python（3.13）而非 conda 的 3.11，
> 编译出的 .so 在 conda 环境里无法加载。
>
> ```bash
> # 正确
> PYO3_PYTHON=/root/miniconda3/envs/monarch_ascend/bin/python cargo build ...
>
> # 或者（conda 环境已激活时）
> PYO3_PYTHON=$(which python) cargo build ...
> ```

> **⚠️ 坑 5: 不要用默认 features 编译**
>
> ```bash
> # ❌ 错误 — 会拉 tensor_engine → rdma-core（需要外网 git clone）
> cargo build -p monarch_extension
>
> # ❌ 错误 — 会触发 rdma-core 编译
> cargo build -p monarch_rdma --features hixl
>
> # ✅ 正确 — 只编译 Ascend 后端
> cargo build -p monarch_extension \
>   --no-default-features \
>   --features "ascend_engine,distributed_sql_telemetry,extension-module"
> ```
>
> 默认 features 包含 `tensor_engine`，它依赖 `monarch_cpp_static_libs`，
> 后者会从 GitHub 克隆 `rdma-core` 源码。在无外网环境下会报：
> `error: RPC failed; curl 16 Error in the HTTP2 framing layer`

> **⚠️ 坑 6: ASCEND_HOME 必须指向架构子目录**
>
> ```bash
> # ❌ 错误 — 这是 CANN 根目录，下面没有 include/
> export ASCEND_HOME=/root/hzz/cann-9.0.0-beta.1
>
> # ✅ 正确 — 架构子目录，下面有 include/ 和 lib64/
> export ASCEND_HOME=/root/hzz/cann-9.0.0-beta.1/aarch64-linux
> ```
>
> build.rs 会检查 `$ASCEND_HOME/include` 是否存在，路径不对会报
> `Ascend CANN installation not found`。

### Step 8: 验证安装

```bash
# 1. 基础导入
python -c "
import torch, torch_npu, monarch, monarch._rust_bindings
print('torch:', torch.__version__)
print('NPU available:', torch.npu.is_available())
print('NPU count:', torch.npu.device_count())
print('monarch OK')
"

# 2. HiXL 桥接最小路径（跨 mesh RDMA read/write，约 30s）
python tests/hixl/e2e/test_hixl_bridge_minimal.py

# 3. Ping-Pong 跨 mesh 通信（含 NPU tensor 运算验证）
python tests/hixl/app/test_ping_pong_npu.py

# 4. GRPO 端到端训练测试（核心验收，需要至少 2 张卡，约 40s）
python tests/hixl/app/test_grpo_npu.py
```

三个脚本均以 `PASS` 或 `exit_code: 0` 结束为通过。

---

## 运行时注意事项

### 每次开新终端都要做

```bash
conda activate monarch_ascend
source /root/hzz/cann-9.0.0-beta.1/set_env.sh
```

不 source CANN 环境会导致 `libascendcl.so` / `libcann_hixl.so` 找不到。

### NPU 设备隔离

HiXL 不支持同卡两个 engine 通信（底层 HCCL 通信域约束），
所以每个 mesh 必须绑定到不同物理卡：

```python
def npu_device(dev_id: int):
    def _bootstrap():
        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = str(dev_id)
        import torch, torch_npu  # noqa
        torch.npu.set_device(0)
    return _bootstrap

mesh_a = this_host().spawn_procs(per_host={"npus": 1}, bootstrap=npu_device(0))
mesh_b = this_host().spawn_procs(per_host={"npus": 1}, bootstrap=npu_device(1))
```

### HiXL DMA 与 torch.npu.synchronize()

NPU 内存操作是异步的。在使用 HiXL DMA 传输前后必须调用
`torch.npu.synchronize()`，否则会出现竞态（例如 `torch.zeros()` 的零填充
kernel 覆盖 DMA 写入的数据，导致 `read_into` 返回全零）：

```python
local = torch.zeros(4, 4, dtype=torch.float32, device="npu")
torch.npu.synchronize()   # 确保零填充完成，页面已分配
await remote.read_into(local.view(torch.uint8).flatten(), timeout=20)
torch.npu.synchronize()   # 确保 DMA 数据对后续操作可见
result = local.sum().cpu().item()
```

### HCCS 2MB 内存对齐

默认传输模式 HCCS 要求所有 RDMA buffer 地址 2MB 对齐：

```python
from monarch._src.rdma.xdma import alloc_aligned_tensor
buf = alloc_aligned_tensor((size,), dtype=torch.float32, device="npu:0")
```

不对齐会报 `rtsIpcMemGetExportKey execution failed (error 503900)`。

### 关键环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `ASCEND_HOME` | CANN 架构子目录（编译时） | 自动检测 |
| `ASCEND_RT_VISIBLE_DEVICES` | 进程可见的物理 NPU（运行时） | 所有卡 |
| `MONARCH_HIXL_TRANSPORT` | HiXL 传输模式 | `hccs` |
| `MONARCH_NPU_DEVICE` | HiXL 使用的逻辑设备号 | `0` |
| `HCCL_INTRA_ROCE_ENABLE` | 跨机必须设为 `1`；由于 CANN 在库加载时缓存该值，**必须在 Python 启动前 export**，只靠 `MONARCH_HIXL_TRANSPORT=roce`（Rust 在运行时再 set_var）太晚 | 未设置 |
| `HCCL_CONNECT_TIMEOUT` | 秒；CANN 9.0 要求 `[120, 7200]`，小于 120 会被 `HcclCommInitClusterInfoMemConfig` 拒绝（`EI0001`） | 未设置 |
| `PYO3_PYTHON` | Rust 编译链接的 Python（编译时） | 自动检测 |

---

## HiXL 单边通信带宽实测

测试环境：Ascend 910B1，HCCS 模式（默认），NPU 5 ↔ NPU 6，TransferSync 同步传输。

测试脚本：`tests/hixl/bench_hixl_bandwidth.py`

```bash
python tests/hixl/bench_hixl_bandwidth.py 5 6
```

| 操作 | 数据量 | 带宽 (GB/s) | 延迟 (ms) |
|------|--------|------------|-----------|
| READ | 1 MB | 7.65 | 0.128 |
| WRITE | 1 MB | 7.88 | 0.124 |
| READ | 2 MB | 10.84 | 0.180 |
| WRITE | 2 MB | 11.52 | 0.170 |
| READ | 4 MB | 14.59 | 0.268 |
| WRITE | 4 MB | 14.49 | 0.270 |
| READ | 8 MB | 16.54 | 0.472 |
| WRITE | 8 MB | 16.69 | 0.468 |
| READ | 16 MB | 17.97 | 0.870 |
| WRITE | 16 MB | 17.99 | 0.869 |
| READ | 32 MB | 18.65 | 1.675 |
| WRITE | 32 MB | 18.53 | 1.687 |
| READ | 64 MB | 18.89 | 3.308 |
| WRITE | 64 MB | 18.46 | 3.385 |
| READ | 128 MB | 19.28 | 6.483 |
| WRITE | 128 MB | 19.32 | 6.469 |
| READ | 256 MB | 19.26 | 12.982 |
| WRITE | 256 MB | 19.42 | 12.870 |
| **READ** | **512 MB** | **19.45** | **25.703** |
| **WRITE** | **512 MB** | **19.48** | **25.668** |

**分析：**

- 峰值带宽约 **19.5 GB/s**，128 MB 以上趋于饱和
- READ 和 WRITE 性能对称，差异 < 1%
- 小数据延迟优秀：1 MB 仅需 ~0.13 ms
- 910B HCCS 单边理论带宽 ~28 GB/s，实测利用率约 **70%**（TransferSync 同步开销，异步流水线可更高）

---

## HiXL 跨机 (RoCE) 实测

测试环境：两台独立 910B 节点，每卡一个 200 Gb/s RoCE port，Monarch actor 分布在两机，通过 `RDMABuffer.read_into` 做单边 READ。

测试脚本：`tests/hixl/e2e/test_multinode_rdma.py`

Worker 启动前必须 export：

```bash
export MONARCH_HIXL_TRANSPORT=roce
export HCCL_INTRA_ROCE_ENABLE=1
export HCCL_CONNECT_TIMEOUT=120
```

| 操作 | 数据量 | 带宽 (GB/s) | 延迟 (ms) |
|------|--------|------------|-----------|
| READ | 16 MB | 14.2 | 1.18 |
| READ | 256 MB | 23.4 | 11.47 |

理论上 200 Gb/s ≈ 25 GB/s，256 MB 实测 **23.4 GB/s ≈ 94%** 线速利用率。

### 跨机已知限制：同 engine pair 只能有一个已注册 region

CANN 9.0 / HiXL：若一对 engine 之间同时存在多个已注册内存区域，第二次之后的 `TransferSync` 会返回 `503900`。注册本身成功，失败发生在 transfer 阶段。

Monarch 侧已内置软解决方案（`monarch_rdma/src/backend/hixl/manager_actor.rs`）：`register_mem_if_needed` 在真实 HiXL 注册之外额外维护 `aliased_addrs` 表，新 `addr..addr+size` 若整段落在已注册区间内，就记作 alias 并跳过第二次 `hixl_register_mem`。上层只要保证所有并发 buffer 的物理地址都落在同一段大区间内（典型做法是预注册一个大 staging pool），就能无限复用子切片而不触发 `503900`。`deregister_mem` 采用引用计数，所有 alias 都释放且 owner 自身也无引用时才真正下发 `hixl_deregister_mem`。

推荐模式：
- **staging pool**（推荐，torchstore 已默认启用）：进程级预注册一个大 `RDMABuffer`，后续所有小 buffer 都从该 pool 切子切片，Rust 层自动识别为 alias。
- **单大 buffer + 切片访问**：一次 `alloc_aligned_tensor` 出最大容量，`RDMABuffer` 覆盖整个 tensor，各子区域用 offset 切片。
- **串行注册**：上一个 `RDMABuffer` drop 之后再创建下一个。禁止 overlap（overlap 会在 `RegisterMem` 阶段就报 `103900`）。

---

## 踩坑速查表

| # | 现象 | 原因 | 解决 |
|---|------|------|------|
| 1 | `undefined symbol: PyErr_SetRaisedException` | Rust 链接了 Python 3.12+，运行在 3.11 | 设置 `PYO3_PYTHON=$(which python)` 重新编译 |
| 2 | `curl 16 Error in the HTTP2 framing layer` | 默认 features 从 GitHub 拉 rdma-core | 用 `--no-default-features --features ascend_engine,...` |
| 3 | `Ascend CANN installation not found` | ASCEND_HOME 路径不对 | 指向架构子目录（含 include/ 和 lib64/） |
| 4 | `rtsIpcMemGetExportKey failed (503900)` | HCCS 要求 2MB 对齐 | 使用 `alloc_aligned_tensor()` 或切 RoCE |
| 5 | `not support connect with self device` | 两个 HiXL engine 在同一张卡 | 每个 mesh 绑定不同物理卡 |
| 6 | `libascendcl.so: cannot open` | 没 source CANN 环境 | `source /path/to/cann/set_env.sh` |
| 7 | `_GLIBCXX_USE_CXX11_ABI` 链接错误 | C++ ABI 不匹配 | pip install 会自动处理；手动编译需加 `CXXFLAGS` |
| 8 | pip install 编了 GPU 版本 | 环境有 CUDA，优先选了 tensor_engine | 显式 `USE_ASCEND_ENGINE=1 USE_TENSOR_ENGINE=0 pip install -e .` |
| 9 | `read_into` 返回全零 | torch 异步零填充与 HiXL DMA 竞态 | `read_into` 前后都加 `torch.npu.synchronize()` |
| 10 | 升级 torch 后 `import monarch` 崩溃 | Rust 扩展链接了旧 libtorch ABI | 重新 `USE_ASCEND_ENGINE=1 USE_TENSOR_ENGINE=0 pip install -e . --no-build-isolation` |
| 11 | `pip install -e .` 时 `rustc -V` 卡住 | rust-toolchain 触发 rustup 下载新工具链，网络慢 | 设置 `RUSTUP_TOOLCHAIN=nightly-2025-12-05` 绕过，或等待下载完成 |
| 12 | `pip install` 丢失 CANN/torch 路径 | pip 默认 build isolation 创建临时环境 | 始终加 `--no-build-isolation` |
| 13 | merge 上游后 `could not find ibverbs/tcp in backend` | `rdma_components.rs` 的 `use` 语句缺少 `#[cfg(not(feature = "hixl"))]` | 在 `IbvManagerActor`/`TcpManagerActor` 的 `use` 行前加 `#[cfg(not(feature = "hixl"))]` |
| 14 | 跨机 Monarch HiXL 启动挂在 `HcclCommInitClusterInfoMemConfig (EI0001)` | `HCCL_CONNECT_TIMEOUT` 没设或 < 120 | 在 worker 启动脚本里 `export HCCL_CONNECT_TIMEOUT=120`（Python 启动前） |
| 15 | 跨机 `hixl_transfer_read` 卡住或超时 | `HCCL_INTRA_ROCE_ENABLE` 在 Python 启动后才被 set_var，CANN 已经缓存了旧值 | 在 worker 启动脚本里 `export HCCL_INTRA_ROCE_ENABLE=1`（不能只靠 `MONARCH_HIXL_TRANSPORT=roce`） |
| 16 | 跨机多 buffer 场景 `TransferSync ret=503900` | 同一 engine pair 同时注册了多个 region，CANN 9.0 HiXL 限制 | Monarch Rust 侧 `register_mem_if_needed` 已做 range-containment aliasing；上层用 staging pool（torchstore 默认）或单大 buffer + 切片；必要时串行创建/drop `RDMABuffer` |
