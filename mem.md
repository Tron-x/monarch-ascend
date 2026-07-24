# Monarch / TorchStore / TorchForge NPU 适配踩坑记录

本文档记录在 Monarch、TorchStore、TorchForge 适配华为 Ascend NPU 过程中遇到的问题、解决方案和注意事项。

---

## 一、Monarch HiXL 集成

### 1.1 构建编译类问题

| 问题 | 根因 | 解决 |
|------|------|------|
| Feature 选择错误 | `USE_TENSOR_ENGINE` 默认为 `"1"`，编译用了 tensor_engine 而非 ascend_engine | 设置 `USE_TENSOR_ENGINE=0` + `USE_ASCEND_ENGINE=1` |
| 默认 Feature 拉入 CUDA | `monarch_extension/Cargo.toml` 的 default 含 tensor_engine | 构建 Ascend 时加 `--no-default-features` |
| monarch_tensor_worker 传染 | 对 monarch_tensor_worker 未设 `default-features = false`，拉入 cuda_backend → nccl-sys | 在 monarch_extension/Cargo.toml 中对该依赖加 `default-features = false` |
| prost 版本冲突 | hyperactor_telemetry 用 prost 0.14，tracing-perfetto-sdk-schema 用 prost ^0.13 | 升级 tracing-perfetto-sdk-schema 到 0.13.1 |
| 缺少 protoc | tracing-perfetto-sdk-schema 编译需要 protoc | 安装 `protobuf-compiler` |
| 缺少 libclang | hccl-sys 用 bindgen 生成 FFI | 安装 clang/libclang-dev |
| group_end(ticket) 接口不一致 | HCCL 后端 group_end() 不接受参数，CUDA 后端接受 | 统一两个后端都不传参数 |
| C++ ABI 冲突 | PyTorch 用 ABI=1，HIXL/CANN 用 ABI=0 | 在 build.rs 中显式清除继承的 CXXFLAGS，设置与 HIXL 一致的 ABI |

### 1.2 CANN / 驱动环境问题

| 问题 | 根因 | 解决 |
|------|------|------|
| CANN 版本不匹配 | CANN 8.1 缺少 HIXL 所需头文件 | 升级到 CANN 9.0+ |
| CANN 安装权限 | 安装器要求 /root 权限至少 755 | 安装到自定义目录如 `/root/hzz/cann-9.0.0` |
| 多版本 CANN 共存 | 环境变量指向旧版本 | source 新版本 set_env.sh 覆盖，或将旧目录重命名备份 |
| 驱动/CANN/torch_npu 三方不匹配 | 版本不一致导致 error code 507033、设备卡死 | 更新驱动 + CANN 9.0 + torch 2.7.1 + torch_npu 匹配 |
| NPU 设备卡死 | 版本不匹配导致挂死 | `npu-smi set -t device-reset -i 0` |

### 1.3 HiXL 协议与初始化

| 问题 | 根因 | 解决 |
|------|------|------|
| TransferSync 失败 503900 | 缺少 `HCCL_INTRA_ROCE_ENABLE=1` | 设置该环境变量（官方示例 run_example.sh 中有） |
| Connect 成功但 Transfer 失败 | 缺少 `BufferPool=0:0` 初始化参数 | Initialize 时传入 `options["BufferPool"] = "0:0"` |
| 单向 Connect 失败 | HIXL 要求双向 Connect | 双方都调用 Connect，或使用 `AutoConnect=1` |
| engine_id 格式 | `ip:port` 走 RoCE，`ip` 或 `ip:0` 走 HCCS | 根据拓扑选择；无 IB/RoCE 网卡时用 `ip:port` + `HCCL_INTRA_ROCE_ENABLE=1` 可走通机内 HCCS |

### 1.4 进程与设备上下文

| 问题 | 根因 | 解决 |
|------|------|------|
| ACL 设备上下文线程局部 | `aclrtSetDevice()` 只对当前线程生效 | 每个 FFI 调用点前显式设置设备 |
| ASCEND_RT_VISIBLE_DEVICES 对 HIXL 不生效 | 只对 torch_npu 有效，HIXL 用物理设备号 | 通过 `aclrtSetDevice(物理设备号)` 直接指定 |
| 设备绑定时机 | HIXL 构造函数内部已读取设备上下文 | 必须在 `new hixl::Hixl()` 之前调用 `aclrtSetDevice` |
| fork() 导致 Connect/Transfer 失败 | 子进程继承 HIXL 内部状态 | 用独立进程（exec 方式）启动 |
| HIXL 双重初始化 503900 | 同一 engine_id 被初始化两次 | 统一由 Rust `HixlManagerActor` 创建和管理 engine |

### 1.5 架构与即插即用

- **唯一生产路线**：`xdma.py` 薄封装调用 `_rust_bindings.rdma`，注册、建链和传输均由 `monarch_rdma`/`hixl-sys` 完成
- **Engine 生命周期**：`HixlManagerActor` 与 backend handle 共享受控 state，不使用 Python ctypes engine 或进程级全局 engine
- **原生诊断**：`tests/hixl/native` 和测试 shim 只用于隔离 CANN/HiXL 问题，不被 Monarch Python 包加载

---

## 二、TorchStore NPU 适配

### 2.1 RDMA / HCCS 测试问题

| 问题 | 根因 | 解决 |
|------|------|------|
| read_into 返回全零 | NPU 缓存未同步 | 在 `read_into` 后加 `torch.npu.synchronize()` |
| test_torchstore_rdma_transport 卡死 | HiXL 不支持 self-connection | 必须跨设备测试，不能同一设备读写 |
| HCCS channel 创建失败 | 内存需 2MB 对齐 | CPU malloc 不满足；保持张量在 NPU 或使用 alloc_aligned_tensor |
| 2MB 对齐 | NPU 首次分配通常对齐，后续小分配可能不对齐 | 源端用单个大 buffer（如 512×512）在 __init__ 分配，所有操作复用 |

### 2.2 环境变量

| 变量 | 说明 |
|------|------|
| `MONARCH_RDMA_EAGER_D2H=1` | 默认先转 CPU 再建 RDMABuffer；HCCS 模式下会失败 |
| `TORCHSTORE_MONARCH_RDMA_EAGER_D2H=0` | 保持张量在 NPU，避免 2MB 对齐问题 |
| `HCCL_INTRA_ROCE_ENABLE=1` | 强制 RoCE，规避 HCCS 2MB 对齐 |

### 2.3 测试执行

- 全量测试 `--only all` 可能 hang：用 `run_all_npu.sh` 分组串行跑，加 timeout 和 sleep
- `conda run` 可能缓冲输出：先 `eval "$(conda shell.bash hook)" && conda activate monarch_ascend` 再执行

---

## 三、TorchForge NPU 适配

### 3.1 Monarch API 变更（版本高于 TorchForge 预期）

| 缺失 API | 替代方案 |
|----------|----------|
| `monarch.utils.setup_env_for_distributed` | `monarch.spmd.setup_torch_elastic_env_async` |
| `monarch.actor.proc_mesh` | 用 `this_host().spawn_procs(per_host={"gpus": N})` 或兼容 shim |

### 3.2 已完成的 TorchForge 修改

| 文件 | 修改 |
|------|------|
| provisioner.py | monarch.utils → monarch.spmd 兼容层；`_VISIBLE_DEVICES_ENV_MAP` 增加 `"npu": "ASCEND_RT_VISIBLE_DEVICES"` |
| spawn.py | proc_mesh 工厂函数兼容层 |
| reference_model.py | PYTORCH_CUDA_ALLOC_CONF 条件设置；`to("cuda")` → `torch.accelerator` |
| titan.py | PYTORCH_CUDA_ALLOC_CONF 条件设置 |
| monarch_executor.py | CUDA_VISIBLE_DEVICES → 设备自适应（NPU 用 ASCEND_RT_VISIBLE_DEVICES） |
| forge_executor.py | `.cuda()` → `torch.accelerator` |
| packed.py | flex attention 检测兼容 NPU（torch.cuda 不可用时返回 False） |

### 3.3 版本依赖

- **TorchForge** 要求 `torch==2.9.0`、`torchtitan==0.2.0`、`vllm>=0.13.0`
- **TorchTitan v0.2.0** 依赖 PyTorch 2.9 新 API：`HuggingFaceStorageWriter`、`DefaultStager`、`StagingOptions` 等
- **PyTorch 2.7.1** 缺少上述 API，需升级到 2.9 才能完整跑通 TorchForge
- **torch_npu** 需与 PyTorch 版本匹配（如 2.9 对应 torch_npu 2.9.x）

---

## 四、TorchTitan 兼容性（PyTorch 2.7 环境下的临时补丁）

若暂未升级到 PyTorch 2.9，可对 TorchTitan v0.2.0 做以下补丁（升级后建议还原）：

| 文件 | 修改 |
|------|------|
| protocols/state_dict_adapter.py | `HuggingFaceStorageReader` try/except ImportError |
| models/deepseek_v3/model/state_dict_adapter.py | 同上 |
| components/checkpoint.py | `HuggingFaceStorageWriter`、`consolidate_safetensors_files_on_every_rank`、`DefaultStager`、`StagingOptions` try/except |
| models/moe/utils.py | `from .kernels import generate_permute_indices` try/except（triton 在 NPU 上不可用） |

---

## 五、调试方法论

1. **逐层对比**：纯 C++ HIXL 成功 → 对比 Monarch + HIXL 失败，每次修一个差异
2. **官方示例**：HIXL 源码仓库 `/tmp/hixl/examples/` 和 `run_example.sh` 含关键环境变量
3. **最小复现**：从完整测试缩减到最小用例（两卡两 mesh、一个 write_from），层层剥离
4. **进程模型**：涉及硬件驱动的库不能 fork 后直接用，需独立进程重新初始化

---

## 六、最终跑通两卡 HIXL 的完整条件

1. `HCCL_INTRA_ROCE_ENABLE=1`
2. Initialize 时传入 `BufferPool=0:0` 和 `AutoConnect=1`
3. `aclrtSetDevice(物理设备号)` 在 `new hixl::Hixl()` 之前
4. engine_id 使用 `ip:port` 格式（port > 0）
5. 两端在不同物理 NPU 上
6. 不使用 fork()，使用独立进程
7. HIXL 初始化入口统一，不重复初始化
8. bridge.cpp 编译时 ABI 与 HIXL 库一致
9. Python 路线（ctypes）统一管理 init/connect/register/transfer

---

## 七、测试用例索引

| 测试 | 用途 |
|------|------|
| tests/hixl/e2e/test_hixl_bridge_minimal.py | 两卡 HIXL 通信核心验收用例 |
| torchstore/tests/test_torchstore_npu.py | TorchStore NPU 全量测试 |
| torchstore/tests/run_all_npu.sh | 分组串行执行脚本 |
