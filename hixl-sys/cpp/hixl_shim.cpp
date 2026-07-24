/**
 * hixl_shim — flat C API wrapping the HiXL C++ library.
 *
 * Compiled as a static library by hixl-sys/build.rs and linked into
 * the Rust _rust_bindings shared library.
 *
 * All HiXL engine operations are called directly on the caller's thread.
 * Before each operation the saved ACL context is restored via
 * aclrtSetCurrentContext, which is safe to call from any thread per the
 * CANN documentation.
 */
#include <hixl/hixl.h>
#include <hixl/hixl_types.h>
#include <acl/acl.h>
#include <acl/acl_rt.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <strings.h>
#include <map>
#include <mutex>
#include <string>
#include <unistd.h>
using namespace hixl;

extern "C" {

struct HixlTestCtx {
    Hixl* engine = nullptr;
    int device = -1;
    aclrtContext acl_ctx = nullptr;

    std::mutex mem_mtx;
    std::map<uintptr_t, MemHandle> mem_handles;
};

static void restore_acl_ctx(HixlTestCtx* ctx) {
    if (ctx->acl_ctx) {
        aclrtSetCurrentContext(ctx->acl_ctx);
    }
}

uintptr_t get_acl_context() {
    aclrtContext ctx = nullptr;
    aclrtGetCurrentContext(&ctx);
    fprintf(stderr, "[hixl_test] get_acl_context: %p\n", ctx);
    return reinterpret_cast<uintptr_t>(ctx);
}

void set_acl_context(uintptr_t ctx_ptr) {
    aclrtContext ctx = reinterpret_cast<aclrtContext>(ctx_ptr);
    aclError ret = aclrtSetCurrentContext(ctx);
    fprintf(stderr, "[hixl_test] set_acl_context(%p): %d\n", ctx, ret);
}

// Build the canonical options map for `hixl::Hixl::Initialize`.
//
// HiXL's C++ Initialize accepts four options (libcann_hixl.so error message
// enumerates them as OPTION_LOCAL_COMM_RES, OPTION_BUFFER_POOL,
// OPTION_RDMA_TRAFFIC_CLASS, OPTION_RDMA_SERVICE_LEVEL).
//
//   BufferPool=0:0
//      Always set.  Disables the internal staging pool; Monarch manages
//      NPU memory itself via aclrtMalloc-aligned tensors.
//
//   LocalCommRes={"version":"<v>"}
//      Opt-in via `MONARCH_HIXL_USE_LOCAL_COMM_RES=1`.  Tells HiXL to
//      auto-generate the endpoint descriptor (net_instance_id,
//      endpoint_list) from the local device — the new replacement for the
//      legacy rankTable JSON.  Recommended by the HiXL team to avoid
//      contending with `torch.distributed`'s HCCL process_group for the
//      HcclAdapter singleton (the historical `HcclCommPrepare ret=0x13`).
//
//      WARNING: enabling LocalCommRes makes HiXL route all transfers
//      through the AICPU kernel `libcann_hixl_kernel.so` instead of the
//      legacy device-direct path.  That .so must be present in the NPU
//      device-side LD_LIBRARY_PATH
//      (`/usr/lib64/aicpu_kernels/0/aicpu_kernels_device/`); otherwise
//      `hixl_transfer_*` fails with `ret=507018`
//      (`ACL_ERROR_RT_AICPU_EXCEPTION`).  See
//      `tests/hixl/run_cs_smoke.sh` for a deployment readiness check.
//
//      Version overridable via `MONARCH_HIXL_LOCAL_COMM_RES_VERSION`
//      (default "1.3").
//
// On the current CANN 9.0.0 release + driver 25.5.0 environment the
// AICPU kernel SO is not deployed yet, so LocalCommRes is off by default
// to keep the legacy path working.  Flip the env flag once HiXL team
// confirms the deployment.
static void populate_init_opts(std::map<AscendString, AscendString>& opts) {
    opts["BufferPool"] = "0:0";

    const char* enable = std::getenv("MONARCH_HIXL_USE_LOCAL_COMM_RES");
    bool use_lcr = (enable && (std::strcmp(enable, "1") == 0 ||
                               strcasecmp(enable, "true") == 0));
    if (use_lcr) {
        const char* ver_env =
            std::getenv("MONARCH_HIXL_LOCAL_COMM_RES_VERSION");
        std::string ver = (ver_env && *ver_env) ? ver_env : "1.3";
        std::string json = std::string("{\"version\":\"") + ver + "\"}";
        opts["LocalCommRes"] = AscendString(json.c_str());
        fprintf(stderr,
                "[hixl_shim] LocalCommRes enabled (version=%s); transfers "
                "will go through AICPU kernel libcann_hixl_kernel.so\n",
                ver.c_str());
    }
}

void* hixl_init_engine(int dev, const char* engine_id) {
    auto* ctx = new HixlTestCtx();
    ctx->device = dev;

    aclrtSetDevice(dev);
    aclrtGetCurrentContext(&ctx->acl_ctx);

    ctx->engine = new Hixl();
    std::string eid(engine_id);
    std::map<AscendString, AscendString> opts;
    populate_init_opts(opts);
    auto s = ctx->engine->Initialize(AscendString(eid.c_str()), opts);
    fprintf(stderr, "[hixl_test] Init(%s) dev=%d tid=%ld: %u\n",
            eid.c_str(), dev, (long)pthread_self(), s);

    if (s != SUCCESS) {
        if (ctx->engine) { delete ctx->engine; }
        delete ctx;
        return nullptr;
    }
    return ctx;
}

void* hixl_init_engine_with_ctx(uintptr_t acl_ctx_ptr, const char* engine_id) {
    auto* ctx = new HixlTestCtx();
    ctx->acl_ctx = reinterpret_cast<aclrtContext>(acl_ctx_ptr);

    aclrtSetCurrentContext(ctx->acl_ctx);

    ctx->engine = new Hixl();
    std::string eid(engine_id);
    std::map<AscendString, AscendString> opts;
    populate_init_opts(opts);
    auto s = ctx->engine->Initialize(AscendString(eid.c_str()), opts);
    fprintf(stderr, "[hixl_test] InitWithCtx(%s) ctx=%p tid=%ld: %u\n",
            eid.c_str(), ctx->acl_ctx, (long)pthread_self(), s);

    if (s != SUCCESS) {
        if (ctx->engine) { delete ctx->engine; }
        delete ctx;
        return nullptr;
    }
    return ctx;
}

int hixl_register_mem(void* ctx_ptr, uintptr_t addr, size_t size) {
    auto* ctx = static_cast<HixlTestCtx*>(ctx_ptr);
    restore_acl_ctx(ctx);

    constexpr size_t HCCS_ALIGN = 2UL * 1024 * 1024;
    if (addr % HCCS_ALIGN != 0) {
        fprintf(stderr, "[hixl] WARNING: addr=%p is NOT 2MB-aligned (off=%zu). "
                "HCCS transfers may fail.\n", (void*)addr, addr % HCCS_ALIGN);
    }

    MemHandle handle;
    MemDesc d{}; d.addr = addr; d.len = size;
    auto s = ctx->engine->RegisterMem(d, MEM_DEVICE, handle);
    fprintf(stderr, "[hixl] RegMem addr=%p size=%zu aligned=%d: %u (tid=%ld)\n",
            (void*)addr, size, (int)(addr % HCCS_ALIGN == 0), s, (long)pthread_self());
    if (s == SUCCESS) {
        std::lock_guard<std::mutex> lk(ctx->mem_mtx);
        ctx->mem_handles[addr] = handle;
    }
    return (int)s;
}

int hixl_deregister_mem(void* ctx_ptr, uintptr_t addr) {
    auto* ctx = static_cast<HixlTestCtx*>(ctx_ptr);
    restore_acl_ctx(ctx);
    MemHandle handle;
    {
        std::lock_guard<std::mutex> lk(ctx->mem_mtx);
        auto it = ctx->mem_handles.find(addr);
        if (it == ctx->mem_handles.end()) {
            fprintf(stderr, "[hixl] DeregMem addr=%p: not found\n", (void*)addr);
            return -1;
        }
        handle = it->second;
        ctx->mem_handles.erase(it);
    }
    auto s = ctx->engine->DeregisterMem(handle);
    fprintf(stderr, "[hixl] DeregMem addr=%p: %u (tid=%ld)\n",
            (void*)addr, s, (long)pthread_self());
    return (int)s;
}

int hixl_connect(void* ctx_ptr, const char* remote_engine_id, int timeout_ms) {
    auto* ctx = static_cast<HixlTestCtx*>(ctx_ptr);
    restore_acl_ctx(ctx);
    std::string eid(remote_engine_id);
    auto s = ctx->engine->Connect(AscendString(eid.c_str()), timeout_ms);
    fprintf(stderr, "[hixl_test] Connect(%s): %u pid=%d (tid=%ld)\n",
            eid.c_str(), s, getpid(), (long)pthread_self());
    if (s != SUCCESS) {
        auto* msg = aclGetRecentErrMsg();
        fprintf(stderr, "[hixl_test] Connect ACL error: %s\n", msg ? msg : "none");
    }
    return (int)s;
}

int hixl_transfer_write(void* ctx_ptr, const char* remote_engine_id,
                         uintptr_t local_addr, uintptr_t remote_addr, size_t len,
                         int timeout_ms) {
    auto* ctx = static_cast<HixlTestCtx*>(ctx_ptr);
    restore_acl_ctx(ctx);
    TransferOpDesc td{};
    td.local_addr = local_addr;
    td.remote_addr = remote_addr;
    td.len = len;
    fprintf(stderr, "[hixl_test] TransferSync(WRITE) local=%p remote=%p len=%zu to=%s (tid=%ld)\n",
            (void*)local_addr, (void*)remote_addr, len, remote_engine_id, (long)pthread_self());
    auto s = ctx->engine->TransferSync(AscendString(remote_engine_id), WRITE, {td}, timeout_ms);
    fprintf(stderr, "[hixl_test] TransferSync WRITE result: %u\n", s);
    if (s != SUCCESS) {
        auto* msg = aclGetRecentErrMsg();
        fprintf(stderr, "[hixl_test] ACL error: %s\n", msg ? msg : "none");
    }
    return (int)s;
}

int hixl_transfer_read(void* ctx_ptr, const char* remote_engine_id,
                        uintptr_t local_addr, uintptr_t remote_addr, size_t len,
                        int timeout_ms) {
    auto* ctx = static_cast<HixlTestCtx*>(ctx_ptr);
    restore_acl_ctx(ctx);
    TransferOpDesc td{};
    td.local_addr = local_addr;
    td.remote_addr = remote_addr;
    td.len = len;
    fprintf(stderr, "[hixl_test] TransferSync(READ) local=%p remote=%p len=%zu from=%s (tid=%ld)\n",
            (void*)local_addr, (void*)remote_addr, len, remote_engine_id, (long)pthread_self());
    auto s = ctx->engine->TransferSync(AscendString(remote_engine_id), READ, {td}, timeout_ms);
    fprintf(stderr, "[hixl_test] TransferSync READ result: %u\n", s);
    if (s != SUCCESS) {
        auto* msg = aclGetRecentErrMsg();
        fprintf(stderr, "[hixl_test] ACL error: %s\n", msg ? msg : "none");
    }
    return (int)s;
}

/// Return the saved ACL context from the engine's init.
uintptr_t hixl_get_acl_context(void* ctx_ptr) {
    auto* ctx = static_cast<HixlTestCtx*>(ctx_ptr);
    return reinterpret_cast<uintptr_t>(ctx->acl_ctx);
}

/// Probe whether HCCS IPC memory export works on the given device.
/// Returns 0 if HCCS is likely to work, non-zero otherwise.
int hixl_probe_hccs(int dev) {
    aclError ret = aclrtSetDevice(dev);
    if (ret != 0) {
        fprintf(stderr, "[hixl] probe_hccs: aclrtSetDevice(%d) failed: %d\n", dev, ret);
        return -1;
    }

    void* ptr = nullptr;
    constexpr size_t PROBE_SIZE = 2UL * 1024 * 1024; // 2 MB aligned alloc
    ret = aclrtMalloc(&ptr, PROBE_SIZE, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != 0 || ptr == nullptr) {
        fprintf(stderr, "[hixl] probe_hccs: aclrtMalloc failed: %d\n", ret);
        return -2;
    }

    char key[128] = {};
    ret = aclrtIpcMemGetExportKey(ptr, PROBE_SIZE, key, sizeof(key),
                                  ACL_RT_IPC_MEM_EXPORT_FLAG_DEFAULT);
    fprintf(stderr, "[hixl] probe_hccs: aclrtIpcMemGetExportKey dev=%d ptr=%p ret=%d\n",
            dev, ptr, ret);

    if (ret == 0) {
        aclrtIpcMemClose(key);
    }
    aclrtFree(ptr);
    return (int)ret;
}

void hixl_cleanup(void* ctx_ptr) {
    auto* ctx = static_cast<HixlTestCtx*>(ctx_ptr);
    restore_acl_ctx(ctx);
    {
        std::lock_guard<std::mutex> lk(ctx->mem_mtx);
        for (auto& [addr, handle] : ctx->mem_handles) {
            ctx->engine->DeregisterMem(handle);
        }
        ctx->mem_handles.clear();
    }
    ctx->engine->Finalize();
    fprintf(stderr, "[hixl] Finalize done (tid=%ld)\n", (long)pthread_self());
    delete ctx->engine;
    delete ctx;
}

} // extern "C"
