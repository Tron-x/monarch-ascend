// Smoke test for the new HixlCS (ClientServer) API in CANN 9.0.0 release.
//
// Two-process e2e via real NPU RoCE NICs on a single host:
//   ./test_hixl_cs_smoke server <dev> <host_ip> <nic_ip> <port>
//   ./test_hixl_cs_smoke client <dev> <host_ip> <local_nic_ip>
//                               <server_host_ip> <server_nic_ip> <port>
//
// `host_ip`  — host's normal TCP IP (e.g. `192.168.0.26`).  Used as the CS
//              control-plane listen / connect address (`server_ip` in both
//              `HixlServerDesc` and `HixlClientDesc`).
// `nic_ip`   — the NPU's RoCE NIC IP (output of `hccn_tool -i N -ip -g`,
//              e.g. `29.191.137.76`).  Used as the RDMA-plane address in
//              `EndpointDesc::commAddr`.  These are *two separate planes*:
//              the CS API splits control over host TCP from data over RoCE.
//
// Call ordering rules (per HiXL team constraints):
//   server: Create -> Listen -> RegMem (×N) -> wait for clients
//   client: Create -> RegMem (all local mem) -> Connect -> GetRemoteMem
//                  -> Batch{Put,Get}{Sync,Async}
//
// IMPORTANT: `HixlCSServerListen` MUST be called *before* `ServerRegMem`.
//            Calling it after RegMem returns 503900 (HIXL_FAILED).
//
// IMPORTANT: the client MUST register ALL local memory *before*
//            `HixlCSClientConnect`.  Registering local memory after Connect
//            leaves the MR unbound to the RDMA QP and the AICPU BatchPut
//            kernel faults with 507018 (ACL_ERROR_RT_AICPU_EXCEPTION).
//
// NOTE on local/remote memory: host-RoCE NICs do NOT support registering
// memory from `aclrtMallocHost`; device memory must come from `aclrtMalloc`
// (we use `aclrtMalloc` + `COMM_MEM_TYPE_DEVICE` for both planes).
//
// Status on this build (CANN 9.0.0 release + driver 25.5.0):
//   ✅ Control plane (Create / Listen / Connect / GetRemoteMem / RegMem /
//      Unreg / Destroy) — all `ret=0`.
//   ✅ Async submit (`BatchPutAsync`) — `ret=0`, completion handle issued.
//   ✅ Multi-region per server natively supported (each `mem_tag` is its
//      own region; we register N user regions plus see the auto-registered
//      `_hixl_builtin_dev_trans_flag` metadata region).
//   ❌ Data-plane completion (`BatchPutSync`, `QueryCompleteStatus`) —
//      Sync returns `507018 = ACL_ERROR_RT_AICPU_EXCEPTION`; Async stays
//      in `HIXL_COMPLETE_STATUS_WAITING` forever.
//
//      Root cause: `BatchPut/BatchGet` are implemented as AICPU kernels in
//      `libcann_hixl_kernel.so` (see
//      `opp/built-in/op_impl/aicpu/config/libcann_hixl_kernel.json`), and
//      the device-side AICPU scheduler on this image cannot find that .so:
//
//          ae_so_manager.cc: Single so manager init failed, soFile is
//          /usr/lib64/aicpu_kernels/0/aicpu_kernels_device/<pid>_0/libcann_hixl_kernel.so.
//          ae_so_manager.cc: Load so libcann_hixl_kernel.so failed.
//
//      The host filesystem only contains the .json descriptor; the matching
//      device-side .so is expected to be pushed via `cann-hixl-compat.tar.gz`
//      but hdcd reports "same as device, skip load" while the device image
//      still doesn't carry it.  This is a CANN/driver deployment issue, not
//      an API misuse; the HiXL team's own validation environment likely has
//      a CANN/driver bundle that does deploy the kernel SO.
//
// Action: run `bash tests/hixl/run_cs_smoke.sh` after every CANN/driver
// upgrade.  Migration from the legacy P2P API to the CS API can proceed
// once this test reports `VERDICT: CS API is READY` (i.e. BatchPut also
// returns 0).

#include "hixl/hixl.h"           // legacy API (for AclRt helpers via ascendcl)
#include "cs/hixl_cs.h"          // new ClientServer API under test
#include "hcomm/hcomm_res_defs.h"
#include "acl/acl.h"

#include <chrono>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

#define CHECK_ACL(call)                                                        \
    do {                                                                       \
        aclError __ret = (call);                                               \
        if (__ret != ACL_SUCCESS) {                                            \
            fprintf(stderr, "[ACL FAIL] " #call " ret=%d at %s:%d\n", __ret,   \
                    __FILE__, __LINE__);                                       \
            return __ret;                                                      \
        }                                                                      \
    } while (0)

#define CHECK_HIXL(call, label)                                                \
    do {                                                                       \
        HixlStatus __ret = (call);                                             \
        fprintf(stderr, "[CS] %-30s ret=%u\n", label, __ret);                  \
        if (__ret != HIXL_SUCCESS) { return (int)__ret; }                      \
    } while (0)

static void init_endpoint(EndpointDesc *ep, uint32_t dev_phy_id,
                          const char *ipv4, CommProtocol protocol) {
    EndpointDescInit(ep, 1);
    ep->protocol = protocol;
    ep->commAddr.type = COMM_ADDR_TYPE_IP_V4;
    inet_pton(AF_INET, ipv4, &ep->commAddr.addr);
    ep->loc.locType = ENDPOINT_LOC_TYPE_DEVICE;
    ep->loc.device.devPhyId = dev_phy_id;
    ep->loc.device.superDevId = 0;
    ep->loc.device.serverIdx = 0;
    ep->loc.device.superPodIdx = 0;
}

static int run_server(uint32_t dev_phy_id, const char *host_ip,
                      const char *nic_ip, uint16_t server_port,
                      void **out_mem_addrs, size_t mem_size, int num_regions,
                      CommProtocol protocol) {
    EndpointDesc ep;
    // RDMA-plane address: NPU RoCE NIC IP.
    init_endpoint(&ep, dev_phy_id, nic_ip, protocol);

    HixlServerDesc desc{};
    desc.endpoint_list = &ep;
    desc.endpoint_list_num = 1;
    // Control-plane address: host's normal TCP IP.
    desc.server_ip = host_ip;
    desc.server_port = server_port;

    HixlServerConfig config{};
    HixlServerHandle server = nullptr;
    CHECK_HIXL(HixlCSServerCreate(&desc, &config, &server), "ServerCreate");

    // Listen BEFORE RegMem.  This is the correct call order: Listen starts
    // the host TCP control-plane listener; subsequent RegMem calls register
    // regions that will be advertised to clients on Connect.  Calling Listen
    // *after* RegMem returns 503900 (HIXL_FAILED) on CANN 9.0.0 release.
    CHECK_HIXL(HixlCSServerListen(server, 16), "ServerListen");

    // Register N independent NPU memory regions with distinct tags.
    std::vector<MemHandle> mem_handles(num_regions, nullptr);
    for (int i = 0; i < num_regions; ++i) {
        void *npu_addr = nullptr;
        CHECK_ACL(aclrtMalloc(&npu_addr, mem_size, ACL_MEM_MALLOC_HUGE_FIRST));
        CHECK_ACL(aclrtMemset(npu_addr, mem_size, 0, mem_size));
        out_mem_addrs[i] = npu_addr;

        CommMem cm{};
        cm.type = COMM_MEM_TYPE_DEVICE;
        cm.addr = npu_addr;
        cm.size = mem_size;

        char tag[32];
        snprintf(tag, sizeof(tag), "region_%d", i);
        char label[64];
        snprintf(label, sizeof(label), "ServerRegMem[%s]", tag);
        CHECK_HIXL(HixlCSServerRegMem(server, tag, &cm, &mem_handles[i]),
                   label);
    }

    fprintf(stderr,
            "[S] ready: host_ip=%s:%u rdma_nic=%s, %d region(s); waiting %ds...\n",
            host_ip, server_port, nic_ip, num_regions, 20);

    // Park long enough for the client process to connect, get remote mem,
    // do its puts, and disconnect cleanly.
    std::this_thread::sleep_for(std::chrono::seconds(20));

    for (int i = 0; i < num_regions; ++i) {
        CHECK_HIXL(HixlCSServerUnregMem(server, mem_handles[i]),
                   "ServerUnregMem");
        aclrtFree(out_mem_addrs[i]);
    }
    CHECK_HIXL(HixlCSServerDestroy(server), "ServerDestroy");
    return 0;
}

static int run_client(uint32_t local_dev_phy, uint32_t remote_dev_phy,
                      const char *local_nic_ip,
                      const char *server_host_ip,
                      const char *server_nic_ip,
                      uint16_t server_port, size_t mem_size, int num_regions,
                      CommProtocol protocol) {
    EndpointDesc local_ep, remote_ep;
    init_endpoint(&local_ep, local_dev_phy, local_nic_ip, protocol);
    init_endpoint(&remote_ep, remote_dev_phy, server_nic_ip, protocol);

    HixlClientDesc desc{};
    desc.local_endpoint = &local_ep;
    desc.remote_endpoint = &remote_ep;
    // Control-plane: server's host TCP IP.
    desc.server_ip = server_host_ip;
    desc.server_port = server_port;
    desc.tc = 0;
    desc.sl = 0;

    HixlClientConfig config{};
    HixlClientHandle client = nullptr;
    CHECK_HIXL(HixlCSClientCreate(&desc, &config, &client), "ClientCreate");

    // HiXL constraint: ALL local memory MUST be registered *before*
    // HixlCSClientConnect establishes the link.  Registering after Connect
    // leaves the local MR unbound to the RDMA QP, so the AICPU BatchPut kernel
    // references an unregistered local key and faults (507018).  Allocate +
    // fill + register the local source buffer here, ahead of Connect.
    void *local_addr = nullptr;
    CHECK_ACL(aclrtMalloc(&local_addr, mem_size, ACL_MEM_MALLOC_HUGE_FIRST));
    // Pattern: 1, 2, 3, ..., as float32
    {
        std::vector<float> host_buf(mem_size / sizeof(float));
        for (size_t i = 0; i < host_buf.size(); ++i) host_buf[i] = float(i + 1);
        CHECK_ACL(aclrtMemcpy(local_addr, mem_size, host_buf.data(),
                              mem_size, ACL_MEMCPY_HOST_TO_DEVICE));
    }
    CommMem local_cm{};
    local_cm.type = COMM_MEM_TYPE_DEVICE;
    local_cm.addr = local_addr;
    local_cm.size = mem_size;
    MemHandle local_handle = nullptr;
    CHECK_HIXL(HixlCSClientRegMem(client, "local_src", &local_cm, &local_handle),
               "ClientRegMem");

    CHECK_HIXL(HixlCSClientConnect(client, 5000), "ClientConnect");

    CommMem *remote_mem_list = nullptr;
    char **mem_tag_list = nullptr;
    uint32_t list_num = 0;
    CHECK_HIXL(HixlCSClientGetRemoteMem(client, &remote_mem_list,
                                        &mem_tag_list, &list_num, 5000),
               "ClientGetRemoteMem");
    fprintf(stderr, "[C] discovered %u remote region(s)\n", list_num);

    // Filter out internal CS regions (e.g. `_hixl_builtin_dev_trans_flag` is
    // registered automatically by ServerCreate as a metadata buffer); only
    // operate on user-supplied tags (prefix "region_").
    std::vector<uint32_t> user_idx;
    for (uint32_t i = 0; i < list_num; ++i) {
        bool is_user = std::strncmp(mem_tag_list[i], "region_", 7) == 0;
        fprintf(stderr,
                "    [%u] tag=\"%s\" addr=%p size=%lu %s\n", i,
                mem_tag_list[i], remote_mem_list[i].addr,
                (unsigned long)remote_mem_list[i].size,
                is_user ? "(user)" : "(internal)");
        if (is_user) user_idx.push_back(i);
    }
    if ((int)user_idx.size() != num_regions) {
        fprintf(stderr,
                "[FAIL] expected %d user region(s), got %zu\n",
                num_regions, user_idx.size());
        return 99;
    }

    // Local source buffer was already allocated + registered before Connect
    // (HiXL requires all local-memory registration to precede link setup).
    // Give the RDMA QP a moment to fully settle after Connect/GetRemoteMem
    // before issuing data-plane ops.
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    for (uint32_t idx : user_idx) {
        HixlOneSideOpDesc op{};
        op.local_buf = local_addr;
        op.remote_buf = remote_mem_list[idx].addr;
        op.len = mem_size;

        // Try Async first (gives separate error channel for the kernel
        // submission vs. the actual completion).
        CompleteHandle h = nullptr;
        char lbl_async[64];
        snprintf(lbl_async, sizeof(lbl_async),
                 "ClientBatchPutAsync[tag=%s]", mem_tag_list[idx]);
        HixlStatus r_async = HixlCSClientBatchPutAsync(client, 1, &op, &h);
        fprintf(stderr, "[CS] %-32s ret=%u\n", lbl_async, r_async);

        if (r_async == HIXL_SUCCESS && h != nullptr) {
            // Poll completion for up to 5s.
            for (int i = 0; i < 50; ++i) {
                HixlCompleteStatus st = HIXL_COMPLETE_STATUS_WAITING;
                HixlStatus q = HixlCSClientQueryCompleteStatus(client, h, &st);
                fprintf(stderr,
                        "[CS] QueryCompleteStatus[%d] q=%u st=%d\n",
                        i, q, (int)st);
                if (q != HIXL_SUCCESS) break;
                if (st == HIXL_COMPLETE_STATUS_COMPLETED) { break; }
                if (st == HIXL_COMPLETE_STATUS_FAILED ||
                    st == HIXL_COMPLETE_STATUS_TIMEOUT) {
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        }

        // Also try Sync for the same descriptor — same op, separate error.
        char lbl_sync[64];
        snprintf(lbl_sync, sizeof(lbl_sync),
                 "ClientBatchPutSync[tag=%s]", mem_tag_list[idx]);
        HixlStatus r_sync = HixlCSClientBatchPutSync(client, 1, &op, 5000);
        fprintf(stderr, "[CS] %-32s ret=%u\n", lbl_sync, r_sync);
        if (r_sync != HIXL_SUCCESS) return (int)r_sync;
    }

    CHECK_HIXL(HixlCSClientUnregMem(client, local_handle), "ClientUnregMem");
    aclrtFree(local_addr);
    CHECK_HIXL(HixlCSClientDestroy(client), "ClientDestroy");
    return 0;
}

// ---------------------------------------------------------------------------
// Two-process mode.  Run the same binary twice:
//   ./test_hixl_cs_smoke server <dev> <local_ip> <port>
//   ./test_hixl_cs_smoke client <dev> <local_ip> <server_ip> <port>
//
// Same-process server+client was observed to make HixlCSServerListen fail
// with 503900 (the API expects independent client-server processes).
// ---------------------------------------------------------------------------

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr,
                "Usage:\n"
                "  %s server <dev> <local_ip> <port>\n"
                "  %s client <dev> <local_ip> <server_ip> <port>\n",
                argv[0], argv[0]);
        return 2;
    }
    const std::string mode(argv[1]);
    const size_t MEM_SIZE = (size_t)std::atoll(
        getenv("CS_SMOKE_MEM_SIZE") ?: "4096");
    const int NUM_REGIONS = 2;
    const CommProtocol PROTOCOL = COMM_PROTOCOL_ROCE;

    if (mode == "server") {
        if (argc < 6) {
            fprintf(stderr,
                    "server needs: <dev> <host_ip> <nic_ip> <port>\n");
            return 2;
        }
        uint32_t dev = std::atoi(argv[2]);
        const char *host_ip = argv[3];
        const char *nic_ip = argv[4];
        uint16_t port = (uint16_t)std::atoi(argv[5]);
        fprintf(stderr,
                "[smoke server] dev=%u host_ip=%s nic_ip=%s port=%u\n",
                dev, host_ip, nic_ip, port);

        CHECK_ACL(aclInit(nullptr));
        CHECK_ACL(aclrtSetDevice((int)dev));

        void *server_mem_addrs[NUM_REGIONS] = {nullptr};
        int rc = run_server(dev, host_ip, nic_ip, port, server_mem_addrs,
                            MEM_SIZE, NUM_REGIONS, PROTOCOL);
        aclFinalize();
        fprintf(stderr, "\n=== SERVER RC = %d ===\n", rc);
        return (rc == 0) ? 0 : 1;
    }

    if (mode == "client") {
        if (argc < 8) {
            fprintf(stderr,
                    "client needs: <dev> <host_ip> <local_nic_ip> "
                    "<server_host_ip> <server_nic_ip> <port>\n");
            return 2;
        }
        uint32_t dev = std::atoi(argv[2]);
        const char *host_ip = argv[3];  // currently unused; reserved
        const char *local_nic_ip = argv[4];
        const char *server_host_ip = argv[5];
        const char *server_nic_ip = argv[6];
        uint16_t port = (uint16_t)std::atoi(argv[7]);
        (void)host_ip;
        fprintf(stderr,
                "[smoke client] dev=%u local_nic=%s server=%s:%u rdma_dst=%s\n",
                dev, local_nic_ip, server_host_ip, port, server_nic_ip);

        CHECK_ACL(aclInit(nullptr));
        CHECK_ACL(aclrtSetDevice((int)dev));

        uint32_t remote_dev = std::atoi(getenv("CS_SMOKE_REMOTE_DEV") ?: "0");
        int rc = run_client(dev, remote_dev, local_nic_ip, server_host_ip,
                            server_nic_ip, port, MEM_SIZE, NUM_REGIONS,
                            PROTOCOL);
        aclFinalize();
        fprintf(stderr, "\n=== CLIENT RC = %d ===\n", rc);
        if (rc == 0) {
            fprintf(stderr,
                    "PASS: HixlCS client put-multi-region succeeded\n");
        }
        return (rc == 0) ? 0 : 1;
    }

    fprintf(stderr, "unknown mode: %s\n", mode.c_str());
    return 2;
}
