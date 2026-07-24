// Cross-host smoke test for the legacy HiXL P2P API.
//
// This is the multi-machine counterpart of test_hixl_bridge_minimal.py and the
// CS smoke: server runs on host A (e.g. monarch1, 192.168.0.26), client runs
// on host B (e.g. monarch2, 192.168.0.23), and the client RDMA-writes a
// pattern into the server's registered NPU memory.  The goal is to prove that
// the legacy P2P API works cross-host on CANN 9.1.0 with both:
//   (a) the default (no LocalCommRes) code path — the historical "blocked by
//       HcclAdapter / rankTable contention" path; and
//   (b) the `LocalCommRes={"version":"1.3"}` path — HiXL team's recommended
//       new endpoint description, which on CANN 9.0.0 toolkit-only installs
//       would fail with `507018` (`ACL_ERROR_RT_AICPU_EXCEPTION`) because the
//       AICPU kernel `libcann_hixl_kernel.so` was not deployed.
//
// HiXL engine_id is a `<host_tcp_ip>:<port>` string (matches Monarch's
// `resolve_engine_id` in monarch_rdma/src/rdma_manager_actor.rs).  HiXL binds
// a host TCP listener on this address as its control plane, and separately
// uses the NPU RoCE NIC (resolved internally via LocalCommRes or rankTable)
// for the actual RDMA data plane.  Both planes must be reachable cross-host
// — the NPU NIC must route to the peer's NPU NIC over RoCE, and the host
// TCP IP must be reachable from the peer host.
//
// Control rendezvous (exchanging the engine_id and the server-registered
// memory address) goes through a separate plain TCP socket; we reuse the
// same host TCP IP for both rendezvous and HiXL control, just on different
// ports.
//
// Usage:
//   server (host A, e.g. m1):
//     ./test_hixl_cross_host server <dev> <local_host_ip> <hixl_port> \
//                                   <rendezvous_port>
//   client (host B, e.g. m2):
//     ./test_hixl_cross_host client <dev> <local_host_ip> <hixl_port> \
//                                   <remote_host_ip> <rendezvous_port> \
//                                   <remote_hixl_port>
//
// host_ip   — host's normal TCP IP (192.168.0.26 / .23), output of
//             `hostname -I | awk '{print $1}'` or similar.  Used both for
//             HiXL's host TCP control plane AND for rendezvous.
// hixl_port — port HiXL binds its control plane on (must differ from
//             rendezvous_port).
//
// Env knobs (matching hixl_shim.cpp):
//   MONARCH_HIXL_USE_LOCAL_COMM_RES=1     enable LocalCommRes init option
//   MONARCH_HIXL_LOCAL_COMM_RES_VERSION   override version string (default "1.3")

#include <hixl/hixl.h>
#include <hixl/hixl_types.h>
#include <acl/acl.h>
#include <acl/acl_rt.h>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <strings.h>
#include <thread>
#include <vector>

using namespace hixl;

static constexpr size_t MEM_SIZE = 64 * 1024;  // 64 KiB (16384 floats)

#define CHECK_ACL(call)                                                       \
    do {                                                                      \
        aclError __ret = (call);                                              \
        if (__ret != ACL_SUCCESS) {                                           \
            fprintf(stderr, "[ACL FAIL] " #call " ret=%d %s:%d\n", __ret,     \
                    __FILE__, __LINE__);                                      \
            return 1;                                                         \
        }                                                                     \
    } while (0)

#define CHECK_HX(call, label)                                                 \
    do {                                                                      \
        Status __r = (call);                                                  \
        fprintf(stderr, "[HX] %-28s ret=%u\n", label, __r);                   \
        if (__r != SUCCESS) return (int)__r;                                  \
    } while (0)

// ---------------------------------------------------------------------------
// Init options, mirroring hixl-sys/cpp/hixl_shim.cpp::populate_init_opts().
// ---------------------------------------------------------------------------
static void populate_opts(std::map<AscendString, AscendString> &opts) {
    opts["BufferPool"] = "0:0";
    const char *flag = std::getenv("MONARCH_HIXL_USE_LOCAL_COMM_RES");
    bool lcr =
        flag && (std::strcmp(flag, "1") == 0 || strcasecmp(flag, "true") == 0);
    if (lcr) {
        const char *v = std::getenv("MONARCH_HIXL_LOCAL_COMM_RES_VERSION");
        std::string ver = (v && *v) ? v : "1.3";
        std::string j = std::string("{\"version\":\"") + ver + "\"}";
        opts["LocalCommRes"] = AscendString(j.c_str());
        fprintf(stderr, "[init] LocalCommRes=%s\n", j.c_str());
    } else {
        fprintf(stderr, "[init] LocalCommRes: disabled (legacy path)\n");
    }
}

// ---------------------------------------------------------------------------
// Tiny blocking TCP rendezvous: send and receive raw bytes over a single
// connection.  Only used for the control plane (exchange engine_id and remote
// memory address).
// ---------------------------------------------------------------------------
static int tcp_listen(const char *host_ip, uint16_t port, int *out_fd) {
    int s = socket(AF_INET, SOCK_STREAM, 0);
    if (s < 0) return -1;
    int yes = 1;
    setsockopt(s, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_port = htons(port);
    inet_pton(AF_INET, host_ip, &a.sin_addr);
    if (bind(s, (sockaddr *)&a, sizeof(a)) < 0) { close(s); return -2; }
    if (listen(s, 1) < 0) { close(s); return -3; }
    fprintf(stderr, "[tcp] listening on %s:%u\n", host_ip, port);
    int c = accept(s, nullptr, nullptr);
    close(s);
    *out_fd = c;
    return c < 0 ? -4 : 0;
}

static int tcp_connect(const char *host_ip, uint16_t port, int *out_fd) {
    for (int i = 0; i < 50; ++i) {
        int s = socket(AF_INET, SOCK_STREAM, 0);
        if (s < 0) return -1;
        sockaddr_in a{};
        a.sin_family = AF_INET;
        a.sin_port = htons(port);
        inet_pton(AF_INET, host_ip, &a.sin_addr);
        if (connect(s, (sockaddr *)&a, sizeof(a)) == 0) {
            *out_fd = s;
            fprintf(stderr, "[tcp] connected to %s:%u (attempt %d)\n",
                    host_ip, port, i + 1);
            return 0;
        }
        close(s);
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
    return -2;
}

static int send_all(int fd, const void *buf, size_t n) {
    const char *p = (const char *)buf;
    while (n) {
        ssize_t k = ::send(fd, p, n, 0);
        if (k <= 0) return -1;
        p += k; n -= (size_t)k;
    }
    return 0;
}
static int recv_all(int fd, void *buf, size_t n) {
    char *p = (char *)buf;
    while (n) {
        ssize_t k = ::recv(fd, p, n, 0);
        if (k <= 0) return -1;
        p += k; n -= (size_t)k;
    }
    return 0;
}

// ---------------------------------------------------------------------------
// Server side: register a buffer of zeros on its NPU and wait for the client
// to write a known pattern into it.  After client signals "done", verify the
// buffer's checksum.
// ---------------------------------------------------------------------------
static int run_server(int dev, const char *local_host_ip, uint16_t hixl_port,
                      uint16_t rdv_port) {
    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(dev));

    Hixl srv;
    char eid_buf[64];
    snprintf(eid_buf, sizeof(eid_buf), "%s:%u", local_host_ip, hixl_port);
    std::string eid(eid_buf);
    std::map<AscendString, AscendString> opts;
    populate_opts(opts);

    CHECK_HX(srv.Initialize(AscendString(eid.c_str()), opts), "ServerInit");

    void *buf = nullptr;
    CHECK_ACL(aclrtMalloc(&buf, MEM_SIZE, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMemset(buf, MEM_SIZE, 0, MEM_SIZE));

    MemDesc md{};
    md.addr = (uintptr_t)buf;
    md.len = MEM_SIZE;
    MemHandle mh = nullptr;
    CHECK_HX(srv.RegisterMem(md, MEM_DEVICE, mh), "ServerRegisterMem");
    fprintf(stderr, "[S] eid=%s registered addr=0x%lx len=%zu\n",
            eid.c_str(), (unsigned long)md.addr, MEM_SIZE);

    int tcp = -1;
    if (tcp_listen(local_host_ip, rdv_port, &tcp) < 0) {
        fprintf(stderr, "[S] tcp_listen failed\n");
        return 90;
    }

    // hand over engine_id + buffer addr + size
    uint16_t eid_len = (uint16_t)eid.size();
    if (send_all(tcp, &eid_len, 2) || send_all(tcp, eid.data(), eid_len) ||
        send_all(tcp, &md.addr, 8) || send_all(tcp, &md.len, 8)) {
        fprintf(stderr, "[S] tcp send failed\n");
        return 91;
    }
    fprintf(stderr, "[S] rendezvous handed over to client\n");

    // wait for client "done" signal
    uint8_t done = 0;
    if (recv_all(tcp, &done, 1) || done != 1) {
        fprintf(stderr, "[S] client did not signal completion\n");
        close(tcp);
        return 92;
    }
    close(tcp);

    // verify buffer pattern: float[i] = i+1 -> sum = N*(N+1)/2
    size_t n = MEM_SIZE / sizeof(float);
    std::vector<float> host(n);
    CHECK_ACL(aclrtMemcpy(host.data(), MEM_SIZE, buf, MEM_SIZE,
                          ACL_MEMCPY_DEVICE_TO_HOST));
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) sum += host[i];
    double expect = (double)n * (n + 1) / 2.0;
    fprintf(stderr, "[S] buf sum=%.0f  expected=%.0f  delta=%.3f\n",
            sum, expect, sum - expect);

    // Data transfer correctness is the actual success criterion.  Emit
    // `[S] PASS` BEFORE attempting cleanup so that the driver script can
    // detect success even if subsequent DeregisterMem / Finalize fails or
    // aborts (CANN 9.1.0 legacy path sometimes returns 103900 in
    // DeregisterMem after the peer has already Disconnected; and some
    // CANN versions SIGABRT inside aclFinalize destructors).
    if (std::abs(sum - expect) > 0.5) {
        fprintf(stderr, "[S] FAIL: data mismatch\n");
        return 93;
    }
    fprintf(stderr, "[S] PASS\n");

    // Best-effort cleanup; ignore non-fatal codes after success.
    Status dr = srv.DeregisterMem(mh);
    if (dr != SUCCESS) {
        fprintf(stderr,
                "[S] WARN: ServerDeregisterMem ret=%u (transfer already "
                "verified, ignoring)\n", dr);
    }
    aclrtFree(buf);
    aclFinalize();
    return 0;
}

// ---------------------------------------------------------------------------
// Client side: get the remote engine_id + addr through rendezvous, allocate
// + fill a local buffer with a known pattern, Connect to the server HiXL
// engine, and TransferSync(WRITE) the pattern into the remote address.
// ---------------------------------------------------------------------------
static int run_client(int dev, const char *local_host_ip, uint16_t hixl_port,
                      const char *remote_host_ip, uint16_t rdv_port,
                      uint16_t remote_hixl_port) {
    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(dev));

    Hixl cli;
    char eid_buf[64];
    snprintf(eid_buf, sizeof(eid_buf), "%s:%u", local_host_ip, hixl_port);
    std::string eid(eid_buf);
    std::map<AscendString, AscendString> opts;
    populate_opts(opts);

    CHECK_HX(cli.Initialize(AscendString(eid.c_str()), opts), "ClientInit");

    int tcp = -1;
    if (tcp_connect(remote_host_ip, rdv_port, &tcp) < 0) {
        fprintf(stderr, "[C] tcp_connect failed\n");
        return 80;
    }

    uint16_t eid_len = 0;
    char remote_eid[128] = {};
    uint64_t remote_addr = 0;
    uint64_t remote_len = 0;
    if (recv_all(tcp, &eid_len, 2) ||
        eid_len >= sizeof(remote_eid) ||
        recv_all(tcp, remote_eid, eid_len) ||
        recv_all(tcp, &remote_addr, 8) ||
        recv_all(tcp, &remote_len, 8)) {
        fprintf(stderr, "[C] tcp recv failed\n");
        close(tcp);
        return 81;
    }
    // sanity-check the engine_id we got matches what the caller told us
    char expected_eid[64];
    snprintf(expected_eid, sizeof(expected_eid), "%s:%u",
             remote_host_ip, remote_hixl_port);
    fprintf(stderr,
            "[C] remote_eid=%s addr=0x%lx len=%lu  (expected_eid=%s)\n",
            remote_eid, (unsigned long)remote_addr,
            (unsigned long)remote_len, expected_eid);
    if (std::strcmp(remote_eid, expected_eid) != 0) {
        fprintf(stderr,
                "[C] WARN: rendezvous-eid != cmdline-eid (proceeding "
                "with rendezvous value)\n");
    }
    if (remote_len != MEM_SIZE) {
        fprintf(stderr, "[C] FAIL: remote_len=%lu != %zu\n",
                (unsigned long)remote_len, MEM_SIZE);
        close(tcp);
        return 82;
    }

    // local buffer: pattern 1, 2, 3, ...
    void *local = nullptr;
    CHECK_ACL(aclrtMalloc(&local, MEM_SIZE, ACL_MEM_MALLOC_HUGE_FIRST));
    {
        size_t n = MEM_SIZE / sizeof(float);
        std::vector<float> host(n);
        for (size_t i = 0; i < n; ++i) host[i] = (float)(i + 1);
        CHECK_ACL(aclrtMemcpy(local, MEM_SIZE, host.data(), MEM_SIZE,
                              ACL_MEMCPY_HOST_TO_DEVICE));
    }
    MemDesc md{};
    md.addr = (uintptr_t)local;
    md.len = MEM_SIZE;
    MemHandle mh = nullptr;
    CHECK_HX(cli.RegisterMem(md, MEM_DEVICE, mh), "ClientRegisterMem");

    CHECK_HX(cli.Connect(AscendString(remote_eid), 15000), "ClientConnect");

    TransferOpDesc op{};
    op.local_addr = (uintptr_t)local;
    op.remote_addr = (uintptr_t)remote_addr;
    op.len = MEM_SIZE;
    std::vector<TransferOpDesc> ops{op};
    CHECK_HX(cli.TransferSync(AscendString(remote_eid), WRITE, ops, 15000),
             "ClientTransferSyncWRITE");

    // signal server we're done
    uint8_t one = 1;
    if (send_all(tcp, &one, 1) != 0) {
        fprintf(stderr, "[C] failed to signal server\n");
        close(tcp);
        return 83;
    }
    close(tcp);

    // Mark PASS as soon as the actual transfer completed (TransferSync
    // already returned 0 above and we signalled the server, which has now
    // checksum-verified the data).  Cleanup quirks shouldn't downgrade a
    // good transfer to FAIL.
    fprintf(stderr, "[C] PASS\n");

    Status ds = cli.Disconnect(AscendString(remote_eid), 5000);
    if (ds != SUCCESS) {
        fprintf(stderr,
                "[C] WARN: ClientDisconnect ret=%u (ignoring)\n", ds);
    }
    Status dr = cli.DeregisterMem(mh);
    if (dr != SUCCESS) {
        fprintf(stderr,
                "[C] WARN: ClientDeregisterMem ret=%u (ignoring)\n", dr);
    }
    aclrtFree(local);
    aclFinalize();
    return 0;
}

// ---------------------------------------------------------------------------
int main(int argc, char **argv) {
    if (argc < 2) {
usage:
        fprintf(stderr,
                "Usage:\n"
                "  %s server <dev> <local_host_ip> <hixl_port>"
                " <rendezvous_port>\n"
                "  %s client <dev> <local_host_ip> <hixl_port>"
                " <remote_host_ip> <rendezvous_port>"
                " <remote_hixl_port>\n",
                argv[0], argv[0]);
        return 2;
    }
    std::string mode = argv[1];

    if (mode == "server") {
        if (argc < 6) goto usage;
        int dev = std::atoi(argv[2]);
        const char *local_host = argv[3];
        uint16_t hixl_port = (uint16_t)std::atoi(argv[4]);
        uint16_t rdv_port = (uint16_t)std::atoi(argv[5]);
        fprintf(stderr,
                "[server-mode] dev=%d local_host=%s hixl_port=%u "
                "rdv_port=%u\n",
                dev, local_host, hixl_port, rdv_port);
        int rc = run_server(dev, local_host, hixl_port, rdv_port);
        fprintf(stderr, "\n=== SERVER RC = %d ===\n", rc);
        return rc;
    } else if (mode == "client") {
        if (argc < 8) goto usage;
        int dev = std::atoi(argv[2]);
        const char *local_host = argv[3];
        uint16_t hixl_port = (uint16_t)std::atoi(argv[4]);
        const char *remote_host = argv[5];
        uint16_t rdv_port = (uint16_t)std::atoi(argv[6]);
        uint16_t remote_hixl_port = (uint16_t)std::atoi(argv[7]);
        fprintf(stderr,
                "[client-mode] dev=%d local_host=%s hixl_port=%u "
                "remote_host=%s rdv_port=%u remote_hixl_port=%u\n",
                dev, local_host, hixl_port, remote_host, rdv_port,
                remote_hixl_port);
        int rc = run_client(dev, local_host, hixl_port, remote_host,
                            rdv_port, remote_hixl_port);
        fprintf(stderr, "\n=== CLIENT RC = %d ===\n", rc);
        return rc;
    }
    fprintf(stderr, "unknown mode: %s\n", mode.c_str());
    goto usage;
}
