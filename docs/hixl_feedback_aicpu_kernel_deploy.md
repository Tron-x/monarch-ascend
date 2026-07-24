# HiXL feedback — AICPU kernel (`libcann_hixl_kernel.so`) device-side deploy is unreliable on CANN 9.1.0 + HDK 25.5.1

**Status**: open, reproducible.
**Affected**: any code path that touches `HixlBatchPut` / `HixlBatchGet` AICPU
kernels — both the new `HixlCS*` ClientServer API and the legacy
`hixl::Hixl::Initialize(..., {"LocalCommRes":"{\"version\":\"1.3\"}"})`
option.
**Not affected**: legacy `hixl::Hixl` API with default options (no
`LocalCommRes`, no CS API) and any cross-host path that internally uses HCCL
ROCE transport via `libccl_kernel.so`.

## 1. Environment

| Component | Version |
| --- | --- |
| HDK driver  | `25.5.1`, `Innerversion=V100R001C23SPC006B220`, package `25.5.1` |
| Firmware    | `7.8.0.6.201` |
| CANN toolkit | `9.1.0`, `innerversion=V100R001C11B063` |
| CANN 910B ops | `Ascend-cann-910b-ops 9.1.0` (separate `.run`, properly installed — `ascend_ops_install.info` present) |
| Hardware    | Atlas 910B (`Product Name=IT21HMDA_Bin0`), 8 NPUs/node, RoCE 200 Gbps |
| OS          | linux 5.10.0-136 aarch64, container deploy |
| Hosts in test | monarch1 `192.168.0.26`, monarch2 `192.168.0.23` |

### Files on host

```
/usr/local/Ascend/cann-9.1.0/opp/built-in/op_impl/aicpu/
├── config/libcann_hixl_kernel.json                       # 380 B, present
└── kernel/cann-hixl-compat.tar.gz                        # 48514 B, md5=25574e55fcdea8cad3311cdca5e3ef9f
                                                          # magic AA 55 AA 55 00 00 ... (Huawei device-side fw container)
```

```
/usr/local/Ascend/cann-9.1.0/aarch64-linux/lib64/libcann_hixl.so   # 2.3 MB, dated Feb 8 — host-side user library
                                                                   # libcann_hixl_kernel.so (device-side) is NOT on host
```

### Files on device (under `/usr/lib64/aicpu_kernels/0/aicpu_kernels_device/`)

```
(empty) — no libcann_hixl_kernel.so observed at any time on host filesystem;
the per-PID subdir `<pid>_0/` is what device-side AICPU scheduler tries to
load from, and the .so is missing there in every failing run.
```

## 2. Observed failure & root cause

### 2.1 The two device-side SO load paths

`aicpu_scheduler` on the device loads its AICPU kernels through two
distinct mechanisms. Output below is from a single failing run's device
plog:

```
[INFO] CCECPU(...aicpu_scheduler) [main.cpp:360]
       LD_LIBRARY_PATH=/usr/lib64/aicpu_kernels/0/aicpu_kernels_device/:
                       /usr/lib64/aicpu_kernels/0/aicpu_kernels_device/sand_box/:
                       /home/HwHiAiUser/hs/0/runtime/lib64/:
                       /usr/lib64/device-compat-plugin/0/:
                       /usr/lib64/device-sw-plugin/0/device-sw-plugin/:
[INFO] CCECPU(...aicpu_scheduler) [aicpusd_cust_so_manager.cpp:83]
       Cust so manager init successfully,
       dir=/home/CustAiCpuUser/cust_aicpu_0_0_<host_pid>/, runMode=0, schedMode=0.

# Path 1 — driver-bundled, always-available
[INFO] [ae_so_manager.cc:258]
       Get api RunAicpuThreadInit  from so libccl_kernel.so success.
[INFO] [ae_so_manager.cc:258]
       Get api RunAicpuChannelInitV3 from so libccl_kernel.so success.

# Path 2 — per-host-pid "custom so", deployed at runtime via HDC,
# cleared on host-process exit
[WARN] [ae_so_manager.cc:348][GetApi]
       Load so libcann_hixl_kernel.so failed.
```

* **Path 1** ("standard AICPU kernels", e.g. `libccl_kernel.so`,
  `libhcom_kernel.so`) is satisfied by the device-side `LD_LIBRARY_PATH`,
  which the driver always provisions when the chip comes up. No per-process
  deploy is needed.
* **Path 2** ("custom AICPU kernels") is for SOs that the runtime needs to
  push from the host package on demand. The device-side cust-so manager
  expects them at `/home/CustAiCpuUser/cust_aicpu_0_0_<host_pid>/`. They
  are tied to the host process lifetime and the device cleans the directory
  when the host process exits.

**`libcann_hixl_kernel.so` is the only AICPU kernel in CANN 9.1.0 that goes
through Path 2.** `libccl_kernel.so` / `libhcom_kernel.so` are on Path 1,
which is exactly why HCCL — and the HCCL-backed cross-host HiXL transport
internally — keeps working when same-host HiXL transfers fail.

### 2.2 Host-side runtime cache vs. device-side per-PID cleanup

The host-side runtime decides whether to push `cann-hixl-compat.tar.gz` at
each `aclInit`. The host plog records this decision verbatim every time:

```
[INFO] TDT(<pid>,...) [package_process_config.cpp:264][SetConfigDataOnHost]
       insert package:cann-hixl-compat.tar.gz config
[INFO] TDT(<pid>,...) [process_mode_manager.cpp:2259][LoadPackageToDeviceByConfig]
       begin to load package:cann-hixl-compat.tar.gz to device:0
[INFO] TDT(<pid>,...) [package_process_config.cpp:323][GetPkgHostAndDeviceDstPath]
       get orgFile:.../cann-hixl-compat.tar.gz,
           dstFile:/home/HwHiAiUser/hdcd/device0/aicpu/<host_pid>_cann-hixl-compat.tar.gz
[INFO] TDT(<pid>,...) [process_mode_manager.cpp:2274][LoadPackageToDeviceByConfig]
       current package:cann-hixl-compat.tar.gz is same as device, skip load
```

The same trace appears for `aicpu_hccl.tar.gz` and `aicpu_hcomm.tar.gz` —
**all three packages report "is same as device, skip load"**. The first
two packages still work, because their SOs live in the Path-1 location
(populated at chip boot, never cleaned).
`cann-hixl-compat.tar.gz` does **not** work, because:

1. The host-side `LoadPackageToDeviceByConfig` cache says "already deployed",
   so the package is never re-pushed.
2. But the device-side `cust_aicpu_0_0_<host_pid>/` directory has been
   cleaned by the time a new host process starts — `cust_aicpu_*` lifetime
   is per-host-PID, not per-package.
3. The new host process therefore never sees `libcann_hixl_kernel.so` in
   the cust-so directory, and `ae_so_manager` falls through to
   `[WARN] Load so libcann_hixl_kernel.so failed.`

This is a clear **cache-coherency bug between the host runtime
(`LoadPackageToDeviceByConfig`) and the device-side cust-so manager** on
CANN 9.1.0. The host runtime should either:

* invalidate its per-package cache when the device-side `cust_aicpu_*`
  directory is reaped, or
* always push `cann-hixl-compat.tar.gz` (treat it as Path-1-equivalent and
  install into a permanent `LD_LIBRARY_PATH` directory at chip boot),
  matching what's already done for `aicpu_hccl.tar.gz` and
  `aicpu_hcomm.tar.gz`.

### 2.3 User-space symptom

User-space symptom (downstream of §2.2):

* `HixlCSClientBatchPutAsync(...)`     -> `ret=0`        (kernel submission accepted)
* `HixlCSClientQueryCompleteStatus(...)` -> repeatedly returns
  `st=0  (HIXL_COMPLETE_STATUS_WAITING)` — never advances to COMPLETED.
* `HixlCSClientBatchPutSync(...)`      -> `ret=507018 == ACL_ERROR_RT_AICPU_EXCEPTION`
* Same for the legacy P2P API when the `LocalCommRes` Init option is set:
  `hixl_transfer_write(...)` -> `ret=507018`.

## 3. What does work

The same machine on the same day, two paths in parallel:

| Path | AICPU kernel used | Status |
| --- | --- | --- |
| Same-host, legacy P2P, **no** `LocalCommRes` | none (device-direct) | ✅ 100% PASS |
| Same-host, legacy P2P, **with** `LocalCommRes={"version":"1.3"}` | `libcann_hixl_kernel.so` :: `HixlBatchPut` | ❌ `507018` |
| Same-host, CS API (`HixlCSClient*`) | `libcann_hixl_kernel.so` :: `HixlBatchPut` | ❌ `507018` (10/10) |
| **Cross-host**, legacy P2P + `LocalCommRes` | `libccl_kernel.so` (HCCL ROCE transport, driver-bundled) | ✅ PASS, data bit-exact |

The cross-host pass is what tripped us into initially thinking
`LocalCommRes` was working — but the device plog for that run clearly shows
HCCL ROCE transport (`hcomm_aicpu_ts_roce`, `libccl_kernel.so`), never
`HixlBatchPut` / `libcann_hixl_kernel.so`. Same-host transfers can't fall
back to HCCL transport, so they hit the AICPU path and fail.

## 4. The "sometimes it works" datapoint (now explained)

On 2026-05-19 around 14:14 UTC (shortly after our CANN 9.1.0 upgrade and a
fresh container start), 4 process invocations did successfully load the
kernel and PASS. The device plog of those runs contains:

```
[INFO] CCECPU(9797,aicpu_scheduler): ... aicpusd_cust_so_manager.cpp:83
       Cust so manager init successfully, dir=/home/CustAiCpuUser/cust_aicpu_0_0_159659/...
[INFO] CCECPU(9797,aicpu_scheduler): ... ae_so_manager.cc:258  Get api HixlBatchPut from so libcann_hixl_kernel.so success.
```

Reading this together with §2.2: the very first process after a fresh
container boot **did** trigger a real `LoadPackageToDeviceByConfig` push
(the host cache was empty), `aicpusd` deployed `cann-hixl-compat.tar.gz`
into `cust_aicpu_0_0_<pid>/`, and the kernel loaded. As soon as that host
process exited, the device's cust-so manager reaped the per-PID directory.
Every subsequent host process re-uses the host-side "already deployed"
cache, so the package is never pushed again, so every subsequent
`HixlBatchPut` lookup fails. This is consistent with the 10/10
back-to-back failure observed afterwards.

In other words: the SO load is not "racy per process" or "time-windowed".
It is **exactly once per fresh host-runtime instance, and only on the
process that wins the cache miss**. After that, the cache says "deployed"
forever (until host runtime restart / container restart), but the device
side has cleaned the deployment.

## 5. Reproducer

Two binaries live in this repo:

```
tests/hixl/native/test_hixl_cs_smoke.cpp           # CS API smoke
tests/hixl/run_cs_smoke.sh                         # driver, prints VERDICT
tests/hixl/e2e/test_hixl_bridge_minimal.py         # legacy P2P + LocalCommRes via Monarch
```

To repro in a fresh container against CANN 9.1.0 + HDK 25.5.1:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# (1) CS API — should be 100% FAIL with 507018 on the SO-not-deployed window
bash tests/hixl/run_cs_smoke.sh
# Expect:
#   ClientBatchPutAsync (submit): OK         (ret=0)
#   ClientBatchPutSync  (data)  : BROKEN     (ret=507018)
#   Diagnosis: kernel SO is loaded successfully on the device, but... OR
#              kernel SO missing in the NPU image...

# (2) Legacy P2P + LocalCommRes — same failure
MONARCH_HIXL_USE_LOCAL_COMM_RES=1 \
    python tests/hixl/e2e/test_hixl_bridge_minimal.py
# Expect: Exception "hixl_transfer_write failed: ... ret=507018"

# (3) Legacy P2P without LocalCommRes — works
python tests/hixl/e2e/test_hixl_bridge_minimal.py
# Expect: PASS: minimal HIXL bridge read/write path works

# (4) Capture device plog for any of the above failing runs:
ls -t /root/ascend/log/run/device-*/*.log | head -1 | \
    xargs grep -E "Load so|Get api|hixl_kernel"
# Expect: 'Load so libcann_hixl_kernel.so failed.'
```

## 6. Questions to the HiXL / CANN runtime team

1. **Why is `cann-hixl-compat.tar.gz` shipped as a per-host-PID `cust_so`
   instead of a permanent driver-bundled AICPU kernel** (the same way
   `libccl_kernel.so` and `libhcom_kernel.so` are shipped via
   `aicpu_hccl.tar.gz` / `aicpu_hcomm.tar.gz`)? All three packages have
   the identical `AA 55 AA 55 …` container format and live in the same
   `opp/built-in/op_impl/aicpu/kernel/` directory; the two HCCL packages
   land in the device-side `LD_LIBRARY_PATH` and are visible to every
   process, while the HiXL package lands in `cust_aicpu_0_0_<host_pid>/`
   and gets reaped on each host-process exit.
2. **`LoadPackageToDeviceByConfig` caches "is same as device, skip load"
   for the entire host-runtime lifetime**, but the device-side `cust_so`
   directory is per-host-PID and is cleaned by `aicpusd` when each host
   process exits. After the first successful deploy, all subsequent host
   processes hit the cache, the package is never re-pushed, and
   `aicpusd_cust_so_manager` can no longer find
   `libcann_hixl_kernel.so` for new host PIDs. Is this an intentional
   design, or a coherence bug? If intentional, what is the supported
   workflow for keeping the SO available across multiple host processes
   without restarting the host runtime?
3. **Is there a host-side knob to force re-push of `cann-hixl-compat.tar.gz`**
   on every `aclInit`? An env variable, a `npu-smi` command, or an explicit
   API would let us work around the cache for now.
4. **Recommended workaround**: it would be straightforward, in our
   container image, to copy `libcann_hixl_kernel.so` into the device-side
   `LD_LIBRARY_PATH` (`/usr/lib64/aicpu_kernels/0/aicpu_kernels_device/`)
   at boot so that the AICPU scheduler picks it up via Path 1 instead of
   the per-PID Path 2. Is the extracted `.so` available anywhere outside
   the encrypted `cann-hixl-compat.tar.gz`? On host the `.tar.gz` is
   48,514 B of opaque binary with a Huawei container header, not a plain
   gzip/tar archive — we cannot extract it ourselves.
5. **Does signature verification factor in?** The header is `AA 55 AA 55
   00 00 00 00 …` (16 B header + 32 B per-package hash + format trailer),
   which matches the two HCCL `.tar.gz` packages exactly. The signed-package
   verifier knobs (`npu-smi info -t custom-op-secverify-{enable,mode}`)
   return `Error parameter in querying chip info` on this image, so we
   can't even read the current state.
6. **For the legacy P2P API with `LocalCommRes`, is there a way to opt
   the data plane out of the AICPU kernel** and stay on the device-direct
   path that the default ("no `LocalCommRes`") option uses successfully?
   Right now our shim treats `LocalCommRes` as an opt-in env knob
   (`MONARCH_HIXL_USE_LOCAL_COMM_RES`) defaulting to off, so production
   traffic stays on the working path until this is resolved.

## 7. Workaround we have shipped to production

In `hixl-sys/cpp/hixl_shim.cpp::populate_init_opts()` (Monarch's HiXL C
shim) we only inject `LocalCommRes={"version":"1.3"}` when
`MONARCH_HIXL_USE_LOCAL_COMM_RES=1` is set. Default behaviour:

* `BufferPool=0:0` (always, Monarch manages its own NPU memory)
* No `LocalCommRes` => HiXL uses the device-direct path => no dependency
  on `libcann_hixl_kernel.so` => same-host transfers work end-to-end.

The historical "per-engine-pair one region (503900)" and "HcclAdapter
contention (0x13)" issues that originally drove the `LocalCommRes`
recommendation are themselves **resolved on CANN 9.1.0 even on the
default path** — we have direct tests for both
(`tests/hixl/e2e/test_multi_region_per_pair.py`,
`tests/hixl/e2e/test_hixl_hccl_coexist.py`) and both pass with
`LocalCommRes` off. So in the absence of a working AICPU kernel deploy,
turning `LocalCommRes` off costs us nothing functionally; we only lose
the future-proofing benefit.
