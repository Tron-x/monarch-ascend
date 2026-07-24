#!/usr/bin/env bash
# Smoke driver for the new HixlCS (ClientServer) API.
#
# Auto-discovers two NPU RoCE NIC IPs via hccn_tool, spawns the server on
# NPU srv_dev and the client on NPU cli_dev, and prints a compact summary.
#
# Usage:
#   ./run_cs_smoke.sh [srv_dev] [cli_dev] [port]
# Defaults: srv_dev=0, cli_dev=1, port=random in 55000-60000.

set -uo pipefail

SRV_DEV=${1:-0}
CLI_DEV=${2:-1}
PORT=${3:-$((RANDOM % 5000 + 55000))}

# Auto-detect the host's primary TCP IP (control plane).  We prefer
# `gethostbyname(hostname)` because that resolves to the host's "main" address
# (e.g. 192.168.x.x) rather than the outbound gateway IP returned by a
# connect-to-8.8.8.8 probe.  Override via `$HOST_IP`.
detect_host_ip() {
    if [[ -n "${HOST_IP:-}" ]]; then echo "$HOST_IP"; return; fi
    python3 - <<'PY' 2>/dev/null
import socket
try:
    h = socket.gethostname()
    ips = sorted({a[4][0] for a in socket.getaddrinfo(h, None)
                  if ":" not in a[4][0] and not a[4][0].startswith("127.")})
    if ips:
        print(ips[0]); raise SystemExit
except Exception:
    pass
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
try:
    s.connect(("8.8.8.8", 80))
    print(s.getsockname()[0])
finally:
    s.close()
PY
}
HOST_IP_LOCAL=$(detect_host_ip)

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
BIN=$SCRIPT_DIR/build/test_hixl_cs_smoke
HCCN=/usr/local/Ascend/driver/tools/hccn_tool

if [[ ! -x "$BIN" ]]; then
    echo "binary missing: $BIN"
    echo "build via: (cd $SCRIPT_DIR && make build/test_hixl_cs_smoke)"
    exit 2
fi
if [[ ! -x "$HCCN" ]]; then
    echo "hccn_tool missing at $HCCN — cannot discover NPU NIC IPs"
    exit 2
fi

resolve_ip() {
    "$HCCN" -i "$1" -ip -g 2>/dev/null | awk -F: '/ipaddr/ {print $2; exit}'
}

SRV_NIC=$(resolve_ip "$SRV_DEV")
CLI_NIC=$(resolve_ip "$CLI_DEV")
if [[ -z "$SRV_NIC" || -z "$CLI_NIC" ]]; then
    echo "could not resolve NIC IPs (srv=$SRV_NIC cli=$CLI_NIC)"
    exit 2
fi
if [[ -z "$HOST_IP_LOCAL" ]]; then
    echo "could not auto-detect host IP — set \$HOST_IP and retry"
    exit 2
fi

export HCCL_INTRA_ROCE_ENABLE=1
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

SRV_LOG=$(mktemp -t cs_srv.XXXX.log)
CLI_LOG=$(mktemp -t cs_cli.XXXX.log)

echo "=== HixlCS smoke ==="
echo "  host TCP plane:   $HOST_IP_LOCAL:$PORT"
echo "  server  NPU $SRV_DEV  rdma_nic=$SRV_NIC"
echo "  client  NPU $CLI_DEV  rdma_nic=$CLI_NIC"

"$BIN" server "$SRV_DEV" "$HOST_IP_LOCAL" "$SRV_NIC" "$PORT" >"$SRV_LOG" 2>&1 &
SRV_PID=$!
sleep 3

CS_SMOKE_REMOTE_DEV=$SRV_DEV \
    timeout 15s "$BIN" client "$CLI_DEV" "$HOST_IP_LOCAL" "$CLI_NIC" \
                              "$HOST_IP_LOCAL" "$SRV_NIC" "$PORT" \
    >"$CLI_LOG" 2>&1
CLI_RC=$?

wait $SRV_PID 2>/dev/null
SRV_RC=$?

echo
echo "--- SERVER ($SRV_LOG, rc=$SRV_RC) ---"
grep -E "\[CS\]|\[S\]|RC|smoke" "$SRV_LOG"
echo
echo "--- CLIENT ($CLI_LOG, rc=$CLI_RC) ---"
grep -E "\[CS\]|\[C\]|RC|discovered|tag=|PASS|smoke" "$CLI_LOG"

# Categorize the result so it's obvious whether CS API is "ready" yet.
LISTEN_OK=$(grep -cE "ServerListen[^[]*ret=0( |$)" "$SRV_LOG" || true)
CONNECT_OK=$(grep -cE "ClientConnect[^[]*ret=0( |$)" "$CLI_LOG" || true)
GETMEM_OK=$(grep -cE "ClientGetRemoteMem.*ret=0( |$)" "$CLI_LOG" || true)
PUT_ASYNC_OK=$(grep -cE "ClientBatchPutAsync.*ret=0( |$)" "$CLI_LOG" || true)
PUT_SYNC_OK=$(grep -cE "ClientBatchPutSync.*ret=0( |$)" "$CLI_LOG" || true)
REG_OK=$(grep -cE "ServerRegMem.*ret=0( |$)" "$SRV_LOG" || true)

echo
echo "=== SUMMARY ==="
printf '  ServerListen                : %s\n' "$([[ $LISTEN_OK -ge 1 ]] && echo OK || echo FAIL)"
printf '  ServerRegMem (multi-region) : %s\n' "$([[ $REG_OK -ge 2 ]] && echo OK || echo FAIL)"
printf '  ClientConnect               : %s\n' "$([[ $CONNECT_OK -ge 1 ]] && echo OK || echo FAIL)"
printf '  ClientGetRemoteMem          : %s\n' "$([[ $GETMEM_OK -ge 1 ]] && echo OK || echo FAIL)"
printf '  ClientBatchPutAsync (submit): %s\n' "$([[ $PUT_ASYNC_OK -ge 1 ]] && echo OK || echo FAIL)"
printf '  ClientBatchPutSync (data)   : %s\n' "$([[ $PUT_SYNC_OK -ge 1 ]] && echo OK || echo BROKEN)"
echo
CONTROL_OK=$([[ $LISTEN_OK -ge 1 && $REG_OK -ge 2 && $CONNECT_OK -ge 1 && \
              $GETMEM_OK -ge 1 && $PUT_ASYNC_OK -ge 1 ]] && echo 1 || echo 0)

if [[ "$CONTROL_OK" == "1" && $PUT_SYNC_OK -ge 1 ]]; then
    echo "VERDICT: CS API is READY for Monarch migration (control + data)."
    exit 0
elif [[ "$CONTROL_OK" == "1" ]]; then
    # All same-host CS-API + LocalCommRes data-plane failures we have observed
    # so far are the SAME failure shape: BatchPutAsync ret=0, QueryComplete
    # stays at st=0 (WAITING) forever, BatchPutSync returns 507018
    # (ACL_ERROR_RT_AICPU_EXCEPTION), and the device-side aicpu_scheduler plog
    # has 'Load so libcann_hixl_kernel.so failed' (the new HiXL AICPU kernel
    # was not deployed to the device).  Cross-host transfers work because they
    # take a different in-driver code path (HCCL ROCE transport via
    # libccl_kernel.so) that does not depend on libcann_hixl_kernel.so.
    #
    # We observed exactly one window on 2026-05-19 where the SO did get loaded
    # ('Get api HixlBatchPut from so libcann_hixl_kernel.so success' in plog)
    # and the data plane PASSed for ~4 invocations; the SO was lost a few
    # hours later without driver reload or container restart and has not come
    # back since.  We report this as a CANN-9.1.0/HDK-25.5.1 deploy bug, not
    # a Monarch-side issue — see docs/hixl_feedback_aicpu_kernel_deploy.md.
    DEV_LOG_LATEST=$(ls -t /root/ascend/log/run/device-*/*.log 2>/dev/null | head -3)
    KERNEL_LOAD_OK=$(grep -lE "Get api HixlBatchPut.*success" $DEV_LOG_LATEST 2>/dev/null | wc -l)
    KERNEL_LOAD_FAIL=$(grep -lE "Load so libcann_hixl_kernel\.so failed" $DEV_LOG_LATEST 2>/dev/null | wc -l)
    SYNC_ERR=$(grep -oE "ClientBatchPutSync.*ret=[0-9]+" "$CLI_LOG" | grep -oE "ret=[0-9]+" | head -1)

    echo "VERDICT: CS API control plane is READY, but same-host data-plane (BatchPut) is broken."
    echo "         sync_err=$SYNC_ERR"
    echo "         device-side libcann_hixl_kernel.so:"
    echo "           Get api success markers (in latest 3 plog): $KERNEL_LOAD_OK"
    echo "           Load so  failed  markers (in latest 3 plog): $KERNEL_LOAD_FAIL"
    if [[ $KERNEL_LOAD_FAIL -ge 1 && $KERNEL_LOAD_OK -lt 1 ]]; then
        echo "  -> Diagnosis: libcann_hixl_kernel.so is NOT deployed to the device."
        echo "                Host has cann-hixl-compat.tar.gz under opp/built-in/op_impl/"
        echo "                aicpu/kernel/ but device-side AICPU scheduler cannot load"
        echo "                it.  This is a CANN/driver deploy bug — see"
        echo "                docs/hixl_feedback_aicpu_kernel_deploy.md for the full"
        echo "                writeup and reproducer that should be sent to the HiXL team."
        echo "                Monarch's hixl_shim defaults MONARCH_HIXL_USE_LOCAL_COMM_RES"
        echo "                to OFF, so production traffic stays on the device-direct"
        echo "                path that does NOT touch this kernel — unaffected."
    elif [[ $KERNEL_LOAD_OK -ge 1 ]]; then
        echo "  -> Diagnosis: SO load is succeeding intermittently right now."
        echo "                If this run still returns sync=$SYNC_ERR, something else"
        echo "                in the BatchPut data plane is wrong; capture the full"
        echo "                aicpu_scheduler plog and attach it to the HiXL ticket"
        echo "                in docs/hixl_feedback_aicpu_kernel_deploy.md."
    else
        echo "  -> Diagnosis: undetermined — check plog under /root/ascend/log/run/"
        echo "                manually for 'Load so' / 'Get api' messages."
    fi
    exit 2
else
    echo "VERDICT: CS API is NOT ready (control plane failing). Stay on legacy P2P."
    exit 1
fi
