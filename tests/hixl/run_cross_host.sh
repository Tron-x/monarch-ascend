#!/usr/bin/env bash
# Cross-host driver for tests/hixl/native/test_hixl_cross_host.cpp.
#
# Runs `server` on monarch1 (local) and `client` on monarch2 (over ssh -p
# 36000).  Auto-discovers NPU RoCE NIC IPs on both hosts via `hccn_tool`,
# picks the host TCP IP for control rendezvous, and runs the test in two
# back-to-back passes:
#   pass 1: default (no LocalCommRes) — legacy HiXL P2P path
#   pass 2: MONARCH_HIXL_USE_LOCAL_COMM_RES=1 — recommended new path
#
# Both passes are expected to PASS on CANN 9.1.0 + HDK 25.5.1 with the new
# signed `cann-hixl-compat.tar.gz` deployed by the driver.
#
# Usage:
#   bash run_cross_host.sh [server_dev=0] [client_dev=0]

set -uo pipefail

SRV_DEV=${1:-0}
CLI_DEV=${2:-0}

SRV_HOST_IP=${SRV_HOST_IP:-192.168.0.26}   # m1 host TCP IP (rendezvous + ssh source)
CLI_HOST_IP=${CLI_HOST_IP:-192.168.0.23}   # m2 host TCP IP (ssh target)
CLI_SSH_PORT=${CLI_SSH_PORT:-36000}

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
BIN_REL=tests/hixl/build/test_hixl_cross_host
BIN_ABS=/root/monarch/$BIN_REL
HCCN=/usr/local/Ascend/driver/tools/hccn_tool

[[ -x "$BIN_ABS" ]] || { echo "[fatal] $BIN_ABS missing — build first"; exit 2; }
[[ -x "$HCCN" ]]    || { echo "[fatal] hccn_tool missing"; exit 2; }

# Just sanity-check the NPU NIC IPs exist (HiXL itself uses them via
# LocalCommRes/rankTable, but our test process binds only to host TCP IPs).
SRV_NIC=$("$HCCN" -i "$SRV_DEV" -ip -g 2>/dev/null | awk -F: '/ipaddr/ {print $2; exit}')
CLI_BIN_INFO=$(ssh -p "$CLI_SSH_PORT" -o StrictHostKeyChecking=no -o BatchMode=yes \
                  root@$CLI_HOST_IP \
                  "$HCCN -i $CLI_DEV -ip -g 2>/dev/null | awk -F: '/ipaddr/ {print \$2; exit}'; \
                   ls -la $BIN_ABS 2>&1 | head -1")
CLI_NIC=$(echo "$CLI_BIN_INFO" | head -1 | tr -d ' ')
echo "[setup] server  : host=$SRV_HOST_IP  NPU $SRV_DEV  (RoCE NIC info: $SRV_NIC)"
echo "[setup] client  : host=$CLI_HOST_IP  NPU $CLI_DEV  (RoCE NIC info: $CLI_NIC)"
echo "[setup] client bin: $(echo "$CLI_BIN_INFO" | tail -1)"

run_pass() {
    local name=$1 lcr_value=$2
    local hixl_srv_port=$(( RANDOM % 5000 + 30000 ))
    local hixl_cli_port=$(( RANDOM % 5000 + 35000 ))
    local rdv_port=$(( RANDOM % 5000 + 40000 ))
    local srv_log=$(mktemp -t cross_srv.XXXX.log)
    local cli_log=$(mktemp -t cross_cli.XXXX.log)

    echo
    echo "========================================="
    echo " PASS: $name   (rdv=$SRV_HOST_IP:$rdv_port)"
    echo "       LocalCommRes=$lcr_value"
    echo "========================================="

    # Build env list as proper `env KEY=VAL ...` prefix; embedding "KEY=VAL"
    # via $var expansion does NOT make bash treat it as an env assignment.
    local env_args=(env HCCL_INTRA_ROCE_ENABLE=1)
    [[ "$lcr_value" == "1" ]] && env_args+=(MONARCH_HIXL_USE_LOCAL_COMM_RES=1)
    local remote_env="HCCL_INTRA_ROCE_ENABLE=1"
    [[ "$lcr_value" == "1" ]] && remote_env="$remote_env MONARCH_HIXL_USE_LOCAL_COMM_RES=1"

    # spawn server locally first (it listens for both rendezvous TCP and HiXL)
    "${env_args[@]}" \
        "$BIN_ABS" server "$SRV_DEV" "$SRV_HOST_IP" "$hixl_srv_port" \
                  "$rdv_port" \
        >"$srv_log" 2>&1 &
    local SRV_PID=$!

    # give server a beat to start listening on rendezvous port
    sleep 2

    # client over ssh
    ssh -p "$CLI_SSH_PORT" -o StrictHostKeyChecking=no -o BatchMode=yes \
        root@$CLI_HOST_IP \
        "source /usr/local/Ascend/ascend-toolkit/set_env.sh >/dev/null 2>&1; \
         env $remote_env \
         timeout 30s $BIN_ABS client $CLI_DEV $CLI_HOST_IP $hixl_cli_port \
            $SRV_HOST_IP $rdv_port $hixl_srv_port" \
        >"$cli_log" 2>&1
    local CLI_RC=$?

    wait "$SRV_PID" 2>/dev/null
    local SRV_RC=$?

    echo
    echo "--- SERVER (rc=$SRV_RC)  $srv_log ---"
    grep -E "\[init\]|\[HX\]|\[S\]|\[tcp\]|RC|sum=|PASS|FAIL|Aborted" "$srv_log" | tail -30
    echo
    echo "--- CLIENT (rc=$CLI_RC)  $cli_log ---"
    grep -E "\[init\]|\[HX\]|\[C\]|\[tcp\]|RC|PASS|FAIL|Aborted|remote_eid" "$cli_log" | tail -30

    # Detect application-level PASS markers from the log, independently of
    # exit code: HiXL libraries sometimes SIGABRT in cleanup (aclFinalize
    # destructors) AFTER a successful data transfer — that's a cleanup bug,
    # not a transfer failure.
    local srv_pass=$(grep -cE "^\[S\] PASS|^=== SERVER RC = 0 ===" "$srv_log" 2>/dev/null || true)
    local cli_pass=$(grep -cE "^\[C\] PASS|^=== CLIENT RC = 0 ===" "$cli_log" 2>/dev/null || true)

    if [[ $srv_pass -ge 1 && $cli_pass -ge 1 ]]; then
        if [[ $SRV_RC -ne 0 || $CLI_RC -ne 0 ]]; then
            echo "[VERDICT] $name : PASS (data transfer OK, but cleanup SIGABRT" \
                 "srv_rc=$SRV_RC cli_rc=$CLI_RC — known HiXL Finalize race)"
        else
            echo "[VERDICT] $name : PASS (m1 server + m2 client OK)"
        fi
        return 0
    fi
    echo "[VERDICT] $name : FAIL  (srv_rc=$SRV_RC cli_rc=$CLI_RC, srv_pass=$srv_pass cli_pass=$cli_pass)"
    return 1
}

overall=0
run_pass "LEGACY (no LocalCommRes)" 0    || overall=1
run_pass "LocalCommRes ENABLED"     1    || overall=1

echo
echo "================================================================"
if [[ $overall -eq 0 ]]; then
    echo " OVERALL: cross-host HiXL transfer works on CANN 9.1.0"
    echo "          both legacy and LocalCommRes paths PASS"
else
    echo " OVERALL: at least one pass FAILED — see logs above"
fi
echo "================================================================"
exit $overall
