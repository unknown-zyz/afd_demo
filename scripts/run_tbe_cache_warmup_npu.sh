#!/usr/bin/env bash
# Run an observable Ascend TBE/kernel_meta warmup for Qwen3 NPU experiments.
#
# The plain run_npu.sh command redirects every rank to results/logs, so a cold
# EP7 compile can look idle from the foreground. This wrapper keeps polling
# rank count, kernel_meta size, and rank logs while run_npu.sh is active.

set -uo pipefail

cd "$(dirname "$0")/.."

usage() {
    cat <<'EOF'
Usage:
  bash scripts/run_tbe_cache_warmup_npu.sh --profile <host1-attn|host2-ep7> [options]

Profiles:
  host1-attn   Local 1A1F warmup on Host1 using ASCEND_VISIBLE_DEVICES=0,1.
  host2-ep7    Local 1A7F/EP7 warmup on Host2 using ASCEND_VISIBLE_DEVICES=0..7.

Options:
  --batch N             Batch size (default: 2)
  --seq N               Prefill sequence length (default: 128)
  --tokens N            max_new_tokens, kept for suffix compatibility (default: 5)
  --model-name PATH     Model path (default: /models/Qwen3-30B-A3B)
  --timeout-sec N       Wall timeout for run_npu.sh (default: 7200)
  --poll-sec N          Status poll interval (default: 30)
  --no-tail             Do not print rank log tails during polling
  -h, --help            Show this help

Examples:
  # Host1
  bash scripts/run_tbe_cache_warmup_npu.sh --profile host1-attn --timeout-sec 7200

  # Host2
  bash scripts/run_tbe_cache_warmup_npu.sh --profile host2-ep7 --timeout-sec 7200
EOF
}

PROFILE=""
BATCH=2
SEQ=128
TOKENS=5
MODEL_NAME="${MODEL_NAME:-/models/Qwen3-30B-A3B}"
TIMEOUT_SEC=7200
POLL_SEC=30
TAIL_LOGS=true

while [[ $# -gt 0 ]]; do
    case "$1" in
        --profile) PROFILE="$2"; shift 2 ;;
        --batch) BATCH="$2"; shift 2 ;;
        --seq) SEQ="$2"; shift 2 ;;
        --tokens) TOKENS="$2"; shift 2 ;;
        --model-name) MODEL_NAME="$2"; shift 2 ;;
        --timeout-sec) TIMEOUT_SEC="$2"; shift 2 ;;
        --poll-sec) POLL_SEC="$2"; shift 2 ;;
        --no-tail) TAIL_LOGS=false; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ -z "$PROFILE" ]]; then
    echo "ERROR: --profile is required" >&2
    usage >&2
    exit 2
fi

if ! [[ "$TIMEOUT_SEC" =~ ^[0-9]+$ ]] || ! [[ "$POLL_SEC" =~ ^[0-9]+$ ]]; then
    echo "ERROR: --timeout-sec and --poll-sec must be positive integers" >&2
    exit 2
fi

mkdir -p results/logs

COMMON_ARGS=(
    --batch "$BATCH"
    --seq "$SEQ"
    --tokens "$TOKENS"
    --model-name "$MODEL_NAME"
    --no-dbo
    --no-generate
    --prefill-warmup-rounds 1
)

case "$PROFILE" in
    host1-attn)
        PROFILE_DESC="Host1 local 1A1F warmup"
        VISIBLE_DEVICES="0,1"
        RUN_ARGS=(
            --attn-size 1
            --ffn-size 1
            --ffn-tp-size 1
            "${COMMON_ARGS[@]}"
        )
        SUFFIX="serial-prefill_npu_b${BATCH}_s${SEQ}_t${TOKENS}"
        ;;
    host2-ep7)
        PROFILE_DESC="Host2 local 1A7F/EP7 warmup"
        VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
        RUN_ARGS=(
            --preset npu-ep7
            --ffn-ep-backend broadcast_reduce_overlap
            "${COMMON_ARGS[@]}"
        )
        SUFFIX="serial-prefill_npu_ep7_broadcast_reduce_overlap_b${BATCH}_s${SEQ}_t${TOKENS}"
        ;;
    *)
        echo "ERROR: unknown profile '$PROFILE'" >&2
        usage >&2
        exit 2
        ;;
esac

export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-3600}"
export HCCL_EXEC_TIMEOUT="${HCCL_EXEC_TIMEOUT:-3600}"
export AFD_DIST_TIMEOUT_SEC="${AFD_DIST_TIMEOUT_SEC:-7200}"
export ASCEND_GLOBAL_CACHE_ENABLE="${ASCEND_GLOBAL_CACHE_ENABLE:-1}"

LOG_GLOB="results/logs/npu_${SUFFIX}_r"*.log
WRAPPER_LOG="results/logs/tbe_cache_warmup_${PROFILE}_b${BATCH}_s${SEQ}_t${TOKENS}.log"

matching_rank_pids() {
    ps -eo pid,args | awk -v suffix="$SUFFIX" '
        {
            cmd = $2
            sub(".*/", "", cmd)
            has_main = 0
            for (i = 3; i < NF; i++) {
                if ($i == "-m" && $(i + 1) == "src.main") {
                    has_main = 1
                    break
                }
            }
        }
        cmd ~ /^python[0-9.]*$/ && has_main && index($0, suffix) > 0 {
            print $1
        }
    '
}

count_matching_ranks() {
    matching_rank_pids | wc -l | tr -d ' '
}

cleanup_matching_ranks() {
    mapfile -t pids < <(matching_rank_pids)
    if [[ "${#pids[@]}" -eq 0 ]]; then
        return 0
    fi
    echo "[warmup-cleanup] force terminating rank PIDs: ${pids[*]}"
    kill -KILL "${pids[@]}" 2>/dev/null || true
}

cleanup_launcher() {
    local pid="${RUN_PID:-}"
    if [[ -z "$pid" ]] || ! ps -p "$pid" >/dev/null 2>&1; then
        return 0
    fi
    echo "[warmup-cleanup] terminating launcher PID: $pid"
    if [[ "${LAUNCHER_SETSID:-false}" == true ]]; then
        kill -TERM "-$pid" 2>/dev/null || true
    fi
    kill -TERM "$pid" 2>/dev/null || true
    for _ in 1 2 3 4 5; do
        ps -p "$pid" >/dev/null 2>&1 || return 0
        sleep 1
    done
    if [[ "${LAUNCHER_SETSID:-false}" == true ]]; then
        kill -KILL "-$pid" 2>/dev/null || true
    fi
    kill -KILL "$pid" 2>/dev/null || true
}

on_signal() {
    echo "[warmup-cleanup] signal received; cleaning matching ranks"
    cleanup_matching_ranks
    cleanup_launcher
}
trap on_signal INT TERM

echo "=== TBE cache warmup ==="
echo "profile=$PROFILE ($PROFILE_DESC)"
echo "batch=$BATCH seq=$SEQ tokens=$TOKENS model=$MODEL_NAME"
echo "visible_devices=$VISIBLE_DEVICES"
echo "timeout_sec=$TIMEOUT_SEC poll_sec=$POLL_SEC"
echo "HCCL_CONNECT_TIMEOUT=$HCCL_CONNECT_TIMEOUT HCCL_EXEC_TIMEOUT=$HCCL_EXEC_TIMEOUT AFD_DIST_TIMEOUT_SEC=$AFD_DIST_TIMEOUT_SEC"
echo "ASCEND_GLOBAL_CACHE_ENABLE=$ASCEND_GLOBAL_CACHE_ENABLE"
echo "kernel_meta_before=$(du -sh kernel_meta 2>/dev/null | awk '{print $1}' || echo missing)"
echo "logs=results/logs/npu_${SUFFIX}_r*.log"
echo "wrapper_log=$WRAPPER_LOG"
echo ""

LAUNCHER_SETSID=false
if command -v setsid >/dev/null 2>&1; then
    LAUNCHER_SETSID=true
fi

(
    export ASCEND_VISIBLE_DEVICES="$VISIBLE_DEVICES"
    export ASCEND_RT_VISIBLE_DEVICES="$VISIBLE_DEVICES"
    if [[ "$LAUNCHER_SETSID" == true ]]; then
        exec setsid bash scripts/run_npu.sh "${RUN_ARGS[@]}"
    else
        exec bash scripts/run_npu.sh "${RUN_ARGS[@]}"
    fi
) > "$WRAPPER_LOG" 2>&1 &
RUN_PID=$!

start_ts=$(date +%s)
rc=0
timed_out=false
while ps -p "$RUN_PID" >/dev/null 2>&1; do
    now_ts=$(date +%s)
    elapsed=$((now_ts - start_ts))
    rank_count=$(count_matching_ranks)
    meta_size=$(du -sh kernel_meta 2>/dev/null | awk '{print $1}' || echo missing)
    echo ""
    echo "[warmup-status] elapsed=${elapsed}s run_pid=$RUN_PID active_src_main=${rank_count:-0} kernel_meta=$meta_size"

    if [[ "$TAIL_LOGS" == true ]]; then
        if [[ -f "$WRAPPER_LOG" ]]; then
            echo "--- tail: $WRAPPER_LOG ---"
            tail -n 8 "$WRAPPER_LOG"
        fi
        found=false
        for log in $LOG_GLOB; do
            if [[ -f "$log" ]]; then
                found=true
                echo "--- tail: $log ---"
                tail -n 6 "$log"
            fi
        done
        if [[ "$found" == false ]]; then
            echo "(rank logs not created yet)"
        fi
    fi

    if (( elapsed >= TIMEOUT_SEC )); then
        echo "[warmup-timeout] elapsed=${elapsed}s reached timeout_sec=${TIMEOUT_SEC}; cleaning up"
        timed_out=true
        cleanup_matching_ranks
        cleanup_launcher
        break
    fi

    sleep "$POLL_SEC"
done

if [[ "$timed_out" == true ]]; then
    rc=124
    wait "$RUN_PID" 2>/dev/null || true
elif ! wait "$RUN_PID"; then
    rc=$?
fi
if [[ "$rc" -ne 0 && "$timed_out" != true ]]; then
    cleanup_matching_ranks
fi

echo ""
echo "=== TBE cache warmup done ==="
echo "exit=$rc"
echo "kernel_meta_after=$(du -sh kernel_meta 2>/dev/null | awk '{print $1}' || echo missing)"
echo "logs=results/logs/npu_${SUFFIX}_r*.log"
echo "wrapper_log=$WRAPPER_LOG"
if [[ "$rc" -eq 0 ]]; then
    echo "Warmup completed. If kernel_meta grew, reuse this worktree/cache for cross-host P3."
else
    echo "Warmup failed or timed out. Inspect the rank logs above and results/logs/npu_${SUFFIX}_r*.log."
fi

exit "$rc"
