#!/bin/bash
# Launch AFD on Ascend NPU-910C.
#
# Layout presets are legacy scaffolding and remain overrideable via
# --attn-size/--ffn-size/--ffn-tp-size. The validated 910C experiment matrix
# uses the explicit 2-rank topology:
#   --attn-size 1 --ffn-size 1 --ffn-tp-size 1
#
# Legacy presets:
#   npu-4card  : 2 ATT (DP) + 2 FFN (TP=2)     [script default]
#   npu-16card : 4 ATT (DP) + 12 FFN (TP=12)
# EP presets:
#   npu-ep4    : 1 ATT + 4 FFN EP ranks
#   npu-ep7    : 1 ATT + 7 FFN EP ranks
#   npu-ep8    : 1 ATT + 8 FFN EP ranks
#   npu-ep15   : 1 ATT + 15 FFN EP ranks (all 16 cards)
#
# Usage:
#   ./scripts/run_npu.sh --attn-size 1 --ffn-size 1 --ffn-tp-size 1 [--tokens N] [other run_single flags]
#   Add --no-timing for profiling-overhead runs.
#   Add --af-comm-mode controller-cpu for the centralized CPU relay baseline.
#
# This script spawns one torchrun-style process per role on the local node
# using HCCL as the distributed backend. For multi-node, set MASTER_ADDR and
# launch this script on each node with NODE_RANK.

set -eu
cd "$(dirname "$0")/.."

# ── Preset defaults ───────────────────────────────────────────────
PRESET="npu-4card"
ATTN_SIZE=2
FFN_SIZE=2
FFN_TP_SIZE=2
FFN_EP_SIZE=1
FFN_EP_BACKEND="broadcast_reduce_sync"
EP_EXPERT_POLICY="round_robin"
TOKENS=5
BATCH=8
SEQ=128
EXTRA_ARGS=()
TIMING_ARGS=(--timing)
AF_COMM_MODE="direct-hccl"
CONTROLLER_HOST="127.0.0.1"
CONTROLLER_PORT=""
CONTROLLER_PID=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --preset)
            PRESET="$2"
            case "$PRESET" in
                npu-4card)  ATTN_SIZE=2; FFN_SIZE=2;  FFN_TP_SIZE=2;  FFN_EP_SIZE=1 ;;
                npu-16card) ATTN_SIZE=4; FFN_SIZE=12; FFN_TP_SIZE=12; FFN_EP_SIZE=1 ;;
                npu-ep4)    ATTN_SIZE=1; FFN_SIZE=4;  FFN_TP_SIZE=1;  FFN_EP_SIZE=4 ;;
                npu-ep7)    ATTN_SIZE=1; FFN_SIZE=7;  FFN_TP_SIZE=1;  FFN_EP_SIZE=7 ;;
                npu-ep8)    ATTN_SIZE=1; FFN_SIZE=8;  FFN_TP_SIZE=1;  FFN_EP_SIZE=8 ;;
                npu-ep15)   ATTN_SIZE=1; FFN_SIZE=15; FFN_TP_SIZE=1;  FFN_EP_SIZE=15 ;;
                *) echo "Unknown preset: $PRESET" >&2; exit 1 ;;
            esac
            shift 2 ;;
        --attn-size)    ATTN_SIZE="$2"; shift 2 ;;
        --ffn-size)     FFN_SIZE="$2";  shift 2 ;;
        --ffn-tp-size)  FFN_TP_SIZE="$2"; shift 2 ;;
        --ffn-ep-size)  FFN_EP_SIZE="$2"; shift 2 ;;
        --ffn-ep-backend) FFN_EP_BACKEND="$2"; shift 2 ;;
        --ep-expert-policy) EP_EXPERT_POLICY="$2"; shift 2 ;;
        --tokens)       TOKENS="$2"; shift 2 ;;
        --batch)        BATCH="$2";  shift 2 ;;
        --seq)          SEQ="$2";    shift 2 ;;
        --no-timing)    TIMING_ARGS=(); shift ;;
        --af-comm-mode) AF_COMM_MODE="$2"; EXTRA_ARGS+=("$1" "$2"); shift 2 ;;
        --controller-host) CONTROLLER_HOST="$2"; EXTRA_ARGS+=("$1" "$2"); shift 2 ;;
        --controller-port) CONTROLLER_PORT="$2"; EXTRA_ARGS+=("$1" "$2"); shift 2 ;;
        *) EXTRA_ARGS+=("$1"); shift ;;
    esac
done

if [ -z "$CONTROLLER_PORT" ]; then
    CONTROLLER_PORT=$((40100 + (RANDOM % 2000)))
    if [ "$AF_COMM_MODE" = "controller-cpu" ]; then
        EXTRA_ARGS+=(--controller-port "$CONTROLLER_PORT")
    fi
fi

WORLD_SIZE=$((ATTN_SIZE + FFN_SIZE))
if (( FFN_SIZE % FFN_TP_SIZE != 0 )); then
    echo "ERROR: --ffn-size=$FFN_SIZE must be divisible by --ffn-tp-size=$FFN_TP_SIZE" >&2
    exit 1
fi
if (( FFN_EP_SIZE > 1 && FFN_EP_SIZE != FFN_SIZE )); then
    echo "ERROR: EP MVP requires --ffn-ep-size=$FFN_EP_SIZE to equal --ffn-size=$FFN_SIZE" >&2
    exit 1
fi

echo "=== NPU-910C launch ==="
echo "preset=$PRESET  attn_size=$ATTN_SIZE  ffn_size=$FFN_SIZE  ffn_tp_size=$FFN_TP_SIZE  ffn_ep_size=$FFN_EP_SIZE"
echo "world_size=$WORLD_SIZE  batch=$BATCH  seq=$SEQ  tokens=$TOKENS"
echo "af_comm_mode=$AF_COMM_MODE controller=${CONTROLLER_HOST}:${CONTROLLER_PORT}"

# ── NPU / HCCL environment ───────────────────────────────────────
# Per-rank device visibility: ATTN_DEVICES (for attention ranks), FFN_DEVICES (for ffn ranks).
# This isolates each role's layer-sharding pool, so both ranks don't compete for the
# same physical chips (on shared 910C boxes with limited free HBM per chip).
# For the validated 2-rank NPU topology, split ASCEND_VISIBLE_DEVICES in half by
# default. Other topologies keep the legacy fallback unless ATTN_DEVICES/FFN_DEVICES
# are explicitly provided.
DEFAULT_DEVS=$(seq -s, 0 $((WORLD_SIZE-1)))
export ASCEND_VISIBLE_DEVICES="${ASCEND_VISIBLE_DEVICES:-$DEFAULT_DEVS}"
ATTN_DEVICES="${ATTN_DEVICES:-}"
FFN_DEVICES="${FFN_DEVICES:-}"
if [ -z "$ATTN_DEVICES" ] && [ -z "$FFN_DEVICES" ] && [ "$ATTN_SIZE" -eq 1 ] && [ "$FFN_SIZE" -eq 1 ]; then
    IFS=',' read -ra VISIBLE_DEV_ARR <<< "$ASCEND_VISIBLE_DEVICES"
    if [ "${#VISIBLE_DEV_ARR[@]}" -ge 2 ]; then
        split_idx=$(( ${#VISIBLE_DEV_ARR[@]} / 2 ))
        if [ "$split_idx" -lt 1 ]; then split_idx=1; fi
        ATTN_DEVICES=$(IFS=','; echo "${VISIBLE_DEV_ARR[*]:0:$split_idx}")
        FFN_DEVICES=$(IFS=','; echo "${VISIBLE_DEV_ARR[*]:$split_idx}")
    fi
fi
echo "visible_devices=$ASCEND_VISIBLE_DEVICES  attn_devices=${ATTN_DEVICES:-<global>}  ffn_devices=${FFN_DEVICES:-<global>}"
export HCCL_BUFFSIZE="${HCCL_BUFFSIZE:-200}"           # MB
export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-600}"
export HCCL_EXEC_TIMEOUT="${HCCL_EXEC_TIMEOUT:-1800}"
# Disable NCCL envs that don't apply on HCCL
unset NCCL_BUFFSIZE NCCL_NCHANNELS_PER_NET_PEER 2>/dev/null || true

MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29650}"

# Per-role timing suffix (readable output naming). Keep the mode in the
# suffix so matrix retries cannot accidentally pick up stale JSON from a
# different run mode with the same batch/seq/tokens.
HAS_NO_DBO=false
HAS_NO_GENERATE=false
HAS_CROSSLAYER=false
for arg in "${EXTRA_ARGS[@]}"; do
    case "$arg" in
        --no-dbo) HAS_NO_DBO=true ;;
        --no-generate) HAS_NO_GENERATE=true ;;
        --crosslayer) HAS_CROSSLAYER=true ;;
    esac
done

if [ "$HAS_NO_DBO" = true ] && [ "$HAS_NO_GENERATE" = true ]; then
    MODE_TAG="serial-prefill"
elif [ "$HAS_NO_DBO" = true ]; then
    MODE_TAG="serial"
elif [ "$HAS_NO_GENERATE" = true ]; then
    MODE_TAG="prefill-dbo"
elif [ "$HAS_CROSSLAYER" = true ]; then
    MODE_TAG="decode-dbo-crosslayer"
else
    MODE_TAG="decode-dbo"
fi
if (( FFN_EP_SIZE > 1 )); then
    SUFFIX="${MODE_TAG}_npu_ep${FFN_EP_SIZE}_${FFN_EP_BACKEND}_b${BATCH}_s${SEQ}_t${TOKENS}"
else
    SUFFIX="${MODE_TAG}_npu_b${BATCH}_s${SEQ}_t${TOKENS}"
fi

# Source python venv if present
if [ -f venv/bin/activate ]; then source venv/bin/activate; fi

if [ "$AF_COMM_MODE" = "controller-cpu" ]; then
    mkdir -p results/logs results/controller_baseline
    CONTROLLER_LOG="results/logs/controller_${SUFFIX}.log"
    CONTROLLER_CSV="results/controller_baseline/relay_${SUFFIX}.csv"
    python -u scripts/run_controller.py \
        --host "$CONTROLLER_HOST" \
        --port "$CONTROLLER_PORT" \
        --output "$CONTROLLER_CSV" \
        > "$CONTROLLER_LOG" 2>&1 &
    CONTROLLER_PID=$!
    echo "controller_pid=$CONTROLLER_PID log=$CONTROLLER_LOG csv=$CONTROLLER_CSV"
    sleep 1
fi

# ── Spawn ranks ──────────────────────────────────────────────────
# Attention ranks: 0..ATTN_SIZE-1
# FFN ranks: ATTN_SIZE..WORLD_SIZE-1
PIDS=()
for (( R=0; R<WORLD_SIZE; R++ )); do
    if (( R < ATTN_SIZE )); then ROLE=attention; RANK_DEVS="$ATTN_DEVICES"
    else ROLE=ffn;                               RANK_DEVS="$FFN_DEVICES"
    fi
    if (( FFN_EP_SIZE > 1 )); then
        RANK_DEVS="$R"
    fi
    LOCAL_RANK=$R
    RANK=$R
    RUN_LOG="results/logs/npu_${SUFFIX}_r${RANK}.log"
    mkdir -p results/logs
    (
        # Per-rank device pool (falls back to global ASCEND_VISIBLE_DEVICES if empty)
        if [ -n "$RANK_DEVS" ]; then
            export ASCEND_VISIBLE_DEVICES="$RANK_DEVS"
            export ASCEND_RT_VISIBLE_DEVICES="$RANK_DEVS"
            LOCAL_RANK=0  # rank sees only its own devs starting at 0
        fi
        RANK=$RANK LOCAL_RANK=$LOCAL_RANK WORLD_SIZE=$WORLD_SIZE \
        ATTN_SIZE=$ATTN_SIZE FFN_SIZE=$FFN_SIZE FFN_EP_SIZE=$FFN_EP_SIZE \
        FFN_COORDINATOR_RANK=$ATTN_SIZE FFN_EP_BACKEND=$FFN_EP_BACKEND EP_EXPERT_POLICY=$EP_EXPERT_POLICY \
        MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT \
        python -u -m src.main \
            --backend npu \
            --role "$ROLE" \
            --world-size "$WORLD_SIZE" \
            --rank "$RANK" \
            --local-rank "$LOCAL_RANK" \
            --attn-node-rank 0 \
            --ffn-node-rank "$ATTN_SIZE" \
            --attn-size "$ATTN_SIZE" \
            --ffn-size "$FFN_SIZE" \
            --ffn-tp-size "$FFN_TP_SIZE" \
            --ffn-ep-size "$FFN_EP_SIZE" \
            --ffn-ep-backend "$FFN_EP_BACKEND" \
            --ffn-coordinator-rank "$ATTN_SIZE" \
            --ep-expert-policy "$EP_EXPERT_POLICY" \
            --batch-size "$BATCH" \
            --prefill-seq-len "$SEQ" \
            --max-new-tokens "$TOKENS" \
            "${TIMING_ARGS[@]}" \
            --timing-suffix "$SUFFIX" \
            --master-addr "$MASTER_ADDR" \
            --master-port "$MASTER_PORT" \
            "${EXTRA_ARGS[@]}" \
            > "$RUN_LOG" 2>&1
    ) &
    PIDS+=($!)
done

# Wait all
rc=0
for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then rc=1; fi
done

if [ -n "$CONTROLLER_PID" ]; then
    if kill -0 "$CONTROLLER_PID" 2>/dev/null; then
        kill "$CONTROLLER_PID" 2>/dev/null || true
    fi
    wait "$CONTROLLER_PID" 2>/dev/null || true
fi

echo "Exit=$rc; logs in results/logs/npu_${SUFFIX}_r*.log"
exit $rc
