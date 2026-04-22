#!/bin/bash
# Launch AFD on Ascend NPU-910C.
#
# Layout presets (overrideable via --attn-size/--ffn-size/--ffn-tp-size):
#   npu-4card  : 2 ATT (DP) + 2 FFN (TP=2)     [default]
#   npu-16card : 4 ATT (DP) + 12 FFN (TP=12)
#
# Usage:
#   ./scripts/run_npu.sh [--preset npu-4card|npu-16card] [--tokens N] [other run_single flags]
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
TOKENS=5
BATCH=8
SEQ=128
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --preset)
            PRESET="$2"
            case "$PRESET" in
                npu-4card)  ATTN_SIZE=2; FFN_SIZE=2;  FFN_TP_SIZE=2 ;;
                npu-16card) ATTN_SIZE=4; FFN_SIZE=12; FFN_TP_SIZE=12 ;;
                *) echo "Unknown preset: $PRESET" >&2; exit 1 ;;
            esac
            shift 2 ;;
        --attn-size)    ATTN_SIZE="$2"; shift 2 ;;
        --ffn-size)     FFN_SIZE="$2";  shift 2 ;;
        --ffn-tp-size)  FFN_TP_SIZE="$2"; shift 2 ;;
        --tokens)       TOKENS="$2"; shift 2 ;;
        --batch)        BATCH="$2";  shift 2 ;;
        --seq)          SEQ="$2";    shift 2 ;;
        *) EXTRA_ARGS+=("$1"); shift ;;
    esac
done

WORLD_SIZE=$((ATTN_SIZE + FFN_SIZE))
if (( FFN_SIZE % FFN_TP_SIZE != 0 )); then
    echo "ERROR: --ffn-size=$FFN_SIZE must be divisible by --ffn-tp-size=$FFN_TP_SIZE" >&2
    exit 1
fi

echo "=== NPU-910C launch ==="
echo "preset=$PRESET  attn_size=$ATTN_SIZE  ffn_size=$FFN_SIZE  ffn_tp_size=$FFN_TP_SIZE"
echo "world_size=$WORLD_SIZE  batch=$BATCH  seq=$SEQ  tokens=$TOKENS"

# ── NPU / HCCL environment ───────────────────────────────────────
# Exposed via `npu-smi info` enumeration. Override ASCEND_VISIBLE_DEVICES
# to restrict cards on shared machines (e.g. "0,1,2,3" for 4-card runs).
export ASCEND_VISIBLE_DEVICES="${ASCEND_VISIBLE_DEVICES:-$(seq -s, 0 $((WORLD_SIZE-1)))}"
export HCCL_BUFFSIZE="${HCCL_BUFFSIZE:-200}"           # MB
export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-600}"
export HCCL_EXEC_TIMEOUT="${HCCL_EXEC_TIMEOUT:-1800}"
# Disable NCCL envs that don't apply on HCCL
unset NCCL_BUFFSIZE NCCL_NCHANNELS_PER_NET_PEER 2>/dev/null || true

MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29650}"

# Per-role timing suffix (readable output naming)
SUFFIX="npu_b${BATCH}_s${SEQ}_t${TOKENS}"

# Source python venv if present
if [ -f venv/bin/activate ]; then source venv/bin/activate; fi

# ── Spawn ranks ──────────────────────────────────────────────────
# Attention ranks: 0..ATTN_SIZE-1
# FFN ranks: ATTN_SIZE..WORLD_SIZE-1
PIDS=()
for (( R=0; R<WORLD_SIZE; R++ )); do
    if (( R < ATTN_SIZE )); then ROLE=attention
    else ROLE=ffn
    fi
    LOCAL_RANK=$R
    RANK=$R
    RUN_LOG="results/logs/npu_${SUFFIX}_r${RANK}.log"
    mkdir -p results/logs
    (
        RANK=$RANK LOCAL_RANK=$LOCAL_RANK WORLD_SIZE=$WORLD_SIZE \
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
            --batch-size "$BATCH" \
            --prefill-seq-len "$SEQ" \
            --max-new-tokens "$TOKENS" \
            --timing \
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

echo "Exit=$rc; logs in results/logs/npu_${SUFFIX}_r*.log"
exit $rc
