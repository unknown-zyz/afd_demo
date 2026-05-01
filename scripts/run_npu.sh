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
#
# Usage:
#   ./scripts/run_npu.sh --attn-size 1 --ffn-size 1 --ffn-tp-size 1 [--tokens N] [other run_single flags]
#   ./scripts/run_npu.sh ... --attn-layer-devices 4 --ffn-layer-devices 12 --layer-shard-policy contiguous
#   Add --no-timing for profiling-overhead runs.
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
TIMING_ARGS=(--timing)
ATTN_LAYER_DEVICES="${AFD_ATTENTION_LAYER_DEVICES:-}"
FFN_LAYER_DEVICES="${AFD_FFN_LAYER_DEVICES:-}"
LAYER_SHARD_POLICY="${AFD_LAYER_SHARD_POLICY:-legacy}"
LAYER_SHARD_POLICY_EXPLICIT=false

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
        --attn-layer-devices) ATTN_LAYER_DEVICES="$2"; shift 2 ;;
        --ffn-layer-devices)  FFN_LAYER_DEVICES="$2";  shift 2 ;;
        --layer-shard-policy) LAYER_SHARD_POLICY="$2"; LAYER_SHARD_POLICY_EXPLICIT=true; shift 2 ;;
        --no-timing)    TIMING_ARGS=(); shift ;;
        *) EXTRA_ARGS+=("$1"); shift ;;
    esac
done

WORLD_SIZE=$((ATTN_SIZE + FFN_SIZE))
if (( FFN_SIZE % FFN_TP_SIZE != 0 )); then
    echo "ERROR: --ffn-size=$FFN_SIZE must be divisible by --ffn-tp-size=$FFN_TP_SIZE" >&2
    exit 1
fi
if [ -n "$ATTN_LAYER_DEVICES" ] && ! [[ "$ATTN_LAYER_DEVICES" =~ ^[0-9]+$ ]]; then
    echo "ERROR: --attn-layer-devices must be a positive integer" >&2
    exit 1
fi
if [ -n "$FFN_LAYER_DEVICES" ] && ! [[ "$FFN_LAYER_DEVICES" =~ ^[0-9]+$ ]]; then
    echo "ERROR: --ffn-layer-devices must be a positive integer" >&2
    exit 1
fi
if [ -n "$ATTN_LAYER_DEVICES" ] && [ "$ATTN_LAYER_DEVICES" -lt 1 ]; then
    echo "ERROR: --attn-layer-devices must be >= 1" >&2
    exit 1
fi
if [ -n "$FFN_LAYER_DEVICES" ] && [ "$FFN_LAYER_DEVICES" -lt 1 ]; then
    echo "ERROR: --ffn-layer-devices must be >= 1" >&2
    exit 1
fi
if [ "$LAYER_SHARD_POLICY_EXPLICIT" = false ] && { [ -n "$ATTN_LAYER_DEVICES" ] || [ -n "$FFN_LAYER_DEVICES" ]; }; then
    LAYER_SHARD_POLICY="contiguous"
fi
case "$LAYER_SHARD_POLICY" in
    legacy|contiguous|round_robin) ;;
    *) echo "ERROR: --layer-shard-policy must be legacy, contiguous, or round_robin" >&2; exit 1 ;;
esac

echo "=== NPU-910C launch ==="
echo "preset=$PRESET  attn_size=$ATTN_SIZE  ffn_size=$FFN_SIZE  ffn_tp_size=$FFN_TP_SIZE"
echo "world_size=$WORLD_SIZE  batch=$BATCH  seq=$SEQ  tokens=$TOKENS"

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
echo "layer_shard_policy=$LAYER_SHARD_POLICY  attn_layer_devices=${ATTN_LAYER_DEVICES:-<all>}  ffn_layer_devices=${FFN_LAYER_DEVICES:-<all>}"
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
PLACEMENT_TAG=""
if [ "$LAYER_SHARD_POLICY" != "legacy" ] || [ -n "$ATTN_LAYER_DEVICES" ] || [ -n "$FFN_LAYER_DEVICES" ]; then
    PLACEMENT_TAG="_layer-a${ATTN_LAYER_DEVICES:-all}-f${FFN_LAYER_DEVICES:-all}-${LAYER_SHARD_POLICY}"
fi
SUFFIX="${MODE_TAG}_npu_b${BATCH}_s${SEQ}_t${TOKENS}${PLACEMENT_TAG}"

# Source python venv if present
if [ -f venv/bin/activate ]; then source venv/bin/activate; fi

# ── Spawn ranks ──────────────────────────────────────────────────
# Attention ranks: 0..ATTN_SIZE-1
# FFN ranks: ATTN_SIZE..WORLD_SIZE-1
PIDS=()
for (( R=0; R<WORLD_SIZE; R++ )); do
    if (( R < ATTN_SIZE )); then ROLE=attention; RANK_DEVS="$ATTN_DEVICES"
    else ROLE=ffn;                               RANK_DEVS="$FFN_DEVICES"
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
        export AFD_LAYER_SHARD_POLICY="$LAYER_SHARD_POLICY"
        if [ "$ROLE" = "attention" ] && [ -n "$ATTN_LAYER_DEVICES" ]; then
            export AFD_ROLE_LAYER_DEVICES="$ATTN_LAYER_DEVICES"
        elif [ "$ROLE" = "ffn" ] && [ -n "$FFN_LAYER_DEVICES" ]; then
            export AFD_ROLE_LAYER_DEVICES="$FFN_LAYER_DEVICES"
        else
            unset AFD_ROLE_LAYER_DEVICES
        fi
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

echo "Exit=$rc; logs in results/logs/npu_${SUFFIX}_r*.log"
exit $rc
