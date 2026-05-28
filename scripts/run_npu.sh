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
#   Add --msprof [--msprof-output PATH] to collect CANN/NPU profiles.
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
FFN_EP_SIZE=1
FFN_EP_BACKEND="broadcast_reduce_sync"
EP_EXPERT_POLICY="round_robin"
EP_EXPERT_POLICY_SET=false
TOKENS=5
BATCH=8
SEQ=128
NUM_MICRO_BATCHES=2
RESERVED_NPUS="${AFD_RESERVED_NPUS:-}"
MSPROF=false
MSPROF_OUTPUT="${MSPROF_OUTPUT:-results/msprof}"
MSPROF_OP=false
MSPROF_ANALYZE=false
EXTRA_ARGS=()
TIMING_ARGS=(--timing)

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
        --ep-expert-policy) EP_EXPERT_POLICY="$2"; EP_EXPERT_POLICY_SET=true; shift 2 ;;
        --tokens)       TOKENS="$2"; shift 2 ;;
        --batch)        BATCH="$2";  shift 2 ;;
        --seq)          SEQ="$2";    shift 2 ;;
        --num-micro-batches) NUM_MICRO_BATCHES="$2"; shift 2 ;;
        --reserved-npus) RESERVED_NPUS="$2"; shift 2 ;;
        --msprof)      MSPROF=true; shift ;;
        --msprof-output) MSPROF_OUTPUT="$2"; shift 2 ;;
        --msprof-op)   MSPROF=true; MSPROF_OP=true; shift ;;
        --msprof-analyze) MSPROF_ANALYZE=true; shift ;;
        --no-timing)    TIMING_ARGS=(); shift ;;
        *) EXTRA_ARGS+=("$1"); shift ;;
    esac
done

WORLD_SIZE=$((ATTN_SIZE + FFN_SIZE))
if (( FFN_SIZE % FFN_TP_SIZE != 0 )); then
    echo "ERROR: --ffn-size=$FFN_SIZE must be divisible by --ffn-tp-size=$FFN_TP_SIZE" >&2
    exit 1
fi
if (( FFN_EP_SIZE > 1 && FFN_EP_SIZE != FFN_SIZE )); then
    echo "ERROR: EP MVP requires --ffn-ep-size=$FFN_EP_SIZE to equal --ffn-size=$FFN_SIZE" >&2
    exit 1
fi
if [ "$FFN_EP_BACKEND" = "npu_moe_v2" ] && [ "$EP_EXPERT_POLICY_SET" = false ]; then
    EP_EXPERT_POLICY="contiguous"
fi

echo "=== NPU-910C launch ==="
echo "preset=$PRESET  attn_size=$ATTN_SIZE  ffn_size=$FFN_SIZE  ffn_tp_size=$FFN_TP_SIZE  ffn_ep_size=$FFN_EP_SIZE"
echo "world_size=$WORLD_SIZE  batch=$BATCH  seq=$SEQ  tokens=$TOKENS  num_micro_batches=$NUM_MICRO_BATCHES"

# ── NPU / HCCL environment ───────────────────────────────────────
# Per-rank device visibility: ATTN_DEVICES (for attention ranks), FFN_DEVICES (for ffn ranks).
# This isolates each role's layer-sharding pool, so both ranks don't compete for the
# same physical chips (on shared 910C boxes with limited free HBM per chip).
# For the validated 2-rank NPU topology, split ASCEND_VISIBLE_DEVICES in half by
# default. Other topologies keep the legacy fallback unless ATTN_DEVICES/FFN_DEVICES
# are explicitly provided.
if [ -n "$RESERVED_NPUS" ] && [[ "$RESERVED_NPUS" =~ ^[0-9]+$ ]]; then
    TOTAL_NPUS="${AFD_TOTAL_NPUS:-16}"
    DEFAULT_DEVS=$(seq -s, 0 $((TOTAL_NPUS-1)))
else
    DEFAULT_DEVS=$(seq -s, 0 $((WORLD_SIZE-1)))
fi
export ASCEND_VISIBLE_DEVICES="${ASCEND_VISIBLE_DEVICES:-$DEFAULT_DEVS}"
ATTN_DEVICES="${ATTN_DEVICES:-}"
FFN_DEVICES="${FFN_DEVICES:-}"

join_by_comma() {
    local IFS=,
    echo "$*"
}

contains_item() {
    local needle="$1"; shift
    local item
    for item in "$@"; do
        [ "$item" = "$needle" ] && return 0
    done
    return 1
}

if [ -n "$RESERVED_NPUS" ]; then
    IFS=',' read -ra VISIBLE_FOR_RESERVE <<< "$ASCEND_VISIBLE_DEVICES"
    RESERVED_LIST=()
    if [[ "$RESERVED_NPUS" =~ ^[0-9]+$ ]]; then
        reserve_count="$RESERVED_NPUS"
        if (( reserve_count > ${#VISIBLE_FOR_RESERVE[@]} )); then
            echo "ERROR: --reserved-npus=$reserve_count exceeds visible device count ${#VISIBLE_FOR_RESERVE[@]}" >&2
            exit 1
        fi
        start_idx=$(( ${#VISIBLE_FOR_RESERVE[@]} - reserve_count ))
        RESERVED_LIST=("${VISIBLE_FOR_RESERVE[@]:$start_idx}")
    else
        IFS=',' read -ra RESERVED_LIST <<< "$RESERVED_NPUS"
    fi
    ACTIVE_LIST=()
    for dev in "${VISIBLE_FOR_RESERVE[@]}"; do
        if ! contains_item "$dev" "${RESERVED_LIST[@]}"; then
            ACTIVE_LIST+=("$dev")
        fi
    done
    if (( ${#ACTIVE_LIST[@]} < WORLD_SIZE )); then
        echo "ERROR: active NPU count ${#ACTIVE_LIST[@]} after reservation is less than world_size=$WORLD_SIZE" >&2
        exit 1
    fi
    ASCEND_VISIBLE_DEVICES=$(join_by_comma "${ACTIVE_LIST[@]}")
    export ASCEND_VISIBLE_DEVICES
    export AFD_RESERVED_NPUS
    AFD_RESERVED_NPUS=$(join_by_comma "${RESERVED_LIST[@]}")
    export AFD_ACTIVE_NPUS="$ASCEND_VISIBLE_DEVICES"
fi

if [ -n "$ATTN_DEVICES" ] || [ -n "$FFN_DEVICES" ]; then
    IFS=',' read -ra RESERVED_CHECK <<< "${AFD_RESERVED_NPUS:-}"
    IFS=',' read -ra ATTN_CHECK <<< "$ATTN_DEVICES"
    IFS=',' read -ra FFN_CHECK <<< "$FFN_DEVICES"
    for dev in "${RESERVED_CHECK[@]}"; do
        [ -z "$dev" ] && continue
        if contains_item "$dev" "${ATTN_CHECK[@]}" || contains_item "$dev" "${FFN_CHECK[@]}"; then
            echo "ERROR: reserved NPU $dev is present in ATTN_DEVICES/FFN_DEVICES" >&2
            exit 1
        fi
    done
fi

if [ -z "$ATTN_DEVICES" ] && [ -z "$FFN_DEVICES" ] && [ "$ATTN_SIZE" -eq 1 ] && [ "$FFN_SIZE" -eq 1 ]; then
    IFS=',' read -ra VISIBLE_DEV_ARR <<< "$ASCEND_VISIBLE_DEVICES"
    if [ "${#VISIBLE_DEV_ARR[@]}" -ge 2 ]; then
        split_idx=$(( ${#VISIBLE_DEV_ARR[@]} / 2 ))
        if [ "$split_idx" -lt 1 ]; then split_idx=1; fi
        ATTN_DEVICES=$(IFS=','; echo "${VISIBLE_DEV_ARR[*]:0:$split_idx}")
        FFN_DEVICES=$(IFS=','; echo "${VISIBLE_DEV_ARR[*]:$split_idx}")
    fi
fi
export AFD_ACTIVE_NPUS="${AFD_ACTIVE_NPUS:-$ASCEND_VISIBLE_DEVICES}"
echo "visible_devices=$ASCEND_VISIBLE_DEVICES  reserved_npus=${AFD_RESERVED_NPUS:-<none>}  attn_devices=${ATTN_DEVICES:-<global>}  ffn_devices=${FFN_DEVICES:-<global>}"
if [ "$MSPROF" = true ]; then
    if ! command -v msprof >/dev/null 2>&1; then
        echo "ERROR: --msprof requested but msprof is not on PATH" >&2
        exit 1
    fi
    echo "msprof=enabled  op_mode=$MSPROF_OP  output=$MSPROF_OUTPUT"
fi
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
if (( NUM_MICRO_BATCHES != 2 )); then
    MB_TAG="_mb${NUM_MICRO_BATCHES}"
else
    MB_TAG=""
fi
if (( FFN_EP_SIZE > 1 )); then
    SUFFIX="${MODE_TAG}_npu_ep${FFN_EP_SIZE}_${FFN_EP_BACKEND}${MB_TAG}_b${BATCH}_s${SEQ}_t${TOKENS}"
else
    SUFFIX="${MODE_TAG}_npu${MB_TAG}_b${BATCH}_s${SEQ}_t${TOKENS}"
fi

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
        if (( FFN_EP_SIZE > 1 )); then
            # EP ranks must keep distinct physical LOCAL_RANK values so HCCL can
            # build the FFN EP subgroup topology. The workers force per-rank
            # primary-device loading in EP mode, so all ranks may share the
            # global visible device pool without loading weights on every chip.
            LOCAL_RANK=$R
        fi
        export RANK LOCAL_RANK WORLD_SIZE
        export ATTN_SIZE FFN_SIZE FFN_EP_SIZE
        export FFN_COORDINATOR_RANK="$ATTN_SIZE"
        export FFN_EP_BACKEND EP_EXPERT_POLICY
        export AFD_RESERVED_NPUS="${AFD_RESERVED_NPUS:-}"
        export AFD_ACTIVE_NPUS="${AFD_ACTIVE_NPUS:-$ASCEND_VISIBLE_DEVICES}"
        export MASTER_ADDR MASTER_PORT

        PY_CMD=(python -u -m src.main \
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
            --num-micro-batches "$NUM_MICRO_BATCHES" \
            "${TIMING_ARGS[@]}" \
            --timing-suffix "$SUFFIX" \
            --master-addr "$MASTER_ADDR" \
            --master-port "$MASTER_PORT" \
            "${EXTRA_ARGS[@]}")
        if [ -n "${AFD_RESERVED_NPUS:-}" ]; then
            PY_CMD+=(--reserved-npus "$AFD_RESERVED_NPUS")
        fi
        if [ "$MSPROF" = true ]; then
            RANK_MSPROF_OUTPUT="$MSPROF_OUTPUT/$SUFFIX/rank${RANK}"
            mkdir -p "$RANK_MSPROF_OUTPUT"
            APP_CMD=$(printf " %q" "${PY_CMD[@]}")
            APP_CMD=${APP_CMD:1}
            MSPROF_CMD=(msprof)
            if [ "$MSPROF_OP" = true ]; then
                MSPROF_CMD+=(op)
            else
                MSPROF_CMD+=(--hccl=on --runtime-api=on --task-time=on --sys-hardware-mem=on --sys-io-profiling=on)
            fi
            MSPROF_CMD+=(--output="$RANK_MSPROF_OUTPUT" --application="$APP_CMD")
            msprof_rc=0
            "${MSPROF_CMD[@]}" > "$RUN_LOG" 2>&1 || msprof_rc=$?
            if grep -q "specified program cannot exit normally" "$RUN_LOG"; then
                exit 1
            fi
            exit "$msprof_rc"
        else
            "${PY_CMD[@]}" > "$RUN_LOG" 2>&1
        fi
    ) &
    PIDS+=($!)
done

# Wait all
rc=0
for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then rc=1; fi
done

if [ "$MSPROF" = true ] && [ "$MSPROF_ANALYZE" = true ]; then
    for profile_dir in "$MSPROF_OUTPUT/$SUFFIX"/rank*; do
        [ -d "$profile_dir" ] || continue
        if ! msprof --analyze=on --rule=communication,communication_matrix --output="$profile_dir" > "$profile_dir/msprof-analyze.log" 2>&1; then
            echo "ERROR: msprof analyze failed for $profile_dir" >&2
            rc=1
        fi
    done
fi

echo "Exit=$rc; logs in results/logs/npu_${SUFFIX}_r*.log"
if [ "$MSPROF" = true ]; then
    echo "msprof output: $MSPROF_OUTPUT/$SUFFIX"
fi
exit $rc
