#!/usr/bin/env bash
# Run the real src.main Qwen3 path with the coordinator control-plane enabled
# on a single host (currently supported topology: 1 attention + 1 FFN, no EP).

set -euo pipefail

cd "$(dirname "$0")/.."

COORD_BIND="127.0.0.1:50071"
COORD_MODE="low_latency"
NUM_EXPERTS=128
BATCH=2
SEQ=128
TOKENS=5
MODEL_NAME="${MODEL_NAME:-/models/Qwen3-30B-A3B}"
LOG_DIR="results/logs/coordinator_single_host"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --coord-bind) COORD_BIND="$2"; shift 2 ;;
        --coord-mode) COORD_MODE="$2"; shift 2 ;;
        --num-experts) NUM_EXPERTS="$2"; shift 2 ;;
        --batch) BATCH="$2"; shift 2 ;;
        --seq) SEQ="$2"; shift 2 ;;
        --tokens) TOKENS="$2"; shift 2 ;;
        --model-name) MODEL_NAME="$2"; shift 2 ;;
        --log-dir) LOG_DIR="$2"; shift 2 ;;
        *) EXTRA_ARGS+=("$1"); shift ;;
    esac
done

mkdir -p "$LOG_DIR"
COORD_LOG="$LOG_DIR/coordinator_$(date +%Y%m%d_%H%M%S).log"

COORD_PID=""
cleanup() {
    if [[ -n "$COORD_PID" ]] && kill -0 "$COORD_PID" 2>/dev/null; then
        kill "$COORD_PID" 2>/dev/null || true
        wait "$COORD_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

echo "=== Single-host coordinator decode launcher ==="
echo "coord_bind=$COORD_BIND coord_mode=$COORD_MODE num_experts=$NUM_EXPERTS"
echo "batch=$BATCH seq=$SEQ tokens=$TOKENS model=$MODEL_NAME"
echo "coord_log=$COORD_LOG"

bash scripts/launch_coordinator.sh \
    --bind "$COORD_BIND" \
    --num-experts "$NUM_EXPERTS" \
    --attn-world 1 \
    --ffn-world 1 \
    --mode "$COORD_MODE" \
    --log-file "$COORD_LOG" &
COORD_PID=$!

sleep 2

bash scripts/run_npu.sh \
    --attn-size 1 \
    --ffn-size 1 \
    --ffn-tp-size 1 \
    --batch "$BATCH" \
    --seq "$SEQ" \
    --tokens "$TOKENS" \
    --model-name "$MODEL_NAME" \
    --routing-backend coordinator \
    --coord-addr "$COORD_BIND" \
    "${EXTRA_ARGS[@]}"
