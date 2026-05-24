#!/usr/bin/env bash
# Launch a single AttentionWorker rank.
#
# Usage:
#   bash scripts/launch_attn_worker.sh \
#     --coord-addr 127.0.0.1:50051 \
#     --rank 0 --world 2 --ffn-world 2 \
#     --master-addr 127.0.0.1 --master-port 29500 \
#     --device-id 0 [--use-deepep]
set -euo pipefail

COORD_ADDR="127.0.0.1:50051"
RANK=0
WORLD=1
FFN_WORLD=1
MASTER_ADDR="127.0.0.1"
MASTER_PORT=29500
HIDDEN=2048
NUM_EXPERTS=128
TOPK=8
MODE="low_latency"
DEVICE_ID=0
EXTRA_FLAGS=()
LOG_FILE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --coord-addr) COORD_ADDR="$2"; shift 2;;
    --rank) RANK="$2"; shift 2;;
    --world) WORLD="$2"; shift 2;;
    --ffn-world) FFN_WORLD="$2"; shift 2;;
    --master-addr) MASTER_ADDR="$2"; shift 2;;
    --master-port) MASTER_PORT="$2"; shift 2;;
    --hidden-size) HIDDEN="$2"; shift 2;;
    --num-experts) NUM_EXPERTS="$2"; shift 2;;
    --topk) TOPK="$2"; shift 2;;
    --mode) MODE="$2"; shift 2;;
    --device-id) DEVICE_ID="$2"; shift 2;;
    --use-deepep) EXTRA_FLAGS+=("--use-deepep"); shift;;
    --use-fallback) shift;;  # Deprecated: fallback is the default.
    --no-init-dist) EXTRA_FLAGS+=("--no-init-dist"); shift;;
    --log-file) LOG_FILE="$2"; shift 2;;
    *) echo "Unknown arg: $1" >&2; exit 1;;
  esac
done

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

CMD=(python3 -m src.coordinator_arch.workers.attention_worker
     --coord-addr "$COORD_ADDR"
     --rank "$RANK" --world "$WORLD" --ffn-world "$FFN_WORLD"
     --master-addr "$MASTER_ADDR" --master-port "$MASTER_PORT"
     --hidden-size "$HIDDEN" --num-experts "$NUM_EXPERTS" --topk "$TOPK"
     --mode "$MODE" --device-id "$DEVICE_ID"
     "${EXTRA_FLAGS[@]}")

echo "[launch_attn_worker rank=$RANK] ${CMD[*]}"
if [[ -n "$LOG_FILE" ]]; then
  mkdir -p "$(dirname "$LOG_FILE")"
  exec "${CMD[@]}" >"$LOG_FILE" 2>&1
else
  exec "${CMD[@]}"
fi
