#!/usr/bin/env bash
# Launch the Coordinator gRPC server.
#
# Usage:
#   bash scripts/launch_coordinator.sh \
#     --bind 0.0.0.0:50051 \
#     --num-experts 128 \
#     --attn-world 16 \
#     --ffn-world 16 \
#     --mode low_latency
set -euo pipefail

BIND="0.0.0.0:50051"
NUM_EXPERTS=128
ATTN_WORLD=16
FFN_WORLD=16
MODE="low_latency"
LOG_LEVEL="INFO"
LOG_FILE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bind) BIND="$2"; shift 2;;
    --num-experts) NUM_EXPERTS="$2"; shift 2;;
    --attn-world) ATTN_WORLD="$2"; shift 2;;
    --ffn-world) FFN_WORLD="$2"; shift 2;;
    --mode) MODE="$2"; shift 2;;
    --log-level) LOG_LEVEL="$2"; shift 2;;
    --log-file) LOG_FILE="$2"; shift 2;;
    *) echo "Unknown arg: $1" >&2; exit 1;;
  esac
done

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

CMD=(python3 -m src.coordinator_arch.coordinator_server
     --bind "$BIND"
     --num-experts "$NUM_EXPERTS"
     --attn-world "$ATTN_WORLD"
     --ffn-world "$FFN_WORLD"
     --mode "$MODE"
     --log-level "$LOG_LEVEL")

echo "[launch_coordinator] ${CMD[*]}"
if [[ -n "$LOG_FILE" ]]; then
  mkdir -p "$(dirname "$LOG_FILE")"
  exec "${CMD[@]}" >"$LOG_FILE" 2>&1
else
  exec "${CMD[@]}"
fi
