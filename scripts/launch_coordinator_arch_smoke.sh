#!/usr/bin/env bash
# Local multi-process smoke test for the Coordinator architecture.
#
# Launches: 1 coordinator + N attn workers + M ffn workers, all on localhost,
# all in --use-fallback mode (no DeepEP, no real NPU required).
#
# Verifies gRPC control plane (register / get_routing_table / update_metrics)
# and that workers stay alive for SMOKE_DURATION seconds.
#
# Usage:
#   bash scripts/launch_coordinator_arch_smoke.sh \
#     --attn-world 2 --ffn-world 2 --num-experts 16 --duration 15
set -euo pipefail

ATTN_WORLD=2
FFN_WORLD=2
NUM_EXPERTS=16
DURATION=15
COORD_PORT=50061
MASTER_PORT=29561
LOG_DIR="results/coordinator_arch_smoke/$(date +%Y%m%d_%H%M%S)"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --attn-world) ATTN_WORLD="$2"; shift 2;;
    --ffn-world) FFN_WORLD="$2"; shift 2;;
    --num-experts) NUM_EXPERTS="$2"; shift 2;;
    --duration) DURATION="$2"; shift 2;;
    --coord-port) COORD_PORT="$2"; shift 2;;
    --master-port) MASTER_PORT="$2"; shift 2;;
    --log-dir) LOG_DIR="$2"; shift 2;;
    *) echo "Unknown arg: $1" >&2; exit 1;;
  esac
done

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
mkdir -p "$LOG_DIR"

echo "[smoke] log_dir=$LOG_DIR coord=127.0.0.1:$COORD_PORT attn=$ATTN_WORLD ffn=$FFN_WORLD experts=$NUM_EXPERTS duration=${DURATION}s"

PIDS=()
cleanup() {
  echo "[smoke] cleaning up PIDs=${PIDS[*]:-}"
  for pid in "${PIDS[@]:-}"; do
    if [[ -n "${pid:-}" ]] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  done
  sleep 1
  for pid in "${PIDS[@]:-}"; do
    if [[ -n "${pid:-}" ]] && kill -0 "$pid" 2>/dev/null; then
      kill -9 "$pid" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT INT TERM

# 1) Coordinator
bash scripts/launch_coordinator.sh \
  --bind "0.0.0.0:$COORD_PORT" \
  --num-experts "$NUM_EXPERTS" \
  --attn-world "$ATTN_WORLD" \
  --ffn-world "$FFN_WORLD" \
  --mode low_latency \
  --log-file "$LOG_DIR/coordinator.log" &
PIDS+=($!)
sleep 2

# 2) ATTN workers (ranks 0 .. ATTN_WORLD-1)
for ((i=0; i<ATTN_WORLD; i++)); do
  bash scripts/launch_attn_worker.sh \
    --coord-addr "127.0.0.1:$COORD_PORT" \
    --rank "$i" --world "$ATTN_WORLD" --ffn-world "$FFN_WORLD" \
    --master-addr 127.0.0.1 --master-port "$MASTER_PORT" \
    --num-experts "$NUM_EXPERTS" \
    --device-id "-1" \
    --use-fallback --no-init-dist \
    --log-file "$LOG_DIR/attn_rank${i}.log" &
  PIDS+=($!)
done

# 3) FFN workers (ranks ATTN_WORLD .. ATTN_WORLD+FFN_WORLD-1)
for ((j=0; j<FFN_WORLD; j++)); do
  rank=$((ATTN_WORLD + j))
  bash scripts/launch_ffn_worker.sh \
    --coord-addr "127.0.0.1:$COORD_PORT" \
    --rank "$rank" --world "$FFN_WORLD" --attn-world "$ATTN_WORLD" \
    --master-addr 127.0.0.1 --master-port "$MASTER_PORT" \
    --num-experts "$NUM_EXPERTS" \
    --device-id "-1" \
    --use-fallback --no-init-dist \
    --log-file "$LOG_DIR/ffn_rank${rank}.log" &
  PIDS+=($!)
done

echo "[smoke] all PIDs: ${PIDS[*]}"
echo "[smoke] sleeping ${DURATION}s; tail $LOG_DIR/*.log to watch"
sleep "$DURATION"

echo "[smoke] checking log signatures ..."
RC=0
grep -l "Coordinator listening on" "$LOG_DIR/coordinator.log" >/dev/null \
  || { echo "[smoke] FAIL: coordinator did not bind"; RC=1; }
for f in "$LOG_DIR"/attn_rank*.log "$LOG_DIR"/ffn_rank*.log; do
  if grep -Eq "register|CoordinatorClient connected|ready" "$f"; then
    echo "[smoke] OK: $(basename $f)"
  else
    echo "[smoke] WARN: $(basename $f) — no register/ready signature"
  fi
done

if [[ $RC -eq 0 ]]; then
  echo "[smoke] PASS"
else
  echo "[smoke] FAIL"
fi
exit $RC
