#!/usr/bin/env bash
# Run one side of a cross-host 1A7F coordinator smoke.
#
# Start Host2 first with --side host2, then Host1 with --side host1. The script
# intentionally prints stale-process state before and after each run; kill only
# exact PIDs that belong to the current run.

set -euo pipefail

cd "$(dirname "$0")/.."

SIDE=""
MASTER_ADDR="${MASTER_ADDR:-192.168.0.125}"
MASTER_PORT="${MASTER_PORT:-35201}"
HCCL_IF_BASE_PORT="${HCCL_IF_BASE_PORT:-37300}"
HCCL_IF_IP="${HCCL_IF_IP:-}"
COORD_BIND="${COORD_BIND:-0.0.0.0:50091}"
COORD_ADDR="${COORD_ADDR:-127.0.0.1:50091}"
MODEL_NAME="${MODEL_NAME:-/models/Qwen3-30B-A3B}"
BATCH=2
SEQ=128
TOKENS=5
TIMEOUT_SEC=1200
OUT_DIR="results_npu/coordinator_arch/crosshost_1a7f_smoke"
TIMING_SUFFIX=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --side) SIDE="$2"; shift 2 ;;
    --master-addr) MASTER_ADDR="$2"; shift 2 ;;
    --master-port) MASTER_PORT="$2"; shift 2 ;;
    --hccl-if-base-port) HCCL_IF_BASE_PORT="$2"; shift 2 ;;
    --hccl-if-ip) HCCL_IF_IP="$2"; shift 2 ;;
    --coord-bind) COORD_BIND="$2"; shift 2 ;;
    --coord-addr) COORD_ADDR="$2"; shift 2 ;;
    --model-name) MODEL_NAME="$2"; shift 2 ;;
    --batch) BATCH="$2"; shift 2 ;;
    --seq) SEQ="$2"; shift 2 ;;
    --tokens) TOKENS="$2"; shift 2 ;;
    --timeout-sec) TIMEOUT_SEC="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --timing-suffix) TIMING_SUFFIX="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [[ "$SIDE" != "host1" && "$SIDE" != "host2" ]]; then
  echo "ERROR: --side must be host1 or host2" >&2
  exit 2
fi

if [[ -z "$TIMING_SUFFIX" ]]; then
  TIMING_SUFFIX="xhost_coord_decode_b${BATCH}_s${SEQ}_t${TOKENS}"
fi

mkdir -p "$OUT_DIR" results/prefill_dbo

export MASTER_ADDR MASTER_PORT HCCL_IF_BASE_PORT
export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-3600}"
export HCCL_EXEC_TIMEOUT="${HCCL_EXEC_TIMEOUT:-3600}"
export HCCL_BUFFSIZE="${HCCL_BUFFSIZE:-200}"
export AFD_DIST_TIMEOUT_SEC="${AFD_DIST_TIMEOUT_SEC:-7200}"
if [[ -n "$HCCL_IF_IP" ]]; then
  export HCCL_IF_IP
fi

show_stale() {
  echo "=== stale process check ($SIDE) ==="
  ps -ef | grep -E "src.main|coordinator_server|launch_coordinator" | grep -v grep || true
}

COORD_PID=""
PIDS=()

cleanup() {
  if [[ -n "$COORD_PID" ]] && kill -0 "$COORD_PID" 2>/dev/null; then
    kill "$COORD_PID" 2>/dev/null || true
    wait "$COORD_PID" 2>/dev/null || true
  fi
  for pid in "${PIDS[@]:-}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT INT TERM

echo "=== cross-host coordinator 1A7F smoke ==="
echo "side=$SIDE master=$MASTER_ADDR:$MASTER_PORT hccl_if_base_port=$HCCL_IF_BASE_PORT hccl_if_ip=${HCCL_IF_IP:-<unset>}"
echo "coord_bind=$COORD_BIND coord_addr=$COORD_ADDR batch=$BATCH seq=$SEQ tokens=$TOKENS timeout_sec=$TIMEOUT_SEC"
echo "out_dir=$OUT_DIR timing_suffix=$TIMING_SUFFIX model=$MODEL_NAME"
show_stale

COMMON_ARGS=(
  --backend npu
  --world-size 8
  --attn-node-rank 0
  --ffn-node-rank 1
  --attn-size 1
  --ffn-size 7
  --ffn-tp-size 1
  --ffn-ep-size 7
  --ffn-ep-backend broadcast_reduce_overlap
  --ffn-coordinator-rank 1
  --ep-expert-policy round_robin
  --routing-backend coordinator
  --routing-update-mode oneshot
  --batch-size "$BATCH"
  --prefill-seq-len "$SEQ"
  --max-new-tokens "$TOKENS"
  --num-micro-batches 2
  --timing
  --timing-suffix "$TIMING_SUFFIX"
  --master-addr "$MASTER_ADDR"
  --master-port "$MASTER_PORT"
  --model-name "$MODEL_NAME"
)

if [[ "$SIDE" == "host1" ]]; then
  COORD_LOG="$OUT_DIR/coordinator.log"
  bash scripts/launch_coordinator.sh \
    --bind "$COORD_BIND" \
    --num-experts 128 \
    --attn-world 1 \
    --ffn-world 7 \
    --mode low_latency \
    --log-file "$COORD_LOG" &
  COORD_PID=$!
  echo "coordinator_pid=$COORD_PID log=$COORD_LOG"
  sleep 3
  if ! kill -0 "$COORD_PID" 2>/dev/null; then
    echo "ERROR: coordinator exited before attention launch; log=$COORD_LOG" >&2
    tail -n 80 "$COORD_LOG" || true
    exit 1
  fi

  export ASCEND_VISIBLE_DEVICES="${ASCEND_VISIBLE_DEVICES:-0}"
  export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-$ASCEND_VISIBLE_DEVICES}"
  LOG="$OUT_DIR/h1_rank0.log"
  timeout "$TIMEOUT_SEC" python3 -u -m src.main \
    "${COMMON_ARGS[@]}" \
    --role attention \
    --rank 0 \
    --local-rank 0 \
    --coord-addr "$COORD_ADDR" \
    >"$LOG" 2>&1
  rc=$?
  echo "host1_attention_rc=$rc log=$LOG"
  tail -n 80 "$LOG" || true
  show_stale
  exit "$rc"
fi

rc=0
for rank in 1 2 3 4 5 6 7; do
  lrank=$((rank - 1))
  LOG="$OUT_DIR/h2_rank${rank}.log"
  (
    export ASCEND_VISIBLE_DEVICES="$lrank"
    export ASCEND_RT_VISIBLE_DEVICES="$lrank"
    timeout "$TIMEOUT_SEC" python3 -u -m src.main \
      "${COMMON_ARGS[@]}" \
      --role ffn \
      --rank "$rank" \
      --local-rank 0 \
      --coord-addr "$COORD_ADDR" \
      >"$LOG" 2>&1
  ) &
  PIDS+=("$!")
  echo "rank=$rank pid=${PIDS[-1]} log=$LOG"
done

for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then
    rc=1
  fi
done

echo "host2_ffn_rc=$rc"
tail -n 60 "$OUT_DIR/h2_rank1.log" || true
tail -n 40 "$OUT_DIR/h2_rank7.log" || true
show_stale
exit "$rc"
