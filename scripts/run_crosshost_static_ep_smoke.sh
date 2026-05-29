#!/usr/bin/env bash
# Run one side of a cross-host static EP decode-DBO experiment.
#
# Topology:
#   Host1: rank0 Attention
#   Host2: rank1..EP FFN EP ranks
#
# Start Host2 first, then Host1. The matrix orchestrator does this
# automatically. Manual cleanup must use explicit PIDs only.

set -euo pipefail

cd "$(dirname "$0")/.."

SIDE=""
MASTER_ADDR="${MASTER_ADDR:-192.168.0.125}"
MASTER_PORT="${MASTER_PORT:-35601}"
HCCL_IF_BASE_PORT="${HCCL_IF_BASE_PORT:-42000}"
HCCL_IF_IP="${HCCL_IF_IP:-}"
MODEL_NAME="${MODEL_NAME:-/models/Qwen3-30B-A3B}"
FFN_EP_SIZE=8
FFN_EP_BACKEND="broadcast_reduce_overlap"
EP_EXPERT_POLICY="round_robin"
HOST2_FFN_DEVICES=""
BATCH=32
SEQ=256
TOKENS=20
NUM_MICRO_BATCHES=2
MODE="decode-dbo"
COMM_TIMING_MODE="completion"
TIMEOUT_SEC=3600
OUT_DIR="results_npu/crosshost_static_ep/smoke"
TIMING_SUFFIX=""
DEBUG_MAX_LAYERS=""
RESOURCE_MONITOR=0
RESOURCE_MONITOR_INTERVAL=1
MSPROF=0
MSPROF_OUTPUT_ROOT=""
MSPROF_OP=0
MSPROF_ANALYZE=0
MSPROF_RANKS="all"
MSPROF_STORAGE_LIMIT_MB=""
ATTN_KERNEL="${AFD_ATTN_KERNEL:-hf}"
ATTN_PRECOPY_LAYER_INPUTS=0
ATTN_FUSED_RMSNORM=0
ATTN_FUSED_ROPE=0
ATTN_STREAM_OVERLAP=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --side) SIDE="$2"; shift 2 ;;
    --master-addr) MASTER_ADDR="$2"; shift 2 ;;
    --master-port) MASTER_PORT="$2"; shift 2 ;;
    --hccl-if-base-port) HCCL_IF_BASE_PORT="$2"; shift 2 ;;
    --hccl-if-ip) HCCL_IF_IP="$2"; shift 2 ;;
    --model-name) MODEL_NAME="$2"; shift 2 ;;
    --ffn-ep-size) FFN_EP_SIZE="$2"; shift 2 ;;
    --ffn-size) FFN_EP_SIZE="$2"; shift 2 ;;
    --ffn-ep-backend) FFN_EP_BACKEND="$2"; shift 2 ;;
    --ep-expert-policy) EP_EXPERT_POLICY="$2"; shift 2 ;;
    --host2-ffn-devices) HOST2_FFN_DEVICES="$2"; shift 2 ;;
    --batch) BATCH="$2"; shift 2 ;;
    --seq) SEQ="$2"; shift 2 ;;
    --tokens) TOKENS="$2"; shift 2 ;;
    --num-micro-batches) NUM_MICRO_BATCHES="$2"; shift 2 ;;
    --mode) MODE="$2"; shift 2 ;;
    --comm-timing-mode) COMM_TIMING_MODE="$2"; shift 2 ;;
    --timeout-sec) TIMEOUT_SEC="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --timing-suffix) TIMING_SUFFIX="$2"; shift 2 ;;
    --debug-max-layers) DEBUG_MAX_LAYERS="$2"; shift 2 ;;
    --resource-monitor) RESOURCE_MONITOR=1; shift ;;
    --resource-monitor-interval) RESOURCE_MONITOR_INTERVAL="$2"; shift 2 ;;
    --msprof) MSPROF=1; shift ;;
    --msprof-output-root) MSPROF_OUTPUT_ROOT="$2"; shift 2 ;;
    --msprof-op) MSPROF=1; MSPROF_OP=1; shift ;;
    --msprof-analyze) MSPROF_ANALYZE=1; shift ;;
    --msprof-ranks) MSPROF_RANKS="$2"; shift 2 ;;
    --msprof-storage-limit-mb) MSPROF_STORAGE_LIMIT_MB="$2"; shift 2 ;;
    --attn-kernel) ATTN_KERNEL="$2"; shift 2 ;;
    --attn-precopy-layer-inputs) ATTN_PRECOPY_LAYER_INPUTS=1; shift ;;
    --attn-fused-rmsnorm) ATTN_FUSED_RMSNORM=1; shift ;;
    --attn-fused-rope) ATTN_FUSED_ROPE=1; shift ;;
    --attn-stream-overlap) ATTN_STREAM_OVERLAP=1; shift ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [[ "$SIDE" != "host1" && "$SIDE" != "host2" ]]; then
  echo "ERROR: --side must be host1 or host2" >&2
  exit 2
fi
if ! [[ "$FFN_EP_SIZE" =~ ^[0-9]+$ ]] || (( FFN_EP_SIZE < 1 )); then
  echo "ERROR: --ffn-ep-size must be a positive integer" >&2
  exit 2
fi
if [[ "$MODE" != "serial" && "$MODE" != "decode-dbo" && "$MODE" != "decode-dbo-crosslayer" ]]; then
  echo "ERROR: --mode must be serial, decode-dbo, or decode-dbo-crosslayer" >&2
  exit 2
fi
if [[ -z "$TIMING_SUFFIX" ]]; then
  mb_tag=""
  if (( NUM_MICRO_BATCHES != 2 )); then
    mb_tag="_mb${NUM_MICRO_BATCHES}"
  fi
  TIMING_SUFFIX="xhost_static_${MODE}_ep${FFN_EP_SIZE}_${FFN_EP_BACKEND}${mb_tag}_b${BATCH}_s${SEQ}_t${TOKENS}"
fi
if [[ -z "$HOST2_FFN_DEVICES" ]]; then
  HOST2_FFN_DEVICES="$(seq -s, 0 $((FFN_EP_SIZE - 1)))"
fi
if [[ -z "$MSPROF_OUTPUT_ROOT" ]]; then
  MSPROF_OUTPUT_ROOT="$OUT_DIR/msprof"
fi

mkdir -p "$OUT_DIR" results/prefill_dbo results/logs

source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || true
# Some vendor OPP set_env scripts call exit instead of return on certain hosts.
# Do not source them in-process; the base Ascend toolkit env is sufficient here.

export MASTER_ADDR MASTER_PORT HCCL_IF_BASE_PORT
export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-600}"
export HCCL_EXEC_TIMEOUT="${HCCL_EXEC_TIMEOUT:-600}"
export HCCL_BUFFSIZE="${HCCL_BUFFSIZE:-200}"
export AFD_DIST_TIMEOUT_SEC="${AFD_DIST_TIMEOUT_SEC:-7200}"
if [[ -n "$HCCL_IF_IP" ]]; then
  export HCCL_IF_IP
fi
if [[ -n "$DEBUG_MAX_LAYERS" ]]; then
  export AFD_DEBUG_MAX_LAYERS="$DEBUG_MAX_LAYERS"
fi

show_stale() {
  echo "=== stale process check ($SIDE) ==="
  ps -ef | grep -E "src[.]main|run_crosshost_static_[e]p" || true
}

copy_timing_files() {
  shopt -s nullglob
  for path in results/prefill_dbo/timing_*_"${TIMING_SUFFIX}".json; do
    cp -f "$path" "$OUT_DIR/"
  done
  shopt -u nullglob
}

PIDS=()
RANK_PIDS=()
cleanup() {
  for pid in "${PIDS[@]:-}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT INT TERM

start_resource_monitor() {
  if (( ! RESOURCE_MONITOR )); then
    return 0
  fi
  local monitor_devices="${ASCEND_VISIBLE_DEVICES:-0}"
  if [[ "$SIDE" == "host2" ]]; then
    monitor_devices="$HOST2_FFN_DEVICES"
  fi
  sample_usages() {
    local dev card chip
    IFS=',' read -ra _monitor_devs <<< "$monitor_devices"
    for dev in "${_monitor_devs[@]}"; do
      [[ -n "$dev" ]] || continue
      card=$((dev / 2))
      chip=$((dev % 2))
      echo "--- npu=$dev card=$card chip=$chip ---"
      npu-smi info -t usages -i "$card" -c "$chip" 2>&1 || true
    done
  }
  {
    echo "resource_monitor_start side=$SIDE ts=$(date -Is)"
    sample_usages
  } >"$OUT_DIR/npu_smi_${SIDE}_start.log"
  (
    while true; do
      echo "=== ts=$(date -Is) side=$SIDE ==="
      sample_usages
      sleep "$RESOURCE_MONITOR_INTERVAL"
    done
  ) >"$OUT_DIR/npu_smi_${SIDE}.log" 2>&1 &
  PIDS+=("$!")
  echo "resource_monitor_pid=${PIDS[-1]} log=$OUT_DIR/npu_smi_${SIDE}.log"
}

run_rank() {
  local rank_label="$1"
  local rank_num="$2"
  local log_path="$3"
  shift 3
  local py_cmd=(python3 -u -m src.main "$@")
  local profile_this_rank=0
  if (( MSPROF )); then
    if [[ "$MSPROF_RANKS" == "all" || ",$MSPROF_RANKS," == *",$rank_num,"* ]]; then
      profile_this_rank=1
    fi
  fi
  if (( profile_this_rank )); then
    if ! command -v msprof >/dev/null 2>&1; then
      echo "ERROR: --msprof requested but msprof is not on PATH" >&2
      return 127
    fi
    local rank_msprof_output="$MSPROF_OUTPUT_ROOT/$TIMING_SUFFIX/${SIDE}_${rank_label}"
    mkdir -p "$rank_msprof_output"
    local app_cmd
    app_cmd=$(printf " %q" "${py_cmd[@]}")
    app_cmd=${app_cmd:1}
    local msprof_cmd=(msprof)
    if (( MSPROF_OP )); then
      msprof_cmd+=(op)
    else
      msprof_cmd+=(--hccl=on --runtime-api=on --task-time=on --sys-hardware-mem=on --sys-io-profiling=on)
    fi
    if [[ -n "$MSPROF_STORAGE_LIMIT_MB" ]]; then
      msprof_cmd+=(--storage-limit="${MSPROF_STORAGE_LIMIT_MB}MB")
    fi
    msprof_cmd+=(--output="$rank_msprof_output" --application="$app_cmd")
    timeout "$TIMEOUT_SEC" "${msprof_cmd[@]}" >"$log_path" 2>&1
  else
    timeout "$TIMEOUT_SEC" "${py_cmd[@]}" >"$log_path" 2>&1
  fi
}

analyze_msprof_outputs() {
  if (( ! MSPROF || ! MSPROF_ANALYZE )); then
    return 0
  fi
  local rc=0
  shopt -s nullglob
  for profile_dir in "$MSPROF_OUTPUT_ROOT/$TIMING_SUFFIX"/"${SIDE}"_*; do
    [[ -d "$profile_dir" ]] || continue
    if ! msprof --analyze=on --rule=communication,communication_matrix --output="$profile_dir" >"$profile_dir/msprof-analyze.log" 2>&1; then
      echo "ERROR: msprof analyze failed for $profile_dir" >&2
      rc=1
    fi
  done
  shopt -u nullglob
  return "$rc"
}

WORLD_SIZE=$((FFN_EP_SIZE + 1))
EXTRA_ARGS=()
if [[ "$MODE" == "decode-dbo-crosslayer" ]]; then
  EXTRA_ARGS+=(--crosslayer)
elif [[ "$MODE" == "serial" ]]; then
  EXTRA_ARGS+=(--no-dbo)
fi

COMMON_ARGS=(
  --backend npu
  --world-size "$WORLD_SIZE"
  --attn-node-rank 0
  --ffn-node-rank 1
  --attn-size 1
  --ffn-size "$FFN_EP_SIZE"
  --ffn-tp-size 1
  --ffn-ep-size "$FFN_EP_SIZE"
  --ffn-ep-backend "$FFN_EP_BACKEND"
  --ffn-coordinator-rank 1
  --ep-expert-policy "$EP_EXPERT_POLICY"
  --batch-size "$BATCH"
  --prefill-seq-len "$SEQ"
  --max-new-tokens "$TOKENS"
  --num-micro-batches "$NUM_MICRO_BATCHES"
  --timing
  --timing-suffix "$TIMING_SUFFIX"
  --comm-timing-mode "$COMM_TIMING_MODE"
  --master-addr "$MASTER_ADDR"
  --master-port "$MASTER_PORT"
  --model-name "$MODEL_NAME"
  --attn-kernel "$ATTN_KERNEL"
  "${EXTRA_ARGS[@]}"
)
if (( ATTN_PRECOPY_LAYER_INPUTS )); then
  COMMON_ARGS+=(--attn-precopy-layer-inputs)
fi
if (( ATTN_FUSED_RMSNORM )); then
  COMMON_ARGS+=(--attn-fused-rmsnorm)
fi
if (( ATTN_FUSED_ROPE )); then
  COMMON_ARGS+=(--attn-fused-rope)
fi
if (( ATTN_STREAM_OVERLAP )); then
  COMMON_ARGS+=(--attn-stream-overlap)
fi

echo "=== cross-host static EP smoke ==="
echo "side=$SIDE world=$WORLD_SIZE ep=$FFN_EP_SIZE backend=$FFN_EP_BACKEND mode=$MODE mb=$NUM_MICRO_BATCHES"
echo "master=$MASTER_ADDR:$MASTER_PORT hccl_if_base_port=$HCCL_IF_BASE_PORT hccl_if_ip=${HCCL_IF_IP:-<unset>}"
echo "batch=$BATCH seq=$SEQ tokens=$TOKENS timeout_sec=$TIMEOUT_SEC debug_max_layers=${DEBUG_MAX_LAYERS:-<none>}"
echo "resource_monitor=$RESOURCE_MONITOR interval=$RESOURCE_MONITOR_INTERVAL"
echo "msprof=$MSPROF op=$MSPROF_OP analyze=$MSPROF_ANALYZE ranks=$MSPROF_RANKS storage_limit_mb=${MSPROF_STORAGE_LIMIT_MB:-<none>} output_root=$MSPROF_OUTPUT_ROOT"
echo "attention kernel=$ATTN_KERNEL precopy=$ATTN_PRECOPY_LAYER_INPUTS fused_rmsnorm=$ATTN_FUSED_RMSNORM fused_rope=$ATTN_FUSED_ROPE stream_overlap=$ATTN_STREAM_OVERLAP"
echo "out_dir=$OUT_DIR timing_suffix=$TIMING_SUFFIX model=$MODEL_NAME"
if [[ "$SIDE" == "host2" ]]; then
  echo "host2_ffn_devices=$HOST2_FFN_DEVICES"
fi
show_stale

rm -f results/prefill_dbo/timing_*_"${TIMING_SUFFIX}".json
start_resource_monitor

if [[ "$SIDE" == "host1" ]]; then
  export ASCEND_VISIBLE_DEVICES="${ASCEND_VISIBLE_DEVICES:-0}"
  export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-$ASCEND_VISIBLE_DEVICES}"
  LOG="$OUT_DIR/h1_rank0.log"
  rc=0
  run_rank "rank0" 0 "$LOG" \
    "${COMMON_ARGS[@]}" \
    --role attention \
    --rank 0 \
    --local-rank 0 || rc=$?
  analyze_msprof_outputs || rc=1
  copy_timing_files
  echo "host1_attention_rc=$rc log=$LOG"
  tail -n 80 "$LOG" || true
  show_stale
  exit "$rc"
fi

IFS=',' read -ra FFN_DEVS <<< "$HOST2_FFN_DEVICES"
if (( ${#FFN_DEVS[@]} < FFN_EP_SIZE )); then
  echo "ERROR: --host2-ffn-devices has ${#FFN_DEVS[@]} devices but EP=$FFN_EP_SIZE" >&2
  exit 2
fi

rc=0
for rank in $(seq 1 "$FFN_EP_SIZE"); do
  dev="${FFN_DEVS[$((rank - 1))]}"
  LOG="$OUT_DIR/h2_rank${rank}.log"
  (
    export ASCEND_VISIBLE_DEVICES="$dev"
    export ASCEND_RT_VISIBLE_DEVICES="$dev"
    run_rank "rank${rank}" "$rank" "$LOG" \
      "${COMMON_ARGS[@]}" \
      --role ffn \
      --rank "$rank" \
      --local-rank 0
  ) &
  PIDS+=("$!")
  RANK_PIDS+=("$!")
  echo "rank=$rank pid=${RANK_PIDS[-1]} device=$dev log=$LOG"
done

for pid in "${RANK_PIDS[@]}"; do
  if ! wait "$pid"; then
    rc=1
  fi
done

copy_timing_files
analyze_msprof_outputs || rc=1
echo "host2_ffn_rc=$rc"
tail -n 60 "$OUT_DIR/h2_rank1.log" || true
tail -n 40 "$OUT_DIR/h2_rank${FFN_EP_SIZE}.log" || true
show_stale
exit "$rc"
