#!/bin/bash
# NPU-910C experiment matrix.
#
# Mirrors scripts/run_experiment_matrix.sh but:
#   - Calls scripts/run_npu.sh (HCCL, default 2-rank topology: attn=1/ffn=1/ffn_tp=1)
#   - Optionally passes an EP preset such as npu-ep7.
#   - Writes all outputs to ${output_root}/{serial,prefill-dbo,decode-dbo,decode-dbo-crosslayer}/
#   - Uses bigger batch ceiling (910C HBM ≈ 62 GB/chip)
#
# Usage:
#   ./scripts/run_experiment_matrix_npu.sh [options]
#
# Options:
#   --modes list    Comma-separated subset of: serial,prefill-dbo,decode-dbo,decode-dbo-crosslayer
#                   (default: all four)
#   --batches list  Comma-separated batch sizes (default: 2,4,8,16,32,64,128,256)
#   --seqs list     Comma-separated seq lens   (default: 128,256,512)
#   --tokens N      max_new_tokens for decode  (default: 20)
#   --visible-devs list  ASCEND_VISIBLE_DEVICES
#                        (default: 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15)
#   --reserved-npus list|N
#                        Reserve physical NPU ids, or reserve N devices from
#                        the tail of --visible-devs, for EPLB replicas.
#   --attn-devs list     Optional per-attention-rank visible devices
#   --ffn-devs list      Optional per-FFN-rank visible devices
#   --preset name        Optional run_npu.sh preset, e.g. npu-ep7
#   --ffn-ep-backend name  EP backend when --preset is an EP preset
#   --ep-expert-policy name Expert ownership policy for EP presets
#   --output-root path   Output root (default: results_npu)
#   --serial-cache-root path  Serial cache root (default: results_npu/serial/cache)
#   --comm-timing-mode enqueue | completion    (default: enqueue)
#   --warmup-p2p / --no-warmup-p2p
#                       Toggle untimed P2P communication warmup before inference
#   --warmup-rounds N   P2P warmup rounds when --warmup-p2p is enabled (default: 5)
#   --prefill-warmup-rounds N
#                       Untimed prefill forward warmup rounds. If omitted, src.main
#                       default applies (NPU: 1, CUDA/CPU: 0). Pass 0 to disable.
#   --routing-backend static | coordinator
#                       Routing backend for src.main (default: static). coordinator
#                       starts a local coordinator for each config and passes
#                       --routing-backend coordinator --coord-addr.
#   --coord-bind host:port
#                       Coordinator bind address. If omitted with coordinator
#                       routing, a per-run localhost port is selected.
#   --coord-mode name   Coordinator mode metadata (default: low_latency)
#   --routing-update-mode oneshot | poll
#                       Safe-point routing update mode for coordinator runs
#                       (default: oneshot)
#   --routing-poll-interval-steps N
#                       Decode-step polling interval for poll mode (default: 16)
#   --routing-rpc-timeout-s SEC
#                       Per-RPC timeout for routing fetch/poll (default: 0.05)
#   --no-timing     Disable detailed timing/report output for overhead checks
#   --no-cache      Force rerun of serial even if cached
#   --append        Append to existing summary instead of replacing it
#   --dry-run       Print commands but don't execute

set -u
cd "$(dirname "$0")/.."

MODES="serial,prefill-dbo,decode-dbo,decode-dbo-crosslayer"
BATCHES="2,4,8,16,32,64,128,256"
SEQS="128,256,512"
TOKENS=20
ATTN_DEVS="${ATTN_DEVS:-}"
FFN_DEVS="${FFN_DEVS:-}"
VISIBLE_DEVS="${VISIBLE_DEVS:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}"
RESERVED_NPUS="${AFD_RESERVED_NPUS:-}"
RUN_PRESET=""
FFN_EP_BACKEND="broadcast_reduce_sync"
EP_EXPERT_POLICY="round_robin"
ROOT_OUT="results_npu"
SERIAL_CACHE_ROOT="${SERIAL_CACHE_ROOT:-results_npu/serial/cache}"
COMM_TIMING_MODE="enqueue"
TIMING_ENABLED=true
NO_CACHE=false
APPEND=false
DRY_RUN=false
CORRECTNESS_TOKENS=0
NUM_MICRO_BATCHES=2
WARMUP_P2P=false
WARMUP_P2P_EXPLICIT=false
WARMUP_ROUNDS=5
PREFILL_WARMUP_ROUNDS=""
PREFILL_WARMUP_EXPLICIT=false
ROUTING_BACKEND="static"
COORD_BIND=""
COORD_MODE="low_latency"
COORD_LOG_DIR=""
NUM_EXPERTS=128
ROUTING_UPDATE_MODE="oneshot"
ROUTING_POLL_INTERVAL_STEPS=16
ROUTING_RPC_TIMEOUT_S=0.05
COORD_STARTUP_SEC=2
CURRENT_COORD_PID=""

cleanup_coord() {
    if [ -n "${CURRENT_COORD_PID:-}" ] && kill -0 "$CURRENT_COORD_PID" 2>/dev/null; then
        kill "$CURRENT_COORD_PID" 2>/dev/null || true
        wait "$CURRENT_COORD_PID" 2>/dev/null || true
    fi
    CURRENT_COORD_PID=""
}
trap cleanup_coord EXIT INT TERM

while [ $# -gt 0 ]; do
    case "$1" in
        --modes) MODES="$2"; shift 2;;
        --batches) BATCHES="$2"; shift 2;;
        --seqs) SEQS="$2"; shift 2;;
        --tokens) TOKENS="$2"; shift 2;;
        --attn-devs) ATTN_DEVS="$2"; shift 2;;
        --ffn-devs) FFN_DEVS="$2"; shift 2;;
        --visible-devs) VISIBLE_DEVS="$2"; shift 2;;
        --reserved-npus) RESERVED_NPUS="$2"; shift 2;;
        --preset) RUN_PRESET="$2"; shift 2;;
        --ffn-ep-backend) FFN_EP_BACKEND="$2"; shift 2;;
        --ep-expert-policy) EP_EXPERT_POLICY="$2"; shift 2;;
        --output-root) ROOT_OUT="$2"; shift 2;;
        --serial-cache-root) SERIAL_CACHE_ROOT="$2"; shift 2;;
        --comm-timing-mode) COMM_TIMING_MODE="$2"; shift 2;;
        --correctness-tokens) CORRECTNESS_TOKENS="$2"; shift 2;;
        --num-micro-batches) NUM_MICRO_BATCHES="$2"; shift 2;;
        --warmup-p2p) WARMUP_P2P=true; WARMUP_P2P_EXPLICIT=true; shift;;
        --no-warmup-p2p) WARMUP_P2P=false; WARMUP_P2P_EXPLICIT=true; shift;;
        --warmup-rounds) WARMUP_ROUNDS="$2"; shift 2;;
        --prefill-warmup-rounds) PREFILL_WARMUP_ROUNDS="$2"; PREFILL_WARMUP_EXPLICIT=true; shift 2;;
        --routing-backend) ROUTING_BACKEND="$2"; shift 2;;
        --coord-bind) COORD_BIND="$2"; shift 2;;
        --coord-mode) COORD_MODE="$2"; shift 2;;
        --coord-log-dir) COORD_LOG_DIR="$2"; shift 2;;
        --num-experts) NUM_EXPERTS="$2"; shift 2;;
        --routing-update-mode) ROUTING_UPDATE_MODE="$2"; shift 2;;
        --routing-poll-interval-steps) ROUTING_POLL_INTERVAL_STEPS="$2"; shift 2;;
        --routing-rpc-timeout-s) ROUTING_RPC_TIMEOUT_S="$2"; shift 2;;
        --coord-startup-sec) COORD_STARTUP_SEC="$2"; shift 2;;
        --no-timing) TIMING_ENABLED=false; shift;;
        --no-cache) NO_CACHE=true; shift;;
        --append) APPEND=true; shift;;
        --dry-run) DRY_RUN=true; shift;;
        -h|--help)
            sed -n '2,54p' "$0"; exit 0;;
        *) echo "Unknown option: $1"; exit 1;;
    esac
done

IFS=',' read -ra MODE_ARR  <<< "$MODES"
IFS=',' read -ra BATCH_ARR <<< "$BATCHES"
IFS=',' read -ra SEQ_ARR   <<< "$SEQS"

if [[ "$COMM_TIMING_MODE" != "enqueue" && "$COMM_TIMING_MODE" != "completion" ]]; then
    echo "ERROR: --comm-timing-mode must be enqueue or completion" >&2
    exit 1
fi
if ! [[ "$WARMUP_ROUNDS" =~ ^[0-9]+$ ]]; then
    echo "ERROR: --warmup-rounds must be a non-negative integer" >&2
    exit 1
fi
if [ -n "$PREFILL_WARMUP_ROUNDS" ] && ! [[ "$PREFILL_WARMUP_ROUNDS" =~ ^[0-9]+$ ]]; then
    echo "ERROR: --prefill-warmup-rounds must be a non-negative integer" >&2
    exit 1
fi
if [[ "$ROUTING_BACKEND" != "static" && "$ROUTING_BACKEND" != "coordinator" ]]; then
    echo "ERROR: --routing-backend must be static or coordinator" >&2
    exit 1
fi
if [[ "$ROUTING_UPDATE_MODE" != "oneshot" && "$ROUTING_UPDATE_MODE" != "poll" ]]; then
    echo "ERROR: --routing-update-mode must be oneshot or poll" >&2
    exit 1
fi
if ! [[ "$ROUTING_POLL_INTERVAL_STEPS" =~ ^[0-9]+$ ]]; then
    echo "ERROR: --routing-poll-interval-steps must be a non-negative integer" >&2
    exit 1
fi
if ! [[ "$NUM_EXPERTS" =~ ^[0-9]+$ ]]; then
    echo "ERROR: --num-experts must be a positive integer" >&2
    exit 1
fi

mkdir -p $ROOT_OUT/serial/cache $ROOT_OUT/prefill-dbo $ROOT_OUT/decode-dbo $ROOT_OUT/decode-dbo-crosslayer
mkdir -p "$SERIAL_CACHE_ROOT"
mkdir -p results/prefill_dbo  # run_npu.sh writes intermediate timing here; we move out
if [ -z "$COORD_LOG_DIR" ]; then
    COORD_LOG_DIR="$ROOT_OUT/coordinator_logs"
fi

: "${MODEL_NAME:=/models/Qwen3-30B-A3B}"

preset_ep_size() {
    case "$1" in
        npu-ep4) echo 4 ;;
        npu-ep7) echo 7 ;;
        npu-ep8) echo 8 ;;
        npu-ep15) echo 15 ;;
        *) echo 1 ;;
    esac
}

EP_SIZE=$(preset_ep_size "$RUN_PRESET")
ACTIVE_WORLD_SIZE=2
if [ "$EP_SIZE" -gt 1 ]; then
    ACTIVE_WORLD_SIZE=$((EP_SIZE + 1))
fi
ATTN_WORLD=$((ACTIVE_WORLD_SIZE - EP_SIZE))
if [ "$EP_SIZE" -le 1 ]; then
    ATTN_WORLD=1
fi
FFN_WORLD=1
if [ "$EP_SIZE" -gt 1 ]; then
    FFN_WORLD=$EP_SIZE
fi

json_field() {
    local path="$1"
    local field="$2"
    if [ ! -f "$path" ]; then
        echo ""
        return 0
    fi
    python3 - "$path" "$field" <<'PY'
import json
import sys
path, field = sys.argv[1], sys.argv[2]
try:
    with open(path) as f:
        value = json.load(f).get(field, "")
except Exception:
    value = ""
if isinstance(value, float):
    print(f"{value:.6f}")
elif value is None:
    print("")
else:
    print(value)
PY
}

choose_coord_bind() {
    if [ -n "$COORD_BIND" ]; then
        echo "$COORD_BIND"
    else
        echo "127.0.0.1:$((50071 + (RANDOM % 1500)))"
    fi
}

WARMUP_TAG=""
if [ "$WARMUP_P2P_EXPLICIT" = true ] || [ "$PREFILL_WARMUP_EXPLICIT" = true ]; then
    WARMUP_P2P_LABEL=0
    if [ "$WARMUP_P2P" = true ]; then
        WARMUP_P2P_LABEL=1
    fi
    PREFILL_WARMUP_LABEL="${PREFILL_WARMUP_ROUNDS:-auto}"
    WARMUP_TAG="_wp2p${WARMUP_P2P_LABEL}_pw${PREFILL_WARMUP_LABEL}"
fi

# Runner wrapper --------------------------------------------------------------
run_one() {
    local mode="$1" batch="$2" seq="$3" tokens="$4"
    local outdir="$5"
    local suffix_extra=""
    if [ "$COMM_TIMING_MODE" = "completion" ]; then
        suffix_extra="_comm-completion"
    fi
    if [ "$TIMING_ENABLED" = false ]; then
        suffix_extra="${suffix_extra}_notiming"
    fi
    suffix_extra="${suffix_extra}${WARMUP_TAG}"
    local mb_tag=""
    if [ "$NUM_MICRO_BATCHES" -ne 2 ]; then
        mb_tag="_mb${NUM_MICRO_BATCHES}"
    fi
    local raw_suffix="${mode}_npu${mb_tag}_b${batch}_s${seq}_t${tokens}"
    if [ "$EP_SIZE" -gt 1 ]; then
        raw_suffix="${mode}_npu_ep${EP_SIZE}_${FFN_EP_BACKEND}${mb_tag}_b${batch}_s${seq}_t${tokens}"
    fi
    local run_suffix="${raw_suffix}${suffix_extra}"
    local extra=""
    case "$mode" in
        serial)                extra="--no-dbo";;
        prefill-dbo)           extra="--no-generate";;
        decode-dbo)            extra="";;
        decode-dbo-crosslayer) extra="--crosslayer";;
    esac
    if [ "$CORRECTNESS_TOKENS" -gt 0 ]; then
        extra="$extra --correctness-check $CORRECTNESS_TOKENS"
    fi
    if [ "$WARMUP_P2P" = true ]; then
        extra="$extra --warmup-p2p --warmup-rounds $WARMUP_ROUNDS"
    fi
    if [ -n "$PREFILL_WARMUP_ROUNDS" ]; then
        extra="$extra --prefill-warmup-rounds $PREFILL_WARMUP_ROUNDS"
    fi
    local reserve_arg=""
    if [ -n "$RESERVED_NPUS" ]; then
        reserve_arg="--reserved-npus $RESERVED_NPUS"
    fi
    local coord_bind_effective=""
    local coord_log=""
    if [ "$ROUTING_BACKEND" = "coordinator" ]; then
        coord_bind_effective="$(choose_coord_bind)"
        coord_log="$COORD_LOG_DIR/coordinator_${run_suffix}.log"
        extra="$extra --routing-backend coordinator --coord-addr $coord_bind_effective"
        extra="$extra --routing-update-mode $ROUTING_UPDATE_MODE"
        extra="$extra --routing-poll-interval-steps $ROUTING_POLL_INTERVAL_STEPS"
        extra="$extra --routing-rpc-timeout-s $ROUTING_RPC_TIMEOUT_S"
    fi

    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "  Running: $run_suffix"
    echo "  Routing: backend=$ROUTING_BACKEND update_mode=$ROUTING_UPDATE_MODE"
    if [ -n "$WARMUP_TAG" ]; then
        echo "  Warmup: p2p=$WARMUP_P2P rounds=$WARMUP_ROUNDS prefill_rounds=${PREFILL_WARMUP_ROUNDS:-auto}"
    fi
    echo "════════════════════════════════════════════════════════════"
    if [ "$DRY_RUN" = true ]; then
        if [ -n "$RUN_PRESET" ]; then
            echo "[dry-run] ASCEND_VISIBLE_DEVICES=$VISIBLE_DEVS MASTER_PORT=<random> bash scripts/run_npu.sh --preset $RUN_PRESET --ffn-ep-backend $FFN_EP_BACKEND --ep-expert-policy $EP_EXPERT_POLICY $reserve_arg --batch $batch --seq $seq --tokens $tokens --model-name $MODEL_NAME --comm-timing-mode $COMM_TIMING_MODE $([ "$TIMING_ENABLED" = false ] && echo --no-timing) $extra"
        else
            echo "[dry-run] ASCEND_VISIBLE_DEVICES=$VISIBLE_DEVS ATTN_DEVICES=$ATTN_DEVS FFN_DEVICES=$FFN_DEVS MASTER_PORT=<random> bash scripts/run_npu.sh --attn-size 1 --ffn-size 1 --ffn-tp-size 1 $reserve_arg --batch $batch --seq $seq --tokens $tokens --model-name $MODEL_NAME --comm-timing-mode $COMM_TIMING_MODE $([ "$TIMING_ENABLED" = false ] && echo --no-timing) $extra"
        fi
        return 0
    fi

    local port=$((29500 + (RANDOM % 2000)))
    if [ "$ROUTING_BACKEND" = "coordinator" ]; then
        mkdir -p "$COORD_LOG_DIR"
        echo "  Coordinator: bind=$coord_bind_effective log=$coord_log"
        bash scripts/launch_coordinator.sh \
            --bind "$coord_bind_effective" \
            --num-experts "$NUM_EXPERTS" \
            --attn-world "$ATTN_WORLD" \
            --ffn-world "$FFN_WORLD" \
            --mode "$COORD_MODE" \
            --log-file "$coord_log" &
        CURRENT_COORD_PID=$!
        sleep "$COORD_STARTUP_SEC"
        if ! kill -0 "$CURRENT_COORD_PID" 2>/dev/null; then
            echo "[FAIL] coordinator exited before run: $coord_log"
            wait "$CURRENT_COORD_PID" 2>/dev/null || true
            CURRENT_COORD_PID=""
            return 1
        fi
    fi
    rm -f "results/prefill_dbo/timing_attention_${raw_suffix}.json" \
          "results/prefill_dbo/timing_ffn_${raw_suffix}.json" \
          "results/prefill_dbo/timing_ffn_coordinator_${raw_suffix}.json" \
          "results/prefill_dbo/timing_ffn_expert_"*"${raw_suffix}.json"
    local timing_flags=(--comm-timing-mode "$COMM_TIMING_MODE")
    if [ "$TIMING_ENABLED" = false ]; then
        timing_flags+=(--no-timing)
    fi
    local run_args=()
    if [ -n "$RUN_PRESET" ]; then
        run_args+=(--preset "$RUN_PRESET" --ffn-ep-backend "$FFN_EP_BACKEND" --ep-expert-policy "$EP_EXPERT_POLICY")
    else
        run_args+=(--attn-size 1 --ffn-size 1 --ffn-tp-size 1)
    fi
    local reserve_flags=()
    if [ -n "$RESERVED_NPUS" ]; then
        reserve_flags+=(--reserved-npus "$RESERVED_NPUS")
    fi
    ASCEND_VISIBLE_DEVICES=$VISIBLE_DEVS ATTN_DEVICES=$ATTN_DEVS FFN_DEVICES=$FFN_DEVS MASTER_PORT=$port bash scripts/run_npu.sh \
        "${run_args[@]}" \
        "${reserve_flags[@]}" \
        --batch "$batch" --seq "$seq" --tokens "$tokens" \
        --num-micro-batches "$NUM_MICRO_BATCHES" \
        --model-name "$MODEL_NAME" \
        "${timing_flags[@]}" \
        $extra
    local rc=$?
    cleanup_coord

    # Inspect logs for OOM
    if grep -q "out of memory\|OutOfMemory\|OOM" results/logs/npu_${raw_suffix}_r*.log 2>/dev/null; then
        echo "[OOM] $run_suffix"
        return 2
    fi
    if [ $rc -ne 0 ]; then
        echo "[FAIL rc=$rc] $run_suffix"
        return $rc
    fi

    if [ "$TIMING_ENABLED" = false ]; then
        echo "[ok] $run_suffix completed without detailed timing"
        return 0
    fi

    # run_npu.sh writes timing to results/prefill_dbo/timing_{attention,ffn*}_${raw_suffix}.json
    local attn_src="results/prefill_dbo/timing_attention_${raw_suffix}.json"
    local ffn_src="results/prefill_dbo/timing_ffn_${raw_suffix}.json"
    if [ "$EP_SIZE" -gt 1 ]; then
        ffn_src="results/prefill_dbo/timing_ffn_coordinator_${raw_suffix}.json"
    fi
    local attn_dst="$outdir/timing_attention_${run_suffix}.json"
    local ffn_dst="$outdir/timing_ffn_${run_suffix}.json"
    if [ "$EP_SIZE" -gt 1 ]; then
        ffn_dst="$outdir/timing_ffn_coordinator_${run_suffix}.json"
    fi

    mkdir -p "$outdir"
    if [ ! -f "$attn_src" ] || [ ! -f "$ffn_src" ]; then
        echo "[FAIL] missing expected timing JSON: $attn_src or $ffn_src"
        return 1
    fi
    mv -f "$attn_src" "$attn_dst"
    mv -f "$ffn_src"  "$ffn_dst"
    if [ "$EP_SIZE" -gt 1 ]; then
        for expert_src in results/prefill_dbo/timing_ffn_expert_*_${raw_suffix}.json; do
            [ -f "$expert_src" ] || continue
            local expert_name
            expert_name=$(basename "$expert_src")
            mv -f "$expert_src" "$outdir/${expert_name/$raw_suffix/$run_suffix}"
        done
    fi

    # Cache serial baselines
    if [ "$mode" = "serial" ] && [ -f "$attn_dst" ]; then
        cp -f "$attn_dst" "$ROOT_OUT/serial/cache/b${batch}_s${seq}_t${tokens}${WARMUP_TAG}.json"
        cp -f "$attn_dst" "$SERIAL_CACHE_ROOT/b${batch}_s${seq}_t${tokens}${WARMUP_TAG}.json"
    fi

    # Generate report (uses serial baseline from cache if available)
    local cache_file="$SERIAL_CACHE_ROOT/b${batch}_s${seq}_t${tokens}${WARMUP_TAG}.json"
    if [ ! -f "$cache_file" ]; then
        cache_file="$ROOT_OUT/serial/cache/b${batch}_s${seq}_t${tokens}${WARMUP_TAG}.json"
    fi
    if [ ! -f "$cache_file" ] && [ -n "$WARMUP_TAG" ]; then
        cache_file="$SERIAL_CACHE_ROOT/b${batch}_s${seq}_t${tokens}.json"
    fi
    if [ ! -f "$cache_file" ]; then
        cache_file="$ROOT_OUT/serial/cache/b${batch}_s${seq}_t${tokens}.json"
    fi
    local cmp_flag=""
    if [ -f "$cache_file" ] && [ "$mode" != "serial" ]; then
        cmp_flag="--serial-baseline $cache_file"
    fi
    if [ -f "$attn_dst" ] && [ -f "$ffn_dst" ]; then
        python3 scripts/gen_experiment_report.py \
            --attn-timing "$attn_dst" --ffn-timing "$ffn_dst" \
            --output "$outdir/report_${run_suffix}.md" \
            --mode "$mode" --batch "$batch" --seq "$seq" --tokens "$tokens" \
            --comm-timing-mode "$COMM_TIMING_MODE" \
            $cmp_flag || true
    fi
    return 0
}

# Main sweep ------------------------------------------------------------------
SUMMARY="$ROOT_OUT/experiment_matrix_summary.csv"
VISIBLE_CHIP_POOL=$(echo "$VISIBLE_DEVS" | tr ',' '\n' | wc -l | tr -d ' ')
if [ "$APPEND" = false ] || [ ! -f "$SUMMARY" ]; then
    echo "mode,batch,seq,tokens,preset,ffn_ep_backend,routing_backend,routing_update_mode,visible_chip_pool,active_world_size,status,tpot_ms,prefill_ms,total_time_ms,routing_table_version,routing_poll_count,routing_poll_ms,report,warmup_p2p,warmup_rounds,prefill_warmup_rounds,warmup_tag" > "$SUMMARY"
fi

for MODE in "${MODE_ARR[@]}"; do
    case "$MODE" in
        serial)                 OUTDIR="$ROOT_OUT/serial";;
        prefill-dbo)            OUTDIR="$ROOT_OUT/prefill-dbo";;
        decode-dbo)             OUTDIR="$ROOT_OUT/decode-dbo";;
        decode-dbo-crosslayer)  OUTDIR="$ROOT_OUT/decode-dbo-crosslayer";;
        *) echo "Unknown mode: $MODE"; exit 1;;
    esac

    for SEQ in "${SEQ_ARR[@]}"; do
        for BATCH in "${BATCH_ARR[@]}"; do
            CACHE="$SERIAL_CACHE_ROOT/b${BATCH}_s${SEQ}_t${TOKENS}${WARMUP_TAG}.json"
            if [ ! -f "$CACHE" ]; then
                CACHE="$ROOT_OUT/serial/cache/b${BATCH}_s${SEQ}_t${TOKENS}${WARMUP_TAG}.json"
            fi
            if [ ! -f "$CACHE" ] && [ -z "$WARMUP_TAG" ]; then
                CACHE="$SERIAL_CACHE_ROOT/b${BATCH}_s${SEQ}_t${TOKENS}.json"
                if [ ! -f "$CACHE" ]; then
                    CACHE="$ROOT_OUT/serial/cache/b${BATCH}_s${SEQ}_t${TOKENS}.json"
                fi
            fi
            if [ "$MODE" = "serial" ] && [ "$NO_CACHE" = false ] && [ -f "$CACHE" ]; then
                echo "[cache-hit] serial b${BATCH}_s${SEQ}_t${TOKENS}  (skipping)"
                CACHE_TPOT=$(json_field "$CACHE" "decode_tpot_ms")
                CACHE_PREFILL=$(json_field "$CACHE" "prefill_ms")
                CACHE_TOTAL=$(json_field "$CACHE" "total_time_ms")
                echo "serial,$BATCH,$SEQ,$TOKENS,$RUN_PRESET,$FFN_EP_BACKEND,$ROUTING_BACKEND,$ROUTING_UPDATE_MODE,$VISIBLE_CHIP_POOL,$ACTIVE_WORLD_SIZE,cached,$CACHE_TPOT,$CACHE_PREFILL,$CACHE_TOTAL,,,,$CACHE,$WARMUP_P2P,$WARMUP_ROUNDS,${PREFILL_WARMUP_ROUNDS:-auto},${WARMUP_TAG#_}" >> "$SUMMARY"
                continue
            fi

            run_one "$MODE" "$BATCH" "$SEQ" "$TOKENS" "$OUTDIR"
            rc=$?
            if [ $rc -eq 2 ]; then
                echo "$MODE,$BATCH,$SEQ,$TOKENS,$RUN_PRESET,$FFN_EP_BACKEND,$ROUTING_BACKEND,$ROUTING_UPDATE_MODE,$VISIBLE_CHIP_POOL,$ACTIVE_WORLD_SIZE,OOM,,,,,,,,$WARMUP_P2P,$WARMUP_ROUNDS,${PREFILL_WARMUP_ROUNDS:-auto},${WARMUP_TAG#_}" >> "$SUMMARY"
                echo "↳ OOM reached for $MODE seq=$SEQ; skipping larger batches."
                break
            elif [ $rc -ne 0 ]; then
                echo "$MODE,$BATCH,$SEQ,$TOKENS,$RUN_PRESET,$FFN_EP_BACKEND,$ROUTING_BACKEND,$ROUTING_UPDATE_MODE,$VISIBLE_CHIP_POOL,$ACTIVE_WORLD_SIZE,FAIL,,,,,,,,$WARMUP_P2P,$WARMUP_ROUNDS,${PREFILL_WARMUP_ROUNDS:-auto},${WARMUP_TAG#_}" >> "$SUMMARY"
            else
                SUFFIX_EXTRA=""
                if [ "$COMM_TIMING_MODE" = "completion" ]; then
                    SUFFIX_EXTRA="_comm-completion"
                fi
                if [ "$TIMING_ENABLED" = false ]; then
                    SUFFIX_EXTRA="${SUFFIX_EXTRA}_notiming"
                fi
                SUFFIX_EXTRA="${SUFFIX_EXTRA}${WARMUP_TAG}"
                MB_TAG_S=""
                if [ "$NUM_MICRO_BATCHES" -ne 2 ]; then
                    MB_TAG_S="_mb${NUM_MICRO_BATCHES}"
                fi
                if [ "$EP_SIZE" -gt 1 ]; then
                    REPORT="$OUTDIR/report_${MODE}_npu_ep${EP_SIZE}_${FFN_EP_BACKEND}${MB_TAG_S}_b${BATCH}_s${SEQ}_t${TOKENS}${SUFFIX_EXTRA}.md"
                else
                    REPORT="$OUTDIR/report_${MODE}_npu${MB_TAG_S}_b${BATCH}_s${SEQ}_t${TOKENS}${SUFFIX_EXTRA}.md"
                fi
                if [ "$TIMING_ENABLED" = false ]; then
                    REPORT=""
                fi
                TIMING_JSON=""
                if [ -n "$REPORT" ]; then
                    TIMING_JSON="${REPORT/report_/timing_attention_}"
                    TIMING_JSON="${TIMING_JSON%.md}.json"
                fi
                TPOT_MS=$(json_field "$TIMING_JSON" "decode_tpot_ms")
                PREFILL_MS=$(json_field "$TIMING_JSON" "prefill_ms")
                TOTAL_MS=$(json_field "$TIMING_JSON" "total_time_ms")
                ROUTING_TABLE_VERSION=$(json_field "$TIMING_JSON" "routing_table_version")
                ROUTING_POLL_COUNT=$(json_field "$TIMING_JSON" "routing_poll_count")
                ROUTING_POLL_MS=$(json_field "$TIMING_JSON" "routing_poll_ms")
                echo "$MODE,$BATCH,$SEQ,$TOKENS,$RUN_PRESET,$FFN_EP_BACKEND,$ROUTING_BACKEND,$ROUTING_UPDATE_MODE,$VISIBLE_CHIP_POOL,$ACTIVE_WORLD_SIZE,ok,$TPOT_MS,$PREFILL_MS,$TOTAL_MS,$ROUTING_TABLE_VERSION,$ROUTING_POLL_COUNT,$ROUTING_POLL_MS,$REPORT,$WARMUP_P2P,$WARMUP_ROUNDS,${PREFILL_WARMUP_ROUNDS:-auto},${WARMUP_TAG#_}" >> "$SUMMARY"
            fi
        done
    done
done

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  NPU matrix complete. Summary: $SUMMARY"
echo "════════════════════════════════════════════════════════════"
cat "$SUMMARY"
