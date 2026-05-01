#!/bin/bash
# NPU-910C experiment matrix.
#
# Mirrors scripts/run_experiment_matrix.sh but:
#   - Calls scripts/run_npu.sh (HCCL, 2-rank topology: attn=1/ffn=1/ffn_tp=1)
#   - Writes all outputs to results_npu/{serial,prefill-dbo,decode-dbo,decode-dbo-crosslayer}/
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
#   --attn-devs list     Optional per-attention-rank visible devices
#   --ffn-devs list      Optional per-FFN-rank visible devices
#   --attn-layer-devices N  Use first N local devices for Attention layer placement
#   --ffn-layer-devices N   Use first N local devices for FFN layer placement
#   --layer-shard-policy legacy | contiguous | round_robin
#   --comm-timing-mode enqueue | completion    (default: enqueue)
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
ATTN_LAYER_DEVICES="${AFD_ATTENTION_LAYER_DEVICES:-}"
FFN_LAYER_DEVICES="${AFD_FFN_LAYER_DEVICES:-}"
LAYER_SHARD_POLICY="${AFD_LAYER_SHARD_POLICY:-legacy}"
LAYER_SHARD_POLICY_EXPLICIT=false
COMM_TIMING_MODE="enqueue"
TIMING_ENABLED=true
NO_CACHE=false
APPEND=false
DRY_RUN=false

while [ $# -gt 0 ]; do
    case "$1" in
        --modes) MODES="$2"; shift 2;;
        --batches) BATCHES="$2"; shift 2;;
        --seqs) SEQS="$2"; shift 2;;
        --tokens) TOKENS="$2"; shift 2;;
        --attn-devs) ATTN_DEVS="$2"; shift 2;;
        --ffn-devs) FFN_DEVS="$2"; shift 2;;
        --visible-devs) VISIBLE_DEVS="$2"; shift 2;;
        --attn-layer-devices) ATTN_LAYER_DEVICES="$2"; shift 2;;
        --ffn-layer-devices) FFN_LAYER_DEVICES="$2"; shift 2;;
        --layer-shard-policy) LAYER_SHARD_POLICY="$2"; LAYER_SHARD_POLICY_EXPLICIT=true; shift 2;;
        --comm-timing-mode) COMM_TIMING_MODE="$2"; shift 2;;
        --no-timing) TIMING_ENABLED=false; shift;;
        --no-cache) NO_CACHE=true; shift;;
        --append) APPEND=true; shift;;
        --dry-run) DRY_RUN=true; shift;;
        -h|--help)
            sed -n '2,26p' "$0"; exit 0;;
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

PLACEMENT_TAG=""
if [ "$LAYER_SHARD_POLICY" != "legacy" ] || [ -n "$ATTN_LAYER_DEVICES" ] || [ -n "$FFN_LAYER_DEVICES" ]; then
    PLACEMENT_TAG="_layer-a${ATTN_LAYER_DEVICES:-all}-f${FFN_LAYER_DEVICES:-all}-${LAYER_SHARD_POLICY}"
fi

ROOT_OUT="results_npu"
mkdir -p $ROOT_OUT/serial/cache $ROOT_OUT/prefill-dbo $ROOT_OUT/decode-dbo $ROOT_OUT/decode-dbo-crosslayer
mkdir -p results/prefill_dbo  # run_npu.sh writes intermediate timing here; we move out

: "${MODEL_NAME:=/models/Qwen3-30B-A3B}"

# Runner wrapper --------------------------------------------------------------
run_one() {
    local mode="$1" batch="$2" seq="$3" tokens="$4"
    local outdir="$5"
    local suffix_extra="$PLACEMENT_TAG"
    if [ "$COMM_TIMING_MODE" = "completion" ]; then
        suffix_extra="${suffix_extra}_comm-completion"
    fi
    if [ "$TIMING_ENABLED" = false ]; then
        suffix_extra="${suffix_extra}_notiming"
    fi
    local run_suffix="${mode}_b${batch}_s${seq}_t${tokens}${suffix_extra}"
    local extra=""
    case "$mode" in
        serial)                extra="--no-dbo";;
        prefill-dbo)           extra="--no-generate";;
        decode-dbo)            extra="";;
        decode-dbo-crosslayer) extra="--crosslayer";;
    esac

    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "  Running: $run_suffix"
    echo "════════════════════════════════════════════════════════════"
    if [ "$DRY_RUN" = true ]; then
        echo "[dry-run] ASCEND_VISIBLE_DEVICES=$VISIBLE_DEVS ATTN_DEVICES=$ATTN_DEVS FFN_DEVICES=$FFN_DEVS MASTER_PORT=<random> bash scripts/run_npu.sh --attn-size 1 --ffn-size 1 --ffn-tp-size 1 --batch $batch --seq $seq --tokens $tokens --model-name $MODEL_NAME --comm-timing-mode $COMM_TIMING_MODE --attn-layer-devices ${ATTN_LAYER_DEVICES:-<all>} --ffn-layer-devices ${FFN_LAYER_DEVICES:-<all>} --layer-shard-policy $LAYER_SHARD_POLICY $([ "$TIMING_ENABLED" = false ] && echo --no-timing) $extra"
        return 0
    fi

    local port=$((29500 + (RANDOM % 2000)))
    local raw_suffix="${mode}_npu_b${batch}_s${seq}_t${tokens}${PLACEMENT_TAG}"
    rm -f "results/prefill_dbo/timing_attention_${raw_suffix}.json" \
          "results/prefill_dbo/timing_ffn_${raw_suffix}.json"
    local timing_flags=(--comm-timing-mode "$COMM_TIMING_MODE")
    if [ "$TIMING_ENABLED" = false ]; then
        timing_flags+=(--no-timing)
    fi
    ASCEND_VISIBLE_DEVICES=$VISIBLE_DEVS ATTN_DEVICES=$ATTN_DEVS FFN_DEVICES=$FFN_DEVS MASTER_PORT=$port bash scripts/run_npu.sh \
        --attn-size 1 --ffn-size 1 --ffn-tp-size 1 \
        --batch "$batch" --seq "$seq" --tokens "$tokens" \
        --model-name "$MODEL_NAME" \
        --attn-layer-devices "${ATTN_LAYER_DEVICES:-}" \
        --ffn-layer-devices "${FFN_LAYER_DEVICES:-}" \
        --layer-shard-policy "$LAYER_SHARD_POLICY" \
        "${timing_flags[@]}" \
        $extra
    local rc=$?

    # Inspect logs for OOM
    local r0_log="results/logs/npu_${mode}_npu_b${batch}_s${seq}_t${tokens}${PLACEMENT_TAG}_r0.log"
    local r1_log="results/logs/npu_${mode}_npu_b${batch}_s${seq}_t${tokens}${PLACEMENT_TAG}_r1.log"
    if grep -q "out of memory\|OutOfMemory\|OOM" "$r0_log" "$r1_log" 2>/dev/null; then
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

    # run_npu.sh writes timing to results/prefill_dbo/timing_{attention,ffn}_${mode}_npu_b{B}_s{S}_t{T}.json
    local attn_src="results/prefill_dbo/timing_attention_${raw_suffix}.json"
    local ffn_src="results/prefill_dbo/timing_ffn_${raw_suffix}.json"
    local attn_dst="$outdir/timing_attention_${run_suffix}.json"
    local ffn_dst="$outdir/timing_ffn_${run_suffix}.json"

    mkdir -p "$outdir"
    if [ ! -f "$attn_src" ] || [ ! -f "$ffn_src" ]; then
        echo "[FAIL] missing expected timing JSON: $attn_src or $ffn_src"
        return 1
    fi
    mv -f "$attn_src" "$attn_dst"
    mv -f "$ffn_src"  "$ffn_dst"

    # Cache serial baselines
    if [ "$mode" = "serial" ] && [ -f "$attn_dst" ]; then
        cp -f "$attn_dst" "$ROOT_OUT/serial/cache/b${batch}_s${seq}_t${tokens}.json"
    fi

    # Generate report (uses serial baseline from cache if available)
    local cache_file="$ROOT_OUT/serial/cache/b${batch}_s${seq}_t${tokens}.json"
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
VISIBLE_CHIP_POOL=$(echo "$VISIBLE_DEVS" | tr ',' '\n' | wc -l)
ACTIVE_WORLD_SIZE=2
if [ "$APPEND" = false ] || [ ! -f "$SUMMARY" ]; then
    echo "mode,batch,seq,tokens,visible_chip_pool,active_world_size,status,report" > "$SUMMARY"
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
            CACHE="$ROOT_OUT/serial/cache/b${BATCH}_s${SEQ}_t${TOKENS}.json"
            if [ "$MODE" = "serial" ] && [ "$NO_CACHE" = false ] && [ -f "$CACHE" ]; then
                echo "[cache-hit] serial b${BATCH}_s${SEQ}_t${TOKENS}  (skipping)"
                echo "serial,$BATCH,$SEQ,$TOKENS,$VISIBLE_CHIP_POOL,$ACTIVE_WORLD_SIZE,cached,$CACHE" >> "$SUMMARY"
                continue
            fi

            run_one "$MODE" "$BATCH" "$SEQ" "$TOKENS" "$OUTDIR"
            rc=$?
            if [ $rc -eq 2 ]; then
                echo "$MODE,$BATCH,$SEQ,$TOKENS,$VISIBLE_CHIP_POOL,$ACTIVE_WORLD_SIZE,OOM," >> "$SUMMARY"
                echo "↳ OOM reached for $MODE seq=$SEQ; skipping larger batches."
                break
            elif [ $rc -ne 0 ]; then
                echo "$MODE,$BATCH,$SEQ,$TOKENS,$VISIBLE_CHIP_POOL,$ACTIVE_WORLD_SIZE,FAIL," >> "$SUMMARY"
            else
                SUFFIX_EXTRA="$PLACEMENT_TAG"
                if [ "$COMM_TIMING_MODE" = "completion" ]; then
                    SUFFIX_EXTRA="_comm-completion"
                    SUFFIX_EXTRA="${PLACEMENT_TAG}${SUFFIX_EXTRA}"
                fi
                if [ "$TIMING_ENABLED" = false ]; then
                    SUFFIX_EXTRA="${SUFFIX_EXTRA}_notiming"
                fi
                REPORT="$OUTDIR/report_${MODE}_b${BATCH}_s${SEQ}_t${TOKENS}${SUFFIX_EXTRA}.md"
                if [ "$TIMING_ENABLED" = false ]; then
                    REPORT=""
                fi
                echo "$MODE,$BATCH,$SEQ,$TOKENS,$VISIBLE_CHIP_POOL,$ACTIVE_WORLD_SIZE,ok,$REPORT" >> "$SUMMARY"
            fi
        done
    done
done

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  NPU matrix complete. Summary: $SUMMARY"
echo "════════════════════════════════════════════════════════════"
cat "$SUMMARY"
