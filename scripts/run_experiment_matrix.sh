#!/bin/bash
# Run a full experiment matrix (serial, prefill-DBO, decode-DBO simple,
# decode-DBO crosslayer) across batch sizes and sequence lengths.
#
# Serial runs are cached by config; if the cache file exists and --no-cache
# is NOT set, the serial run is skipped.
#
# Usage:
#   ./scripts/run_experiment_matrix.sh [options]
#
# Options:
#   --modes list    Comma-separated subset of: serial,prefill-dbo,decode-dbo,decode-dbo-crosslayer
#                   (default: all four)
#   --batches list  Comma-separated batch sizes (default: 2,4,8,16,32,64)
#   --seqs list     Comma-separated seq lens   (default: 128,256,512)
#   --tokens N      max_new_tokens for decode  (default: 20)
#   --deployment    local | multinode          (default: local)
#   --no-cache      Force rerun of serial even if cached
#   --dry-run       Print commands but don't execute
#
# Output directory layout (created as needed):
#   results/serial/                 → full timing JSONs + reports (serial baselines)
#   results/serial/cache/           → deduped index keyed by "b<B>_s<S>_t<T>"
#   results/prefill-dbo/            → prefill DBO timings + reports
#   results/decode-dbo/             → decode DBO (crosslayer OFF) timings + reports
#   results/decode-dbo-crosslayer/  → decode DBO (crosslayer ON) timings + reports
#
# Stops probing larger batches for a (mode, seq) pair as soon as OOM is hit.

set -u
cd "$(dirname "$0")/.."

MODES="serial,prefill-dbo,decode-dbo,decode-dbo-crosslayer"
BATCHES="2,4,8,16,32,64"
SEQS="128,256,512"
TOKENS=20
DEPLOYMENT="local"
NO_CACHE=false
DRY_RUN=false

while [ $# -gt 0 ]; do
    case "$1" in
        --modes) MODES="$2"; shift 2;;
        --batches) BATCHES="$2"; shift 2;;
        --seqs) SEQS="$2"; shift 2;;
        --tokens) TOKENS="$2"; shift 2;;
        --deployment) DEPLOYMENT="$2"; shift 2;;
        --no-cache) NO_CACHE=true; shift;;
        --dry-run) DRY_RUN=true; shift;;
        -h|--help)
            sed -n '2,30p' "$0"; exit 0;;
        *) echo "Unknown option: $1"; exit 1;;
    esac
done

IFS=',' read -ra MODE_ARR  <<< "$MODES"
IFS=',' read -ra BATCH_ARR <<< "$BATCHES"
IFS=',' read -ra SEQ_ARR   <<< "$SEQS"

mkdir -p results/serial/cache results/prefill-dbo results/decode-dbo results/decode-dbo-crosslayer

# Runner wrapper --------------------------------------------------------------
# args: mode batch seq tokens extra_flags output_dir report_suffix
# returns: 0 ok | 2 OOM | other = failure
run_one() {
    local mode="$1" batch="$2" seq="$3" tokens="$4"
    local extra="$5" outdir="$6" rsfx="$7"
    local run_suffix="${mode}_b${batch}_s${seq}_t${tokens}"
    local base_flags="--tokens $tokens"
    case "$mode" in
        serial)                  base_flags="$base_flags --no-dbo --generate";;
        prefill-dbo)             base_flags="$base_flags";;  # prefill-only (default: no-generate)
        decode-dbo)              base_flags="$base_flags --generate";;
        decode-dbo-crosslayer)   base_flags="$base_flags --generate --crosslayer";;
    esac
    # Always warmup P2P for stable timing
    base_flags="$base_flags --warmup-p2p --warmup-rounds 5"

    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "  Running: $run_suffix"
    echo "════════════════════════════════════════════════════════════"
    if [ "$DRY_RUN" = true ]; then
        echo "[dry-run] bash scripts/run_single.sh $DEPLOYMENT $batch $seq $base_flags $extra"
        return 0
    fi

    # Unique MASTER_PORT per run to dodge TIME_WAIT between back-to-back runs
    local port=$((29500 + (RANDOM % 2000)))
    MASTER_PORT=$port bash scripts/run_single.sh "$DEPLOYMENT" "$batch" "$seq" $base_flags $extra
    local rc=$?
    if [ $rc -eq 2 ]; then
        echo "[OOM] $run_suffix"
        return 2
    fi
    if [ $rc -ne 0 ]; then
        echo "[FAIL rc=$rc] $run_suffix"
        return $rc
    fi

    # run_single.sh writes to results/prefill_dbo/timing_*_<suffix>.json where
    # <suffix> = warmup_<deployment>_b<B>_s<S>_t<T>[_crosslayer]
    local raw_suffix="warmup_${DEPLOYMENT}_b${batch}_s${seq}_t${tokens}"
    if [ "$mode" = "serial" ]; then
        raw_suffix="warmup_serial_${DEPLOYMENT}_b${batch}_s${seq}_t${tokens}"
    fi
    if [ "$mode" = "decode-dbo-crosslayer" ]; then
        raw_suffix="${raw_suffix}_crosslayer"
    fi
    local attn_src="results/prefill_dbo/timing_attention_${raw_suffix}.json"
    local ffn_src="results/prefill_dbo/timing_ffn_${raw_suffix}.json"
    local attn_dst="$outdir/timing_attention_${run_suffix}.json"
    local ffn_dst="$outdir/timing_ffn_${run_suffix}.json"

    mkdir -p "$outdir"
    [ -f "$attn_src" ] && mv -f "$attn_src" "$attn_dst"
    [ -f "$ffn_src"  ] && mv -f "$ffn_src"  "$ffn_dst"

    # Copy serial attn timing into cache
    if [ "$mode" = "serial" ] && [ -f "$attn_dst" ]; then
        cp -f "$attn_dst" "results/serial/cache/b${batch}_s${seq}_t${tokens}.json"
    fi

    # Generate markdown report (with optional serial baseline comparison)
    local cache_file="results/serial/cache/b${batch}_s${seq}_t${tokens}.json"
    local cmp_flag=""
    if [ -f "$cache_file" ] && [ "$mode" != "serial" ]; then
        cmp_flag="--serial-baseline $cache_file"
    fi
    python scripts/gen_experiment_report.py \
        --attn-timing "$attn_dst" \
        --ffn-timing "$ffn_dst" \
        --output "$outdir/report_${run_suffix}.md" \
        --mode "$mode" --batch "$batch" --seq "$seq" --tokens "$tokens" \
        $cmp_flag || true
    return 0
}

# Main sweep ------------------------------------------------------------------
SUMMARY="results/experiment_matrix_summary.csv"
echo "mode,batch,seq,tokens,status,report" > "$SUMMARY"

for MODE in "${MODE_ARR[@]}"; do
    case "$MODE" in
        serial)                 OUTDIR="results/serial";;
        prefill-dbo)            OUTDIR="results/prefill-dbo";;
        decode-dbo)             OUTDIR="results/decode-dbo";;
        decode-dbo-crosslayer)  OUTDIR="results/decode-dbo-crosslayer";;
        *) echo "Unknown mode: $MODE"; exit 1;;
    esac

    for SEQ in "${SEQ_ARR[@]}"; do
        for BATCH in "${BATCH_ARR[@]}"; do
            # Cache lookup for serial mode
            CACHE="results/serial/cache/b${BATCH}_s${SEQ}_t${TOKENS}.json"
            if [ "$MODE" = "serial" ] && [ "$NO_CACHE" = false ] && [ -f "$CACHE" ]; then
                echo "[cache-hit] serial b${BATCH}_s${SEQ}_t${TOKENS}  (skipping)"
                echo "serial,$BATCH,$SEQ,$TOKENS,cached,$CACHE" >> "$SUMMARY"
                continue
            fi

            run_one "$MODE" "$BATCH" "$SEQ" "$TOKENS" "" "$OUTDIR" ""
            rc=$?
            if [ $rc -eq 2 ]; then
                echo "$MODE,$BATCH,$SEQ,$TOKENS,OOM," >> "$SUMMARY"
                echo "↳ OOM reached for $MODE seq=$SEQ; skipping larger batches."
                break  # stop escalating batch for this (mode, seq)
            elif [ $rc -ne 0 ]; then
                echo "$MODE,$BATCH,$SEQ,$TOKENS,FAIL," >> "$SUMMARY"
            else
                REPORT="$OUTDIR/report_${MODE}_b${BATCH}_s${SEQ}_t${TOKENS}.md"
                echo "$MODE,$BATCH,$SEQ,$TOKENS,ok,$REPORT" >> "$SUMMARY"
            fi
        done
    done
done

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Matrix complete. Summary: $SUMMARY"
echo "════════════════════════════════════════════════════════════"
if command -v column >/dev/null 2>&1; then
    column -s, -t "$SUMMARY" | head -80
else
    cat "$SUMMARY"
fi

# Clean up transient files from run_single.sh that matrix already moved/re-reported
rm -f results/prefill_dbo/report_warmup_*.md 2>/dev/null || true
rm -f results/prefill_dbo/timing_*_warmup_*.json 2>/dev/null || true
