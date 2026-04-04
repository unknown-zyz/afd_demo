#!/bin/bash
# v3 全量实验: Decode + Prefill × DBO On/Off, push until OOM
# 使用 run_single.sh 确保 timing 文件命名一致
# 输出: results/experiments_qwen3_v3/{decode,prefill}/
set -eo pipefail
cd "$(dirname "$0")/.."
source venv/bin/activate

# Kill ALL python src.main processes (including from rogue agents)
kill_all_experiments() {
    local pids
    pids=$(ps -eo pid,args | grep "python.*src\.main" | grep -v grep | awk '{print $1}' 2>/dev/null || true)
    if [ -n "$pids" ]; then
        echo "  ⚠ Killing lingering processes: $pids"
        for p in $pids; do
            kill -9 "$p" 2>/dev/null || true
        done
        sleep 3
    fi
}
trap kill_all_experiments EXIT

# Clean old timing files to avoid confusion
echo "[INFO] Cleaning old timing files..."
rm -f results/prefill_dbo/timing_*_local_*.json results/prefill_dbo/timing_*_serial_*.json

V3_DIR="results/experiments_qwen3_v3"
PREFILL_DBO_DIR="results/prefill_dbo"

mkdir -p "$V3_DIR/decode" "$V3_DIR/prefill" "$V3_DIR/logs"

PORT_COUNTER=0

# Run a single experiment via run_single.sh
# Usage: run_one <phase_dir> <batch> <seq> <tokens> <dbo|no-dbo> [--generate]
run_one() {
    PORT_COUNTER=$((PORT_COUNTER + 1))
    local port=$((29800 + PORT_COUNTER))
    local phase_dir="$1" batch="$2" seq="$3" tokens="$4" dbo_flag="$5" gen_flag="${6:-}"
    local dbo_label="dbo"
    local extra=""

    if [ "$dbo_flag" = "no-dbo" ]; then
        extra="--no-dbo"
        dbo_label="serial"
    fi
    if [ -n "$gen_flag" ]; then
        extra="$extra --generate"
    fi

    local suffix="b${batch}_s${seq}_${dbo_label}"
    local logfile="$V3_DIR/logs/${phase_dir}_${suffix}.log"

    echo ""
    echo "──── $phase_dir | batch=$batch seq=$seq tokens=$tokens | $dbo_label (port=$port) ────"

    # Pre-flight: kill any competing processes (from rogue agents etc.)
    kill_all_experiments

    # Run experiment, capture full output to log file (not piped through tee)
    if MASTER_PORT=$port timeout 900 bash scripts/run_single.sh local "$batch" "$seq" \
        --tokens "$tokens" $extra > "$logfile" 2>&1; then

        echo "  ✓ Completed"
        # Show key info from log
        grep -E "Generated|prefill_time|timing saved|Timing saved|tokens/s|tok/s" "$logfile" | tail -3 || true

        # Copy timing files from prefill_dbo/ to v3 dir
        local role_suffix="local_b${batch}_s${seq}_t${tokens}"
        if [ "$dbo_flag" = "no-dbo" ]; then
            role_suffix="serial_local_b${batch}_s${seq}_t${tokens}"
        fi

        for role in attention ffn; do
            local src="$PREFILL_DBO_DIR/timing_${role}_${role_suffix}.json"
            if [ -f "$src" ]; then
                cp "$src" "$V3_DIR/$phase_dir/timing_${role}_${suffix}.json"
            fi
        done

        # Generate pipeline image (DBO runs only)
        local attn_f="$V3_DIR/$phase_dir/timing_attention_${suffix}.json"
        local ffn_f="$V3_DIR/$phase_dir/timing_ffn_${suffix}.json"
        if [ "$dbo_flag" != "no-dbo" ] && [ -f "$attn_f" ] && [ -f "$ffn_f" ]; then
            python scripts/visualize_dbo_pipeline.py \
                --attn-timing "$attn_f" \
                --ffn-timing "$ffn_f" \
                --output "$V3_DIR/$phase_dir/pipeline_${suffix}.png" \
                --start-layer 1 --num-layers 3 2>/dev/null && \
                echo "  ✓ Pipeline image" || echo "  ⚠ Pipeline image failed"
        fi
        sleep 8  # Longer cooldown to avoid port TIME_WAIT
        return 0
    else
        local ec=$?
        echo "  ✗ FAILED (exit=$ec)"
        tail -5 "$logfile" 2>/dev/null || true
        # Check for OOM vs other errors
        if grep -q "OutOfMemory\|CUDA out of memory\|OOM" "$logfile" 2>/dev/null; then
            echo "  → OOM detected"
        fi
        sleep 8
        return 1
    fi
}

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  v3 全量实验: Decode + Prefill × DBO On/Off (push→OOM)  ║"
echo "╚══════════════════════════════════════════════════════════╝"

# ═══════════════════════════════════════════════════════
# DECODE EXPERIMENTS (seq=128, tokens=20, --generate)
# ═══════════════════════════════════════════════════════
echo ""
echo "━━━━━━ DECODE (seq=128, tokens=20) ━━━━━━"

for batch in 2 4 8 16 32 64 128; do
    if ! run_one "decode" "$batch" 128 20 "dbo" "--generate"; then
        echo "  → Decode DBO OOM at batch=$batch"
        break
    fi
    run_one "decode" "$batch" 128 20 "no-dbo" "--generate" || true
done

# ═══════════════════════════════════════════════════════
# PREFILL EXPERIMENTS (tokens=5, no --generate)
# ═══════════════════════════════════════════════════════
echo ""
echo "━━━━━━ PREFILL: batch=4, varying seq ━━━━━━"
for seq in 64 128 256 512 1024 2048 4096; do
    run_one "prefill" 4 "$seq" 5 "dbo" || break
    run_one "prefill" 4 "$seq" 5 "no-dbo" || true
done

echo ""
echo "━━━━━━ PREFILL: seq=128, varying batch ━━━━━━"
for batch in 2 4 8 16 32 64 128; do
    run_one "prefill" "$batch" 128 5 "dbo" || break
    run_one "prefill" "$batch" 128 5 "no-dbo" || true
done

echo ""
echo "━━━━━━ PREFILL: cross configs ━━━━━━"
for config in "8 512" "8 1024" "8 2048" "16 256" "16 512" "16 1024" "32 256" "32 512" "64 128" "64 256"; do
    batch=$(echo $config | cut -d' ' -f1)
    seq=$(echo $config | cut -d' ' -f2)
    run_one "prefill" "$batch" "$seq" 5 "dbo" || continue
    run_one "prefill" "$batch" "$seq" 5 "no-dbo" || true
done

echo ""
echo "════════════════════════════════════════════════"
echo "  v3 实验全部完成!"
echo "════════════════════════════════════════════════"
echo "Decode results:"
ls "$V3_DIR/decode/"*.json 2>/dev/null | wc -l
echo "Prefill results:"
ls "$V3_DIR/prefill/"*.json 2>/dev/null | wc -l
