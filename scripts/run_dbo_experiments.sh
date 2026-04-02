#!/bin/bash
# DBO 综合实验脚本
# 在单机 4×V100-32GB 上测试 Prefill/Decode 阶段 DBO 开启/关闭的效果
#
# Usage:
#   ./scripts/run_dbo_experiments.sh [phase]
#   phase: prefill-batch | prefill-seq | decode | all (默认 all)
#
# 输出:
#   results/experiments/prefill/  - Prefill timing JSON + pipeline 图
#   results/experiments/decode/   - Decode 日志 + 汇总 CSV
#   results/experiments/summary.csv - 总汇总

set -e
cd "$(dirname "$0")/.."
source venv/bin/activate

PHASE="${1:-all}"
MODEL="Qwen/Qwen2-1.5B"
GPUS="0,1,2,3"
TOKENS_PREFILL=5
TOKENS_DECODE=20
BASE_DIR="results/experiments"
PREFILL_DIR="$BASE_DIR/prefill"
DECODE_DIR="$BASE_DIR/decode"
MASTER_PORT_BASE=29600
PORT_COUNTER=0

mkdir -p "$PREFILL_DIR/logs" "$DECODE_DIR/logs" "$BASE_DIR"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info()  { echo -e "${BLUE}[INFO]${NC} $*"; }
log_ok()    { echo -e "${GREEN}[OK]${NC} $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }
log_phase() { echo -e "\n${GREEN}========================================${NC}"; echo -e "${GREEN}  $*${NC}"; echo -e "${GREEN}========================================${NC}\n"; }

next_port() {
    PORT_COUNTER=$((PORT_COUNTER + 1))
    echo $((MASTER_PORT_BASE + PORT_COUNTER))
}

# ============================================================
# Prefill 实验：使用 --no-generate 模式
# ============================================================
run_prefill_experiment() {
    local BATCH=$1
    local SEQ=$2
    local DBO=$3  # "on" or "off"
    local PORT
    PORT=$(next_port)

    local DBO_FLAG=""
    local SUFFIX
    if [ "$DBO" = "off" ]; then
        DBO_FLAG="--no-dbo"
        SUFFIX="serial_b${BATCH}_s${SEQ}"
    else
        SUFFIX="dbo_b${BATCH}_s${SEQ}"
    fi

    log_info "Prefill: batch=$BATCH seq=$SEQ DBO=$DBO (port=$PORT)"

    local ATTN_LOG="$PREFILL_DIR/logs/attn_${SUFFIX}.log"
    local FFN_LOG="$PREFILL_DIR/logs/ffn_${SUFFIX}.log"

    # FFN 节点（后台）
    CUDA_VISIBLE_DEVICES=2,3 python -m src.main \
        --model-name "$MODEL" \
        --role ffn \
        --master-addr 127.0.0.1 \
        --master-port "$PORT" \
        --world-size 2 --rank 1 --local-rank 0 \
        --attn-node-rank 0 --ffn-node-rank 1 \
        --batch-size "$BATCH" \
        --prefill-seq-len "$SEQ" \
        --max-new-tokens "$TOKENS_PREFILL" \
        --timing --timing-suffix "$SUFFIX" \
        --no-generate $DBO_FLAG \
        > "$FFN_LOG" 2>&1 &
    local FFN_PID=$!
    sleep 3

    # Attention 节点（前台）
    CUDA_VISIBLE_DEVICES=0,1 python -m src.main \
        --model-name "$MODEL" \
        --role attention \
        --master-addr 127.0.0.1 \
        --master-port "$PORT" \
        --world-size 2 --rank 0 --local-rank 0 \
        --attn-node-rank 0 --ffn-node-rank 1 \
        --batch-size "$BATCH" \
        --prefill-seq-len "$SEQ" \
        --max-new-tokens "$TOKENS_PREFILL" \
        --prompt "Hello world, this is a test prompt for scaling experiments." \
        --timing --timing-suffix "$SUFFIX" \
        --no-generate $DBO_FLAG \
        > "$ATTN_LOG" 2>&1

    wait $FFN_PID 2>/dev/null || true

    # 检查 OOM
    if grep -q "OutOfMemoryError\|CUDA out of memory" "$ATTN_LOG" "$FFN_LOG" 2>/dev/null; then
        log_warn "OOM: batch=$BATCH seq=$SEQ DBO=$DBO"
        echo "$SUFFIX,prefill,$BATCH,$SEQ,$DBO,OOM,OOM" >> "$BASE_DIR/summary.csv"
        return 2
    fi

    # 提取 prefill 时间
    local TIME_MS
    TIME_MS=$(grep -oP 'prefill_time=\K[0-9.]+' "$ATTN_LOG" || echo "N/A")
    log_ok "Prefill $SUFFIX: ${TIME_MS}ms"
    echo "$SUFFIX,prefill,$BATCH,$SEQ,$DBO,$TIME_MS," >> "$BASE_DIR/summary.csv"

    # DBO 模式下生成 pipeline 可视化 (timing 文件由 main.py 保存到 results/prefill_dbo/)
    if [ "$DBO" = "on" ]; then
        local ATTN_TIMING="results/prefill_dbo/timing_attention_${SUFFIX}.json"
        local FFN_TIMING="results/prefill_dbo/timing_ffn_${SUFFIX}.json"
        if [ -f "$ATTN_TIMING" ] && [ -f "$FFN_TIMING" ]; then
            python scripts/visualize_dbo_pipeline.py \
                --attn-timing "$ATTN_TIMING" \
                --ffn-timing "$FFN_TIMING" \
                --output "$PREFILL_DIR/pipeline_${SUFFIX}.png" \
                --start-layer 1 --num-layers 4 2>/dev/null && \
                log_ok "Pipeline 图: pipeline_${SUFFIX}.png" || \
                log_warn "可视化生成失败"
        fi
    fi

    return 0
}

# ============================================================
# Decode 实验：使用 generation 模式
# ============================================================
run_decode_experiment() {
    local BATCH=$1
    local SEQ=$2
    local DBO=$3  # "on" or "off"
    local PORT
    PORT=$(next_port)

    local DBO_FLAG=""
    local SUFFIX
    if [ "$DBO" = "off" ]; then
        DBO_FLAG="--no-dbo"
        SUFFIX="serial_b${BATCH}_s${SEQ}"
    else
        SUFFIX="dbo_b${BATCH}_s${SEQ}"
    fi

    log_info "Decode: batch=$BATCH seq=$SEQ DBO=$DBO tokens=$TOKENS_DECODE (port=$PORT)"

    local ATTN_LOG="$DECODE_DIR/logs/attn_${SUFFIX}.log"
    local FFN_LOG="$DECODE_DIR/logs/ffn_${SUFFIX}.log"

    # 生成指定长度的 prompt
    local PROMPT
    PROMPT=$(python -c "print('Test ' * ($SEQ // 5))")

    # FFN 节点（后台）
    CUDA_VISIBLE_DEVICES=2,3 python -m src.main \
        --model-name "$MODEL" \
        --role ffn \
        --master-addr 127.0.0.1 \
        --master-port "$PORT" \
        --world-size 2 --rank 1 --local-rank 0 \
        --attn-node-rank 0 --ffn-node-rank 1 \
        --batch-size "$BATCH" \
        --max-new-tokens "$TOKENS_DECODE" \
        --greedy \
        $DBO_FLAG \
        > "$FFN_LOG" 2>&1 &
    local FFN_PID=$!
    sleep 3

    # Attention 节点（前台）
    CUDA_VISIBLE_DEVICES=0,1 python -m src.main \
        --model-name "$MODEL" \
        --role attention \
        --master-addr 127.0.0.1 \
        --master-port "$PORT" \
        --world-size 2 --rank 0 --local-rank 0 \
        --attn-node-rank 0 --ffn-node-rank 1 \
        --batch-size "$BATCH" \
        --max-new-tokens "$TOKENS_DECODE" \
        --prompt "$PROMPT" \
        --greedy \
        $DBO_FLAG \
        > "$ATTN_LOG" 2>&1

    wait $FFN_PID 2>/dev/null || true

    # 检查 OOM
    if grep -q "OutOfMemoryError\|CUDA out of memory" "$ATTN_LOG" "$FFN_LOG" 2>/dev/null; then
        log_warn "OOM: batch=$BATCH seq=$SEQ DBO=$DBO"
        echo "$SUFFIX,decode,$BATCH,$SEQ,$DBO,OOM,OOM" >> "$BASE_DIR/summary.csv"
        return 2
    fi

    # 提取吞吐量和时间
    local TIME_MS TOK_S
    TIME_MS=$(grep -oP 'Generated.*in \K[0-9.]+(?=ms)' "$ATTN_LOG" || echo "N/A")
    TOK_S=$(grep -oP '\(\K[0-9.]+(?= tok/s)' "$ATTN_LOG" || echo "N/A")
    log_ok "Decode $SUFFIX: ${TIME_MS}ms, ${TOK_S} tok/s"
    echo "$SUFFIX,decode,$BATCH,$SEQ,$DBO,$TIME_MS,$TOK_S" >> "$BASE_DIR/summary.csv"

    return 0
}

# ============================================================
# Phase 1: Prefill batch 扩展 (seq=128)
# ============================================================
run_prefill_batch() {
    log_phase "Phase 1: Prefill Batch 扩展 (seq=128)"
    local BATCH_SIZES=(1 2 4 8 16 32 64 128)
    local SEQ=128

    for BATCH in "${BATCH_SIZES[@]}"; do
        # DBO ON
        if ! run_prefill_experiment "$BATCH" "$SEQ" "on"; then
            log_warn "Batch=$BATCH OOM，跳过更大的 batch"
            break
        fi
        # DBO OFF (serial)
        run_prefill_experiment "$BATCH" "$SEQ" "off" || break
        sleep 1
    done
}

# ============================================================
# Phase 2: Prefill seq 扩展 (batch=8)
# ============================================================
run_prefill_seq() {
    log_phase "Phase 2: Prefill Seq 扩展 (batch=8)"
    local SEQ_LENGTHS=(32 64 128 256 512 1024 2048)
    local BATCH=8

    for SEQ in "${SEQ_LENGTHS[@]}"; do
        # DBO ON
        if ! run_prefill_experiment "$BATCH" "$SEQ" "on"; then
            log_warn "Seq=$SEQ OOM，跳过更大的 seq"
            break
        fi
        # DBO OFF (serial)
        run_prefill_experiment "$BATCH" "$SEQ" "off" || break
        sleep 1
    done
}

# ============================================================
# Phase 3: Decode batch 扩展 (seq=128)
# ============================================================
run_decode_batch() {
    log_phase "Phase 3: Decode Batch 扩展 (seq=128, tokens=$TOKENS_DECODE)"
    local BATCH_SIZES=(1 2 4 8 16 32 64)
    local SEQ=128

    for BATCH in "${BATCH_SIZES[@]}"; do
        # DBO ON
        if ! run_decode_experiment "$BATCH" "$SEQ" "on"; then
            log_warn "Batch=$BATCH OOM，跳过更大的 batch"
            break
        fi
        # DBO OFF
        run_decode_experiment "$BATCH" "$SEQ" "off" || break
        sleep 1
    done
}

# ============================================================
# 主流程
# ============================================================
case "$PHASE" in
    prefill-batch)  run_prefill_batch ;;
    prefill-seq)    run_prefill_seq ;;
    decode)         run_decode_batch ;;
    all)
        echo "name,phase,batch_size,seq_len,dbo,time_ms,tok_s" > "$BASE_DIR/summary.csv"
        run_prefill_batch
        run_prefill_seq
        run_decode_batch
        ;;
    *)
        echo "Usage: $0 [prefill-batch|prefill-seq|decode|all]"
        exit 1
        ;;
esac

log_phase "实验完成"
echo ""
echo "汇总:"
column -t -s',' "$BASE_DIR/summary.csv" 2>/dev/null || cat "$BASE_DIR/summary.csv"
echo ""
echo "结果目录: $BASE_DIR/"
