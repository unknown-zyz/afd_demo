#!/bin/bash
# AFD + DBO 综合实验脚本
# 在单机 4×V100-32GB 上测试 Prefill/Decode 阶段 DBO 开启/关闭的效果
#
# Usage: ./scripts/run_experiments.sh [phase]
#
# Phases:
#   decode          - Decode batch 扩展 (b1-b128, seq=128, 20 tokens)
#   decode-large    - Decode 大 batch 扩展 (b256-b1024, seq=128)
#   prefill-batch   - Prefill batch 扩展 (b1-b128, seq=128)
#   prefill-seq     - Prefill seq 扩展 (s64-s2048, batch=4)
#   prefill-multimb - Prefill 多 micro-batch 实验 (MB=2,3,4)
#   all             - 运行全部实验
#
# Environment variables:
#   MODEL_PATH      - 模型路径 (默认: /data/Qwen/Qwen3-30B-A3B/)
#   MASTER_PORT_BASE - 起始端口 (默认: 29800)
#   MODEL_LOAD_SLEEP - 模型加载等待时间秒 (默认: 30)
#
# Examples:
#   ./scripts/run_experiments.sh all
#   ./scripts/run_experiments.sh decode
#   ./scripts/run_experiments.sh prefill-multimb
#   MODEL_PATH=/data/my_model/ ./scripts/run_experiments.sh prefill-batch

set -e
cd "$(dirname "$0")/.."
source venv/bin/activate

PHASE="${1:-all}"
MODEL="${MODEL_PATH:-"/data/Qwen/Qwen3-30B-A3B/"}"
TOKENS_PREFILL=5
TOKENS_DECODE=20
BASE_DIR="results/experiments_qwen3"
PREFILL_DIR="$BASE_DIR/prefill"
DECODE_DIR="$BASE_DIR/decode"
TIMING_DIR="results/prefill_dbo"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-29800}"
PORT_COUNTER=0
MODEL_LOAD_SLEEP="${MODEL_LOAD_SLEEP:-30}"

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

cleanup_gpu() {
    sleep 10
}

# ============================================================
# Prefill 实验：使用 --no-generate 模式
# 支持自定义 num_micro_batches（默认 2）
# ============================================================
run_prefill_experiment() {
    local BATCH=$1
    local SEQ=$2
    local DBO=$3  # "on" or "off"
    local NUM_MB=${4:-2}
    local PORT
    PORT=$(next_port)

    local DBO_FLAG="" SUFFIX MB_SUFFIX=""
    if [ "$NUM_MB" -ne 2 ]; then
        MB_SUFFIX="_mb${NUM_MB}"
    fi

    if [ "$DBO" = "off" ]; then
        DBO_FLAG="--no-dbo"
        SUFFIX="serial_b${BATCH}_s${SEQ}${MB_SUFFIX}"
    else
        SUFFIX="dbo_b${BATCH}_s${SEQ}${MB_SUFFIX}"
    fi

    log_info "Prefill: batch=$BATCH seq=$SEQ DBO=$DBO micro_batches=$NUM_MB (port=$PORT)"

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
        --num-micro-batches "$NUM_MB" \
        --timing --timing-suffix "qwen3_${SUFFIX}" \
        --no-generate $DBO_FLAG \
        > "$FFN_LOG" 2>&1 &
    local FFN_PID=$!
    sleep "$MODEL_LOAD_SLEEP"

    # 检查 FFN 是否已崩溃（OOM during loading）
    if ! kill -0 $FFN_PID 2>/dev/null; then
        if grep -q "OutOfMemoryError\|CUDA out of memory" "$FFN_LOG" 2>/dev/null; then
            log_warn "OOM (FFN loading): batch=$BATCH seq=$SEQ DBO=$DBO"
            echo "$SUFFIX,prefill,$BATCH,$SEQ,$DBO,OOM," >> "$BASE_DIR/summary.csv"
            cleanup_gpu
            return 2
        fi
    fi

    # Attention 节点（前台，带超时）
    timeout 600 env CUDA_VISIBLE_DEVICES=0,1 python -m src.main \
        --model-name "$MODEL" \
        --role attention \
        --master-addr 127.0.0.1 \
        --master-port "$PORT" \
        --world-size 2 --rank 0 --local-rank 0 \
        --attn-node-rank 0 --ffn-node-rank 1 \
        --batch-size "$BATCH" \
        --prefill-seq-len "$SEQ" \
        --max-new-tokens "$TOKENS_PREFILL" \
        --num-micro-batches "$NUM_MB" \
        --prompt "Hello world, this is a test prompt for scaling experiments." \
        --timing --timing-suffix "qwen3_${SUFFIX}" \
        --no-generate $DBO_FLAG \
        > "$ATTN_LOG" 2>&1

    wait $FFN_PID 2>/dev/null || true

    # 检查 OOM
    if grep -q "OutOfMemoryError\|CUDA out of memory" "$ATTN_LOG" "$FFN_LOG" 2>/dev/null; then
        log_warn "OOM: batch=$BATCH seq=$SEQ DBO=$DBO"
        echo "$SUFFIX,prefill,$BATCH,$SEQ,$DBO,OOM," >> "$BASE_DIR/summary.csv"
        cleanup_gpu
        return 2
    fi

    # 检查其他错误 (排除 NCCL 清理警告)
    if grep -q "Traceback\|RuntimeError\|AssertionError\|ValueError\|TypeError" "$ATTN_LOG" "$FFN_LOG" 2>/dev/null; then
        # 排除已知的 NCCL 清理错误
        if ! grep -q "Traceback" "$ATTN_LOG" 2>/dev/null && ! grep -q "Traceback" "$FFN_LOG" 2>/dev/null; then
            : # 没有 Python traceback，忽略
        else
            log_error "Error in run: batch=$BATCH seq=$SEQ DBO=$DBO"
            grep -h "Traceback\|Error" "$ATTN_LOG" "$FFN_LOG" 2>/dev/null | grep -v "c10d\|ProcessGroup\|TCPStore\|recvValue\|heartbeat\|Warning" | tail -3
            echo "$SUFFIX,prefill,$BATCH,$SEQ,$DBO,ERROR," >> "$BASE_DIR/summary.csv"
            cleanup_gpu
            return 1
        fi
    fi

    # 提取 prefill 时间
    local TIME_MS
    TIME_MS=$(grep -oP 'prefill_time=\K[0-9.]+' "$ATTN_LOG" || echo "N/A")
    log_ok "Prefill $SUFFIX: ${TIME_MS}ms"
    echo "$SUFFIX,prefill,$BATCH,$SEQ,$DBO,$TIME_MS," >> "$BASE_DIR/summary.csv"

    # DBO 模式下生成 pipeline 可视化
    if [ "$DBO" = "on" ]; then
        local ATTN_TIMING="$TIMING_DIR/timing_attention_qwen3_${SUFFIX}.json"
        local FFN_TIMING="$TIMING_DIR/timing_ffn_qwen3_${SUFFIX}.json"
        # 查找对应的 serial timing 文件
        local SERIAL_SUFFIX="serial_b${BATCH}_s${SEQ}"
        local SERIAL_ATTN_TIMING="$TIMING_DIR/timing_attention_qwen3_${SERIAL_SUFFIX}.json"
        local SERIAL_FLAG=""
        if [ -f "$SERIAL_ATTN_TIMING" ]; then
            SERIAL_FLAG="--serial-timing $SERIAL_ATTN_TIMING"
        fi
        if [ -f "$ATTN_TIMING" ] && [ -f "$FFN_TIMING" ]; then
            python scripts/visualize_dbo_pipeline.py \
                --attn-timing "$ATTN_TIMING" \
                --ffn-timing "$FFN_TIMING" \
                --output "$PREFILL_DIR/pipeline_${SUFFIX}.png" \
                --start-layer 1 --num-layers 4 $SERIAL_FLAG 2>/dev/null && \
                log_ok "Pipeline 图: pipeline_${SUFFIX}.png" || \
                log_warn "可视化生成失败"
        fi
    fi

    cleanup_gpu
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
    sleep "$MODEL_LOAD_SLEEP"

    # 检查 FFN 是否已崩溃（OOM during loading）
    if ! kill -0 $FFN_PID 2>/dev/null; then
        if grep -q "OutOfMemoryError\|CUDA out of memory" "$FFN_LOG" 2>/dev/null; then
            log_warn "OOM (FFN loading): batch=$BATCH seq=$SEQ DBO=$DBO"
            echo "$SUFFIX,decode,$BATCH,$SEQ,$DBO,OOM,OOM" >> "$BASE_DIR/summary.csv"
            cleanup_gpu
            return 2
        fi
    fi

    # Attention 节点（前台，带超时）
    timeout 600 env CUDA_VISIBLE_DEVICES=0,1 python -m src.main \
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
        cleanup_gpu
        return 2
    fi

    # 检查其他错误 (排除 NCCL 清理警告)
    if grep -q "Traceback\|RuntimeError\|AssertionError\|ValueError\|TypeError" "$ATTN_LOG" "$FFN_LOG" 2>/dev/null; then
        if ! grep -q "Traceback" "$ATTN_LOG" 2>/dev/null && ! grep -q "Traceback" "$FFN_LOG" 2>/dev/null; then
            : # 没有 Python traceback，忽略
        else
            log_error "Error in run: batch=$BATCH seq=$SEQ DBO=$DBO"
            grep -h "Traceback\|Error" "$ATTN_LOG" "$FFN_LOG" 2>/dev/null | grep -v "c10d\|ProcessGroup\|TCPStore\|recvValue\|heartbeat\|Warning" | tail -3
            echo "$SUFFIX,decode,$BATCH,$SEQ,$DBO,ERROR,ERROR" >> "$BASE_DIR/summary.csv"
            cleanup_gpu
            return 1
        fi
    fi

    # 提取吞吐量和时间
    local TIME_MS TOK_S
    TIME_MS=$(grep -oP 'Generated.*in \K[0-9.]+(?=ms)' "$ATTN_LOG" || echo "N/A")
    TOK_S=$(grep -oP '\(\K[0-9.]+(?= tok/s)' "$ATTN_LOG" || echo "N/A")
    log_ok "Decode $SUFFIX: ${TIME_MS}ms, ${TOK_S} tok/s"
    echo "$SUFFIX,decode,$BATCH,$SEQ,$DBO,$TIME_MS,$TOK_S" >> "$BASE_DIR/summary.csv"

    cleanup_gpu
    return 0
}

# ============================================================
# Phase: Decode batch 扩展 (seq=128)
# ============================================================
run_decode_batch() {
    log_phase "Decode Batch 扩展 (seq=128, tokens=$TOKENS_DECODE) — Qwen3-30B-A3B"
    local BATCH_SIZES=(1 2 4 8 16 32 64 128)
    local SEQ=128

    for BATCH in "${BATCH_SIZES[@]}"; do
        # Serial first
        if ! run_decode_experiment "$BATCH" "$SEQ" "off"; then
            log_warn "Batch=$BATCH serial OOM/Error, 跳过更大的 batch"
            break
        fi
        # DBO ON
        if ! run_decode_experiment "$BATCH" "$SEQ" "on"; then
            log_warn "Batch=$BATCH DBO OOM/Error, 跳过更大的 batch"
            break
        fi
        sleep 2
    done
}

# ============================================================
# Phase: Decode 大 batch 扩展 (256, 512, 1024)
# ============================================================
run_decode_large() {
    log_phase "Decode 大 Batch 扩展 (seq=128, tokens=$TOKENS_DECODE) — Qwen3-30B-A3B"
    local BATCH_SIZES=(256 512 1024)
    local SEQ=128

    for BATCH in "${BATCH_SIZES[@]}"; do
        if ! run_decode_experiment "$BATCH" "$SEQ" "off"; then
            log_warn "Batch=$BATCH serial OOM/Error, 跳过更大的 batch"
            break
        fi
        if ! run_decode_experiment "$BATCH" "$SEQ" "on"; then
            log_warn "Batch=$BATCH DBO OOM/Error, 跳过更大的 batch"
            break
        fi
        sleep 2
    done
}

# ============================================================
# Phase: Prefill batch 扩展 (seq=128)
# ============================================================
run_prefill_batch() {
    log_phase "Prefill Batch 扩展 (seq=128) — Qwen3-30B-A3B"
    local BATCH_SIZES=(1 2 4 8 16 32 64 128)
    local SEQ=128

    for BATCH in "${BATCH_SIZES[@]}"; do
        # Serial 先跑（供 pipeline 可视化使用实测 serial 时间）
        if ! run_prefill_experiment "$BATCH" "$SEQ" "off"; then
            log_warn "Batch=$BATCH serial OOM/Error, 跳过更大的 batch"
            break
        fi
        # DBO ON
        if ! run_prefill_experiment "$BATCH" "$SEQ" "on"; then
            log_warn "Batch=$BATCH DBO OOM/Error, 跳过更大的 batch"
            break
        fi
        sleep 2
    done
}

# ============================================================
# Phase: Prefill seq 扩展 (batch=4)
# ============================================================
run_prefill_seq() {
    log_phase "Prefill Seq 扩展 (batch=4) — Qwen3-30B-A3B"
    local SEQ_LENGTHS=(64 128 256 512 1024 2048)
    local BATCH=4

    for SEQ in "${SEQ_LENGTHS[@]}"; do
        # Serial first
        if ! run_prefill_experiment "$BATCH" "$SEQ" "off"; then
            log_warn "Seq=$SEQ serial OOM/Error, 跳过更大的 seq"
            break
        fi
        # DBO ON
        if ! run_prefill_experiment "$BATCH" "$SEQ" "on"; then
            log_warn "Seq=$SEQ DBO OOM/Error, 跳过更大的 seq"
            break
        fi
        sleep 2
    done
}

# ============================================================
# Phase: Prefill 多 micro-batch (batch=32,64, MB=2,3,4)
# ============================================================
run_prefill_multimb() {
    log_phase "Prefill 多 Micro-batch 实验 (seq=128) — Qwen3-30B-A3B"
    local BATCH_SIZES=(32 64)
    local SEQ=128
    local MB_COUNTS=(2 3 4)

    for BATCH in "${BATCH_SIZES[@]}"; do
        # serial 数据已有（b32 和 b64 在之前实验中已测过），跳过
        log_info "Skip serial for batch=$BATCH (already tested)"

        for NUM_MB in "${MB_COUNTS[@]}"; do
            if ! run_prefill_experiment "$BATCH" "$SEQ" "on" "$NUM_MB"; then
                log_warn "Batch=$BATCH MB=$NUM_MB OOM/Error, 跳过"
                continue
            fi
            sleep 2
        done
    done
}

# ============================================================
# 主流程
# ============================================================
log_phase "Qwen3-30B-A3B DBO 综合实验"
log_info "模型: $MODEL"
log_info "GPU: 0,1 (Attention) + 2,3 (FFN/MoE)"
log_info "模型加载等待: ${MODEL_LOAD_SLEEP}s"

case "$PHASE" in
    decode)           run_decode_batch ;;
    decode-large)     run_decode_large ;;
    prefill-batch)    run_prefill_batch ;;
    prefill-seq)      run_prefill_seq ;;
    prefill-multimb)  run_prefill_multimb ;;
    all)
        echo "name,phase,batch_size,seq_len,dbo,time_ms,tok_s" > "$BASE_DIR/summary.csv"
        run_decode_batch
        run_prefill_batch
        run_prefill_seq
        ;;
    *)
        echo "Usage: $0 [decode|decode-large|prefill-batch|prefill-seq|prefill-multimb|all]"
        exit 1
        ;;
esac

log_phase "实验完成"
echo ""
echo "汇总:"
column -t -s',' "$BASE_DIR/summary.csv" 2>/dev/null || cat "$BASE_DIR/summary.csv"
echo ""
echo "结果目录: $BASE_DIR/"
