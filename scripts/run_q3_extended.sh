#!/bin/bash
# Qwen3-30B-A3B 扩展实验脚本
# 1) Decode 大 batch 扩展 (256, 512, 1024)
# 2) Prefill 多 micro-batch 实验 (num_micro_batches=2,3,4)
#
# Usage: ./scripts/run_q3_extended.sh [decode-large|prefill-multimb|all]

set -e
cd "$(dirname "$0")/.."
source venv/bin/activate

PHASE="${1:-all}"
MODEL="/data/Qwen/Qwen3-30B-A3B/"
TOKENS_DECODE=20
TOKENS_PREFILL=5
BASE_DIR="results/experiments_q3"
DECODE_DIR="$BASE_DIR/decode"
PREFILL_DIR="$BASE_DIR/prefill"
TIMING_DIR="results/prefill_dbo"
MASTER_PORT_BASE=29850
PORT_COUNTER=0
MODEL_LOAD_SLEEP=30

mkdir -p "$DECODE_DIR/logs" "$PREFILL_DIR/logs"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
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
    # Kill any lingering python processes on our ports
    pkill -f "src.main.*master-port.*298[5-9]" 2>/dev/null || true
    sleep 3
}

# ============================================================
# Decode 实验
# ============================================================
run_decode_experiment() {
    local BATCH=$1
    local SEQ=$2
    local DBO=$3  # "on" or "off"
    local PORT
    PORT=$(next_port)

    local DBO_FLAG="" SUFFIX
    if [ "$DBO" = "off" ]; then
        DBO_FLAG="--no-dbo"
        SUFFIX="serial_b${BATCH}_s${SEQ}"
    else
        SUFFIX="dbo_b${BATCH}_s${SEQ}"
    fi

    log_info "Decode: batch=$BATCH seq=$SEQ DBO=$DBO tokens=$TOKENS_DECODE (port=$PORT)"

    local ATTN_LOG="$DECODE_DIR/logs/attn_${SUFFIX}.log"
    local FFN_LOG="$DECODE_DIR/logs/ffn_${SUFFIX}.log"
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

    # 检查 FFN 是否已崩溃
    if ! kill -0 $FFN_PID 2>/dev/null; then
        if grep -q "OutOfMemoryError\|CUDA out of memory" "$FFN_LOG" 2>/dev/null; then
            log_warn "OOM (FFN loading): batch=$BATCH DBO=$DBO"
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
        log_warn "OOM: batch=$BATCH DBO=$DBO"
        echo "$SUFFIX,decode,$BATCH,$SEQ,$DBO,OOM,OOM" >> "$BASE_DIR/summary.csv"
        cleanup_gpu
        return 2
    fi

    # 检查其他错误（排除 NCCL 清理警告）
    if grep -q "Traceback" "$ATTN_LOG" "$FFN_LOG" 2>/dev/null; then
        log_error "Error: batch=$BATCH DBO=$DBO"
        grep -h "Traceback\|Error" "$ATTN_LOG" "$FFN_LOG" 2>/dev/null | grep -v "c10d\|ProcessGroup\|TCPStore\|recvValue\|heartbeat\|Warning" | tail -3
        echo "$SUFFIX,decode,$BATCH,$SEQ,$DBO,ERROR,ERROR" >> "$BASE_DIR/summary.csv"
        cleanup_gpu
        return 1
    fi

    # 提取结果
    local TIME_MS TOK_S
    TIME_MS=$(grep -oP 'Generated.*in \K[0-9.]+(?=ms)' "$ATTN_LOG" || echo "N/A")
    TOK_S=$(grep -oP '\(\K[0-9.]+(?= tok/s)' "$ATTN_LOG" || echo "N/A")
    log_ok "Decode $SUFFIX: ${TIME_MS}ms, ${TOK_S} tok/s"
    echo "$SUFFIX,decode,$BATCH,$SEQ,$DBO,$TIME_MS,$TOK_S" >> "$BASE_DIR/summary.csv"

    cleanup_gpu
    return 0
}

# ============================================================
# Prefill 实验（支持自定义 num_micro_batches）
# ============================================================
run_prefill_experiment() {
    local BATCH=$1
    local SEQ=$2
    local DBO=$3  # "on" or "off"
    local NUM_MB=${4:-2}  # micro-batch 数量，默认 2
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

    # FFN 节点
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
        --timing --timing-suffix "q3_${SUFFIX}" \
        --no-generate $DBO_FLAG \
        > "$FFN_LOG" 2>&1 &
    local FFN_PID=$!
    sleep "$MODEL_LOAD_SLEEP"

    # 检查 FFN 是否已崩溃
    if ! kill -0 $FFN_PID 2>/dev/null; then
        if grep -q "OutOfMemoryError\|CUDA out of memory" "$FFN_LOG" 2>/dev/null; then
            log_warn "OOM (FFN): batch=$BATCH seq=$SEQ DBO=$DBO MB=$NUM_MB"
            echo "$SUFFIX,prefill,$BATCH,$SEQ,$DBO,OOM,,${NUM_MB}" >> "$BASE_DIR/summary.csv"
            cleanup_gpu
            return 2
        fi
    fi

    # Attention 节点
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
        --timing --timing-suffix "q3_${SUFFIX}" \
        --no-generate $DBO_FLAG \
        > "$ATTN_LOG" 2>&1

    wait $FFN_PID 2>/dev/null || true

    # 检查 OOM
    if grep -q "OutOfMemoryError\|CUDA out of memory" "$ATTN_LOG" "$FFN_LOG" 2>/dev/null; then
        log_warn "OOM: batch=$BATCH seq=$SEQ DBO=$DBO MB=$NUM_MB"
        echo "$SUFFIX,prefill,$BATCH,$SEQ,$DBO,OOM,,${NUM_MB}" >> "$BASE_DIR/summary.csv"
        cleanup_gpu
        return 2
    fi

    # 检查其他错误
    if grep -q "Traceback" "$ATTN_LOG" "$FFN_LOG" 2>/dev/null; then
        log_error "Error: batch=$BATCH seq=$SEQ DBO=$DBO MB=$NUM_MB"
        grep -h "Traceback\|Error" "$ATTN_LOG" "$FFN_LOG" 2>/dev/null | grep -v "c10d\|ProcessGroup\|TCPStore\|recvValue\|heartbeat\|Warning" | tail -3
        echo "$SUFFIX,prefill,$BATCH,$SEQ,$DBO,ERROR,,${NUM_MB}" >> "$BASE_DIR/summary.csv"
        cleanup_gpu
        return 1
    fi

    # 提取 prefill 时间
    local TIME_MS
    TIME_MS=$(grep -oP 'prefill_time=\K[0-9.]+' "$ATTN_LOG" || echo "N/A")
    log_ok "Prefill $SUFFIX: ${TIME_MS}ms"
    echo "$SUFFIX,prefill,$BATCH,$SEQ,$DBO,$TIME_MS,,${NUM_MB}" >> "$BASE_DIR/summary.csv"

    # Pipeline 可视化（仅 DBO 且有 serial 数据时）
    if [ "$DBO" = "on" ]; then
        local SERIAL_TIMING="$TIMING_DIR/timing_attention_q3_serial_b${BATCH}_s${SEQ}${MB_SUFFIX}.json"
        if [ -f "$SERIAL_TIMING" ]; then
            python scripts/visualize_dbo_pipeline.py \
                --attention-json "$TIMING_DIR/timing_attention_q3_${SUFFIX}.json" \
                --ffn-json "$TIMING_DIR/timing_ffn_q3_${SUFFIX}.json" \
                --output "$PREFILL_DIR/pipeline_${SUFFIX}.png" \
                --skip-l0 --start-layer 1 --end-layer 4 \
                --serial-timing "$SERIAL_TIMING" \
                2>&1 || log_warn "Pipeline 图生成失败: $SUFFIX"
            [ -f "$PREFILL_DIR/pipeline_${SUFFIX}.png" ] && log_ok "Pipeline 图: pipeline_${SUFFIX}.png"
        fi
    fi

    cleanup_gpu
    return 0
}

# ============================================================
# Phase 1: Decode 大 Batch (256, 512, 1024)
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
# Phase 2: Prefill 多 Micro-batch (batch=32,64, MB=2,3,4)
# ============================================================
run_prefill_multimb() {
    log_phase "Prefill 多 Micro-batch 实验 (seq=128) — Qwen3-30B-A3B"
    local BATCH_SIZES=(32 64)
    local SEQ=128
    local MB_COUNTS=(2 3 4)

    for BATCH in "${BATCH_SIZES[@]}"; do
        # 先跑一次 serial（MB 数量不影响 serial）
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
log_phase "Qwen3-30B-A3B 扩展实验"

case "$PHASE" in
    decode-large)     run_decode_large ;;
    prefill-multimb)  run_prefill_multimb ;;
    all)
        run_decode_large
        run_prefill_multimb
        ;;
    *)
        echo "Usage: $0 [decode-large|prefill-multimb|all]"
        exit 1
        ;;
esac

log_phase "扩展实验完成"
echo ""
echo "新增结果已追加到: $BASE_DIR/summary.csv"
echo ""
tail -20 "$BASE_DIR/summary.csv" | column -t -s',' 2>/dev/null || tail -20 "$BASE_DIR/summary.csv"
