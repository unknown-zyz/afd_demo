#!/bin/bash
# 单机测试 AFD + DBO (4×GPU)
#
# Usage: ./scripts/test_local.sh [max_tokens] [batch_size] [options]
#
# Options:
#   --timing        - 启用详细 timing 输出和日志保存
#   --no-dbo        - 关闭 DBO (serial baseline)
#   --verbose       - 详细输出
#
# Environment variables:
#   MODEL_PATH      - 模型路径 (默认: /data/Qwen/Qwen3-30B-A3B/)
#   MASTER_PORT     - 通信端口 (默认: 29600)
#
# Examples:
#   ./scripts/test_local.sh                     # 默认: 5 tokens, batch=1, DBO ON
#   ./scripts/test_local.sh 10 2                # 10 tokens, batch=2
#   ./scripts/test_local.sh 30 1 --timing       # benchmark 模式
#   ./scripts/test_local.sh 50 2 --no-dbo       # DBO OFF

set -e
cd "$(dirname "$0")/.."
source venv/bin/activate

# NCCL buffer tuning: 32MB prevents A2F flow-control blocking in DBO pipeline
export NCCL_BUFFSIZE=33554432
export NCCL_NCHANNELS_PER_NET_PEER=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- Positional args ---
MAX_TOKENS="${1:-5}"
BATCH_SIZE="${2:-1}"
shift 2 2>/dev/null || true

# --- Parse optional flags ---
TIMING=false
NO_DBO=false
VERBOSE=false
for arg in "$@"; do
    case "$arg" in
        --timing)  TIMING=true ;;
        --no-dbo)  NO_DBO=true ;;
        --verbose) VERBOSE=true ;;
    esac
done

# --- Config ---
MODEL_PATH="${MODEL_PATH:-/data/Qwen/Qwen3-30B-A3B/}"
MASTER_ADDR="127.0.0.1"
MASTER_PORT="${MASTER_PORT:-29600}"

# Build extra flags for python
EXTRA_FLAGS=""
if $NO_DBO; then
    EXTRA_FLAGS+=" --no-dbo"
fi
if $TIMING; then
    EXTRA_FLAGS+=" --timing --verbose"
elif $VERBOSE; then
    EXTRA_FLAGS+=" --verbose"
fi

DBO_LABEL="ON"
$NO_DBO && DBO_LABEL="OFF"

echo "========================================"
echo "  AFD + DBO 单机测试 (4×GPU)"
echo "========================================"
echo "Model: $MODEL_PATH"
echo "Max tokens: $MAX_TOKENS"
echo "Batch size: $BATCH_SIZE"
echo "DBO: $DBO_LABEL"
echo "Timing: $TIMING"
echo "Master: $MASTER_ADDR:$MASTER_PORT"
echo "----------------------------------------"

# GPU memory before
echo "GPU Memory before:"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
echo ""

echo "Starting test (single-node, 4 GPUs)..."
echo "  Attention node: GPU 0,1"
echo "  FFN node: GPU 2,3"

if $TIMING; then
    mkdir -p results

    # FFN 节点 (后台, GPU 2,3) — tee 保存日志
    CUDA_VISIBLE_DEVICES=2,3 python -m src.main \
        --model-name "$MODEL_PATH" \
        --role ffn \
        --master-addr "$MASTER_ADDR" \
        --master-port "$MASTER_PORT" \
        --world-size 2 \
        --rank 1 \
        --local-rank 0 \
        --attn-node-rank 0 \
        --ffn-node-rank 1 \
        --batch-size "$BATCH_SIZE" \
        --max-new-tokens "$MAX_TOKENS" \
        $EXTRA_FLAGS \
        2>&1 | tee results/ffn_bench.log | sed 's/^/[FFN] /' &
    FFN_PID=$!

    sleep 5

    # Attention 节点 (前台, GPU 0,1) — tee 保存日志
    CUDA_VISIBLE_DEVICES=0,1 python -m src.main \
        --model-name "$MODEL_PATH" \
        --role attention \
        --master-addr "$MASTER_ADDR" \
        --master-port "$MASTER_PORT" \
        --world-size 2 \
        --rank 0 \
        --local-rank 0 \
        --attn-node-rank 0 \
        --ffn-node-rank 1 \
        --batch-size "$BATCH_SIZE" \
        --max-new-tokens "$MAX_TOKENS" \
        --prompt "Hello" \
        --greedy \
        $EXTRA_FLAGS \
        2>&1 | tee results/attn_bench.log | sed 's/^/[ATTN] /'

    wait $FFN_PID 2>/dev/null || true

    echo ""
    echo "========================================"
    echo "  Results saved to results/"
    echo "========================================"
    ls -la results/*.log results/*.json 2>/dev/null || true
else
    # FFN 节点 (后台, GPU 2,3) — 简单模式
    CUDA_VISIBLE_DEVICES=2,3 python -m src.main \
        --model-name "$MODEL_PATH" \
        --role ffn \
        --master-addr "$MASTER_ADDR" \
        --master-port "$MASTER_PORT" \
        --world-size 2 \
        --rank 1 \
        --local-rank 0 \
        --attn-node-rank 0 \
        --ffn-node-rank 1 \
        --batch-size "$BATCH_SIZE" \
        --max-new-tokens "$MAX_TOKENS" \
        $EXTRA_FLAGS \
        2>&1 | sed 's/^/[FFN] /' &
    FFN_PID=$!
    echo "FFN node started (PID: $FFN_PID)"

    sleep 5

    # Attention 节点 (前台, GPU 0,1)
    CUDA_VISIBLE_DEVICES=0,1 python -m src.main \
        --model-name "$MODEL_PATH" \
        --role attention \
        --master-addr "$MASTER_ADDR" \
        --master-port "$MASTER_PORT" \
        --world-size 2 \
        --rank 0 \
        --local-rank 0 \
        --attn-node-rank 0 \
        --ffn-node-rank 1 \
        --batch-size "$BATCH_SIZE" \
        --max-new-tokens "$MAX_TOKENS" \
        --prompt "Hello" \
        --greedy \
        $EXTRA_FLAGS \
        2>&1 | sed 's/^/[ATTN] /'

    wait $FFN_PID 2>/dev/null || true
fi

echo ""
echo "GPU Memory after:"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
