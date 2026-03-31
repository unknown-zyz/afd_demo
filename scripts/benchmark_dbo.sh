#!/bin/bash
# DBO Benchmark: 测量 Attention、通信、FFN 各阶段延迟
# Usage: ./scripts/benchmark_dbo.sh [max_tokens] [batch_size] [dbo]
#
# Examples:
#   ./scripts/benchmark_dbo.sh                 # 默认: 30 tokens, batch=1, DBO ON
#   ./scripts/benchmark_dbo.sh 50 2            # 50 tokens, batch=2, DBO ON
#   ./scripts/benchmark_dbo.sh 50 2 off        # 50 tokens, batch=2, DBO OFF

set -e
cd "$(dirname "$0")/.."
source venv/bin/activate

MAX_TOKENS="${1:-30}"
BATCH_SIZE="${2:-1}"
DBO_MODE="${3:-on}"  # on/off
MODEL_PATH="/data/Qwen/Qwen3-30B-A3B/"
MASTER_PORT="${MASTER_PORT:-29620}"

# 设置 DBO 参数
if [ "$DBO_MODE" = "off" ]; then
    DBO_FLAG="--no-dbo"
    DBO_LABEL="OFF"
else
    DBO_FLAG=""
    DBO_LABEL="ON"
fi

echo "========================================"
echo "  DBO Benchmark (单机 4 GPU)"
echo "========================================"
echo "Model: $MODEL_PATH"
echo "Max tokens: $MAX_TOKENS"
echo "Batch size: $BATCH_SIZE"
echo "DBO: $DBO_LABEL"
echo "----------------------------------------"

# 创建 results 目录
mkdir -p results

# 启动 FFN 节点 (后台)
CUDA_VISIBLE_DEVICES=2,3 python -m src.main \
    --model-name "$MODEL_PATH" \
    --role ffn \
    --master-addr 127.0.0.1 \
    --master-port "$MASTER_PORT" \
    --world-size 2 \
    --rank 1 \
    --local-rank 0 \
    --attn-node-rank 0 \
    --ffn-node-rank 1 \
    --batch-size "$BATCH_SIZE" \
    --max-new-tokens "$MAX_TOKENS" \
    --timing \
    --verbose \
    $DBO_FLAG \
    2>&1 | tee results/ffn_bench.log | sed 's/^/[FFN] /' &
FFN_PID=$!
sleep 5

# 启动 Attention 节点 (前台)
CUDA_VISIBLE_DEVICES=0,1 python -m src.main \
    --model-name "$MODEL_PATH" \
    --role attention \
    --master-addr 127.0.0.1 \
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
    --timing \
    --verbose \
    $DBO_FLAG \
    2>&1 | tee results/attn_bench.log | sed 's/^/[ATTN] /'

wait $FFN_PID 2>/dev/null || true

echo ""
echo "========================================"
echo "  Results saved to results/"
echo "========================================"
ls -la results/*.log results/*.json 2>/dev/null || true
