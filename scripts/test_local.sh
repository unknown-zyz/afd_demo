#!/bin/bash
# 测试 Qwen3-30B-A3B MoE 模型 (需要 4 GPU, 单机)
# Usage: ./scripts/test_qwen3_moe.sh [max_tokens] [batch_size]
#   ./scripts/test_qwen3_moe.sh           # 默认: max_tokens=5, batch_size=1
#   ./scripts/test_qwen3_moe.sh 10 2      # 自定义

set -e
cd "$(dirname "$0")/.."
source venv/bin/activate

MAX_TOKENS="${1:-5}"
BATCH_SIZE="${2:-1}"
MODEL_PATH="/data/Qwen/Qwen3-30B-A3B/"
MASTER_ADDR="127.0.0.1"
MASTER_PORT="${MASTER_PORT:-29600}"

echo "=== Qwen3-30B-A3B MoE 测试 ==="
echo "Model: $MODEL_PATH"
echo "Max tokens: $MAX_TOKENS"
echo "Batch size: $BATCH_SIZE"
echo "Master: $MASTER_ADDR:$MASTER_PORT"
echo ""

# 监控内存使用
echo "GPU Memory before:"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader

echo ""
echo "Starting test (single-node, 4 GPUs)..."
echo "  Attention node: GPU 0,1"
echo "  FFN node: GPU 2,3"

# 启动 FFN 节点 (后台, 使用 GPU 2,3)
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
    --no-decode-dbo \
    2>&1 | sed 's/^/[FFN] /' &
FFN_PID=$!
echo "FFN node started (PID: $FFN_PID)"

# 等待 FFN 节点初始化
sleep 5

# 启动 Attention 节点 (前台, 使用 GPU 0,1)
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
    --no-decode-dbo \
    2>&1 | sed 's/^/[ATTN] /'

# 等待 FFN 节点完成
wait $FFN_PID 2>/dev/null || true

echo ""
echo "GPU Memory after:"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
