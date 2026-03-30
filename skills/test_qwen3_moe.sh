#!/bin/bash
# 测试 Qwen3-30B-A3B MoE 模型 (需要 4 GPU, 单机)
# Usage: ./skills/test_qwen3_moe.sh [max_tokens] [batch_size]
#   ./skills/test_qwen3_moe.sh           # 默认: max_tokens=5, batch_size=1
#   ./skills/test_qwen3_moe.sh 10 2      # 自定义

set -e
cd "$(dirname "$0")/.."
source venv/bin/activate

MAX_TOKENS="${1:-5}"
BATCH_SIZE="${2:-1}"
MODEL_PATH="/data/Qwen/Qwen3-30B-A3B/"

echo "=== Qwen3-30B-A3B MoE 测试 ==="
echo "Model: $MODEL_PATH"
echo "Max tokens: $MAX_TOKENS"
echo "Batch size: $BATCH_SIZE"
echo ""

# 监控内存使用
echo "GPU Memory before:"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader

echo ""
echo "Starting test..."

# 使用 4 GPU, world_size=2 (Attention 用 GPU 0-1, FFN 用 GPU 2-3)
# 注意: AFD 模式下 attention 和 ffn 各自使用独立的 GPU
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=2 -m src.main \
    --model-name "$MODEL_PATH" \
    --local-test \
    --batch-size "$BATCH_SIZE" \
    --max-new-tokens "$MAX_TOKENS" \
    --prompt "Hello" \
    --greedy \
    --no-decode-dbo

echo ""
echo "GPU Memory after:"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
