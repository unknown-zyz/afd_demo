#!/bin/bash
# DBO Pipeline Profiling: 收集用于可视化的 timing 数据
# Usage: ./scripts/profile_dbo_pipeline.sh [deployment] [batch_size] [max_tokens]
#
# 参数:
#   deployment  - 部署类型: local (单机) 或 multinode (多机)
#   batch_size  - 批大小 (默认: 4, 需要 >= 2 才有 2 个 micro-batch)
#   max_tokens  - 生成 token 数 (默认: 5)
#
# 示例:
#   ./scripts/profile_dbo_pipeline.sh local 4 5      # 单机测试
#   ./scripts/profile_dbo_pipeline.sh multinode 4 5  # 多机测试 (需要先启动 FFN 节点)
#
# 输出:
#   results/prefill_dbo/timing_attention_{deployment}_b{batch}_t{tokens}.json
#   results/prefill_dbo/timing_ffn_{deployment}_b{batch}_t{tokens}.json

set -e
cd "$(dirname "$0")/.."
source venv/bin/activate

# 参数
DEPLOYMENT="${1:-local}"
BATCH_SIZE="${2:-4}"
MAX_TOKENS="${3:-5}"
MODEL_PATH="/data/Qwen/Qwen3-30B-A3B/"
MASTER_PORT="${MASTER_PORT:-29650}"

# 构建 timing 后缀
TIMING_SUFFIX="${DEPLOYMENT}_b${BATCH_SIZE}_t${MAX_TOKENS}"

echo "========================================"
echo "  DBO Pipeline Profiling"
echo "========================================"
echo "Deployment: $DEPLOYMENT"
echo "Batch size: $BATCH_SIZE (will have 2 micro-batches)"
echo "Max tokens: $MAX_TOKENS"
echo "Timing suffix: $TIMING_SUFFIX"
echo "Output files:"
echo "  - results/prefill_dbo/timing_attention_${TIMING_SUFFIX}.json"
echo "  - results/prefill_dbo/timing_ffn_${TIMING_SUFFIX}.json"
echo "----------------------------------------"

# 创建 results 目录
mkdir -p results/prefill_dbo

if [ "$DEPLOYMENT" = "local" ]; then
    echo ""
    echo "[INFO] Running local 4-GPU test..."
    
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
        --timing-suffix "$TIMING_SUFFIX" \
        --no-generate \
        --verbose \
        2>&1 | tee results/prefill_dbo/ffn_profile_${TIMING_SUFFIX}.log | sed 's/^/[FFN] /' &
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
        --timing-suffix "$TIMING_SUFFIX" \
        --no-generate \
        --verbose \
        2>&1 | tee results/prefill_dbo/attn_profile_${TIMING_SUFFIX}.log | sed 's/^/[ATTN] /'
    
    wait $FFN_PID 2>/dev/null || true
    
elif [ "$DEPLOYMENT" = "multinode" ]; then
    echo ""
    echo "[INFO] Running multi-node test..."
    echo "[INFO] Make sure FFN node is already running on the remote machine!"
    echo ""
    echo "To start FFN node on remote machine, run:"
    echo "  cd /home/zyz/afd_demo && source venv/bin/activate"
    echo "  ./scripts/run_ffn_node.sh \$(hostname -I | awk '{print \$1}') $MASTER_PORT \\"
    echo "    --batch-size $BATCH_SIZE --max-new-tokens $MAX_TOKENS \\"
    echo "    --timing --timing-suffix $TIMING_SUFFIX --no-generate"
    echo ""
    read -p "Press Enter when FFN node is ready, or Ctrl+C to cancel..."
    
    # 启动 Attention 节点 (本机)
    CUDA_VISIBLE_DEVICES=0,1 python -m src.main \
        --model-name "$MODEL_PATH" \
        --role attention \
        --master-addr "$(hostname -I | awk '{print $1}')" \
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
        --timing-suffix "$TIMING_SUFFIX" \
        --no-generate \
        --verbose \
        2>&1 | tee results/prefill_dbo/attn_profile_${TIMING_SUFFIX}.log | sed 's/^/[ATTN] /'
else
    echo "Error: Unknown deployment type: $DEPLOYMENT"
    echo "Usage: $0 [local|multinode] [batch_size] [max_tokens]"
    exit 1
fi

echo ""
echo "========================================"
echo "  Profiling Complete!"
echo "========================================"
echo "Generated files:"
ls -la results/prefill_dbo/*${TIMING_SUFFIX}* 2>/dev/null || echo "(no files found)"
echo ""
echo "To visualize:"
echo "  python scripts/visualize_dbo_pipeline.py \\"
echo "    --attn-timing results/prefill_dbo/timing_attention_${TIMING_SUFFIX}.json \\"
echo "    --ffn-timing results/prefill_dbo/timing_ffn_${TIMING_SUFFIX}.json \\"
echo "    --output results/prefill_dbo/dbo_pipeline_${TIMING_SUFFIX}.png"
