#!/bin/bash
# 启动单个 AFD 节点（用于多机部署）
#
# Usage: ./scripts/run_node.sh <role> [master_addr] [master_port] [options]
#
# Arguments:
#   role        - attention 或 ffn
#   master_addr - Master 节点地址 (默认: 10.244.64.179)
#   master_port - Master 端口 (默认: 29500)
#
# Environment variables:
#   MODEL_PATH      - 模型路径 (默认: /data/Qwen/Qwen3-30B-A3B/)
#   CUDA_DEVICES    - GPU 设备 (默认: 0,1)
#   WORLD_SIZE      - 总进程数 (默认: 2)
#   BATCH_SIZE      - batch 大小 (默认: 1)
#
# Examples:
#   ./scripts/run_node.sh ffn 10.244.64.179 29500
#   ./scripts/run_node.sh attention 10.244.64.179 29500 --prompt "Hello"
#   ./scripts/run_node.sh ffn 10.244.64.179 29500 --no-dbo

set -e
cd "$(dirname "$0")/.."
source venv/bin/activate

# Use Chinese HuggingFace mirror
export HF_ENDPOINT=${HF_ENDPOINT:-"https://hf-mirror.com"}

# --- Role (required first arg) ---
ROLE="${1:?Usage: run_node.sh <attention|ffn> [master_addr] [master_port] [options]}"
case "$ROLE" in
    attention) RANK=0 ;;
    ffn)       RANK=1 ;;
    *)         echo "Error: role must be 'attention' or 'ffn', got '$ROLE'"; exit 1 ;;
esac

# --- Positional args ---
MASTER_ADDR="${2:-10.244.64.179}"
MASTER_PORT="${3:-29500}"

# --- Config from env ---
MODEL_PATH="${MODEL_PATH:-/data/Qwen/Qwen3-30B-A3B/}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1}"
WORLD_SIZE="${WORLD_SIZE:-2}"
BATCH_SIZE="${BATCH_SIZE:-1}"
LOCAL_RANK="${LOCAL_RANK:-0}"
ATTN_RANK="${ATTN_RANK:-0}"
FFN_RANK="${FFN_RANK:-1}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-32}"

echo "=== ${ROLE^} Node ==="
echo "Model: $MODEL_PATH"
echo "Master: $MASTER_ADDR:$MASTER_PORT"
echo "CUDA_VISIBLE_DEVICES: $CUDA_DEVICES"
echo "Rank: $RANK/$WORLD_SIZE (local_rank=$LOCAL_RANK)"

CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python -m src.main \
    --model-name "$MODEL_PATH" \
    --role "$ROLE" \
    --master-addr "$MASTER_ADDR" \
    --master-port "$MASTER_PORT" \
    --world-size "$WORLD_SIZE" \
    --rank "$RANK" \
    --local-rank "$LOCAL_RANK" \
    --attn-node-rank "$ATTN_RANK" \
    --ffn-node-rank "$FFN_RANK" \
    --batch-size "$BATCH_SIZE" \
    --max-seq-len "$MAX_SEQ_LEN" \
    "${@:4}"
