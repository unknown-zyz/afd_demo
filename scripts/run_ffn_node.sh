#!/bin/bash
# Run FFN node for multi-machine AFD Demo (supports multi-GPU per role)
# Usage: ./run_ffn_node.sh [master_addr] [master_port] [options]

set -e
cd "$(dirname "$0")/.."
source venv/bin/activate

# Use Chinese HuggingFace mirror
export HF_ENDPOINT=${HF_ENDPOINT:-"https://hf-mirror.com"}

MODEL_NAME=${MODEL_NAME:-"/data/Qwen/Qwen3-30B-A3B/"}
MASTER_ADDR=${1:-"10.244.64.173"}
MASTER_PORT=${2:-"29500"}
WORLD_SIZE=${WORLD_SIZE:-2}
RANK=${RANK:-1}
LOCAL_RANK=${LOCAL_RANK:-0}
ATTN_RANK=${ATTN_RANK:-0}
FFN_RANK=${FFN_RANK:-1}
BATCH_SIZE=${BATCH_SIZE:-1}
MAX_SEQ_LEN=${MAX_SEQ_LEN:-32}
NUM_MB=${NUM_MB:-2}
CUDA_DEVICES=${CUDA_DEVICES:-"0,1"}

echo "=== FFN Node ==="
echo "Model: $MODEL_NAME"
echo "Master: $MASTER_ADDR:$MASTER_PORT"
echo "CUDA_VISIBLE_DEVICES: $CUDA_DEVICES"
echo "Rank: $RANK/$WORLD_SIZE (local_rank=$LOCAL_RANK), attn_rank=$ATTN_RANK, ffn_rank=$FFN_RANK"

export RANK=$RANK
export LOCAL_RANK=$LOCAL_RANK
export WORLD_SIZE=$WORLD_SIZE
export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=$MASTER_PORT

CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python -m src.main \
    --model-name "$MODEL_NAME" \
    --role ffn \
    --master-addr "$MASTER_ADDR" \
    --master-port "$MASTER_PORT" \
    --world-size "$WORLD_SIZE" \
    --rank "$RANK" \
    --local-rank "$LOCAL_RANK" \
    --attn-node-rank "$ATTN_RANK" \
    --ffn-node-rank "$FFN_RANK" \
    --batch-size $BATCH_SIZE \
    --max-seq-len $MAX_SEQ_LEN \
    --num-micro-batches $NUM_MB \
    "${@:3}"
