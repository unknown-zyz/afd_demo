#!/bin/bash
# Run DBO demo locally (single machine, multi-GPU)
# Usage: ./run_local.sh [options]
#   ./run_local.sh                          # default async DBO
#   ./run_local.sh --scheduler compare      # compare sync vs async
#   ./run_local.sh --timing                 # enable detailed timing

set -e
cd "$(dirname "$0")/.."
source venv/bin/activate

# Use Chinese HuggingFace mirror
export HF_ENDPOINT=${HF_ENDPOINT:-"https://hf-mirror.com"}

MODEL_NAME=${MODEL_NAME:-"/data/Qwen/Qwen3-30B-A3B/"}
MASTER_PORT=${MASTER_PORT:-"29500"}
WORLD_SIZE=${WORLD_SIZE:-2}
CUDA_DEVICES=${CUDA_DEVICES:-"0,1"}
BATCH_SIZE=${BATCH_SIZE:-1}
NUM_MB=${NUM_MB:-2}
MAX_SEQ_LEN=${MAX_SEQ_LEN:-32}
ATTN_RANK=${ATTN_RANK:-0}
FFN_RANK=${FFN_RANK:-1}

echo "=== AFD Demo (DBO) ==="
echo "Model: $MODEL_NAME"
echo "CUDA_VISIBLE_DEVICES: $CUDA_DEVICES"
echo "WorldSize: $WORLD_SIZE, Batch: $BATCH_SIZE, MicroBatches: $NUM_MB, MaxSeqLen: $MAX_SEQ_LEN"
echo "Tip: 4 卡分角色测试请用 ./scripts/run_local_4gpu.sh"

CUDA_VISIBLE_DEVICES=$CUDA_DEVICES torchrun \
    --nproc_per_node=$WORLD_SIZE \
    --master_port=$MASTER_PORT \
    -m src.main \
    --model-name "$MODEL_NAME" \
    --local-test \
    --world-size $WORLD_SIZE \
    --batch-size $BATCH_SIZE \
    --max-seq-len $MAX_SEQ_LEN \
    --num-micro-batches $NUM_MB \
    --attn-node-rank $ATTN_RANK \
    --ffn-node-rank $FFN_RANK \
    "$@"
