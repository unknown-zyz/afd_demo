#!/bin/bash
# Run local 4-GPU test (2 GPUs for attention role, 2 GPUs for FFN role).
#
# This launches two independent processes instead of torchrun:
# - rank 0 (attention) on GPUs 0,1
# - rank 1 (ffn) on GPUs 2,3

set -euo pipefail
cd "$(dirname "$0")/.."
source venv/bin/activate

export HF_ENDPOINT=${HF_ENDPOINT:-"https://hf-mirror.com"}

MODEL_NAME=${MODEL_NAME:-"/data/Qwen/Qwen3-30B-A3B/"}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-"29630"}
BATCH_SIZE=${BATCH_SIZE:-1}
MAX_SEQ_LEN=${MAX_SEQ_LEN:-32}
NUM_MB=${NUM_MB:-2}

COMMON_ARGS=(
  --model-name "$MODEL_NAME"
  --world-size 2
  --attn-node-rank 0
  --ffn-node-rank 1
  --master-addr "$MASTER_ADDR"
  --master-port "$MASTER_PORT"
  --batch-size "$BATCH_SIZE"
  --max-seq-len "$MAX_SEQ_LEN"
  --num-micro-batches "$NUM_MB"
)

echo "=== Local 4-GPU AFD ==="
echo "Model: $MODEL_NAME"
echo "Master: $MASTER_ADDR:$MASTER_PORT"
echo "Attention GPUs: 0,1 | FFN GPUs: 2,3"

CUDA_VISIBLE_DEVICES=2,3 python -m src.main \
  --role ffn \
  --rank 1 \
  --local-rank 0 \
  "${COMMON_ARGS[@]}" \
  "$@" &
FFN_PID=$!

sleep 3

CUDA_VISIBLE_DEVICES=0,1 python -m src.main \
  --role attention \
  --rank 0 \
  --local-rank 0 \
  "${COMMON_ARGS[@]}" \
  "$@"

wait "$FFN_PID"
