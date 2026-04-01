#!/bin/bash
# 运行单个 DBO 实验配置
# Usage: ./scripts/run_experiment.sh <deployment> <batch> <seq> [tokens]
#
# 参数:
#   deployment  - local (单机) 或 multinode (多机)
#   batch       - batch size
#   seq         - prefill sequence length
#   tokens      - 生成 token 数 (默认: 5)
#
# 示例:
#   ./scripts/run_experiment.sh local 8 128
#   ./scripts/run_experiment.sh multinode 16 128 5
#
# 产出:
#   results/prefill_dbo/timing_attention_<suffix>.json
#   results/prefill_dbo/timing_ffn_<suffix>.json
#   results/prefill_dbo/dbo_pipeline_<suffix>.png

set -e
cd "$(dirname "$0")/.."
source venv/bin/activate

DEPLOYMENT="$1"
BATCH="$2"
SEQ="$3"
TOKENS="${4:-5}"
NO_DBO="${5:-}"  # Pass "nodbo" as 5th arg for serial baseline

if [ -z "$DEPLOYMENT" ] || [ -z "$BATCH" ] || [ -z "$SEQ" ]; then
    echo "Usage: $0 <local|multinode> <batch> <seq> [tokens] [nodbo]"
    exit 1
fi

MODEL_PATH="/data/Qwen/Qwen3-30B-A3B/"
MASTER_PORT="${MASTER_PORT:-29650}"

# SSH 配置 (多机)
REMOTE_HOST="192.168.5.32"
REMOTE_PORT="31310"
REMOTE_KEY="$HOME/.ssh/id_rsa_second"
REMOTE_USER="zyz"
REMOTE_PATH="/home/zyz/afd_demo"

DBO_FLAG=""
if [ "$NO_DBO" = "nodbo" ]; then
    DBO_FLAG="--no-dbo"
    SUFFIX="serial_${DEPLOYMENT}_b${BATCH}_s${SEQ}_t${TOKENS}"
else
    SUFFIX="${DEPLOYMENT}_b${BATCH}_s${SEQ}_t${TOKENS}"
fi

echo "========================================"
echo "  ${NO_DBO:+Serial }Experiment: $SUFFIX"
echo "========================================"
echo "Deployment: $DEPLOYMENT"
echo "Batch: $BATCH, Seq: $SEQ, Tokens: $TOKENS"
echo "DBO: ${NO_DBO:+disabled}${NO_DBO:-enabled}"
echo "Output: results/prefill_dbo/timing_*_${SUFFIX}.json"
echo "========================================"

mkdir -p results/prefill_dbo/logs

run_local() {
    # FFN 节点 (后台)
    CUDA_VISIBLE_DEVICES=2,3 python -m src.main \
        --model-name "$MODEL_PATH" \
        --role ffn \
        --master-addr 127.0.0.1 \
        --master-port "$MASTER_PORT" \
        --world-size 2 --rank 1 --local-rank 0 \
        --attn-node-rank 0 --ffn-node-rank 1 \
        --batch-size "$BATCH" \
        --prefill-seq-len "$SEQ" \
        --max-new-tokens "$TOKENS" \
        --timing --timing-suffix "$SUFFIX" \
        --no-generate --verbose $DBO_FLAG \
        2>&1 | tee "results/prefill_dbo/logs/ffn_${SUFFIX}.log" | sed 's/^/[FFN] /' &
    FFN_PID=$!
    sleep 5

    # Attention 节点 (前台)
    CUDA_VISIBLE_DEVICES=0,1 python -m src.main \
        --model-name "$MODEL_PATH" \
        --role attention \
        --master-addr 127.0.0.1 \
        --master-port "$MASTER_PORT" \
        --world-size 2 --rank 0 --local-rank 0 \
        --attn-node-rank 0 --ffn-node-rank 1 \
        --batch-size "$BATCH" \
        --prefill-seq-len "$SEQ" \
        --max-new-tokens "$TOKENS" \
        --prompt "Hello world, this is a test prompt for batch scaling experiments with a longer text." \
        --timing --timing-suffix "$SUFFIX" \
        --no-generate --verbose $DBO_FLAG \
        2>&1 | tee "results/prefill_dbo/logs/attn_${SUFFIX}.log" | sed 's/^/[ATTN] /'
    
    wait $FFN_PID 2>/dev/null || true
}

run_multinode() {
    local MASTER_ADDR
    MASTER_ADDR="$(hostname -I | awk '{print $1}')"
    echo "[INFO] Master addr: $MASTER_ADDR"
    echo "[INFO] Starting FFN on remote machine..."

    # 远程 FFN 节点 (后台 SSH)
    ssh -p "$REMOTE_PORT" -i "$REMOTE_KEY" "${REMOTE_USER}@${REMOTE_HOST}" \
        "cd $REMOTE_PATH && source venv/bin/activate && \
         CUDA_VISIBLE_DEVICES=2,3 python -m src.main \
         --model-name '$MODEL_PATH' \
         --role ffn \
         --master-addr $MASTER_ADDR \
         --master-port $MASTER_PORT \
         --world-size 2 --rank 1 --local-rank 0 \
         --attn-node-rank 0 --ffn-node-rank 1 \
         --batch-size $BATCH \
         --prefill-seq-len $SEQ \
         --max-new-tokens $TOKENS \
         --timing --timing-suffix '$SUFFIX' \
         --no-generate --verbose $DBO_FLAG" \
         2>&1 | tee "results/prefill_dbo/logs/ffn_${SUFFIX}.log" | sed 's/^/[FFN] /' &
    REMOTE_PID=$!
    sleep 10

    # 本地 Attention 节点 (前台)
    CUDA_VISIBLE_DEVICES=0,1 python -m src.main \
        --model-name "$MODEL_PATH" \
        --role attention \
        --master-addr "$MASTER_ADDR" \
        --master-port "$MASTER_PORT" \
        --world-size 2 --rank 0 --local-rank 0 \
        --attn-node-rank 0 --ffn-node-rank 1 \
        --batch-size "$BATCH" \
        --prefill-seq-len "$SEQ" \
        --max-new-tokens "$TOKENS" \
        --prompt "Hello world, this is a test prompt for batch scaling experiments with a longer text." \
        --timing --timing-suffix "$SUFFIX" \
        --no-generate --verbose $DBO_FLAG \
        2>&1 | tee "results/prefill_dbo/logs/attn_${SUFFIX}.log" | sed 's/^/[ATTN] /'

    wait $REMOTE_PID 2>/dev/null || true

    # 从远程拉取 FFN timing 文件
    echo "[INFO] Fetching FFN timing from remote..."
    scp -P "$REMOTE_PORT" -i "$REMOTE_KEY" \
        "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}/results/prefill_dbo/timing_ffn_${SUFFIX}.json" \
        "results/prefill_dbo/" 2>/dev/null && echo "[INFO] ✓ FFN timing fetched" \
        || echo "[WARN] Failed to fetch FFN timing"
}

# 运行实验
if [ "$DEPLOYMENT" = "local" ]; then
    run_local
elif [ "$DEPLOYMENT" = "multinode" ]; then
    run_multinode
else
    echo "ERROR: Unknown deployment: $DEPLOYMENT (use 'local' or 'multinode')"
    exit 1
fi

# 检查 OOM
if grep -q "OutOfMemoryError\|CUDA out of memory\|OOM" "results/prefill_dbo/logs/attn_${SUFFIX}.log" 2>/dev/null || \
   grep -q "OutOfMemoryError\|CUDA out of memory\|OOM" "results/prefill_dbo/logs/ffn_${SUFFIX}.log" 2>/dev/null; then
    echo ""
    echo "⚠️  OOM DETECTED for $SUFFIX"
    exit 2
fi

# 检查 timing 文件是否生成
ATTN_TIMING="results/prefill_dbo/timing_attention_${SUFFIX}.json"
FFN_TIMING="results/prefill_dbo/timing_ffn_${SUFFIX}.json"

if [ ! -f "$ATTN_TIMING" ] || [ ! -f "$FFN_TIMING" ]; then
    echo "⚠️  Missing timing files for $SUFFIX"
    ls -la results/prefill_dbo/timing_*${SUFFIX}* 2>/dev/null
    exit 1
fi

# 生成可视化 (跳过 Layer 0, 仅 DBO 模式)
if [ "$NO_DBO" != "nodbo" ]; then
    echo ""
    echo "[INFO] Generating visualization..."
    python scripts/visualize_dbo_pipeline.py \
        --attn-timing "$ATTN_TIMING" \
        --ffn-timing "$FFN_TIMING" \
        --output "results/prefill_dbo/dbo_pipeline_${SUFFIX}.png" \
        --start-layer 1 --num-layers 4
    echo "   Plot:   results/prefill_dbo/dbo_pipeline_${SUFFIX}.png"
fi

echo ""
echo "✅ Experiment completed: $SUFFIX"
echo "   Timing: $ATTN_TIMING"
echo "   Timing: $FFN_TIMING"
