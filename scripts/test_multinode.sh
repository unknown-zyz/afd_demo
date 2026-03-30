#!/bin/bash
# 多机测试 Qwen3-30B-A3B MoE 模型
# Attention 节点在本地 (GPU 0,1), FFN 节点在远程机器 (GPU 0,1)
# Usage: ./scripts/test_qwen3_moe_multinode.sh [max_tokens] [batch_size]

set -e
cd "$(dirname "$0")/.."
source venv/bin/activate

MAX_TOKENS="${1:-3}"
BATCH_SIZE="${2:-1}"
MODEL_PATH="/data/Qwen/Qwen3-30B-A3B/"

# 网络配置
LOCAL_IP="10.244.64.179"
MASTER_PORT="${MASTER_PORT:-29700}"
REMOTE_HOST="192.168.5.32"
REMOTE_PORT="31310"
REMOTE_KEY="~/.ssh/id_rsa_second"
REMOTE_USER="zyz"

echo "=== Qwen3-30B-A3B MoE 多机测试 ==="
echo "Model: $MODEL_PATH"
echo "Max tokens: $MAX_TOKENS"
echo "Batch size: $BATCH_SIZE"
echo "Master: $LOCAL_IP:$MASTER_PORT"
echo ""
echo "Node 分布:"
echo "  本地 (Attention): $LOCAL_IP, GPU 0,1"
echo "  远程 (FFN): $REMOTE_HOST, GPU 0,1"
echo ""

# 检查远程连接
echo "检查远程连接..."
ssh -o ConnectTimeout=5 -p $REMOTE_PORT -i $REMOTE_KEY $REMOTE_USER@$REMOTE_HOST "echo 'Remote OK'" || {
    echo "Error: Cannot connect to remote host"
    exit 1
}

# 同步代码到远程 (使用 scp 代替 rsync)
echo "同步代码到远程..."
ssh -p $REMOTE_PORT -i $REMOTE_KEY $REMOTE_USER@$REMOTE_HOST "mkdir -p /home/zyz/afd_demo"
scp -r -P $REMOTE_PORT -i $REMOTE_KEY \
    /home/zyz/afd_demo/src /home/zyz/afd_demo/skills \
    $REMOTE_USER@$REMOTE_HOST:/home/zyz/afd_demo/ 2>/dev/null || true

echo ""
echo "GPU Memory (本地):"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv

echo ""
echo "GPU Memory (远程):"
ssh -p $REMOTE_PORT -i $REMOTE_KEY $REMOTE_USER@$REMOTE_HOST \
    "nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv"

echo ""
echo "Starting multi-node test..."

# 在远程启动 FFN 节点 (后台)
echo "启动远程 FFN 节点..."
ssh -p $REMOTE_PORT -i $REMOTE_KEY $REMOTE_USER@$REMOTE_HOST "
cd /home/zyz/afd_demo
source venv/bin/activate
CUDA_VISIBLE_DEVICES=0,1 python -m src.main \
    --model-name '$MODEL_PATH' \
    --role ffn \
    --master-addr '$LOCAL_IP' \
    --master-port '$MASTER_PORT' \
    --world-size 2 \
    --rank 1 \
    --local-rank 0 \
    --attn-node-rank 0 \
    --ffn-node-rank 1 \
    --batch-size $BATCH_SIZE \
    --max-new-tokens $MAX_TOKENS \
    --no-decode-dbo \
    2>&1
" &
FFN_PID=$!
echo "远程 FFN 节点已启动 (本地 PID: $FFN_PID)"

# 等待远程节点初始化
sleep 10

# 在本地启动 Attention 节点 (前台)
echo "启动本地 Attention 节点..."
CUDA_VISIBLE_DEVICES=0,1 python -m src.main \
    --model-name "$MODEL_PATH" \
    --role attention \
    --master-addr "$LOCAL_IP" \
    --master-port "$MASTER_PORT" \
    --world-size 2 \
    --rank 0 \
    --local-rank 0 \
    --attn-node-rank 0 \
    --ffn-node-rank 1 \
    --batch-size $BATCH_SIZE \
    --max-new-tokens $MAX_TOKENS \
    --prompt "Hello" \
    --greedy \
    --no-decode-dbo \
    2>&1

# 等待远程节点完成
wait $FFN_PID 2>/dev/null || true

echo ""
echo "=== 测试完成 ==="
echo "GPU Memory (本地):"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
