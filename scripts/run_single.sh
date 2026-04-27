#!/bin/bash
# 运行单个 DBO 实验配置（支持本地/多机，可选 pipeline 可视化）
#
# Usage: ./scripts/run_single.sh <deployment> <batch> <seq> [options]
#
# Arguments:
#   deployment  - local (单机 4GPU) 或 multinode (多机)
#   batch       - batch size
#   seq         - prefill sequence length
#
# Options:
#   --tokens N      - 生成 token 数 (默认: 5)
#   --no-dbo        - 关闭 DBO (serial baseline)
#   --visualize     - 自动生成 pipeline Gantt 图
#   --verbose       - 详细输出
#   --generate      - 启用自回归生成 (默认: 仅 prefill)
#
# Environment variables:
#   MODEL_PATH      - 模型路径 (默认: /data/Qwen/Qwen3-30B-A3B/)
#   MASTER_PORT     - 通信端口 (默认: 29650)
#
# Examples:
#   ./scripts/run_single.sh local 8 128
#   ./scripts/run_single.sh local 8 128 --no-dbo
#   ./scripts/run_single.sh local 8 128 --visualize --tokens 5
#   ./scripts/run_single.sh multinode 16 128 --tokens 5
#
# Output:
#   results/prefill_dbo/timing_attention_<suffix>.json
#   results/prefill_dbo/timing_ffn_<suffix>.json
#   results/prefill_dbo/dbo_pipeline_<suffix>.png  (with --visualize)

# set -e disabled: errors are handled explicitly
cd "$(dirname "$0")/.."
source venv/bin/activate

show_usage() {
    cat <<'EOF'
Usage: ./scripts/run_single.sh <local|multinode> <batch> <seq> [options]

Arguments:
  deployment        local (single machine) or multinode
  batch             Batch size
  seq               Prefill sequence length

Options:
  --tokens N        Number of generated tokens (default: 5)
  --no-dbo          Disable DBO and run serial baseline
  --visualize       Generate pipeline visualization after the run
  --verbose         Enable verbose Python logging
  --generate        Enable autoregressive decode (default: prefill only)
  --warmup-p2p      Run untimed NCCL P2P warmup before timing
  --warmup-rounds N Number of warmup rounds (default: 3)
  --crosslayer      Enable decode cross-layer pipeline
  -h, --help        Show this help message
EOF
}

# Avoid CUDA memory fragmentation on tight-memory GPUs (V100-32GB)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# NCCL buffer tuning for DBO pipeline:
# - BUFFSIZE=32MB prevents flow-control blocking on A2F sends (MB0 send was
#   blocking 15-24ms/layer when FFN hadn't drained the recv buffer, causing
#   ~925ms waste across 47 layers).  32MB covers tensors up to batch=16,seq=256.
# - NCHANNELS=1 limits per-peer channels to reduce GPU memory overhead.
# Total NCCL buffer memory: ~2 × 32MB = 64MB (0.2% of V100-32GB).
export NCCL_BUFFSIZE=33554432
export NCCL_NCHANNELS_PER_NET_PEER=1

# ── Positional arguments ──────────────────────────────────────────
if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
    show_usage
    exit 0
fi

DEPLOYMENT="$1"
BATCH="$2"
SEQ="$3"
shift 3 2>/dev/null || true

if [ -z "$DEPLOYMENT" ] || [ -z "$BATCH" ] || [ -z "$SEQ" ]; then
    show_usage
    exit 1
fi

# ── Optional flags ────────────────────────────────────────────────
TOKENS=5
NO_DBO=false
VISUALIZE=false
VERBOSE=false
GENERATE=false
WARMUP_P2P=false
WARMUP_ROUNDS=3
CROSSLAYER=false

while [ $# -gt 0 ]; do
    case "$1" in
        --tokens)
            TOKENS="$2"
            shift 2
            ;;
        --no-dbo)
            NO_DBO=true
            shift
            ;;
        --visualize)
            VISUALIZE=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --generate)
            GENERATE=true
            shift
            ;;
        --warmup-p2p)
            WARMUP_P2P=true
            shift
            ;;
        --warmup-rounds)
            WARMUP_ROUNDS="$2"
            shift 2
            ;;
        --crosslayer)
            CROSSLAYER=true
            shift
            ;;
        *)
            echo "ERROR: Unknown option: $1"
            exit 1
            ;;
    esac
done

# ── Configuration ─────────────────────────────────────────────────
MODEL_PATH="${MODEL_PATH:-"/data/Qwen/Qwen3-30B-A3B/"}"
MASTER_PORT="${MASTER_PORT:-29650}"

# SSH 配置 (多机)
REMOTE_HOST="192.168.5.32"
REMOTE_PORT="31310"
REMOTE_KEY="$HOME/.ssh/id_rsa_second"
REMOTE_USER="zyz"
REMOTE_PATH="/home/zyz/afd_demo"

# ── Build suffix & flags ─────────────────────────────────────────
DBO_FLAG=""
if [ "$NO_DBO" = true ]; then
    DBO_FLAG="--no-dbo"
    SUFFIX="serial_${DEPLOYMENT}_b${BATCH}_s${SEQ}_t${TOKENS}"
else
    SUFFIX="${DEPLOYMENT}_b${BATCH}_s${SEQ}_t${TOKENS}"
fi

GENERATE_FLAG="--no-generate"
if [ "$GENERATE" = true ]; then
    GENERATE_FLAG=""
fi

WARMUP_FLAGS=""
if [ "$WARMUP_P2P" = true ]; then
    WARMUP_FLAGS="--warmup-p2p --warmup-rounds $WARMUP_ROUNDS"
    SUFFIX="warmup_${SUFFIX}"
fi

CROSSLAYER_FLAG=""
if [ "$CROSSLAYER" = true ]; then
    CROSSLAYER_FLAG="--crosslayer"
    SUFFIX="${SUFFIX}_crosslayer"
fi

echo "========================================"
echo "  $([ "$NO_DBO" = true ] && echo 'Serial ' || echo '')Experiment: $SUFFIX"
echo "========================================"
echo "Deployment: $DEPLOYMENT"
echo "Batch: $BATCH, Seq: $SEQ, Tokens: $TOKENS"
echo "DBO: $([ "$NO_DBO" = true ] && echo disabled || echo enabled)"
echo "Generate: $([ "$GENERATE" = true ] && echo enabled || echo disabled)"
echo "Visualize: $([ "$VISUALIZE" = true ] && echo yes || echo no)"
echo "Output: results/prefill_dbo/timing_*_${SUFFIX}.json"
echo "========================================"

mkdir -p results/prefill_dbo/logs

# ── run_local() ───────────────────────────────────────────────────
run_local() {
    # Cleanup trap: kill FFN background process if attention exits/crashes
    local FFN_PID=""
    cleanup_ffn() {
        if [ -n "$FFN_PID" ]; then
            kill "$FFN_PID" 2>/dev/null || true
            wait "$FFN_PID" 2>/dev/null || true
        fi
    }
    trap cleanup_ffn EXIT

    # FFN node (background, output to log file)
    CUDA_VISIBLE_DEVICES=2,3 python -u -m src.main \
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
        --verbose $GENERATE_FLAG $DBO_FLAG $WARMUP_FLAGS $CROSSLAYER_FLAG \
        > "results/prefill_dbo/logs/ffn_${SUFFIX}.log" 2>&1 &
    FFN_PID=$!
    sleep 5

    # Attention node (foreground, output to log file)
    CUDA_VISIBLE_DEVICES=0,1 python -u -m src.main \
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
        --verbose $GENERATE_FLAG $DBO_FLAG $WARMUP_FLAGS $CROSSLAYER_FLAG \
        > "results/prefill_dbo/logs/attn_${SUFFIX}.log" 2>&1

    wait $FFN_PID 2>/dev/null || true

    # Show summary from both logs
    echo ""
    echo "[ATTN] === Log tail ==="
    tail -5 "results/prefill_dbo/logs/attn_${SUFFIX}.log" 2>/dev/null | sed 's/^/[ATTN] /'
}

# ── run_multinode() ───────────────────────────────────────────────
run_multinode() {
    local MASTER_ADDR
    MASTER_ADDR="$(hostname -I | awk '{print $1}')"
    echo "[INFO] Master addr: $MASTER_ADDR"
    echo "[INFO] Starting FFN on remote machine..."

    # 远程 FFN 节点 (后台 SSH)
    ssh -p "$REMOTE_PORT" -i "$REMOTE_KEY" "${REMOTE_USER}@${REMOTE_HOST}" \
        "cd $REMOTE_PATH && source venv/bin/activate && \
         CUDA_VISIBLE_DEVICES=2,3 python -u -m src.main \
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
         --verbose $GENERATE_FLAG $DBO_FLAG $CROSSLAYER_FLAG" \
         2>&1 | tee "results/prefill_dbo/logs/ffn_${SUFFIX}.log" | sed 's/^/[FFN] /' &
    REMOTE_PID=$!
    sleep 10

    # 本地 Attention 节点 (前台)
    CUDA_VISIBLE_DEVICES=0,1 python -u -m src.main \
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
        --verbose $GENERATE_FLAG $DBO_FLAG $CROSSLAYER_FLAG \
        2>&1 | tee "results/prefill_dbo/logs/attn_${SUFFIX}.log" | sed 's/^/[ATTN] /'

    wait $REMOTE_PID 2>/dev/null || true

    # 从远程拉取 FFN timing 文件
    echo "[INFO] Fetching FFN timing from remote..."
    scp -P "$REMOTE_PORT" -i "$REMOTE_KEY" \
        "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}/results/prefill_dbo/timing_ffn_${SUFFIX}.json" \
        "results/prefill_dbo/" 2>/dev/null && echo "[INFO] ✓ FFN timing fetched" \
        || echo "[WARN] Failed to fetch FFN timing"
}

# ── Run experiment ────────────────────────────────────────────────
if [ "$DEPLOYMENT" = "local" ]; then
    run_local
elif [ "$DEPLOYMENT" = "multinode" ]; then
    run_multinode
else
    echo "ERROR: Unknown deployment: $DEPLOYMENT (use 'local' or 'multinode')"
    exit 1
fi

# ── OOM detection ─────────────────────────────────────────────────
if grep -q "OutOfMemoryError\|CUDA out of memory\|OOM" "results/prefill_dbo/logs/attn_${SUFFIX}.log" 2>/dev/null || \
   grep -q "OutOfMemoryError\|CUDA out of memory\|OOM" "results/prefill_dbo/logs/ffn_${SUFFIX}.log" 2>/dev/null; then
    echo ""
    echo "⚠️  OOM DETECTED for $SUFFIX"
    exit 2
fi

# ── Timing file check ────────────────────────────────────────────
ATTN_TIMING="results/prefill_dbo/timing_attention_${SUFFIX}.json"
FFN_TIMING="results/prefill_dbo/timing_ffn_${SUFFIX}.json"

if [ ! -f "$ATTN_TIMING" ] || [ ! -f "$FFN_TIMING" ]; then
    echo "⚠️  Missing timing files for $SUFFIX"
    ls -la results/prefill_dbo/timing_*${SUFFIX}* 2>/dev/null
    # Non-fatal for serial mode (timing may be minimal)
    if [ "$NO_DBO" = false ]; then
        exit 1
    fi
fi

# ── Visualization (DBO only) ─────────────────────────────────────
if [ "$VISUALIZE" = true ] && [ "$NO_DBO" = false ]; then
    echo ""
    echo "[INFO] Generating visualization..."
    python scripts/visualize_dbo_pipeline.py \
        --attn-timing "$ATTN_TIMING" \
        --ffn-timing "$FFN_TIMING" \
        --output "results/prefill_dbo/dbo_pipeline_${SUFFIX}.png" \
        --start-layer 1 --num-layers 4
    echo "   Plot: results/prefill_dbo/dbo_pipeline_${SUFFIX}.png"
fi

# ── Auto-generate markdown report ────────────────────────────────
if [ -f "$ATTN_TIMING" ] && [ -f "$FFN_TIMING" ]; then
    REPORT="results/prefill_dbo/report_${SUFFIX}.md"
    if [ "$NO_DBO" = true ]; then MODE_TAG="serial"
    elif [ "$GENERATE" = false ]; then MODE_TAG="prefill-dbo"
    elif [ "$CROSSLAYER" = true ]; then MODE_TAG="decode-dbo-crosslayer"
    else MODE_TAG="decode-dbo"
    fi
    CACHE="results/serial/cache/b${BATCH}_s${SEQ}_t${TOKENS}.json"
    CMP=""
    [ -f "$CACHE" ] && [ "$NO_DBO" = false ] && CMP="--serial-baseline $CACHE"
    python scripts/gen_experiment_report.py \
        --attn-timing "$ATTN_TIMING" --ffn-timing "$FFN_TIMING" \
        --output "$REPORT" --mode "$MODE_TAG" \
        --batch "$BATCH" --seq "$SEQ" --tokens "$TOKENS" $CMP \
        && echo "   Report: $REPORT" || echo "[WARN] report generation failed"
fi

echo ""
echo "✅ Experiment completed: $SUFFIX"
echo "   Timing: $ATTN_TIMING"
echo "   Timing: $FFN_TIMING"
