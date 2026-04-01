#!/bin/bash
# Batch Scaling Experiments: 自动化测试不同 batch/seq 配置
# Usage: ./scripts/batch_scaling_experiments.sh [phase]
#
# Phases:
#   batch    - Phase 1: Batch size scaling (seq=128)
#   seq      - Phase 2: Sequence length scaling (batch=8)
#   combined - Phase 3: High load combinations
#   all      - Run all phases
#
# 自动检测 OOM 并停止，生成可视化和日志

set -e
cd "$(dirname "$0")/.."

PHASE="${1:-all}"
MODEL_PATH="/data/Qwen/Qwen3-30B-A3B/"
MASTER_PORT=29650
TOKENS=5

# SSH 配置
REMOTE_HOST="192.168.5.32"
REMOTE_PORT="31310"
REMOTE_KEY="$HOME/.ssh/id_rsa_second"
REMOTE_USER="zyz"
REMOTE_PATH="/home/zyz/afd_demo"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查 OOM
check_oom() {
    local log_file="$1"
    if grep -q "OutOfMemoryError\|CUDA out of memory\|OOM" "$log_file"; then
        return 0  # OOM detected
    fi
    return 1  # No OOM
}

# 运行单机测试
run_local_test() {
    local batch=$1
    local seq=$2
    local suffix="local_b${batch}_s${seq}_t${TOKENS}"
    
    log_info "Testing LOCAL: batch=$batch, seq=$seq, tokens=$TOKENS"
    
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
        --batch-size "$batch" \
        --prefill-seq-len "$seq" \
        --max-new-tokens "$TOKENS" \
        --timing \
        --timing-suffix "$suffix" \
        --no-generate \
        --verbose \
        2>&1 | tee "results/prefill_dbo/logs/ffn_${suffix}.log" &
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
        --batch-size "$batch" \
        --prefill-seq-len "$seq" \
        --max-new-tokens "$TOKENS" \
        --prompt "Hello world, this is a test prompt for batch scaling experiments." \
        --timing \
        --timing-suffix "$suffix" \
        --verbose \
        2>&1 | tee "results/prefill_dbo/logs/attn_${suffix}.log"
    
    # 等待 FFN 完成
    wait $FFN_PID 2>/dev/null || true
    
    # 检查 OOM
    if check_oom "results/prefill_dbo/logs/attn_${suffix}.log" || \
       check_oom "results/prefill_dbo/logs/ffn_${suffix}.log"; then
        log_error "OOM detected for batch=$batch, seq=$seq"
        return 1
    fi
    
    # 生成可视化 (跳过 Layer 0)
    log_info "Generating visualization..."
    python scripts/visualize_dbo_pipeline.py \
        --attn-timing "results/prefill_dbo/timing_attention_${suffix}.json" \
        --ffn-timing "results/prefill_dbo/timing_ffn_${suffix}.json" \
        --output "results/prefill_dbo/dbo_pipeline_${suffix}.png" \
        --start-layer 1 --num-layers 5 || log_warn "Visualization failed"
    
    log_info "✓ Completed: $suffix"
    return 0
}

# 运行多机测试
run_multinode_test() {
    local batch=$1
    local seq=$2
    local suffix="multinode_b${batch}_s${seq}_t${TOKENS}"
    
    log_info "Testing MULTINODE: batch=$batch, seq=$seq, tokens=$TOKENS"
    
    # 启动远程 FFN 节点
    log_info "Starting FFN node on remote machine..."
    ssh -p "$REMOTE_PORT" -i "$REMOTE_KEY" "${REMOTE_USER}@${REMOTE_HOST}" \
        "cd $REMOTE_PATH && source venv/bin/activate && \
         CUDA_VISIBLE_DEVICES=2,3 python -m src.main \
         --model-name \"$MODEL_PATH\" \
         --role ffn \
         --master-addr 10.244.64.179 \
         --master-port $MASTER_PORT \
         --world-size 2 \
         --rank 1 \
         --local-rank 0 \
         --attn-node-rank 0 \
         --ffn-node-rank 1 \
         --batch-size $batch \
         --prefill-seq-len $seq \
         --max-new-tokens $TOKENS \
         --timing \
         --timing-suffix \"$suffix\" \
         --no-generate \
         --verbose \
         2>&1 | tee results/prefill_dbo/logs/ffn_${suffix}.log" &
    REMOTE_PID=$!
    sleep 10
    
    # 启动本地 Attention 节点
    CUDA_VISIBLE_DEVICES=0,1 python -m src.main \
        --model-name "$MODEL_PATH" \
        --role attention \
        --master-addr 10.244.64.179 \
        --master-port "$MASTER_PORT" \
        --world-size 2 \
        --rank 0 \
        --local-rank 0 \
        --attn-node-rank 0 \
        --ffn-node-rank 1 \
        --batch-size "$batch" \
        --prefill-seq-len "$seq" \
        --max-new-tokens "$TOKENS" \
        --prompt "Hello world, this is a test prompt for batch scaling experiments." \
        --timing \
        --timing-suffix "$suffix" \
        --verbose \
        2>&1 | tee "results/prefill_dbo/logs/attn_${suffix}.log"
    
    # 等待远程完成
    wait $REMOTE_PID 2>/dev/null || true
    
    # 检查 OOM
    if check_oom "results/prefill_dbo/logs/attn_${suffix}.log"; then
        log_error "OOM detected for batch=$batch, seq=$seq"
        return 1
    fi
    
    # 从远程拉取 FFN timing
    scp -P "$REMOTE_PORT" -i "$REMOTE_KEY" \
        "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}/results/prefill_dbo/timing_ffn_${suffix}.json" \
        "results/prefill_dbo/" || log_warn "Failed to fetch FFN timing"
    
    # 生成可视化
    log_info "Generating visualization..."
    python scripts/visualize_dbo_pipeline.py \
        --attn-timing "results/prefill_dbo/timing_attention_${suffix}.json" \
        --ffn-timing "results/prefill_dbo/timing_ffn_${suffix}.json" \
        --output "results/prefill_dbo/dbo_pipeline_${suffix}.png" \
        --start-layer 1 --num-layers 5 || log_warn "Visualization failed"
    
    log_info "✓ Completed: $suffix"
    return 0
}

# Phase 1: Batch size scaling
run_batch_scaling() {
    log_info "=== Phase 1: Batch Size Scaling (seq=128) ==="
    local seq=128
    
    for batch in 4 8 16 32 64; do
        log_info "--- Batch $batch ---"
        
        # Local test
        if ! run_local_test $batch $seq; then
            log_error "LOCAL test failed at batch=$batch, stopping phase 1"
            break
        fi
        
        # Multinode test
        if ! run_multinode_test $batch $seq; then
            log_error "MULTINODE test failed at batch=$batch, stopping phase 1"
            break
        fi
    done
}

# Phase 2: Sequence length scaling
run_seq_scaling() {
    log_info "=== Phase 2: Sequence Length Scaling (batch=8) ==="
    local batch=8
    
    for seq in 32 64 128 256 512; do
        log_info "--- Seq $seq ---"
        
        # Local test
        if ! run_local_test $batch $seq; then
            log_error "LOCAL test failed at seq=$seq, stopping phase 2"
            break
        fi
        
        # Multinode test
        if ! run_multinode_test $batch $seq; then
            log_error "MULTINODE test failed at seq=$seq, stopping phase 2"
            break
        fi
    done
}

# Phase 3: High load combinations
run_combined() {
    log_info "=== Phase 3: High Load Combinations ==="
    
    local configs=(
        "8 256"
        "16 128"
        "32 64"
    )
    
    for config in "${configs[@]}"; do
        read batch seq <<< "$config"
        log_info "--- Batch $batch, Seq $seq ---"
        
        # Local test
        if ! run_local_test $batch $seq; then
            log_error "LOCAL test failed at (batch=$batch, seq=$seq)"
            continue
        fi
        
        # Multinode test
        if ! run_multinode_test $batch $seq; then
            log_error "MULTINODE test failed at (batch=$batch, seq=$seq)"
            continue
        fi
    done
}

# Main
main() {
    source venv/bin/activate
    
    # 创建日志目录
    mkdir -p results/prefill_dbo/logs
    
    log_info "========================================="
    log_info "  Batch Scaling Experiments"
    log_info "========================================="
    log_info "Phase: $PHASE"
    log_info "Model: $MODEL_PATH"
    log_info "Tokens: $TOKENS"
    log_info ""
    
    case "$PHASE" in
        batch)
            run_batch_scaling
            ;;
        seq)
            run_seq_scaling
            ;;
        combined)
            run_combined
            ;;
        all)
            run_batch_scaling
            run_seq_scaling
            run_combined
            ;;
        *)
            log_error "Unknown phase: $PHASE"
            log_info "Usage: $0 [batch|seq|combined|all]"
            exit 1
            ;;
    esac
    
    log_info ""
    log_info "========================================="
    log_info "  Experiments Completed"
    log_info "========================================="
}

main "$@"
