#!/bin/bash
#
# Decode DBO Performance Benchmark Script
#
# Tests different batch sizes, sequence lengths, and DBO configurations
# Monitors memory usage and saves detailed results
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="$PROJECT_ROOT/results/decode_benchmark"

mkdir -p "$RESULTS_DIR"

# Configuration
MODEL=${MODEL:-"Qwen/Qwen2-1.5B"}
GPUS=${CUDA_VISIBLE_DEVICES:-"0,1,2,3"}
NUM_GPUS=$(echo "$GPUS" | tr ',' '\n' | wc -l)

# Test configurations
BATCH_SIZES=(2 4 8)
SEQ_LENGTHS=(128 512)
MAX_TOKENS=50

echo "=============================================="
echo "Decode DBO Benchmark"
echo "=============================================="
echo "Model: $MODEL"
echo "GPUs: $GPUS (${NUM_GPUS} total)"
echo "Results: $RESULTS_DIR"
echo "=============================================="
echo ""

# Activate venv
source "$PROJECT_ROOT/venv/bin/activate"

# Function to run benchmark
run_benchmark() {
    local batch=$1
    local seq_len=$2
    local dbo=$3
    local dbo_flag=""
    
    if [ "$dbo" = "off" ]; then
        dbo_flag="--no-dbo"
    fi
    
    local name="${MODEL##*/}_batch${batch}_seq${seq_len}_dbo_${dbo}"
    local log_file="$RESULTS_DIR/${name}.log"
    
    echo "----------------------------------------"
    echo "Testing: batch=$batch, seq_len=$seq_len, DBO=$dbo"
    echo "Log: $log_file"
    
    # Generate prompt of desired length
    local prompt=$(python -c "print('Test ' * ($seq_len // 5))")
    
    # Run test
    CUDA_VISIBLE_DEVICES=$GPUS torchrun --nproc_per_node=2 \
        -m src.main \
        --local-test \
        --model-name "$MODEL" \
        --batch-size $batch \
        --max-new-tokens $MAX_TOKENS \
        --prompt "$prompt" \
        --num-micro-batches 2 \
        --timing \
        $dbo_flag \
        > "$log_file" 2>&1
    
    # Extract throughput
    local throughput=$(grep "Generated.*tokens in" "$log_file" | grep -oP '\(\K[0-9.]+(?= tok/s)' || echo "N/A")
    local time_ms=$(grep "Generated.*tokens in" "$log_file" | grep -oP 'in \K[0-9.]+(?=ms)' || echo "N/A")
    
    echo "Result: $time_ms ms, $throughput tok/s"
    echo ""
    
    # Save to summary
    echo "$name,$batch,$seq_len,$dbo,$time_ms,$throughput" >> "$RESULTS_DIR/summary.csv"
}

# Initialize summary file
echo "name,batch_size,seq_len,dbo,time_ms,throughput_tok_s" > "$RESULTS_DIR/summary.csv"

# Run benchmarks
for batch in "${BATCH_SIZES[@]}"; do
    for seq_len in "${SEQ_LENGTHS[@]}"; do
        # Test with DBO
        if ! run_benchmark $batch $seq_len "on"; then
            echo "⚠ Test failed (OOM?), skipping remaining tests with these params"
            continue
        fi
        
        # Test without DBO
        if ! run_benchmark $batch $seq_len "off"; then
            echo "⚠ Test failed (OOM?)"
        fi
        
        # Give system time to recover
        sleep 2
    done
done

echo "=============================================="
echo "Benchmark Complete!"
echo "Results saved to: $RESULTS_DIR"
echo "=============================================="
echo ""
echo "Summary:"
cat "$RESULTS_DIR/summary.csv"
