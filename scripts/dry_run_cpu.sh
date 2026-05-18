#!/bin/bash
# CPU dry-run validation for the NPU branch.
#
# Purpose: verify the code path end-to-end on a plain CPU host (no CUDA, no
# NPU hardware required). Uses a tiny model, max_new_tokens=1, backend=cpu.
# Success criteria:
#   - Imports succeed (no torch_npu on the import path).
#   - Device abstraction picks cpu / gloo.
#   - Distributed init_process_group works.
#   - Forward through at least one decode step without crashing.
#
# Does NOT validate: performance, numerical parity, NPU-specific behaviour.

set -eu
cd "$(dirname "$0")/.."

MODEL="${MODEL:-Qwen/Qwen2-0.5B-Instruct}"
# The dry-run forces a 2-rank setup (1 attn + 1 ffn) which is the minimum
# topology the AFD split supports. It intentionally does not exercise TP.
WORLD_SIZE=2
MASTER_ADDR=127.0.0.1
MASTER_PORT="${MASTER_PORT:-29700}"

if [ -f venv/bin/activate ]; then source venv/bin/activate; fi

mkdir -p results/logs
LOG_DIR="results/logs/dry_run_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
echo "=== CPU dry-run: backend=cpu model=$MODEL world=$WORLD_SIZE ==="
echo "Logs: $LOG_DIR"

PIDS=()
for R in 0 1; do
    ROLE=$([ $R -eq 0 ] && echo attention || echo ffn)
    (
        RANK=$R LOCAL_RANK=0 WORLD_SIZE=$WORLD_SIZE \
        MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT \
        python -u -m src.main \
            --backend cpu \
            --role "$ROLE" \
            --world-size $WORLD_SIZE \
            --rank $R --local-rank 0 \
            --attn-node-rank 0 --ffn-node-rank 1 \
            --model-name "$MODEL" \
            --dtype float32 \
            --batch-size 1 --max-seq-len 16 --prefill-seq-len 8 \
            --max-new-tokens 1 \
            --no-dbo \
            --master-addr "$MASTER_ADDR" --master-port "$MASTER_PORT" \
            > "$LOG_DIR/rank${R}.log" 2>&1
    ) &
    PIDS+=($!)
done

rc=0
for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then rc=1; fi
done

echo ""
echo "=== dry-run result: exit=$rc ==="
if [ "$rc" = 0 ]; then
    echo "PASS: CPU path works end-to-end"
else
    echo "FAIL: See $LOG_DIR/rank*.log"
    echo "--- rank0 tail ---"
    tail -40 "$LOG_DIR/rank0.log" || true
    echo "--- rank1 tail ---"
    tail -40 "$LOG_DIR/rank1.log" || true
fi
exit $rc
