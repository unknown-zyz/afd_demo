#!/bin/bash
# 运行全套 DBO 实验：DBO + Serial 对比，单机 + 多机
# Usage: ./scripts/run_all_experiments.sh [local|multinode|all]
set -e
cd "$(dirname "$0")/.."
source venv/bin/activate

MODE="${1:-all}"
BASE_PORT=29660

run_one() {
    local deploy="$1" batch="$2" seq="$3" tokens="${4:-5}" nodbo="${5:-}"
    local port=$BASE_PORT
    BASE_PORT=$((BASE_PORT + 1))
    echo ""
    echo "================================================================"
    echo "  Running: deploy=$deploy batch=$batch seq=$seq tokens=$tokens ${nodbo:+[SERIAL]}"
    echo "================================================================"
    MASTER_PORT=$port ./scripts/run_experiment.sh "$deploy" "$batch" "$seq" "$tokens" "$nodbo" || {
        local rc=$?
        if [ $rc -eq 2 ]; then
            echo "⚠️  OOM at batch=$batch seq=$seq — skipping larger configs"
            return 2
        fi
        echo "⚠️  Error (rc=$rc) — continuing"
        return 1
    }
}

run_local_suite() {
    echo "==========================================="
    echo "  Phase 1: Local Batch Scaling (seq=128)"
    echo "==========================================="
    for b in 4 8 16 32 64; do
        run_one local "$b" 128 || { [ $? -eq 2 ] && break; }
    done

    echo "==========================================="
    echo "  Phase 2: Local Serial Baselines"
    echo "==========================================="
    for b in 4 8 16 32; do
        run_one local "$b" 128 5 nodbo || true
    done

    echo "==========================================="
    echo "  Phase 3: Local Seq Scaling (batch=8)"
    echo "==========================================="
    for s in 32 64 128 256 512; do
        run_one local 8 "$s" || { [ $? -eq 2 ] && break; }
    done
}

run_multinode_suite() {
    echo "==========================================="
    echo "  Phase 4: Multinode Batch Scaling (seq=128)"
    echo "==========================================="
    for b in 4 8 16 32 64; do
        run_one multinode "$b" 128 || { [ $? -eq 2 ] && break; }
    done

    echo "==========================================="
    echo "  Phase 5: Multinode Serial Baselines"
    echo "==========================================="
    for b in 4 8 16 32; do
        run_one multinode "$b" 128 5 nodbo || true
    done

    echo "==========================================="
    echo "  Phase 6: Multinode Seq Scaling (batch=8)"
    echo "==========================================="
    for s in 32 64 128 256 512; do
        run_one multinode 8 "$s" || { [ $? -eq 2 ] && break; }
    done
}

case "$MODE" in
    local)      run_local_suite ;;
    multinode)  run_multinode_suite ;;
    all)        run_local_suite; run_multinode_suite ;;
    *)          echo "Usage: $0 [local|multinode|all]"; exit 1 ;;
esac

echo ""
echo "============================================"
echo "  All experiments complete!"
echo "  Results in: results/prefill_dbo/"
echo "============================================"
ls results/prefill_dbo/timing_*.json 2>/dev/null | wc -l | xargs -I{} echo "  {} timing files generated"
