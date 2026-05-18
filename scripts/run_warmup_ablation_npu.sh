#!/bin/bash
# Run NPU warmup ablation variants for decode-dbo.
#
# Defaults are intentionally small and focused on the L0 decode question:
#   batches={4,16,64}, seq=512, mode=decode-dbo, preset=npu-ep7.
#
# Override with environment variables, for example:
#   BATCHES=2,4,8,16,32,64,128 SEQS=512 DRY_RUN=true bash scripts/run_warmup_ablation_npu.sh

set -eu
cd "$(dirname "$0")/.."

OUT_ROOT="${OUT_ROOT:-results_npu_ep7/warmup_ablation}"
MODES="${MODES:-decode-dbo}"
BATCHES="${BATCHES:-4,16,64}"
SEQS="${SEQS:-512}"
TOKENS="${TOKENS:-20}"
PRESET="${PRESET:-npu-ep7}"
FFN_EP_BACKEND="${FFN_EP_BACKEND:-broadcast_reduce_overlap}"
EP_EXPERT_POLICY="${EP_EXPERT_POLICY:-round_robin}"
SERIAL_CACHE_ROOT="${SERIAL_CACHE_ROOT:-results_npu_ep7/serial/cache}"
WARMUP_ROUNDS="${WARMUP_ROUNDS:-5}"
DRY_RUN="${DRY_RUN:-false}"

common_args=(
  --modes "$MODES"
  --batches "$BATCHES"
  --seqs "$SEQS"
  --tokens "$TOKENS"
  --preset "$PRESET"
  --ffn-ep-backend "$FFN_EP_BACKEND"
  --ep-expert-policy "$EP_EXPERT_POLICY"
  --serial-cache-root "$SERIAL_CACHE_ROOT"
)

if [ "$DRY_RUN" = true ]; then
  common_args+=(--dry-run)
fi

run_variant() {
  local name="$1"
  shift
  echo ""
  echo "============================================================"
  echo "Warmup ablation variant: $name"
  echo "============================================================"
  bash scripts/run_experiment_matrix_npu.sh \
    "${common_args[@]}" \
    --output-root "$OUT_ROOT/$name" \
    "$@"
}

run_variant both_on     --warmup-p2p --warmup-rounds "$WARMUP_ROUNDS" --prefill-warmup-rounds 1
run_variant p2p_only    --warmup-p2p --warmup-rounds "$WARMUP_ROUNDS" --prefill-warmup-rounds 0
run_variant prefill_only --no-warmup-p2p --warmup-rounds "$WARMUP_ROUNDS" --prefill-warmup-rounds 1
run_variant both_off    --no-warmup-p2p --warmup-rounds "$WARMUP_ROUNDS" --prefill-warmup-rounds 0

if [ "$DRY_RUN" != true ]; then
  python3 scripts/aggregate_warmup_ablation_npu.py --root "$OUT_ROOT"
fi
