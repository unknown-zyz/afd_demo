# Branch consolidation audit

This document records the branch consolidation decision used to prepare
`chore/consolidate-main`.

## Merge source

`origin/exp/mb4-experiment` is the integration source. It contains the useful
mainline sequence:

```text
origin/npu
  -> origin/exp/npu-ep-prototype
  -> origin/exp/npu-ep-overlap-v2
  -> origin/exp/profile-viz-qa-fix
  -> origin/exp/mb4-experiment
```

The branch adds NPU/HCCL execution, EP7 / `broadcast_reduce_overlap`, timing and
reporting fixes, four-lane pipeline visualization, EP7 full-matrix results,
MB4 experiments, fused dispatch, warmup controls, and L0/no-L0 analysis.

## Branch decisions

| Branch | Decision | Rationale |
|---|---|---|
| `origin/exp/mb4-experiment` | Merge into main | Most complete current NPU EP7 line. Includes MB2 fused dispatch positive result and MB4 negative result. |
| `origin/exp/profile-viz-qa-fix` | Delete after merge | Fully contained in `exp/mb4-experiment`. |
| `origin/exp/npu-ep-overlap-v2` | Delete after merge | Superseded by EP7 round-6/round-9 results. |
| `origin/exp/npu-ep-prototype` | Delete after merge | Superseded by overlap v2 and MB4 branches. |
| `origin/npu` | Delete or archive after main is validated | Its useful NPU runtime changes are now folded into main. |
| `origin/feat/comm-transfer-bench` | Delete | No unique commits relative to main. |
| `origin/feat/npu-910c` | Delete after final spot check | Early 910C adaptation branch; later NPU branches cover its runtime changes. |
| `origin/exp/controller-baseline` | Archive conclusion, do not merge runtime path | CPU controller relay is a deliberate negative baseline: 40-56x slower than direct HCCL. |
| `origin/exp/npu-ep-dual-stream` | Archive conclusion, do not merge runtime path | Dual NPU compute streams showed no speedup and remained default-off. |
| `origin/exp/npu-moe-backend` | Archive conclusion, do not merge runtime path | `npu_grouped_matmul` is correct but 4.9-11.1x slower; NPUGraph capture fails. |
| `origin/exp/npu-ffn-balance` | Do not merge now | Configurable layer placement/D2D changes need a separate review before adoption. |
| `origin/feat/nvshmem` | Archive design only, do not merge runtime path | CUDA/NVSHMEM is not part of the current 910C/HCCL mainline. |

## Mainline behavior after consolidation

- GPU / CUDA path remains available.
- NPU / HCCL path is first-class in main through `scripts/run_npu.sh` and
  `scripts/run_experiment_matrix_npu.sh`.
- EP7 `broadcast_reduce_overlap` is the recommended NPU experiment topology.
- MB2 remains the recommended micro-batch count. MB4 support is retained as a
  parameterized experiment path, but current results show it is slower.
- Fused EP dispatch is retained because it improves MB2 decode TPOT by about
  5-8% for small/mid batch sizes.
- Negative experiment runtime paths are intentionally not merged.

## Final SHAs before deletion (recorded 2026-05-18)

Recorded for recovery: `git fetch origin && git branch <local> <sha>` reconstitutes any branch.

| Branch | Final SHA | Status |
|---|---|---|
| `main` | `bb342fc` | Target |
| `chore/consolidate-main` | `9c64ef3` | Open PR to main, then delete |
| `exp/mb4-experiment` | `175c9f5` | Delete after PR merge (= consolidate-main source) |
| `exp/profile-viz-qa-fix` | `d218b77` | Delete after PR merge |
| `exp/npu-ep-overlap-v2` | `f7a2f00` | Delete after PR merge |
| `exp/npu-ep-prototype` | `4943141` | Delete after PR merge |
| `npu` | `7f558e9` | Delete after PR merge |
| `feat/comm-transfer-bench` | `f36bcd0` | Delete (no unique commits) |
| `feat/npu-910c` | `ab7aa45` | Delete (`prefill-warmup-rounds` etc. already in consolidate-main) |
| `feat/nvshmem` | `4082b4b` | Delete (CUDA NVSHMEM, not 910C path) |
| `exp/controller-baseline` | `aac8411` | Delete (negative result archived in `experiment_archive.md`) |
| `exp/npu-ep-dual-stream` | `381f0bb` | Delete (negative result archived) |
| `exp/npu-ffn-balance` | `a68fb7c` | Delete (reviewed; no extracted value) |
| `exp/npu-moe-backend` | `cc1e25c` | Delete (negative result archived) |

Deletion command after PR merge:

```bash
for br in exp/mb4-experiment exp/profile-viz-qa-fix exp/npu-ep-overlap-v2 \
          exp/npu-ep-prototype npu feat/comm-transfer-bench feat/npu-910c \
          feat/nvshmem exp/controller-baseline exp/npu-ep-dual-stream \
          exp/npu-ffn-balance exp/npu-moe-backend; do
  git push origin --delete "$br"
done
git remote prune origin
```

