# Results manifest

This repository intentionally keeps representative experiment outputs in git so
that performance claims can be audited without rerunning expensive NPU jobs.

## Keep

| Path | Reason |
|---|---|
| `results/` | GPU baseline, reports, and historical GPU archive. |
| `results_npu/` | Historical NPU runs, communication benchmarks, and intermediate EP experiments. |
| `results_npu_ep7/` | Canonical current NPU EP7 full matrix and pipeline figures. |
| `results_npu_ep7_mb4_v2/` | Final MB2 fused-dispatch vs MB4 comparison. |
| `results_npu_ep7_mb4/` | First MB4 sweep retained as historical context for the v2 comparison. |

## Cleaned during consolidation

| Path | Action | Reason |
|---|---|---|
| `results/archive/pre-full-rerun-20260429/gpu/archive/pre-full-rerun-20260429/` | Removed | Accidental nested duplicate of the same archive payload. |

## Future cleanup rule

Do not delete timing JSON/CSV/PNG files from the result directories unless a
newer README or summary explicitly supersedes them and the deletion is recorded
in this manifest.
