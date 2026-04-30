# GPU/NPU DBO experiment summary

This report summarizes the fresh full rerun after branch cleanup and exact
TTFT/TPOT reporting. Speedup is always `serial / DBO`; values above `1.0x`
mean DBO is faster.

## Branches and artifacts

| Platform | Branch / source | Result root | Notes |
|---|---|---|---|
| GPU | `main` @ `49bb17c` | `results/` | Fresh default, high-batch, and long-seq matrix; plots and audit regenerated on `main`. |
| Ascend 910C NPU | `npu` after `5b424a6` | `results_npu/` | Fresh matrix executed in the long-lived `afd-npu-test` container and copied back to this branch. |

Key generated files:

| File | Purpose |
|---|---|
| `results/experiment_matrix_summary.csv` | GPU merged matrix status. |
| `results/baseline_audit.csv` | GPU mode-matched baseline audit. |
| `results_npu/experiment_matrix_summary.csv` | NPU merged matrix status. |
| `results_npu/baseline_audit.csv` | NPU mode-matched baseline audit. |
| `results_npu/*/pipelines_index.md` | NPU pipeline figure index pages. |

## Metric definitions

| Mode | Service metric | DBO timing field | Serial baseline field |
|---|---|---|---|
| `prefill-dbo` | Model-side TTFT / TTFT-path | `total_time_ms` from prefill-only DBO timing | `prefill_ms` from serial cache |
| `decode-dbo` | Exact TPOT | `decode_tpot_ms` | `decode_tpot_ms` |
| `decode-dbo-crosslayer` | Exact TPOT | `decode_tpot_ms` | `decode_tpot_ms` |

The pipeline Gantt figures still visualize representative per-layer / per-step
events, but all report and audit speedups use the exact TTFT/TPOT fields above.

## Matrix coverage

### GPU

| Mode | OK / cached | OOM | Notes |
|---|---:|---:|---|
| `serial` | 45 (`27 ok + 18 cached`) | 5 | `b256` OOM for `seq=128/256/512/1024/2048`. |
| `prefill-dbo` | 20 | 7 | OOM boundary tightens with sequence length. |
| `decode-dbo` | 45 | 5 | OK through `b192`; `b256` OOM for all tested seqs. |
| `decode-dbo-crosslayer` | 45 | 5 | Same capacity boundary as decode DBO. |

GPU baseline audit: **110 / 110 OK**, no fallback or missing baseline entries.

### NPU 910C

| Mode | OK | OOM | Notes |
|---|---:|---:|---|
| `serial` | 45 | 5 | OK through `b512` for all tested seqs; `b1024` OOM for all tested seqs. |
| `prefill-dbo` | 25 | 8 | OOM boundary: `b256/s128`, `b128/s256`, `b64/s512`, `b32/s1024`, `b16/s2048`; explicit `b512` probes OOM for `s128/256/512`. |
| `decode-dbo` | 45 | 1 | OK through `b512` for all tested seqs; `b1024/s128` OOM. |
| `decode-dbo-crosslayer` | 45 | 1 | OK through `b512` for all tested seqs; `b1024/s128` OOM. |

NPU baseline audit: **115 / 115 OK**, no fallback or missing baseline entries.
Pipeline plots: **115 / 115 generated** locally after copying results back
because the NPU container did not have `matplotlib` installed.

## Speedup summary

| Platform | Mode | Samples | Median | Min | Max | Interpretation |
|---|---|---:|---:|---:|---:|---|
| GPU | `prefill-dbo` TTFT-path | 20 | 0.50x | 0.31x | 0.72x | Fresh GPU prefill DBO is slower than serial on this rerun. |
| GPU | `decode-dbo` TPOT | 45 | 0.75x | 0.63x | 1.07x | Mostly slower; only a small subset reaches parity or slight positive speedup. |
| GPU | `decode-dbo-crosslayer` TPOT | 45 | 0.75x | 0.64x | 1.09x | Cross-layer does not materially change the median GPU result. |
| NPU | `prefill-dbo` TTFT-path | 25 | 3.32x | 1.14x | 5.18x | Fresh NPU prefill DBO is consistently faster for configs that fit. |
| NPU | `decode-dbo` TPOT | 45 | 0.85x | 0.70x | 1.16x | Median decode DBO is slower; some small/large cells are slightly positive. |
| NPU | `decode-dbo-crosslayer` TPOT | 45 | 0.85x | 0.70x | 1.18x | Similar to decode DBO; cross-layer does not produce a broad median gain. |

## Capacity boundaries

| Platform | Mode | Largest fully OK tested batch | First OOM probes |
|---|---|---|---|
| GPU | `serial` | `b192` for `s1024/2048`; default/high-batch grids stop at `b256` OOM | `b256` for all tested seqs. |
| GPU | `prefill-dbo` | `b64/s128`, `b64/s256`, `b32/s512`, `b8/s1024`, `b4/s2048` | `b96/s128`, `b96/s256`, `b64/s512`, `b16/s1024`, `b8/s2048`. |
| GPU | `decode-dbo` | `b192` for high/long seq grids | `b256` for all tested seqs. |
| GPU | `decode-dbo-crosslayer` | `b192` for high/long seq grids | `b256` for all tested seqs. |
| NPU | `serial` | `b512` for all tested seqs | `b1024` for all tested seqs. |
| NPU | `prefill-dbo` | `b128/s128`, `b64/s256`, `b32/s512`, `b16/s1024`, `b8/s2048` | next larger batch for each seq; `b512` also OOM for `s128/256/512`. |
| NPU | `decode-dbo` | `b512` for all tested seqs | `b1024/s128`. |
| NPU | `decode-dbo-crosslayer` | `b512` for all tested seqs | `b1024/s128`. |

## Conclusions

1. The old headline that NPU decode DBO had a very large speedup should not be
   used after the exact-TPOT rerun. With exact `decode_tpot_ms`, NPU decode DBO
   has a median speedup below `1.0x`, though some cells are slightly positive.
2. The strongest fresh positive result is NPU prefill DBO: all fitting NPU
   prefill samples are faster than serial by the model-side TTFT-path metric.
3. GPU DBO does not show a median speedup in this full rerun. GPU decode and
   cross-layer decode are mostly below `1.0x`, and GPU prefill is also below
   `1.0x` on the refreshed baseline.
4. Baseline completeness is now clean for both result roots: no speedup in the
   active reports relies on legacy representative ITL or `total_time_ms/tokens`
   fallback.
5. OOM entries are expected capacity boundaries, not missing data. The matrix
   summaries keep them explicitly so plots and reports do not silently treat
   them as successful speedup cells.
