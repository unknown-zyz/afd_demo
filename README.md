# AFD Demo: Attention-FFN Disaggregation + DBO

AFD Demo is an experiment repo for **Attention/FFN disaggregated inference** and
**Dual Batch Overlap (DBO)** pipeline scheduling. The current primary model is
Qwen3-30B-A3B. The maintained code paths are:

| Branch | Backend | Main result root | Purpose |
|---|---|---|---|
| `main` | CUDA/NCCL | `results/` | GPU baseline and GPU experiments. |
| `npu` | Ascend NPU/HCCL | `results_npu/` | 910C adaptation and NPU experiments. |

## Current capabilities

| Capability | Status | Main entry |
|---|---|---|
| Attention/FFN disaggregation | Supported | `src/model/disaggregated.py` |
| Serial AF baseline | Supported | `SimplePipelineScheduler` |
| Prefill DBO | Supported | `AsyncPipelineScheduler` |
| Decode DBO | Supported | `DecodeDBOScheduler` |
| Decode cross-layer pipeline | Experimental | `--crosslayer` |
| KV cache / autoregressive generation | Supported | HuggingFace `DynamicCache` |
| TTFT/TPOT reports and pipeline Gantt plots | Supported | `scripts/gen_experiment_report.py`, `scripts/visualize_dbo_pipeline.py` |
| Ascend 910C execution | Supported on `npu` | `scripts/run_npu.sh`, `scripts/run_experiment_matrix_npu.sh` |

## Environment

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pytest tests/ -q
```

GPU runs use:

```bash
export MODEL_PATH=/data/Qwen/Qwen3-30B-A3B/
```

NPU runs use:

```bash
export MODEL_NAME=/models/Qwen3-30B-A3B
```

## Quick commands

### Serial baseline

Serial runs disable DBO and generate tokens so they can provide exact
`decode_tpot_ms` baseline data.

```bash
./scripts/run_single.sh local 4 128 --tokens 20 --no-dbo --generate
```

### Prefill DBO

The default single-run mode is prefill-only DBO.

```bash
./scripts/run_single.sh local 4 128 --tokens 20
```

Equivalent explicit Python mode: DBO enabled + `--no-generate`.

### Decode DBO

```bash
./scripts/run_single.sh local 4 128 --tokens 20 --generate
./scripts/run_single.sh local 4 128 --tokens 20 --generate --crosslayer
```

### Matrix experiments

```bash
# GPU/CUDA
./scripts/run_experiment_matrix.sh \
  --modes serial,prefill-dbo,decode-dbo,decode-dbo-crosslayer \
  --batches 2,4,8,16,32,64 \
  --seqs 128,256,512 \
  --tokens 20

# NPU/910C, run inside the long-lived NPU container
./scripts/run_experiment_matrix_npu.sh \
  --modes serial,prefill-dbo,decode-dbo,decode-dbo-crosslayer \
  --batches 2,4,8,16,32,64,128,256 \
  --seqs 128,256,512 \
  --tokens 20 \
  --no-cache
```

Both matrix scripts stop probing larger batches for the same `(mode, seq)` after
an OOM and record that OOM in the summary CSV.

## Outputs

| Root | Content |
|---|---|
| `results/serial/` / `results_npu/serial/` | Serial timing JSON, reports, and cache baselines. |
| `results/prefill-dbo/` / `results_npu/prefill-dbo/` | Prefill DBO timing, report, PNG. |
| `results/decode-dbo/` / `results_npu/decode-dbo/` | Decode DBO timing, report, PNG. |
| `results/decode-dbo-crosslayer/` / `results_npu/decode-dbo-crosslayer/` | Decode cross-layer timing, report, PNG. |
| `*/experiment_matrix_summary.csv` | Matrix status (`ok`, `cached`, `OOM`, `FAIL`). |
| `*/baseline_audit.csv` | Whether each DBO result has a mode-matched serial baseline. |

Post-processing:

```bash
python scripts/plot_all_pipelines.py --root results
python scripts/audit_experiment_baselines.py --root results --output-csv results/baseline_audit.csv

python scripts/plot_all_pipelines.py --root results_npu
python scripts/audit_experiment_baselines.py --root results_npu --output-csv results_npu/baseline_audit.csv
```

## Metrics

| Mode | Metric | Speedup formula |
|---|---|---|
| `prefill-dbo` | model-side TTFT / TTFT-path | `serial_prefill_ms / dbo_total_time_ms` |
| `decode-dbo` | exact TPOT | `serial_decode_tpot_ms / dbo_decode_tpot_ms` |
| `decode-dbo-crosslayer` | exact TPOT | `serial_decode_tpot_ms / dbo_decode_tpot_ms` |

Pipeline figures visualize representative layer/step events, but report
speedups use the exact TTFT/TPOT fields above.

## Documentation

- `doc/01-architecture.md` - current architecture and scheduler design.
- `doc/02-usage.md` - detailed serial / prefill / decode command manual.
- `doc/03-api-reference.md` - current public APIs and script interfaces.
- `doc/04-deployment.md` - GPU local/multinode and NPU container deployment.
- `doc/05-code-review-guide.md` - review checklist for scheduler/timing/result paths.
- `doc/npu_910c_adaptation.md` - Ascend 910C backend notes.
- `doc/gpu_npu_experiment_summary.md` - latest GPU/NPU coverage, speedups, and OOM boundaries.
- `doc/npu_vs_gpu_experiment_analysis.md` - TTFT/TPOT metric interpretation.

## Current result conclusion

See `doc/gpu_npu_experiment_summary.md` for the exact matrix. In brief:

- Active GPU and NPU result roots now have clean baseline audits.
- The old “NPU decode 5x” headline should not be used after exact-TPOT rerun.
- The strongest fresh positive result is NPU prefill DBO.
- OOM rows are expected capacity boundaries and are kept explicitly in summaries.
