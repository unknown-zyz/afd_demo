# Usage and experiment command reference

## 1. Setup

```bash
cd /path/to/afd_demo
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pytest tests/ -q
```

GPU model path:

```bash
export MODEL_PATH=/data/Qwen/Qwen3-30B-A3B/
```

NPU model path:

```bash
export MODEL_NAME=/models/Qwen3-30B-A3B
```

## 2. Single-run GPU entry

```bash
./scripts/run_single.sh <local|multinode> <batch> <seq> [options]
```

Common options:

| Option | Meaning |
|---|---|
| `--tokens N` | Number of generated tokens / `max_new_tokens`; default `5` for single runs. |
| `--no-dbo` | Disable DBO and run the serial AF baseline. |
| `--generate` | Run prefill + autoregressive decode. Without it, `run_single.sh` runs prefill-only. |
| `--crosslayer` | Enable decode cross-layer pipeline; only meaningful with generation and DBO. |
| `--visualize` | Generate one pipeline PNG after the run. |
| `--warmup-p2p` | Run untimed NCCL P2P warmup before timing. |
| `--warmup-rounds N` | Number of P2P warmup rounds; default `3`, matrix uses `5`. |
| `--verbose` | More detailed Python logging. |

## 3. Serial baseline

Serial means A/F separation is still used, but DBO is disabled.

### Single config

```bash
# Serial generation baseline: needed for exact decode TPOT.
./scripts/run_single.sh local 4 128 --tokens 20 --no-dbo --generate

# Serial prefill-only baseline: useful when explicitly capturing prefill_ms.
./scripts/run_single.sh local 4 128 --tokens 20 --no-dbo
```

Matrix serial baseline:

```bash
./scripts/run_experiment_matrix.sh \
  --modes serial \
  --batches 2,4,8,16,32,64 \
  --seqs 128,256,512 \
  --tokens 20 \
  --no-cache
```

Serial outputs:

| Path | Meaning |
|---|---|
| `results/serial/timing_attention_serial_b<B>_s<S>_t<T>.json` | Attention-side serial timing. |
| `results/serial/timing_ffn_serial_b<B>_s<S>_t<T>.json` | FFN-side serial timing. |
| `results/serial/cache/b<B>_s<S>_t<T>.json` | Deduplicated baseline cache used by reports/plots. |

## 4. Prefill DBO

Prefill DBO is DBO enabled with generation disabled. `run_single.sh` defaults to
prefill-only unless `--generate` is passed.

### Single config

```bash
./scripts/run_single.sh local 4 128 --tokens 20
./scripts/run_single.sh local 4 128 --tokens 20 --warmup-p2p --warmup-rounds 5
```

Matrix:

```bash
./scripts/run_experiment_matrix.sh \
  --modes prefill-dbo \
  --batches 2,4,8,16,32,64 \
  --seqs 128,256,512 \
  --tokens 20
```

Prefill DBO outputs:

```text
results/prefill-dbo/timing_attention_prefill-dbo_b<B>_s<S>_t<T>.json
results/prefill-dbo/timing_ffn_prefill-dbo_b<B>_s<S>_t<T>.json
results/prefill-dbo/report_prefill-dbo_b<B>_s<S>_t<T>.md
results/prefill-dbo/pipeline_prefill-dbo_b<B>_s<S>_t<T>.png
```

Speedup uses model-side TTFT-path:

```text
serial_prefill_ms / dbo_total_time_ms
```

## 5. Decode DBO and crosslayer

Decode DBO requires generation.

### Single config

```bash
# Decode DBO
./scripts/run_single.sh local 4 128 --tokens 20 --generate

# Decode DBO with cross-layer pipeline
./scripts/run_single.sh local 4 128 --tokens 20 --generate --crosslayer
```

Matrix:

```bash
./scripts/run_experiment_matrix.sh \
  --modes decode-dbo,decode-dbo-crosslayer \
  --batches 2,4,8,16,32,64 \
  --seqs 128,256,512 \
  --tokens 20
```

Decode outputs:

```text
results/decode-dbo/timing_attention_decode-dbo_b<B>_s<S>_t<T>.json
results/decode-dbo/timing_ffn_decode-dbo_b<B>_s<S>_t<T>.json
results/decode-dbo-crosslayer/timing_attention_decode-dbo-crosslayer_b<B>_s<S>_t<T>.json
results/decode-dbo-crosslayer/timing_ffn_decode-dbo-crosslayer_b<B>_s<S>_t<T>.json
```

Speedup uses exact TPOT:

```text
serial_decode_tpot_ms / dbo_decode_tpot_ms
```

Representative ITL events in PNGs are for visual inspection only.

## 6. GPU matrix options

```bash
./scripts/run_experiment_matrix.sh [options]
```

| Option | Default | Meaning |
|---|---|---|
| `--modes list` | `serial,prefill-dbo,decode-dbo,decode-dbo-crosslayer` | Comma-separated mode subset. |
| `--batches list` | `2,4,8,16,32,64` | Comma-separated batch sizes. |
| `--seqs list` | `128,256,512` | Comma-separated prefill sequence lengths. |
| `--tokens N` | `20` | Decode token count for matrix runs. |
| `--deployment local|multinode` | `local` | Use local 4-GPU or multinode `run_single.sh` mode. |
| `--no-cache` | false | Force serial rerun even if cache exists. |
| `--dry-run` | false | Print commands without executing. |

Representative GPU rerun phases:

```bash
# Default grid
./scripts/run_experiment_matrix.sh \
  --modes serial,prefill-dbo,decode-dbo,decode-dbo-crosslayer \
  --batches 2,4,8,16,32,64 \
  --seqs 128,256,512 \
  --tokens 20

# High-batch expansion
./scripts/run_experiment_matrix.sh \
  --modes serial,prefill-dbo,decode-dbo,decode-dbo-crosslayer \
  --batches 96,128,192,256 \
  --seqs 128,256,512 \
  --tokens 20

# Long-sequence expansion
./scripts/run_experiment_matrix.sh \
  --modes serial,prefill-dbo,decode-dbo,decode-dbo-crosslayer \
  --batches 2,4,8,16,32,64,96,128,192,256 \
  --seqs 1024,2048 \
  --tokens 20
```

The script rewrites `results/experiment_matrix_summary.csv` at start. Save or
rename phase summaries if you run multiple phases.

## 7. NPU matrix options

Run inside the 910C container/worktree.

```bash
./scripts/run_experiment_matrix_npu.sh [options]
```

| Option | Default | Meaning |
|---|---|---|
| `--modes list` | all four modes | Same modes as GPU matrix. |
| `--batches list` | `2,4,8,16,32,64,128,256` | Comma-separated batch sizes. |
| `--seqs list` | `128,256,512` | Comma-separated prefill sequence lengths. |
| `--tokens N` | `20` | Decode token count. |
| `--visible-devs list` | `0..15` | `ASCEND_VISIBLE_DEVICES` pool. |
| `--attn-devs list` | empty | Per-attention-rank device pool override. |
| `--ffn-devs list` | empty | Per-FFN-rank device pool override. |
| `--no-cache` | false | Force serial rerun. |
| `--append` | false | Append to existing summary instead of replacing it. |
| `--dry-run` | false | Print commands without executing. |

Representative NPU rerun phases:

```bash
# Default grid
./scripts/run_experiment_matrix_npu.sh \
  --modes serial,prefill-dbo,decode-dbo,decode-dbo-crosslayer \
  --batches 2,4,8,16,32,64,128,256 \
  --seqs 128,256,512 \
  --tokens 20 \
  --no-cache

# High-batch b512 probe
./scripts/run_experiment_matrix_npu.sh \
  --append \
  --modes serial,prefill-dbo,decode-dbo,decode-dbo-crosslayer \
  --batches 512 \
  --seqs 128,256,512 \
  --tokens 20 \
  --no-cache

# Long-sequence grid
./scripts/run_experiment_matrix_npu.sh \
  --append \
  --modes serial,prefill-dbo,decode-dbo,decode-dbo-crosslayer \
  --batches 2,4,8,16,32,64,128,256,512 \
  --seqs 1024,2048 \
  --tokens 20 \
  --no-cache

# b1024 boundary probes
./scripts/run_experiment_matrix_npu.sh \
  --append \
  --modes serial,decode-dbo,decode-dbo-crosslayer \
  --batches 1024 \
  --seqs 128,256,512,1024,2048 \
  --tokens 20 \
  --no-cache
```

NPU outputs use `results_npu/` with the same subdirectory layout as GPU. The NPU
summary has extra columns for visible chip pool and active world size.

## 8. Reports, plots, and audits

Generate reports during matrix runs; regenerate figures/audits afterward:

```bash
python scripts/plot_all_pipelines.py --root results
python scripts/audit_experiment_baselines.py --root results --output-csv results/baseline_audit.csv

python scripts/plot_all_pipelines.py --root results_npu
python scripts/audit_experiment_baselines.py --root results_npu --output-csv results_npu/baseline_audit.csv
```

Interpret audit statuses:

| Status | Meaning |
|---|---|
| `ok` | Mode-matched baseline exists and speedup is meaningful. |
| `serial-cache-missing` | Run serial for the same `(batch, seq, tokens)`. |
| `baseline-missing` | Cache exists but lacks the required `prefill_ms` or `decode_tpot_ms`. |
| `serial-cache-invalid` | Cache JSON could not be parsed. |

## 9. Troubleshooting

| Symptom | Action |
|---|---|
| CUDA/NPU OOM | Reduce batch/seq; matrix summaries record the first OOM boundary per `(mode, seq)`. |
| First P2P send is slow | Use `--warmup-p2p --warmup-rounds 5`; matrix scripts already do this on GPU. |
| Speedup is `N/A` | Check `baseline_audit.csv`; run or repair the matching serial cache. |
| NPU plot generation fails in container | Copy results back to an environment with `matplotlib`, or install it in the container. |
| HCCL/NCCL peer waits forever after OOM | Confirm the peer log contains OOM, then terminate only the matching stuck peer process. |
