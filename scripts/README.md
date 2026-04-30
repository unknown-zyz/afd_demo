# Script reference

The canonical experiment commands are documented in `doc/02-usage.md`. This
file is a compact index for scripts in this directory.

## Main entrypoints

| Script | Purpose |
|---|---|
| `run_single.sh` | GPU single run: serial, prefill DBO, decode DBO, or decode crosslayer. |
| `run_experiment_matrix.sh` | GPU batch x sequence x mode sweep; writes `results/experiment_matrix_summary.csv`. |
| `run_node.sh` | Manual multinode Attention/FFN process launcher. |

NPU/HCCL scripts live on the `npu` branch:

| Script | Purpose |
|---|---|
| `run_npu.sh` | Ascend 910C single run with HCCL. Use explicit `--attn-size 1 --ffn-size 1 --ffn-tp-size 1` for the validated topology. |
| `run_experiment_matrix_npu.sh` | Ascend 910C matrix sweep; writes `results_npu/experiment_matrix_summary.csv`. |

## Reporting and validation

| Script | Purpose |
|---|---|
| `gen_experiment_report.py` | Generate per-run markdown reports from timing JSON. |
| `visualize_dbo_pipeline.py` | Generate one pipeline Gantt figure from timing JSON. |
| `plot_all_pipelines.py` | Scan a result root and generate pipeline figures for all valid DBO rows. |
| `audit_experiment_baselines.py` | Check that every DBO row has a mode-matched serial baseline. |
| `capture_serial_split.py` | Re-capture serial prefill-only timing and merge `prefill_ms` / `decode_tpot_ms` into cache files. |
| `capture_serial_prefill.sh` | Legacy GPU-only helper kept for old workflows; prefer `capture_serial_split.py`. |

## GPU examples

```bash
# serial baseline
./scripts/run_single.sh local 8 128 --tokens 20 --no-dbo --generate

# prefill DBO, default single-run mode
./scripts/run_single.sh local 8 128 --tokens 20

# decode DBO / decode crosslayer
./scripts/run_single.sh local 8 128 --tokens 20 --generate
./scripts/run_single.sh local 8 128 --tokens 20 --generate --crosslayer

# matrix
./scripts/run_experiment_matrix.sh \
  --modes serial,prefill-dbo,decode-dbo,decode-dbo-crosslayer \
  --batches 2,4,8,16,32,64 \
  --seqs 128,256,512 \
  --tokens 20
```

## NPU examples

Run these on the `npu` branch inside the prepared 910C container/worktree.

```bash
# serial baseline
./scripts/run_npu.sh --attn-size 1 --ffn-size 1 --ffn-tp-size 1 \
  --batch 8 --seq 128 --tokens 20 --model-name "$MODEL_NAME" --no-dbo

# prefill DBO
./scripts/run_npu.sh --attn-size 1 --ffn-size 1 --ffn-tp-size 1 \
  --batch 8 --seq 128 --tokens 20 --model-name "$MODEL_NAME" --no-generate

# decode DBO / decode crosslayer
./scripts/run_npu.sh --attn-size 1 --ffn-size 1 --ffn-tp-size 1 \
  --batch 8 --seq 128 --tokens 20 --model-name "$MODEL_NAME"
./scripts/run_npu.sh --attn-size 1 --ffn-size 1 --ffn-tp-size 1 \
  --batch 8 --seq 128 --tokens 20 --model-name "$MODEL_NAME" --crosslayer

# matrix
./scripts/run_experiment_matrix_npu.sh \
  --modes serial,prefill-dbo,decode-dbo,decode-dbo-crosslayer \
  --batches 2,4,8,16,32,64,128,256 \
  --seqs 128,256,512 \
  --tokens 20 \
  --visible-devs 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
```

## Post-processing

```bash
# GPU
python scripts/plot_all_pipelines.py --root results
python scripts/audit_experiment_baselines.py --root results --output-csv results/baseline_audit.csv

# NPU
python scripts/plot_all_pipelines.py --root results_npu
python scripts/audit_experiment_baselines.py --root results_npu --output-csv results_npu/baseline_audit.csv
```

Use `--help` on the shell scripts and Python scripts for the complete option
list.
