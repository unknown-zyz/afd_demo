# Ascend NPU-910C adaptation

This document describes the NPU/HCCL adaptation on the maintained `npu` branch.
The fresh 910C results live under `results_npu/`; see
[gpu_npu_experiment_summary.md](gpu_npu_experiment_summary.md) for coverage,
speedups, and OOM boundaries.

## 1. Validated environment

The validated remote workflow uses the persistent container:

```bash
ssh schedTeam@1.95.114.229 -p 22 -i ~/.ssh/id_rsa_second
docker exec -it afd-npu-test bash
```

Do not delete `afd-npu-test`.

Inside the container:

```bash
torch 2.6.0+cpu
torch_npu 2.6.0
torch_npu.npu.is_available() == True
MODEL_NAME=/models/Qwen3-30B-A3B
```

## 2. Backend model

| Concern | CUDA branch | NPU branch |
|---|---|---|
| Accelerator API | `torch.cuda` | `torch.npu` via `torch_npu.contrib.transfer_to_npu` |
| Distributed backend | NCCL | HCCL |
| Launch script | `scripts/run_single.sh` | `scripts/run_npu.sh` |
| Matrix script | `scripts/run_experiment_matrix.sh` | `scripts/run_experiment_matrix_npu.sh` |
| Result root | `results/` | `results_npu/` |

The device abstraction is in `src/utils/device.py`. `src/main.py` initializes
the backend before constructing the model or schedulers.

## 3. NPU launch path

Single config:

```bash
# serial
bash scripts/run_npu.sh --attn-size 1 --ffn-size 1 --ffn-tp-size 1 \
  --batch 4 --seq 128 --tokens 20 --model-name "$MODEL_NAME" --no-dbo

# prefill DBO
bash scripts/run_npu.sh --attn-size 1 --ffn-size 1 --ffn-tp-size 1 \
  --batch 4 --seq 128 --tokens 20 --model-name "$MODEL_NAME" --no-generate

# decode DBO
bash scripts/run_npu.sh --attn-size 1 --ffn-size 1 --ffn-tp-size 1 \
  --batch 4 --seq 128 --tokens 20 --model-name "$MODEL_NAME"

# decode crosslayer
bash scripts/run_npu.sh --attn-size 1 --ffn-size 1 --ffn-tp-size 1 \
  --batch 4 --seq 128 --tokens 20 --model-name "$MODEL_NAME" --crosslayer
```

Matrix:

```bash
./scripts/run_experiment_matrix_npu.sh \
  --modes serial,prefill-dbo,decode-dbo,decode-dbo-crosslayer \
  --batches 2,4,8,16,32,64,128,256 \
  --seqs 128,256,512 \
  --tokens 20 \
  --no-cache
```

The matrix script moves intermediate files from `results/prefill_dbo/` into
`results_npu/{serial,prefill-dbo,decode-dbo,decode-dbo-crosslayer}/` and writes
`results_npu/experiment_matrix_summary.csv`.

## 4. Device and HCCL environment variables

| Variable / option | Meaning |
|---|---|
| `ASCEND_VISIBLE_DEVICES` / `--visible-devs` | Visible chip pool for the run. |
| `ATTN_DEVICES` / `--attn-devs` | Optional per-attention-rank visible devices. |
| `FFN_DEVICES` / `--ffn-devs` | Optional per-FFN-rank visible devices. |
| `HCCL_BUFFSIZE` | HCCL communication buffer size in MB. |
| `HCCL_CONNECT_TIMEOUT` | HCCL rendezvous timeout. |
| `HCCL_EXEC_TIMEOUT` | HCCL execution timeout. |

`scripts/run_npu.sh` unsets NCCL-only environment variables because they do not
apply to HCCL.

## 5. Validated topology and limitations

The fresh matrix uses:

```text
attn_size=1
ffn_size=1
ffn_tp_size=1
active_world_size=2
visible_chip_pool=16
```

Known limitations:

1. Older multi-rank presets (`attn_size=2`, `ffn_size=2`, `ffn_tp_size=2`) are
   still scaffolding and have known memory/device-mapping issues unless fixed
   separately.
2. FFN tensor parallelism beyond `ffn_tp_size=1` is not the validated path.
3. Attention data parallelism beyond `attn_size=1` is not the validated path.
4. NVSHMEM is CUDA-only in this repo; NPU uses HCCL P2P/collectives.
5. FP8 compute is not viable on the validated `torch_npu 2.6.0` stack; FP8
   tensors can be stored but NPU ops on them are unsupported.

## 6. NPU-specific warmup behavior

Ascend/HCCL can show per-shape JIT/graph compilation effects in the first
prefill pass. `src/main.py` supports `--prefill-warmup-rounds`; default behavior
is one untimed prefill warmup on NPU and zero on CUDA/CPU unless explicitly set.

This avoids polluting layer timing with compilation overhead.

## 7. Result summary

Fresh NPU matrix status:

| Mode | OK | OOM |
|---|---:|---:|
| `serial` | 45 | 5 |
| `prefill-dbo` | 25 | 8 |
| `decode-dbo` | 45 | 1 |
| `decode-dbo-crosslayer` | 45 | 1 |

NPU baseline audit: 115 / 115 valid DBO rows.

Pipeline plots were generated locally after copying `results_npu/` back because
the NPU container did not have `matplotlib` installed.

## 8. Troubleshooting

| Symptom | Action |
|---|---|
| Rank hangs after peer OOM | Inspect both rank logs; if OOM is confirmed, kill only the stuck peer PIDs. |
| SSH resets while monitoring | Reduce concurrent SSH sessions and retry after a short delay. |
| Plotting fails in container | Generate plots in local venv with `matplotlib`. |
| `Float8_e4m3fn has not been supported` | FP8 compute is unsupported on the current NPU stack. |
| HCCL timeout | Check visible devices, stale rank processes, and `MASTER_PORT` conflicts. |
