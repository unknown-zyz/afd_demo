# Deployment guide

## 1. Resource requirements

| Scenario | Recommended resources |
|---|---|
| GPU local development | 4 x V100-32GB or similar. |
| GPU minimal smoke | 2 GPUs with enough HBM for the selected model/config. |
| GPU multinode | 2 nodes, each with the required role GPUs and reachable NCCL port. |
| Ascend 910C NPU | Use the prepared privileged long-lived container on the NPU host. |

Qwen3-30B-A3B is MoE-heavy; FFN/MoE weights dominate memory. OOM boundaries are
tracked in `results/experiment_matrix_summary.csv` and
`results_npu/experiment_matrix_summary.csv`.

## 2. Software setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pytest tests/ -q
```

GPU model path:

```bash
export MODEL_PATH=/data/Qwen/Qwen3-30B-A3B/
```

NPU model path inside the validated container:

```bash
export MODEL_NAME=/models/Qwen3-30B-A3B
```

## 3. GPU local workflow

Check resources first:

```bash
bash .github/skills/testing-workflow/check_resources.sh
```

Run representative configs:

```bash
# serial baseline
./scripts/run_single.sh local 4 128 --tokens 20 --no-dbo --generate

# prefill DBO
./scripts/run_single.sh local 4 128 --tokens 20

# decode DBO and crosslayer
./scripts/run_single.sh local 4 128 --tokens 20 --generate
./scripts/run_single.sh local 4 128 --tokens 20 --generate --crosslayer
```

GPU tuning defaults set by `run_single.sh`:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
NCCL_BUFFSIZE=33554432
NCCL_NCHANNELS_PER_NET_PEER=1
```

`NCCL_BUFFSIZE` should be at least the tensor size being sent to avoid P2P
flow-control stalls.

## 4. GPU multinode workflow

The non-NPU remote used by `run_single.sh multinode` is:

```bash
ssh zyz@192.168.5.32 -p 31310 -i ~/.ssh/id_rsa_second
```

Both machines need the same code, Python environment, model path, and a reachable
master port.

Automatic mode:

```bash
./scripts/run_single.sh multinode 4 128 --tokens 20 --generate
```

Manual mode:

```bash
# Remote / FFN node
cd /path/to/afd_demo
source venv/bin/activate
./scripts/run_node.sh ffn <master_ip> 29500

# Local / attention node
cd /path/to/afd_demo
source venv/bin/activate
./scripts/run_node.sh attention <master_ip> 29500 \
  --batch-size 4 \
  --prefill-seq-len 128 \
  --max-new-tokens 20
```

## 5. Ascend 910C NPU workflow

Use the NPU host and long-lived container:

```bash
ssh schedTeam@1.95.114.229 -p 22 -i ~/.ssh/id_rsa_second
docker exec -it afd-npu-test bash
```

Do not delete `afd-npu-test`; it is the shared persistent test container.

Inside the container:

```bash
cd /workspace/afd_demo_npu_rerun_20260429
export MODEL_NAME=/models/Qwen3-30B-A3B
```

Single NPU config:

```bash
# serial
bash scripts/run_npu.sh --attn-size 1 --ffn-size 1 --ffn-tp-size 1 \
  --batch 4 --seq 128 --tokens 20 --model-name "$MODEL_NAME" --no-dbo

# prefill DBO
bash scripts/run_npu.sh --attn-size 1 --ffn-size 1 --ffn-tp-size 1 \
  --batch 4 --seq 128 --tokens 20 --model-name "$MODEL_NAME" --no-generate

# decode DBO / crosslayer
bash scripts/run_npu.sh --attn-size 1 --ffn-size 1 --ffn-tp-size 1 \
  --batch 4 --seq 128 --tokens 20 --model-name "$MODEL_NAME"
bash scripts/run_npu.sh --attn-size 1 --ffn-size 1 --ffn-tp-size 1 \
  --batch 4 --seq 128 --tokens 20 --model-name "$MODEL_NAME" --crosslayer
```

NPU matrix:

```bash
./scripts/run_experiment_matrix_npu.sh \
  --modes serial,prefill-dbo,decode-dbo,decode-dbo-crosslayer \
  --batches 2,4,8,16,32,64,128,256 \
  --seqs 128,256,512 \
  --tokens 20 \
  --no-cache
```

Relevant environment variables:

| Variable | Meaning |
|---|---|
| `ASCEND_VISIBLE_DEVICES` / `--visible-devs` | Visible NPU chip pool. |
| `ATTN_DEVICES` / `--attn-devs` | Per-attention-rank device pool. |
| `FFN_DEVICES` / `--ffn-devs` | Per-FFN-rank device pool. |
| `HCCL_BUFFSIZE` | HCCL communication buffer size in MB. |
| `HCCL_CONNECT_TIMEOUT`, `HCCL_EXEC_TIMEOUT` | HCCL connection/execution timeouts. |

The fresh validated matrix uses active world size `2` even when the visible pool
contains 16 chips.

## 6. Matrix experiments

GPU:

```bash
./scripts/run_experiment_matrix.sh \
  --modes serial,prefill-dbo,decode-dbo,decode-dbo-crosslayer \
  --batches 2,4,8,16,32,64 \
  --seqs 128,256,512 \
  --tokens 20
```

NPU:

```bash
./scripts/run_experiment_matrix_npu.sh \
  --modes serial,prefill-dbo,decode-dbo,decode-dbo-crosslayer \
  --batches 2,4,8,16,32,64,128,256 \
  --seqs 128,256,512 \
  --tokens 20 \
  --no-cache
```

For expanded grids and b1024 boundary probes, use the commands in
`doc/02-usage.md`.

## 7. Post-processing

```bash
python scripts/plot_all_pipelines.py --root results
python scripts/audit_experiment_baselines.py --root results --output-csv results/baseline_audit.csv

python scripts/plot_all_pipelines.py --root results_npu
python scripts/audit_experiment_baselines.py --root results_npu --output-csv results_npu/baseline_audit.csv
```

If the NPU container lacks `matplotlib`, copy `results_npu/` back to a local
environment with plotting dependencies and run `plot_all_pipelines.py` there.

## 8. Troubleshooting

| Problem | Recommendation |
|---|---|
| CUDA/NPU OOM | Treat it as a capacity boundary; reduce batch/seq or stop larger probes for that `(mode, seq)`. |
| Peer hangs after the other rank OOMs | Inspect logs first; if OOM is confirmed, kill only the matching stuck peer PID. |
| First NCCL/HCCL transfer is slow | Use explicit warmup. GPU matrix already adds `--warmup-p2p --warmup-rounds 5`; NPU uses prefill warmup to absorb JIT/HCCL compile effects. |
| Speedup is missing or suspicious | Run `audit_experiment_baselines.py` and check for exact `prefill_ms` / `decode_tpot_ms`. |
| NPU SSH resets under load | Avoid many concurrent SSH sessions while long matrix jobs are running. |
