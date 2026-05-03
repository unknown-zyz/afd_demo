# AFD Demo：Attention/FFN 分离与 DBO 实验

AFD Demo 是一个用于研究 **Attention/FFN 分离推理**（Attention-FFN
Disaggregation, AFD）和 **Dual Batch Overlap**（DBO）流水调度的实验仓库。
当前主要模型为 **Qwen3-30B-A3B**。

当前维护两条主线：

| 分支 | 后端 | 结果目录 | 用途 |
|---|---|---|---|
| `main` | CUDA / NCCL | `results/` | GPU 基线与 GPU 实验。 |
| `npu` | Ascend NPU / HCCL | `results_npu/` | 910C 适配与 NPU 实验。 |

## 当前能力

| 能力 | 状态 | 入口 |
|---|---|---|
| Attention/FFN 分离 | 支持 | `src/model/disaggregated.py` |
| Serial AF baseline | 支持 | `SimplePipelineScheduler` |
| Prefill DBO | 支持 | `AsyncPipelineScheduler` |
| Decode DBO | 支持 | `DecodeDBOScheduler` |
| Decode cross-layer pipeline | 实验性支持 | `--crosslayer` |
| NPU EP overlap | 实验性支持 | `--ffn-ep-backend broadcast_reduce_overlap` |
| KV cache / 自回归生成 | 支持 | HuggingFace `DynamicCache` |
| TTFT/TPOT 报告与 pipeline Gantt 图 | 支持 | `scripts/gen_experiment_report.py`、`scripts/visualize_dbo_pipeline.py` |
| Ascend 910C 运行 | 在 `npu` 分支支持 | `scripts/run_npu.sh`、`scripts/run_experiment_matrix_npu.sh` |

## 环境准备

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pytest tests/ -q
```

GPU 路径：

```bash
export MODEL_PATH=/data/Qwen/Qwen3-30B-A3B/
```

NPU 路径：

```bash
export MODEL_NAME=/models/Qwen3-30B-A3B
```

## 快速命令

### 串行基线（Serial baseline）

Serial 关闭 DBO，但仍然使用 A/F 分离结构。Decode baseline 需要生成 token，
这样才能得到准确的 `decode_tpot_ms`。

```bash
./scripts/run_single.sh local 4 128 --tokens 20 --no-dbo --generate
```

### 预填充 DBO（Prefill DBO）

`run_single.sh` 默认是 prefill-only DBO；不加 `--generate` 时不会进入自回归
decode loop。

```bash
./scripts/run_single.sh local 4 128 --tokens 20
```

### 解码 DBO / 跨层流水（Decode / Crosslayer）

```bash
./scripts/run_single.sh local 4 128 --tokens 20 --generate
./scripts/run_single.sh local 4 128 --tokens 20 --generate --crosslayer
```

### 矩阵实验

```bash
# GPU / CUDA 矩阵
./scripts/run_experiment_matrix.sh \
  --modes serial,prefill-dbo,decode-dbo,decode-dbo-crosslayer \
  --batches 2,4,8,16,32,64 \
  --seqs 128,256,512 \
  --tokens 20

# NPU / 910C 矩阵：在 npu 分支的长期容器内运行
./scripts/run_experiment_matrix_npu.sh \
  --modes serial,prefill-dbo,decode-dbo,decode-dbo-crosslayer \
  --batches 2,4,8,16,32,64,128,256 \
  --seqs 128,256,512 \
  --tokens 20 \
  --no-cache
```

两个矩阵脚本都会在同一个 `(mode, seq)` 下遇到 OOM 后停止继续探测更大的
batch，并把 OOM 明确写入 summary CSV。

`batch` 默认是同一个 prompt 复制成多份；指定 `seq` 时，prefill 和 generation
都会 pad/truncate 到该长度。Timing JSON 中的 `prefill_seq_len` 与
`actual_prompt_len` 可用于确认文件名里的 `s<seq>` 是否等于真实 prompt/KV cache
长度。

## 输出目录

| 目录 / 文件 | 含义 |
|---|---|
| `results/serial/` / `results_npu/serial/` | Serial timing JSON、报告和 baseline cache。 |
| `results/prefill-dbo/` / `results_npu/prefill-dbo/` | Prefill DBO timing、报告、PNG。 |
| `results/decode-dbo/` / `results_npu/decode-dbo/` | Decode DBO timing、报告、PNG。 |
| `results/decode-dbo-crosslayer/` / `results_npu/decode-dbo-crosslayer/` | Decode cross-layer timing、报告、PNG。 |
| `results_npu/ep4_broadcast_reduce_sync/` | NPU EP4 + `broadcast_reduce_sync` 同步版负结果，保留用于复盘探索过程。 |
| `results_npu/ep_overlap/` | NPU EP overlap 修复结果，包含 EP4/EP7 对比和首个 decode 正收益配置。 |
| `*/experiment_matrix_summary.csv` | 矩阵状态：`ok`、`cached`、`OOM`、`FAIL`。 |
| `*/baseline_audit.csv` | 每条 DBO 结果是否存在 mode-matched serial baseline。 |

后处理：

```bash
python scripts/plot_all_pipelines.py --root results
python scripts/audit_experiment_baselines.py --root results --output-csv results/baseline_audit.csv

python scripts/plot_all_pipelines.py --root results_npu
python scripts/audit_experiment_baselines.py --root results_npu --output-csv results_npu/baseline_audit.csv
```

## 指标口径

统一使用：

```text
speedup = serial / DBO
```

大于 `1.0x` 才表示 DBO 更快。

| 模式 | 指标 | 加速比公式 |
|---|---|---|
| `prefill-dbo` | 模型侧 TTFT-path | `serial_prefill_ms / dbo_total_time_ms` |
| `decode-dbo` | 准确 TPOT | `serial_decode_tpot_ms / dbo_decode_tpot_ms` |
| `decode-dbo-crosslayer` | 准确 TPOT | `serial_decode_tpot_ms / dbo_decode_tpot_ms` |

Decode DBO 的 pipeline 图固定展示 0-based decode step 1（第 2 个 decode-loop
iteration），只用于观察 overlap、气泡和层间事件，不能作为最终 speedup 分母。

多 batch 下，`decode_tpot_ms` 表示整个 batch 每推进 1 个 decode step 的平均
wall time。一次 step 会为 batch 内每条序列各生成 1 个 token；如需吞吐，使用
`1000 * batch / decode_tpot_ms` 换算。

Pipeline 图里的 A2F/F2A send bars 取决于 `comm_timing_mode`：默认 `enqueue`
只显示 `isend()` 返回/排队开销；`completion` 显示有效 Work 完成跨度，用于观察
通信是否被 Attention/FFN 计算掩盖。`completion` 不是纯硬件链路时延，可能包含
队列、接收端 readiness 和完成通知开销。用 `--no-timing`、`enqueue`、`completion`
跑同一配置可评估 profiling 开销。

如果需要校准“有效完成跨度”和独立通信耗时的关系，可用通信 microbenchmark：

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 \
  scripts/bench_comm_transfer.py \
  --backend cuda --sizes-mib 0.004,0.031,1,16,32 \
  --warmup 5 --iters 50 --blocking \
  --output results/comm_bench/gpu_comm.json
```

旧实验中曾出现 “NPU decode DBO 约 5x 加速” 的误判，根因是把 decode step 1
timing 或 fallback 口径当成了准确 TPOT。当前结论以
[`doc/08-gpu-npu-experiment-summary.md`](doc/08-gpu-npu-experiment-summary.md)
为准。

## 文档

| 文档 | 内容 |
|---|---|
| [`doc/README.md`](doc/README.md) | 文档目录与推荐阅读顺序。 |
| [`doc/01-architecture.md`](doc/01-architecture.md) | 架构、scheduler、KV cache、backend abstraction。 |
| [`doc/02-usage.md`](doc/02-usage.md) | Serial / prefill / decode / matrix 命令手册。 |
| [`doc/03-api-reference.md`](doc/03-api-reference.md) | 当前代码与脚本 API 参考。 |
| [`doc/04-deployment.md`](doc/04-deployment.md) | GPU local、多机、NPU 910C 部署。 |
| [`doc/05-code-review-guide.md`](doc/05-code-review-guide.md) | 代码审查清单。 |
| [`doc/06-npu-910c-adaptation.md`](doc/06-npu-910c-adaptation.md) | Ascend 910C / HCCL 适配说明。 |
| [`doc/07-npu-vs-gpu-experiment-analysis.md`](doc/07-npu-vs-gpu-experiment-analysis.md) | GPU/NPU 指标解释与旧 5x 误判原因。 |
| [`doc/08-gpu-npu-experiment-summary.md`](doc/08-gpu-npu-experiment-summary.md) | 最新 GPU/NPU 覆盖率、speedup 和 OOM 边界。 |
