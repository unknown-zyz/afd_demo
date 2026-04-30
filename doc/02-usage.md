# 02. 使用与实验命令

本文是当前实验命令手册，按 serial、prefill DBO、decode DBO / crosslayer、
GPU matrix、NPU matrix 和后处理组织。

## 1. 环境准备

```bash
cd /path/to/afd_demo
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pytest tests/ -q
```

GPU 模型路径：

```bash
export MODEL_PATH=/data/Qwen/Qwen3-30B-A3B/
```

NPU 模型路径：

```bash
export MODEL_NAME=/models/Qwen3-30B-A3B
```

## 2. GPU 单次入口

```bash
./scripts/run_single.sh <local|multinode> <batch> <seq> [options]
```

常用参数：

| 参数 | 含义 |
|---|---|
| `--tokens N` | 生成 token 数 / `max_new_tokens`；单次默认 `5`。 |
| `--no-dbo` | 关闭 DBO，运行 serial AF baseline。 |
| `--generate` | 启用 prefill + autoregressive decode；不加时只跑 prefill。 |
| `--crosslayer` | 启用 decode cross-layer pipeline；只在 DBO + generation 下有意义。 |
| `--visualize` | 单次运行结束后生成 pipeline PNG。 |
| `--warmup-p2p` | timing 前执行未计时 NCCL P2P warmup。 |
| `--warmup-rounds N` | P2P warmup 轮数；单次默认 `3`，matrix 使用 `5`。 |
| `--verbose` | 打印更详细日志。 |

## 3. 串行基线（Serial baseline）

Serial 表示关闭 DBO，但仍使用 A/F 分离。

```bash
# 串行生成基线：用于准确 decode TPOT。
./scripts/run_single.sh local 4 128 --tokens 20 --no-dbo --generate

# 串行预填充基线：用于补采 prefill_ms。
./scripts/run_single.sh local 4 128 --tokens 20 --no-dbo
```

矩阵：

```bash
./scripts/run_experiment_matrix.sh \
  --modes serial \
  --batches 2,4,8,16,32,64 \
  --seqs 128,256,512 \
  --tokens 20 \
  --no-cache
```

输出：

| 路径 | 含义 |
|---|---|
| `results/serial/timing_attention_serial_b<B>_s<S>_t<T>.json` | Attention 侧 serial timing。 |
| `results/serial/timing_ffn_serial_b<B>_s<S>_t<T>.json` | FFN 侧 serial timing。 |
| `results/serial/cache/b<B>_s<S>_t<T>.json` | 报告和图表使用的 baseline cache。 |

## 4. 预填充 DBO（Prefill DBO）

Prefill DBO 表示 DBO 开启、generation 关闭。`run_single.sh` 默认就是
prefill-only，除非显式加 `--generate`。

```bash
./scripts/run_single.sh local 4 128 --tokens 20
```

矩阵：

```bash
./scripts/run_experiment_matrix.sh \
  --modes prefill-dbo \
  --batches 2,4,8,16,32,64 \
  --seqs 128,256,512 \
  --tokens 20
```

Prefill speedup 使用模型侧 TTFT-path：

```text
serial_prefill_ms / dbo_total_time_ms
```

其中 `serial_prefill_ms` 来自 serial cache，`dbo_total_time_ms` 来自 prefill DBO
timing JSON。

## 5. 解码 DBO / 跨层流水（Decode DBO / Crosslayer）

Decode DBO 必须加 `--generate`。

```bash
# 解码 DBO
./scripts/run_single.sh local 4 128 --tokens 20 --generate

# 解码跨层流水
./scripts/run_single.sh local 4 128 --tokens 20 --generate --crosslayer
```

矩阵：

```bash
./scripts/run_experiment_matrix.sh \
  --modes decode-dbo,decode-dbo-crosslayer \
  --batches 2,4,8,16,32,64 \
  --seqs 128,256,512 \
  --tokens 20
```

Decode speedup 使用准确 TPOT：

```text
serial_decode_tpot_ms / dbo_decode_tpot_ms
```

Pipeline PNG 中的 representative ITL / representative step 只用于观察 overlap，
不用于最终 speedup。

## 6. GPU 矩阵参数

```bash
./scripts/run_experiment_matrix.sh [options]
```

| 参数 | 默认值 | 含义 |
|---|---|---|
| `--modes list` | 四种模式全部运行 | `serial,prefill-dbo,decode-dbo,decode-dbo-crosslayer` 的子集。 |
| `--batches list` | `2,4,8,16,32,64` | batch size 列表。 |
| `--seqs list` | `128,256,512` | prefill sequence length 列表。 |
| `--tokens N` | `20` | decode token 数。 |
| `--deployment` | `local` | `local` 或 `multinode`。 |
| `--no-cache` | false | 强制重跑 serial，不复用 cache。 |
| `--dry-run` | false | 只打印命令，不执行。 |

代表性 GPU rerun phases：

```bash
# 默认网格
./scripts/run_experiment_matrix.sh \
  --modes serial,prefill-dbo,decode-dbo,decode-dbo-crosslayer \
  --batches 2,4,8,16,32,64 \
  --seqs 128,256,512 \
  --tokens 20

# 高 batch 扩展
./scripts/run_experiment_matrix.sh \
  --modes serial,decode-dbo,decode-dbo-crosslayer \
  --batches 96,128,192,256 \
  --seqs 128,256,512 \
  --tokens 20

# 长序列扩展
./scripts/run_experiment_matrix.sh \
  --modes serial,prefill-dbo,decode-dbo,decode-dbo-crosslayer \
  --batches 2,4,8,16,32,64,96,128,192,256 \
  --seqs 1024,2048 \
  --tokens 20
```

脚本启动时会重写 `results/experiment_matrix_summary.csv`。如果分阶段运行，需要
手动保存或合并阶段 summary。

## 7. NPU 矩阵参数

NPU matrix 需要在 910C 容器 / worktree 内运行，脚本位于 `npu` 分支。

```bash
./scripts/run_experiment_matrix_npu.sh [options]
```

| 参数 | 默认值 | 含义 |
|---|---|---|
| `--modes list` | 四种模式全部运行 | 与 GPU matrix 相同。 |
| `--batches list` | `2,4,8,16,32,64,128,256` | batch size 列表。 |
| `--seqs list` | `128,256,512` | prefill sequence length 列表。 |
| `--tokens N` | `20` | decode token 数。 |
| `--visible-devs list` | `0..15` | `ASCEND_VISIBLE_DEVICES` 设备池。 |
| `--attn-devs list` | 空 | Attention rank 的设备池覆盖。 |
| `--ffn-devs list` | 空 | FFN rank 的设备池覆盖。 |
| `--no-cache` | false | 强制重跑 serial。 |
| `--append` | false | 追加到已有 summary，而不是重写。 |
| `--dry-run` | false | 只打印命令，不执行。 |

代表性 NPU rerun phases：

```bash
# 默认网格
./scripts/run_experiment_matrix_npu.sh \
  --modes serial,prefill-dbo,decode-dbo,decode-dbo-crosslayer \
  --batches 2,4,8,16,32,64,128,256 \
  --seqs 128,256,512 \
  --tokens 20 \
  --no-cache

# b512 高 batch 探测
./scripts/run_experiment_matrix_npu.sh \
  --append \
  --modes serial,prefill-dbo,decode-dbo,decode-dbo-crosslayer \
  --batches 512 \
  --seqs 128,256,512 \
  --tokens 20 \
  --no-cache

# 长序列网格
./scripts/run_experiment_matrix_npu.sh \
  --append \
  --modes serial,prefill-dbo,decode-dbo,decode-dbo-crosslayer \
  --batches 2,4,8,16,32,64,128,256,512 \
  --seqs 1024,2048 \
  --tokens 20 \
  --no-cache

# b1024 边界探测
./scripts/run_experiment_matrix_npu.sh \
  --append \
  --modes serial,decode-dbo,decode-dbo-crosslayer \
  --batches 1024 \
  --seqs 128,256,512,1024,2048 \
  --tokens 20 \
  --no-cache
```

NPU 输出使用 `results_npu/`，目录布局与 GPU 相同，并额外记录 visible chip pool
和 active world size。

## 8. 后处理

```bash
# GPU
python scripts/plot_all_pipelines.py --root results
python scripts/audit_experiment_baselines.py --root results --output-csv results/baseline_audit.csv

# NPU
python scripts/plot_all_pipelines.py --root results_npu
python scripts/audit_experiment_baselines.py --root results_npu --output-csv results_npu/baseline_audit.csv
```

`baseline_audit.csv` 的常见状态：

| 状态 | 含义 |
|---|---|
| `ok` | 存在 mode-matched baseline，speedup 可信。 |
| `serial-cache-missing` | 缺少相同 `(batch, seq, tokens)` 的 serial cache。 |
| `baseline-missing` | cache 存在，但缺少 `prefill_ms` 或 `decode_tpot_ms`。 |
| `serial-cache-invalid` | cache 无法解析。 |

## 9. 常见问题

| 问题 | 处理方式 |
|---|---|
| Speedup 是 `N/A` | 检查 `baseline_audit.csv`，补跑或修复对应 serial cache。 |
| 启动到输出耗时远大于 timing | 多数时间在模型加载、进程启动、warmup，不属于 scheduler timing。 |
| 只有一个 token 输出 | 确认是否加了 `--generate`，以及 `--tokens` 是否大于 1。 |
| NPU rank 挂住 | 先看双 rank 日志；确认一侧 OOM 后，只 kill 对应 stuck peer PID。 |
