# 脚本索引

完整实验命令请看 [`doc/02-usage.md`](../doc/02-usage.md)。本文件只作为
`scripts/` 目录的快速索引。

## 主要入口

| 脚本 | 用途 |
|---|---|
| `run_single.sh` | GPU 单配置运行：serial、prefill DBO、decode DBO 或 decode crosslayer。 |
| `run_experiment_matrix.sh` | GPU batch × seq × mode 矩阵扫描，写入 `results/experiment_matrix_summary.csv`。 |
| `run_node.sh` | 手动启动单个 Attention 或 FFN 节点，用于多机调试。 |

NPU/HCCL 脚本位于 `npu` 分支：

| 脚本 | 用途 |
|---|---|
| `run_npu.sh` | Ascend 910C 单配置运行。当前验证拓扑显式使用 `--attn-size 1 --ffn-size 1 --ffn-tp-size 1`。 |
| `run_experiment_matrix_npu.sh` | Ascend 910C 矩阵扫描，写入 `results_npu/experiment_matrix_summary.csv`。 |

## 报告与验证

| 脚本 | 用途 |
|---|---|
| `gen_experiment_report.py` | 从 timing JSON 生成单次运行 Markdown 报告；prefill/decode 对比读取 serial cache。 |
| `visualize_dbo_pipeline.py` | 从一组 timing JSON 生成 pipeline Gantt 图。 |
| `plot_all_pipelines.py` | 扫描结果目录，为所有有效 DBO 行生成 pipeline 图。 |
| `audit_experiment_baselines.py` | 检查每条 DBO 结果是否有 mode-matched serial baseline。 |
| `capture_serial_split.py` | 重新采集 serial prefill-only 时间，并把 `prefill_ms` / `decode_tpot_ms` 合并进 cache。 |
| `capture_serial_prefill.sh` | 旧 GPU-only 辅助脚本；新流程优先使用 `capture_serial_split.py`。 |

报告口径：

- Prefill 报告中的 Serial TTFT 来自 `results/serial/cache/b<B>_s<S>_t<T>.json` 的 `prefill_ms`。
- Decode 报告中的 Serial TPOT 来自同一 cache 的 `decode_tpot_ms`。
- `decode_tpot_ms` 是 batch-level per-step latency：一次 step 同时为 batch 内每条
  序列各生成 1 个 token；吞吐可用 `1000 * batch / decode_tpot_ms` 换算。
- Timing JSON 中的 `prefill_seq_len` 是请求的 prefill 长度，`actual_prompt_len`
  是 tokenizer 后实际输入长度；两者一致时，`s<seq>` 标签才对应真实上下文长度。
- `comm_timing_mode=enqueue` 时，A2F/F2A bars 是非阻塞 `isend()` 返回开销，不是
  传输完成时间；`comm_timing_mode=completion` 时，bars 是有效 Work 完成跨度，
  用于观察通信是否被计算掩盖。
- 使用 `--no-timing` 可关闭详细 timing，与 enqueue/completion runs 对比 profiling
  开销。
- Decode DBO 的 pipeline 明细固定来自 0-based decode step 1（第 2 个 decode-loop iteration）；step 0 被跳过以避开 warmup / 冷启动，该 step 1 timing 不用于最终 speedup。

## GPU 示例

```bash
# 串行基线
./scripts/run_single.sh local 8 128 --tokens 20 --no-dbo --generate

# 预填充 DBO，单次运行默认模式
./scripts/run_single.sh local 8 128 --tokens 20

# 解码 DBO / 解码跨层流水
./scripts/run_single.sh local 8 128 --tokens 20 --generate
./scripts/run_single.sh local 8 128 --tokens 20 --generate --crosslayer

# GPU 矩阵
./scripts/run_experiment_matrix.sh \
  --modes serial,prefill-dbo,decode-dbo,decode-dbo-crosslayer \
  --batches 2,4,8,16,32,64 \
  --seqs 128,256,512 \
  --tokens 20
```

## NPU 示例

以下命令需要在 `npu` 分支和已准备好的 910C 容器 / worktree 内运行。

```bash
# 串行基线
./scripts/run_npu.sh --attn-size 1 --ffn-size 1 --ffn-tp-size 1 \
  --batch 8 --seq 128 --tokens 20 --model-name "$MODEL_NAME" --no-dbo

# 预填充 DBO
./scripts/run_npu.sh --attn-size 1 --ffn-size 1 --ffn-tp-size 1 \
  --batch 8 --seq 128 --tokens 20 --model-name "$MODEL_NAME" --no-generate

# 解码 DBO / 解码跨层流水
./scripts/run_npu.sh --attn-size 1 --ffn-size 1 --ffn-tp-size 1 \
  --batch 8 --seq 128 --tokens 20 --model-name "$MODEL_NAME"
./scripts/run_npu.sh --attn-size 1 --ffn-size 1 --ffn-tp-size 1 \
  --batch 8 --seq 128 --tokens 20 --model-name "$MODEL_NAME" --crosslayer

# NPU 矩阵
./scripts/run_experiment_matrix_npu.sh \
  --modes serial,prefill-dbo,decode-dbo,decode-dbo-crosslayer \
  --batches 2,4,8,16,32,64,128,256 \
  --seqs 128,256,512 \
  --tokens 20 \
  --visible-devs 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
```

## 后处理

```bash
# GPU
python scripts/plot_all_pipelines.py --root results
python scripts/audit_experiment_baselines.py --root results --output-csv results/baseline_audit.csv

# NPU
python scripts/plot_all_pipelines.py --root results_npu
python scripts/audit_experiment_baselines.py --root results_npu --output-csv results_npu/baseline_audit.csv
```

每个脚本的完整参数以 `--help` 输出为准。
