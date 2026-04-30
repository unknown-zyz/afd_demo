# 05. 代码审查指南

本文用于审查当前 AFD/DBO 路径的代码和实验变更，重点关注会影响 timing、
speedup 和分布式正确性的风险。

## 1. 当前执行模式

| 模式 | 命令特征 | Scheduler | 主要指标 |
|---|---|---|---|
| Serial | `--no-dbo --generate` | `SimplePipelineScheduler` + generation path | `prefill_ms`、`decode_tpot_ms` |
| Prefill DBO | 默认 prefill-only | `AsyncPipelineScheduler` | 模型侧 TTFT-path |
| Decode DBO | `--generate` | `DecodeDBOScheduler` | 准确 TPOT |
| Decode crosslayer | `--generate --crosslayer` | `DecodeDBOScheduler(use_crosslayer=True)` | 准确 TPOT |

审查任何变更时，先判断它影响哪个模式，以及 serial 和 DBO 是否仍使用相同语义的
指标对比。

## 2. 优先检查的文件

| 领域 | 文件 |
|---|---|
| CLI / 运行编排 | `src/main.py`、`scripts/run_single.sh`、`scripts/run_npu.sh` |
| 分布式状态 | `src/distributed/__init__.py`、`src/distributed/warmup.py` |
| Worker | `src/model/attention_worker.py`、`src/model/ffn_worker.py` |
| 生成 / KV cache | `src/model/disaggregated.py` |
| Scheduler | `src/pipeline/scheduler.py`、`src/pipeline/async_scheduler.py`、`src/pipeline/decode_scheduler.py` |
| Timing / 报告 | `src/utils/timing.py`、`scripts/gen_experiment_report.py`、`scripts/experiment_baselines.py` |
| 图表 / 审计 | `scripts/visualize_dbo_pipeline.py`、`scripts/plot_all_pipelines.py`、`scripts/audit_experiment_baselines.py` |

## 3. 分布式与张量生命周期

1. Async send 必须在 `handle.wait()` 完成前保留 tensor 引用。
2. 不要过早提交会阻塞同一内部 stream 的 NCCL/HCCL receive。
3. OOM 后一侧 rank 可能一直等对端；必须先检查日志，再只 kill 对应 stuck peer。
4. GPU 上 `NCCL_BUFFSIZE` 必须覆盖发送 tensor 大小，否则 `isend` 可能被流控阻塞。

## 4. 计时审查

有效 speedup 字段：

| 模式 | Serial 字段 | DBO 字段 |
|---|---|---|
| Prefill | `prefill_ms` | `total_time_ms` |
| Decode | `decode_tpot_ms` | `decode_tpot_ms` |
| Crosslayer | `decode_tpot_ms` | `decode_tpot_ms` |

不要重新引入：

- `total_time_ms / max_new_tokens` 作为 fallback speedup；
- legacy `decode_step_ms` 作为最终 TPOT；
- decode step 1 timing 作为最终 speedup 分母。

Decode step 1 timing 只能用于 pipeline Gantt 图解释。

## 5. 调度器审查点

### 串行基线

- Serial 是稳定 baseline，不应被改成语义不同的优化路径。
- Serial cache 必须按 `(batch, seq, tokens)` 区分。
- Serial generation 必须写出准确 decode loop 字段。

### 预填充 DBO

- 修改 micro-batch 逻辑时要检查峰值显存和 in-flight buffer。
- Prefill OOM 边界通常比 decode 更紧，因为 activation 与 `batch * seq` 强相关。
- NPU prefill 使用未计时 warmup 吸收 HCCL/JIT 编译开销。

### 解码 DBO

- KV cache slicing 必须保持 batch 顺序。
- KV cache 所有权仍在 Attention role。
- `decode_steps` 必须等于真实 decode-loop iteration 数。
- Crosslayer 修改要重点检查 deadlock 风险和方向性 group。

## 6. 脚本审查点

### GPU 矩阵

`scripts/run_experiment_matrix.sh`：

- 默认重写 `results/experiment_matrix_summary.csv`；
- 除非 `--no-cache`，否则复用 serial baseline；
- 自动加入 `--warmup-p2p --warmup-rounds 5`；
- 遇到 OOM 后停止同一 `(mode, seq)` 的更大 batch。

### NPU 矩阵

`scripts/run_experiment_matrix_npu.sh`：

- 写入 `results_npu/`；
- 支持 `--append`；
- 记录 visible chip pool 和 active world size；
- 需要检查 rank 日志识别 OOM。

当前验证 NPU 拓扑：

```text
attn_size=1, ffn_size=1, ffn_tp_size=1
```

## 7. 结果审查清单

```bash
python scripts/plot_all_pipelines.py --root results
python scripts/audit_experiment_baselines.py --root results --output-csv results/baseline_audit.csv

python scripts/plot_all_pipelines.py --root results_npu
python scripts/audit_experiment_baselines.py --root results_npu --output-csv results_npu/baseline_audit.csv
```

检查：

1. 结论引用的 cell 在 `baseline_audit.csv` 中必须是 `ok`。
2. OOM 行必须保留在 matrix summary 中。
3. Report speedup 必须等于 `serial / DBO`。
4. Pipeline PNG 只用于诊断 overlap，不作为最终 speedup 来源。
5. 最新结论以 [08-gpu-npu-experiment-summary.md](08-gpu-npu-experiment-summary.md) 为准。

## 8. 合并前验证

文档变更：

```bash
git diff --check
```

代码或脚本行为变更：

```bash
source venv/bin/activate
python -m compileall -q src scripts tests
pytest tests/ -q
```

实验流程变更需要先跑小配置 smoke，再跑大矩阵。
