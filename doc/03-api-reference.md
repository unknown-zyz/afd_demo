# 03. API 参考

本文列出当前维护路径中应该被视为公开或半公开的代码 / 脚本接口。未列出的旧
helper、legacy fallback 和实验残留不应作为新功能依赖。

## 1. CLI：`src/main.py`

关键参数：

| 参数 | 含义 |
|---|---|
| `--model-name` | HuggingFace 模型路径或名称。 |
| `--prompt` | 输入 prompt。 |
| `--batch-size` | batch size。 |
| `--prefill-seq-len` | prefill sequence length。 |
| `--max-new-tokens` | 生成 token 数。 |
| `--comm-timing-mode` | send event 计时模式：`enqueue` 或 `completion`。 |
| `--backend` | `cuda`、`npu` 或 `cpu`。 |
| `--attn-size` | Attention role rank 数。 |
| `--ffn-size` | FFN role rank 数。 |
| `--ffn-tp-size` | FFN tensor parallel size。 |
| `--no-dbo` | 关闭 DBO。 |
| `--no-generate` | 只跑 prefill，不进入 autoregressive decode。 |
| `--crosslayer` | 启用 decode crosslayer。 |
| `--prefill-warmup-rounds` | prefill 未计时 warmup 轮数；NPU 默认会进行 warmup。 |
| `--enable-timing` | 输出 timing JSON。 |
| `--timing-output` | timing JSON 输出路径。 |

## 2. 设备与后端：`src/utils/device.py`

| 接口 | 含义 |
|---|---|
| `get_device()` | 返回当前 accelerator device。 |
| `get_backend()` | 返回分布式 backend：NCCL、HCCL 或 Gloo。 |
| `is_npu_available()` | 检查 Ascend NPU 是否可用。 |
| `initialize_device()` | 初始化 CUDA / NPU / CPU 环境。 |

调用方不应直接假设只有 CUDA；新增代码应通过这些 helper 选择 device/backend。

## 3. 分布式上下文：`src/distributed/__init__.py`

| 对象 / 字段 | 含义 |
|---|---|
| `DistributedContext` | 当前 rank、world size、role、process group。 |
| `role` | `attention` 或 `ffn`。 |
| `rank` / `world_size` | 全局 rank 信息。 |
| `role_rank` / `role_size` | 当前 role 内的 rank 信息。 |
| `a2f_group` / `f2a_group` | crosslayer 使用的方向性通信组。 |

Review distributed 变更时要确认 group 创建顺序和所有 rank 的调用顺序一致。

## 4. 模型与工作器

| 模块 | 接口 | 含义 |
|---|---|---|
| `src/model/disaggregated.py` | `DisaggregatedQwenModel` | A/F 分离模型封装。 |
| `src/model/attention_worker.py` | `AttentionWorker` | embedding、attention、KV cache、lm_head。 |
| `src/model/ffn_worker.py` | `FFNWorker` | FFN / MoE 计算。 |

Attention role 持有 KV cache；FFN role 不应依赖 cache。

## 5. 调度器

| Scheduler | 文件 | 用途 |
|---|---|---|
| `SimplePipelineScheduler` | `src/pipeline/scheduler.py` | Serial AF baseline。 |
| `AsyncPipelineScheduler` | `src/pipeline/async_scheduler.py` | Prefill DBO。 |
| `DecodeDBOScheduler` | `src/pipeline/decode_scheduler.py` | Decode DBO / crosslayer。 |

新增 scheduler timing 时必须输出能被报告脚本消费的字段，尤其是
`prefill_ms`、`decode_loop_ms`、`decode_steps`、`decode_tpot_ms`。

## 6. 计时 JSON

常用字段：

| 字段 | 含义 |
|---|---|
| `batch_size`、`prefill_seq_len`、`max_new_tokens` | 配置参数。 |
| `actual_prompt_len` | tokenizer 后实际输入长度；用于审计 `s<seq>` 标签是否真实生效。 |
| `mode` | `serial`、`prefill-dbo`、`decode-dbo` 或 `decode-dbo-crosslayer`。 |
| `total_time_ms` | scheduler 记录的总时间。 |
| `prefill_ms` | serial prefill-only 时间。 |
| `decode_loop_ms` | decode loop 总时间。 |
| `decode_steps` | decode loop step 数。 |
| `decode_tpot_ms` | 准确 TPOT。 |
| `comm_timing_mode` | send event 口径：`enqueue` 或 `completion`。 |
| `tensor_bytes` / `tensor_mib` | send event payload 大小。 |
| `completion_source` | send event 完成来源：`enqueue`、`future_callback`、`observed_wait` 等。 |
| `layers` | 每层 timing。 |
| `events` | Gantt 图使用的 decode step 1 事件。 |

`events` 是可视化数据，不是 speedup denominator。

## 7. 脚本接口

| 脚本 | 主要用途 |
|---|---|
| `scripts/run_single.sh` | GPU 单配置运行。 |
| `scripts/run_experiment_matrix.sh` | GPU 矩阵实验。 |
| `scripts/run_npu.sh` | NPU 单配置运行；位于 `npu` 分支。 |
| `scripts/run_experiment_matrix_npu.sh` | NPU 矩阵实验；位于 `npu` 分支。 |
| `scripts/gen_experiment_report.py` | 生成 Markdown 报告。 |
| `scripts/visualize_dbo_pipeline.py` | 生成单张 pipeline 图。 |
| `scripts/plot_all_pipelines.py` | 批量生成 pipeline 图。 |
| `scripts/audit_experiment_baselines.py` | 审计 DBO 行是否有 mode-matched serial baseline。 |
| `scripts/capture_serial_split.py` | 补采 serial prefill / decode TPOT split。 |

## 8. 不再作为公开接口的内容

以下内容不要在新功能中依赖：

- legacy `decode_step_ms` 作为最终 TPOT；
- `total_time_ms / max_new_tokens` fallback speedup；
- 已删除或历史遗留的 validation helper；
- 未维护的通用 pipeline base class；
- `results*/` 中历史实验产物的内部格式。
