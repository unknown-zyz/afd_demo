# GPU / NPU DBO 实验结果口径审计

本文说明当前 GPU `results/` 与 NPU `results_npu/` 实验结果如何用 TTFT / TPOT 口径比较、哪些图上的 `Speedup: N/A` 可以补齐、NPU 分支相对 GPU 主线做了哪些改动，以及目前结论的可信边界。

> **当前状态:** 本文保留口径审计和历史问题说明。GPU/NPU 已完成 fresh
> full rerun，且 active result roots 的 baseline audit 均为 OK；最终覆盖率、
> OOM 边界和 speedup 结论以
> [gpu_npu_experiment_summary.md](gpu_npu_experiment_summary.md) 为准。

## 1. 统一比较口径

DBO 图和 report 中继续使用 **Speedup = serial / DBO**（>1 表示 DBO 更快），但 baseline 必须按服务指标匹配：

| 模式 | DBO 时间 | serial baseline | speedup |
|---|---:|---:|---:|
| prefill-dbo | model-side TTFT / TTFT-path latency，即一次 prefill path 的 `total_time_ms` | serial model-side TTFT，即 `prefill_ms` | `serial_TTFT / DBO_TTFT` |
| decode-dbo / crosslayer | exact decode-loop TPOT，即 `decode_tpot_ms`；representative ITL 只用于 pipeline Gantt 细节 | serial exact TPOT，即 `decode_tpot_ms` | `serial_TPOT / DBO_TPOT` |

不能把 serial full-generation 的 `total_time_ms / max_new_tokens` 当成 TTFT baseline。它是 per-output-token / TPOT 风格口径，不能代表首 token 延迟。

### 完整线上 TTFT 与本项目 TTFT-path 的边界

线上服务里的 TTFT（Time To First Token）通常是从请求到达服务端，到第一个 token 能返回给用户之间的端到端延迟，可能包含：

1. 请求排队、scheduler batching。
2. tokenizer / prompt 处理。
3. prefix cache / KV cache 检查或构建。
4. 模型 prefill forward。
5. 首 token logits、sampling / greedy decode。
6. detokenize、流式返回和网络发送。

当前实验 JSON 里的 prefill timing 主要覆盖模型侧 prefill path，不包含完整线上服务的排队、tokenizer、网络返回等开销。因此文档中使用 **model-side TTFT / TTFT-path** 表示它；它适合比较本仓库里的 prefill DBO，但不等同于线上服务观测到的完整 TTFT。

### TPOT 与 `decode_tpot_ms`

TPOT（Time Per Output Token）指 decode loop 每输出一个 decode token 的平均耗时。当前代码显式记录：

```text
decode_tpot_ms = decode_loop_ms / decode_steps
decode_steps = max_new_tokens - 1
```

其中 prefill 后采样出的第一个 token 属于 TTFT-path，不属于 decode loop。旧结果中常见的：

```text
total_time_ms / max_new_tokens
```

会把 TTFT 均摊进每个输出 token，且分母也包含首 token，因此不再用于 speedup。

本轮修复后：

- `scripts/experiment_baselines.py` 统一封装 baseline 解析逻辑。
- `scripts/visualize_dbo_pipeline.py` 与 `scripts/gen_experiment_report.py` 使用同一口径。
- TTFT 缺 `prefill_ms` 时明确显示 `Speedup: N/A`，不再生成误导性 speedup。
- TPOT 缺 `decode_tpot_ms` 时明确显示 `Speedup: N/A`，不再用 fallback 生成 speedup。

## 2. N/A 清单与能否补齐

使用 `scripts/audit_experiment_baselines.py` 重新扫描现有 JSON：

```bash
python scripts/audit_experiment_baselines.py --root results
python scripts/audit_experiment_baselines.py --root results_npu
```

### GPU / NPU 当前结果

当前最终结论和补采状态见 [gpu_npu_experiment_summary.md](gpu_npu_experiment_summary.md)。
本文件只保留口径说明：decode speedup 现在必须基于 `decode_tpot_ms`，
旧结果中只有 `total_time_ms / max_new_tokens` 或 legacy `decode_step_ms` 的
配置会显示 `Speedup: N/A`，直到重新运行生成 exact TPOT 字段。

### serial cache 字段语义

| 字段 | 建议解释 | 是否可直接算 speedup |
|---|---|---|
| `prefill_ms` | serial model-side TTFT / TTFT-path baseline | 可用于 TTFT speedup |
| `decode_tpot_ms` | exact decode-loop TPOT | 可用于 TPOT speedup |
| `decode_loop_ms` / `decode_steps` | exact TPOT 的分子/分母 | 用于审计 `decode_tpot_ms` |
| `total_time_ms` | mode-dependent total；decode DBO 中仍是 representative ITL sample | 不能用于 decode speedup |
| `max_new_tokens` | 输出 token 数 | 不能直接作为 exact TPOT 分母；decode loop steps 是 `max_new_tokens - 1` |

严格 speedup 应优先使用 `prefill_ms` 和 `decode_tpot_ms`。
`scripts/capture_serial_split.py` 会对同一配置额外跑 serial prefill-only，
把 `prefill_ms` 写回 cache，并推导 `decode_tpot_ms`，用于补全拆分字段。

## 3. NPU 分支相对 GPU 主线的主要改动

NPU 分支 `feat/npu-910c` 基于旧提交 `e0d1118` 开发，早于 main 上的 cleanup 合并。因此它同时包含“真实 NPU 适配”和“cleanup 前旧代码回流”。

真实 NPU 适配包括：

| 类别 | 文件 | 作用 |
|---|---|---|
| backend 抽象 | `src/utils/device.py` | 增加 `--backend auto/cuda/npu/cpu`，NPU 时选择 HCCL，并通过 `torch_npu.contrib.transfer_to_npu` 兼容部分 `torch.cuda.*` 调用 |
| CLI 参数 | `src/main.py` | 增加 `--backend`、`--attn-size`、`--ffn-size`、`--ffn-tp-size`、`--prefill-warmup-rounds` |
| NPU 启动 | `scripts/run_npu.sh` | 按 rank 启动 NPU 进程，设置 HCCL 环境变量，支持 `ATTN_DEVICES` / `FFN_DEVICES` 设备池 |
| NPU 矩阵 | `scripts/run_experiment_matrix_npu.sh` | 输出到 `results_npu/`，记录 `chip_pool`，支持 append / dry-run |
| 可视化 root | `scripts/plot_all_pipelines.py` | 支持 `--root results_npu` |
| L1/JIT 修复 | `src/main.py` / scheduler 相关逻辑 | NPU 默认跑未计时 prefill warmup，避免首个 shape 编译污染 L0/L1 timing |

这些改动是 NPU 能跑通的必要适配，但还没有干净地 rebase 到当前 main。

## 4. 影响结果可信度的问题

### 4.1 TTFT baseline 错误

NPU 分支中的 `scripts/gen_experiment_report.py` 会把 serial `total_time_ms / max_new_tokens` 当作 token-level baseline，无论当前 report 是 prefill 还是 decode。这会导致 prefill-dbo report 用 TPOT 风格口径比较 TTFT-path。

修复后 prefill 缺 `prefill_ms` 时应显示：

```text
Serial baseline: N/A
Speedup: N/A
```

因此，现有 NPU prefill 图/报告不能用于判断 NPU TTFT-path 是否比 GPU 更快或更慢。

### 4.2 TPOT fallback 可能高估 speedup

`total_time_ms / max_new_tokens` 会把 TTFT-path 均摊到每个 token 上，
并且分母包含 prefill 采样出的首 token，因此不再用于 speedup。
旧结果若缺 `decode_tpot_ms`，应显示 N/A，而不是继续给出近似 speedup。

### 4.3 representative ITL sample 不是 TPOT 平均值

当前 DBO decode 图中的 Gantt 事件仍来自一个 representative ITL sample。
实现上，`DecodeDBOScheduler` 初始化 `_timing_step = 1`，每次
`forward_decode_dbo()` 后递增 `_current_step`；只有 `_current_step == 1`
时创建 `TimingTracker`。也就是说，图里的 pipeline bars 记录的是第 2 个
DBO decode call，第 0 个 step 被跳过以规避 lazy init、allocator、cache
和通信冷启动。

这类 representative ITL sample 适合画 pipeline Gantt、看每层
overlap/bubble，但不再用于 speedup。服务指标对比使用 `decode_tpot_ms`。

Representative timing 可能缺失的情况包括：未开启 `--timing`、使用 `--no-dbo`、`--no-generate`、`max_new_tokens` 太少导致达不到 step 1、`batch_size < num_micro_batches` 未创建 decode scheduler，或运行失败/OOM。

### 4.4 GPU 与 NPU 不是同一分支代码

NPU 分支带回了 cleanup 前的一些旧文件或旧路径，例如：

- `src/model/kv_cache.py`
- `src/utils/validation.py`
- `src/pipeline/scheduler.py` 的大段旧逻辑
- `scripts/capture_serial_prefill.sh`
- `P2PKeepalive` / `--keepalive` 相关路径

这些在当前 main 已被删除或简化。即使核心 DBO 路径相近，GPU 与 NPU 当前不是完全同一代码基线，跨平台性能差异不能只解释为硬件差异。

### 4.5 NPU 脚本命名和覆盖风险

`scripts/run_npu.sh` 的中间 timing suffix 只有 `npu_b{B}_s{S}_t{T}`，没有把 `serial`、`prefill-dbo`、`decode-dbo-crosslayer` 编进文件名。矩阵脚本依赖“单个模式跑完立刻 move”避免覆盖；如果失败重试或并行运行，可能读到上一个模式残留的 JSON。

建议 NPU suffix 与 GPU `run_single.sh` 对齐，包含 mode 和 crosslayer 标记。

### 4.6 NPU 覆盖范围不完整

NPU decode DBO 有 `seq=1024/2048` 和 `b512` 结果，但对应 serial cache 缺失。当前只能说这些 DBO 配置跑出了 timing，不能给可信 speedup。

另外，`chip_pool` 记录的是可见设备池大小，不等于当前实际拓扑中被 `--attn-size 1 --ffn-size 1 --ffn-tp-size 1` 使用的 rank/device 数；报告中应区分“可见池”和“实际 active topology”。

## 5. 如何解释当前 GPU/NPU 差异

当前能支持的结论：

1. **GPU TTFT-path 是可信正收益。** 现有 15 个 prefill 配置都有 `prefill_ms`，speedup 约 1.05-1.44x。
2. **旧 GPU TPOT 结论需要重跑。** 旧 decode 图/report 没有 exact DBO `decode_tpot_ms`，只能保留为历史 representative-step 分析。
3. **旧 NPU TPOT 结论需要重跑。** 旧 decode 图/report 没有 exact DBO `decode_tpot_ms`，不能继续使用 fallback speedup 做结论。
4. **NPU TTFT-path 目前不能下结论。** 旧图/报告里的 prefill/TTFT speedup 是错误 baseline 口径，应视为无效。

因此，“GPU prefill 正、decode 负；NPU prefill 负、decode 5x 正”这个现象不能直接作为最终结论。更准确的说法是：

- GPU TTFT-path 正收益可信；
- GPU TPOT 需要用新代码重跑 serial 与 DBO，生成双方的 `decode_tpot_ms`；
- NPU TPOT 需要用新代码重跑 serial 与 DBO，生成双方的 `decode_tpot_ms`；
- NPU TTFT-path 缺 serial `prefill_ms` baseline，当前 speedup 不可信。

## 6. 建议的后续修复顺序

1. 将 `feat/npu-910c` rebase 到当前 main，剔除 cleanup 前旧代码回流，只保留 backend/HCCL/NPU 脚本和 JIT warmup 修复。
2. 在 serial 和 DBO decode 路径显式写出 `decode_loop_ms`、`decode_steps`、`decode_tpot_ms`，避免长期依赖 representative step 或 fallback。
3. 补跑 NPU serial baseline：至少覆盖 `seq=1024/2048` 与 `b512`，并补 prefill-only capture。
4. 修正 NPU `run_npu.sh` timing suffix，加入 mode/crosslayer，降低失败重试时误读旧 JSON 的风险。
5. 对 GPU 与 NPU 只在“同一代码基线、同一 mode-matched baseline、同一 warmup 策略”的配置上做最终跨平台结论。
