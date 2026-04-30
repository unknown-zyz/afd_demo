# 07. GPU / NPU 实验指标分析

本文解释如何阅读 GPU `results/` 与 NPU `results_npu/` 的实验输出，并重点说明
为什么旧实验曾误认为 NPU decode DBO 有约 5 倍加速。最终覆盖率和 speedup 结论见
[08-gpu-npu-experiment-summary.md](08-gpu-npu-experiment-summary.md)。

## 1. 统一加速比公式

所有报告和 pipeline 图都应使用：

```text
speedup = serial_baseline / DBO_time
```

大于 `1.0x` 表示 DBO 更快。关键要求是 serial baseline 必须和 DBO 模式匹配：

| 模式 | DBO 指标 | Serial baseline | 含义 |
|---|---|---|---|
| `prefill-dbo` | `total_time_ms` | `prefill_ms` | 模型侧 TTFT / TTFT-path |
| `decode-dbo` | `decode_tpot_ms` | `decode_tpot_ms` | 准确 TPOT |
| `decode-dbo-crosslayer` | `decode_tpot_ms` | `decode_tpot_ms` | 准确 TPOT |

不要使用 `total_time_ms / max_new_tokens` 作为 TPOT。它会把 prefill、首 token
路径和其他非 decode-loop 时间混入 decode 对比。

## 2. 为什么旧实验误认为 NPU 有 5 倍加速

旧的 “NPU decode DBO 约 5x” 不是硬件真实性能结论，而是指标口径混用导致的
误判。主要问题有三个：

### 2.1 把 step 1 timing 当成整体 TPOT

DBO decode 的 pipeline 图不是随机挑选样本，而是固定记录 0-based decode step 1：
第 2 个 decode-loop iteration。step 0 被跳过，用来避开 warmup / 冷启动影响。
step 1 timing 可以说明：

- Attention 和 FFN 是否有 overlap；
- 哪些层存在 bubble；
- crosslayer 是否减少等待；
- step 1 的局部耗时结构。

但它不是整个 decode 阶段所有 token 的平均成本。若只拿 step 1 timing 与 serial
decode 全流程平均值对比，就会把一个局部 step 误当成整体 TPOT，容易放大 speedup。

### 2.2 使用了 fallback / legacy 分母

部分旧报告在缺少准确字段时使用过 fallback：

```text
total_time_ms / tokens
legacy decode_step_ms
timed decode step 1 / 旧单步 ITL 字段
```

这些字段和准确 TPOT 语义不同：

- `total_time_ms / tokens` 可能混入 prefill、首 token、warmup 或调度外开销；
- legacy `decode_step_ms` 在旧流程中并不总是由完整 decode loop 平均得到；
- timed decode step 1 只代表图里展示的第 2 个 decode-loop iteration。

把这些字段与 serial 的准确或近似平均值混用，就会产生看似很高的加速比。

### 2.3 Serial 与 DBO 分母不对齐

正确比较必须是同一种服务指标：

```text
serial_decode_tpot_ms / dbo_decode_tpot_ms
```

旧结论中存在 DBO 使用 step 1 timing / fallback，而 serial 使用另一种 TPOT 或总时间
拆分值的情况。分子分母语义不一致时，speedup 数字没有可解释性。

## 3. 修正后的 decode TPOT

当前 decode timing 使用：

```text
decode_tpot_ms = decode_loop_ms / decode_steps
decode_steps = max_new_tokens - 1
```

第一个 token 属于 prefill / TTFT-path；后续 token 才属于 decode loop。

用准确 TPOT 重跑后：

- NPU `decode-dbo` 中位 speedup 为 `0.85x`；
- NPU `decode-dbo-crosslayer` 中位 speedup 为 `0.85x`；
- 少数配置略高于 `1.0x`，但不存在整体 5 倍 decode 加速。

因此旧 “NPU decode 5x” 只能作为历史误判案例，不能继续引用。

## 4. TTFT-path 与在线 TTFT

Prefill 实验测的是模型侧 TTFT-path：

```text
serial_prefill_ms / dbo_total_time_ms
```

它包含 scheduler 计时内的模型 prefill 路径，但不包含生产服务中的排队、tokenizer、
网络传输、流式返回和全局 serving scheduler 开销。因此它适合比较本仓库内的
serial vs DBO，不应直接写成线上端到端 TTFT。

## 5. Decode step 1 timing 的正确用途

Decode step 1 timing 可能在以下情况下缺失：

- timing 未开启；
- DBO 未开启；
- generation 未开启；
- `max_new_tokens` 太小，达不到 decode loop step 1；
- `batch_size < num_micro_batches`；
- 运行失败或 OOM。

它的正确用途是看 pipeline 结构，不是算最终 speedup。最终 decode speedup 必须使用
`decode_tpot_ms`，即所有 decode-loop step 的平均 TPOT。

## 6. 基线审计

```bash
python scripts/audit_experiment_baselines.py --root results --output-csv results/baseline_audit.csv
python scripts/audit_experiment_baselines.py --root results_npu --output-csv results_npu/baseline_audit.csv
```

状态含义：

| 状态 | 含义 |
|---|---|
| `ok` | mode-matched serial baseline 存在，speedup 有意义。 |
| `serial-cache-missing` | 缺少匹配 serial cache。 |
| `baseline-missing` | cache 存在，但缺少 `prefill_ms` 或 `decode_tpot_ms`。 |
| `serial-cache-invalid` | cache 不能解析。 |

Fresh rerun 审计结果：

| Root | 有效 DBO 行 |
|---|---:|
| `results/` | 110 / 110 |
| `results_npu/` | 115 / 115 |

## 7. 解读边界

- OOM 行是容量边界，不是缺失数据。
- GPU 与 NPU 只能在相同 metric 定义和 mode-matched baseline 下比较。
- 旧 “NPU decode 5x” 结论不可复用。
- 当前最终结论见 [08-gpu-npu-experiment-summary.md](08-gpu-npu-experiment-summary.md)。
