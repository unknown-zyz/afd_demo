# GPU / NPU DBO 实验结果口径审计

本文说明当前 GPU `results/` 与 NPU `results_npu/` 实验结果如何比较、哪些图上的 `Speedup: N/A` 可以补齐、NPU 分支相对 GPU 主线做了哪些改动，以及目前结论的可信边界。

## 1. 统一比较口径

DBO 图和 report 中的 speedup 必须按模式匹配：

| 模式 | DBO 时间 | serial baseline | speedup |
|---|---:|---:|---:|
| prefill-dbo | 一次完整 prefill pass 的 `total_time_ms` | serial prefill-only 的 `prefill_ms` | `prefill_ms / dbo_total_time_ms` |
| decode-dbo / crosslayer | 一个 representative decode step 的 `total_time_ms` | `decode_step_ms`，缺失时才用 `total_time_ms / max_new_tokens` | `serial_step_ms / dbo_step_ms` |

不能把 serial full-generation 的 `total_time_ms / max_new_tokens` 当成 prefill baseline。它是 decode/generation 口径，混入了生成阶段，不代表一次 prefill pass。

本轮修复后：

- `scripts/experiment_baselines.py` 统一封装 baseline 解析逻辑。
- `scripts/visualize_dbo_pipeline.py` 与 `scripts/gen_experiment_report.py` 使用同一口径。
- prefill 缺 `prefill_ms` 时明确显示 `Speedup: N/A`，不再生成误导性 speedup。
- decode 缺 `decode_step_ms` 时允许 fallback 到 `total_time_ms / max_new_tokens`，并在 report 中标注 fallback 来源。

## 2. N/A 清单与能否补齐

使用 `scripts/audit_experiment_baselines.py` 重新扫描现有 JSON：

```bash
python scripts/audit_experiment_baselines.py --root results
python scripts/audit_experiment_baselines.py --root results_npu
```

### GPU `results/`

| 子模式 | baseline 状态 | 数量 | speedup 范围 |
|---|---|---:|---:|
| decode-dbo | native `decode_step_ms` | 15 | 0.340-0.734 |
| decode-dbo | fallback `total_time_ms/tokens` | 9 | 1.080-1.327 |
| decode-dbo-crosslayer | native `decode_step_ms` | 15 | 0.337-0.683 |
| decode-dbo-crosslayer | fallback `total_time_ms/tokens` | 9 | 1.090-1.427 |
| prefill-dbo | native `prefill_ms` | 15 | 1.048-1.439 |

结论：

- GPU decode/crosslayer 图上的可补 N/A 已通过 fallback 重新生成。
- GPU prefill 当前已有 `prefill_ms`，可以给可信 speedup。
- GPU decode 不是“全负收益”：小/中 batch 的 native baseline 多数 <1，大 batch fallback 配置多数 >1；但 fallback 数值可能受 serial full-generation 均摊口径影响，应优先补 `decode_step_ms`。

### NPU `results_npu/`

| 子模式 | baseline 状态 | 数量 | speedup 范围 |
|---|---|---:|---:|
| decode-dbo | fallback `total_time_ms/tokens` | 24 | 1.123-4.049 |
| decode-dbo | serial cache missing | 21 | N/A |
| decode-dbo-crosslayer | fallback `total_time_ms/tokens` | 24 | 1.209-4.063 |
| decode-dbo-crosslayer | serial cache missing | 21 | N/A |
| prefill-dbo | `prefill_ms` missing | 18 | N/A |

结论：

- NPU decode 在已有 serial cache 的 48 个配置上可以补 speedup，但全是 fallback，不是 native `decode_step_ms`。
- NPU `seq=1024/2048` 和部分 `b512` decode 缺 serial cache，不能凭空补 speedup。
- NPU prefill 现有 18 个配置都缺 `prefill_ms`，旧 report 中的 prefill speedup 不可信，应改为 N/A，直到补跑 serial prefill baseline。

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

### 4.1 prefill baseline 错误

NPU 分支中的 `scripts/gen_experiment_report.py` 会把 serial `total_time_ms / max_new_tokens` 当作 per-step baseline，无论当前 report 是 prefill 还是 decode。这会导致 prefill-dbo report 用 decode/generation 口径比较 prefill pass。

修复后 prefill 缺 `prefill_ms` 时应显示：

```text
Serial baseline: N/A
Speedup: N/A
```

因此，现有 NPU prefill 图/报告不能用于判断 NPU prefill 是否比 GPU 更快或更慢。

### 4.2 decode fallback 可能高估 speedup

`total_time_ms / max_new_tokens` 是可接受的 decode 兜底，但它仍然把 serial 端完整生成总时间均摊到每个 token。如果 serial 总时间包含 prefill 或首 token 额外开销，fallback 会偏大，从而高估 DBO decode speedup。

所以 NPU decode 的 1.1-4.1x 可以作为“现有 cache 下的近似对比”，但严格结论需要补 `decode_step_ms` 或重新跑只计 representative decode step 的 serial baseline。

### 4.3 GPU 与 NPU 不是同一分支代码

NPU 分支带回了 cleanup 前的一些旧文件或旧路径，例如：

- `src/model/kv_cache.py`
- `src/utils/validation.py`
- `src/pipeline/scheduler.py` 的大段旧逻辑
- `scripts/capture_serial_prefill.sh`
- `P2PKeepalive` / `--keepalive` 相关路径

这些在当前 main 已被删除或简化。即使核心 DBO 路径相近，GPU 与 NPU 当前不是完全同一代码基线，跨平台性能差异不能只解释为硬件差异。

### 4.4 NPU 脚本命名和覆盖风险

`scripts/run_npu.sh` 的中间 timing suffix 只有 `npu_b{B}_s{S}_t{T}`，没有把 `serial`、`prefill-dbo`、`decode-dbo-crosslayer` 编进文件名。矩阵脚本依赖“单个模式跑完立刻 move”避免覆盖；如果失败重试或并行运行，可能读到上一个模式残留的 JSON。

建议 NPU suffix 与 GPU `run_single.sh` 对齐，包含 mode 和 crosslayer 标记。

### 4.5 NPU 覆盖范围不完整

NPU decode DBO 有 `seq=1024/2048` 和 `b512` 结果，但对应 serial cache 缺失。当前只能说这些 DBO 配置跑出了 timing，不能给可信 speedup。

另外，`chip_pool` 记录的是可见设备池大小，不等于当前实际拓扑中被 `--attn-size 1 --ffn-size 1 --ffn-tp-size 1` 使用的 rank/device 数；报告中应区分“可见池”和“实际 active topology”。

## 5. 如何解释当前 GPU/NPU 差异

当前能支持的结论：

1. **GPU prefill 是可信正收益。** 现有 15 个 prefill 配置都有 `prefill_ms`，speedup 约 1.05-1.44x。
2. **GPU decode 是混合结果。** 小/中 batch native baseline 多数 <1，大 batch fallback 多数 >1；不能简单说 GPU decode 全负。
3. **NPU decode 在已有 cache 的配置上显示明显正收益。** 但 48 个可算 speedup 的配置全部依赖 fallback，且缺失更大 seq/batch 的 serial baseline。
4. **NPU prefill 目前不能下结论。** 旧图/报告里的 prefill speedup 是错误 baseline 口径，应视为无效。

因此，“GPU prefill 正、decode 负；NPU prefill 负、decode 5x 正”这个现象不能直接作为最终结论。更准确的说法是：

- GPU prefill 正收益可信；
- GPU decode 在大 batch 可能转正，但需要 native serial decode step 补强；
- NPU decode 近似对比显示正收益，尤其大 batch，但需要补 serial decode step；
- NPU prefill 缺 serial prefill baseline，当前 speedup 不可信。

## 6. 建议的后续修复顺序

1. 将 `feat/npu-910c` rebase 到当前 main，剔除 cleanup 前旧代码回流，只保留 backend/HCCL/NPU 脚本和 JIT warmup 修复。
2. 在 serial 路径显式写出 `prefill_ms` 和 `decode_step_ms`，避免长期依赖 fallback。
3. 补跑 NPU serial baseline：至少覆盖 `seq=1024/2048` 与 `b512`，并补 prefill-only capture。
4. 修正 NPU `run_npu.sh` timing suffix，加入 mode/crosslayer，降低失败重试时误读旧 JSON 的风险。
5. 对 GPU 与 NPU 只在“同一代码基线、同一 mode-matched baseline、同一 warmup 策略”的配置上做最终跨平台结论。

