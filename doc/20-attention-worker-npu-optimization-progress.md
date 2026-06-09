# AttentionWorker NPU 优化进展

## 总结

当前分支：`feat/attn-worker-npu-optimizations`。

本文结论全部来自 Ascend 910C NPU；GPU 结果不作为验收依据。早期结果主要来自 Host1 `afd-npu-test`，2026-05-27 统一矩阵因 Host1 HBM 被占用，切到 Host2 `afd-npu-test-h2` 执行。

当前最新结论：

- 报告主口径已统一为 4 类：`prefill_core`、`prefill_full_layer`、`decode_core`、`decode_full_layer`，都使用 all-48 layer aggregate，并在同一 `B={1,8,32}` × `S/cache={128,512,1024}` 矩阵上统计。
- Prefill core：官方 PFA 在这 9 个 shape 均慢于 HF/SDPA；例如 b32/s1024 为 `3.461ms vs 2.442ms`。
- Prefill full-layer：`official + RMSNorm/RoPE fusion` 在中/大 shape 明显优于 HF；例如 b32/s1024 为 `14.733ms vs 17.755ms`。
- Decode core：官方 IFA 在所有 batch/cache 下都优于 HF/SDPA；例如 b32/cache1024 为 `0.078ms vs 0.431ms`。
- Decode full-layer：`official + RMSNorm/RoPE fusion` 是当前单层最佳路径，9 个 shape 约 `0.632-0.659ms`，比 HF `0.716-1.051ms` 更稳定。
- 真实 decode-DBO 同一 3x3 矩阵已跑 3 次取中位数并生成 pipeline 图。单层 Attention 收益没有直接转化为 TPOT：b1 TPOT 约 `146-148ms`，b8 约 `326-363ms`，b32 约 `701-755ms`，主要瓶颈仍在 FFN/通信/pipeline。
- pipeline 中的 `attn_compute` 与单层 bench 不是同一口径：b8/b32 的 pipeline 每层 MB-sum Attention 中位数约 `1.8-1.9ms`，约为 MB-equivalent 单层 bench 的 `1.42-1.51x`；b1 会退化为 1 个 microbatch，timing JSON 无 per-layer pipeline events。
- batch/seq scaling 不能简单按 `B*S^2` 解读 full-layer：core、QKV/OProj、RMSNorm/RoPE、layout/mask、kernel launch、设备同步和 pipeline 切分混在一起；本报告后续以 core/full-layer 分开判断。

### 这一阶段已经得到的稳定判断

- Attention 单层优化已经基本收敛：decode 侧 `official IFA + fused RMSNorm/RoPE` 是当前最优单层路径；prefill 侧则不能简单把 official PFA 设为默认，只能在 full-layer 中依赖 fusion 在中/大 shape 上转正。
- 之前几轮补实验已经把几个争议点收敛清楚：社区 `flash-attention-npu` 在当前形状上明显慢于官方 IFA，且不支持本项目所需的 RoPE 融合；precopy 是低风险优化，但主要改善小 shape 的重复跨设备迁移；MB3/MB4 能跑但没有成为更好的默认设置。
- msprof 与 unified matrix 的结论一致：prefill 小 shape 慢不只是 PFA kernel 本身，QKV/OProj、layout/cast/transpose、mask、RMSNorm/RoPE 等固定开销同样重要；decode IFA 的 kernel 优势是真实存在的，但它还不等于端到端 TPOT 优势。
- 因此当前阶段的核心结论不是“Attention 还没优化完”，而是“Attention 单层已拿到主要收益，端到端 decode-DBO 的主瓶颈已经转向 FFN、A/F 通信和 pipeline overlap”。
- MoE distribute 的版本问题已用新建隔离容器验证：当前生产/验证容器仍是 CANN 8.5.0 + torch_npu 2.6，v2 不暴露 `comm_alg`；隔离容器 CANN 8.5.1 + torch_npu 2.9 已暴露 `comm_alg`，且 `dispatch_v2/combine_v2` 在 Qwen3 `H=2048`、`comm_alg=fullmesh_v2` 下可跑通。
- 当前分支最新相关提交包括 `16a7bd1`、`2b36ad6`、`5dfe9ed`；其中后两个提交只做结果 CSV 规范化，不改变实验结论。`git push -u origin HEAD` 仍受本地 GitHub 凭据缺失阻塞。

## 已实现内容

| 项目 | 状态 | 说明 |
|---|---|---|
| 优化配置与 timing 元数据 | 已实现 | CLI / 环境变量已覆盖 attention kernel、precopy、fusion、TP 占位字段、stream overlap、active/reserved NPU 等元数据。 |
| Layer 输入预拷贝 | 已实现并补性能 A/B | `AttentionLayerInputCache` 可将 mask、position ids、RoPE tensors 预放置到各 layer device。 |
| 官方 NPU attention 适配 | 已实现 | Prefill 使用当前环境可用的 `torch_npu.npu_prompt_flash_attention`，decode 使用 `torch_npu.npu_incre_flash_attention`。 |
| official prefill 的 causal mask 修复 | 已实现 | 当未显式传入 mask 且 `seq_len > 1` 时，构造 causal bool mask。 |
| RMSNorm / RoPE 融合 | 已实现并补矩阵 benchmark | RMSNorm 优先使用 `torch_npu.npu_fused_rms_norm`，不可用时用 `torch_npu.npu_rms_norm`；RoPE 使用 `torch_npu.npu_rotary_mul`。 |
| msprof 工作流 | 已实现并用于本轮分析 | `run_npu.sh` / matrix 脚本支持 msprof；本轮另外用 full profiler 采集了单层 benchmark 代表点。 |
| 预留 NPU 资源池 | 已实现 | `--reserved-npus` 会将设备从初始 active pool 中移除，并写入 timing 元数据。 |
| NPU stream overlap | 已实现（静态 A/F 路径） | `--attn-stream-overlap` 已接入 prefill 与 decode DBO scheduler；MB2/MB3/MB4 已补端到端对比。 |
| Async MoE communicator 协议 | 已实现（API 级） | fallback all-to-all payload 已改为 `async_op=True` handle，并保留 tensor 生命周期；coordinator skeleton 已改用 async dispatch/combine API。 |
| Attention TP | 未实现 | 需要多 attention rank 执行语义、Q/K/V/O 切分与 KV cache ownership 设计，当前暂时 blocked。 |

## 剩余 blocked 待办（截至 2026-05-27）

当前只剩 3 个 blocked 项，它们都不是简单“还没做”，而是被当前拓扑设计、torch_npu API 能力或端到端 gate 条件卡住：

| 待办 | 为什么 blocked | 重新打开的前提 | 当前优先级 |
|---|---|---|---|
| `attention-tp` | 当前运行语义默认只有一个 attention coordinator；若直接局部改成多 attention ranks，会同时影响 TP group、Q/K/V/O 权重切分、KV cache ownership、A/F 通信接口和 scheduler 假设。 | 先单独完成 attention TP 拓扑设计，明确 coordinator 语义、KV cache 所属关系与 NPU 验证矩阵，再进入代码实现。 | 中 |
| `bench-moe-dispatch-real-routing` | 当前 `afd-npu-test*` 生产/验证容器仍是 torch_npu 2.6 + CANN 8.5.0，base 只接受 H=7168，v2 不暴露 `comm_alg`；隔离容器已证明 torch_npu 2.9 + CANN 8.5.1 的 v2 可用，但真实 decode path 尚未迁移到该版本。 | 用隔离容器/新环境继续跑 real routing benchmark；若稳定，再规划 official v2 backend 接入，不能直接改现有容器。 | 中 |
| `multinode-decode-dbo-large-batch` | 现阶段 Host1 单机端到端 decode-DBO 还没有出现“official/fusion 路径 TPOT 转正”的 gate 结果；在这种情况下继续跑跨机大 batch，只会把单机未解决瓶颈带到更复杂的 HCCL 场景。 | 先把单机 FFN/通信/pipeline 瓶颈压下去，确认 Host1 单机 TPOT 相比 HF 转正，再继续跨机大 batch。 | 高 |

从后续收益看，真正最值得优先推进的是第三项背后的单机端到端瓶颈，也就是 FFN、通信与 pipeline overlap；另外两项更多是“结构性扩展项”，不适合在当前状态下硬推进。

## 统一 Attention 实验矩阵（2026-05-27）

统一矩阵全部来自 Host2 `afd-npu-test-h2`，模型 `/models/Qwen3-30B-A3B`，dtype `bfloat16`，all-48 layers aggregate：

```text
B ∈ {1, 8, 32}
S/cache ∈ {128, 512, 1024}
```

产物：

- `results_npu/attention_scope_aligned/single_layer_aligned_all48.{json,csv,md}`
- `results_npu/attention_scope_aligned/single_layer_aligned_mb_equiv.{json,csv,md}`
- `results_npu/attention_scope_aligned/decode_dbo/decode_dbo_summary.{csv,md}`
- `results_npu/attention_scope_aligned/decode_dbo/pipeline_plot_index.{csv,md}`
- `results_npu/attention_scope_aligned/alignment_summary.{csv,md}`
- `results_npu/attention_scope_aligned/single_layer_scaling_summary.{csv,md}`

### 四类单层口径

| phase | 计时对象 | 包含 | 不包含 |
|---|---|---|---|
| `prefill_core` | 已准备 q/k/v 后的 prefill attention kernel | SDPA core 或 PFA core | RMSNorm、QKV、RoPE、OProj、FFN、A/F 通信 |
| `prefill_full_layer` | 完整 prefill `AttentionLayer.forward()` | RMSNorm、QKV、q/k norm、RoPE、PFA/SDPA、OProj | FFN、A/F 通信 |
| `decode_core` | 已准备 q/k/v/cache 后的 decode attention kernel | SDPA 或 IFA core | RMSNorm、QKV、RoPE、KV update、OProj、FFN、A/F 通信 |
| `decode_full_layer` | 完整 decode `AttentionLayer.forward()` | RMSNorm、QKV、q/k norm、RoPE、KV cache update/read、IFA/SDPA、OProj | FFN、A/F 通信 |

### Prefill core：HF/SDPA vs official PFA

单位：ms / layer，48 层中位数。

| Batch | Seq | HF/SDPA | official PFA | speedup |
|---:|---:|---:|---:|---:|
| 1 | 128 | 0.079 | 0.180 | 0.44x |
| 1 | 512 | 0.116 | 0.191 | 0.61x |
| 1 | 1024 | 0.166 | 0.219 | 0.76x |
| 8 | 128 | 0.119 | 0.199 | 0.60x |
| 8 | 512 | 0.289 | 0.371 | 0.78x |
| 8 | 1024 | 0.697 | 0.952 | 0.73x |
| 32 | 128 | 0.211 | 0.338 | 0.62x |
| 32 | 512 | 0.878 | 1.160 | 0.76x |
| 32 | 1024 | 2.442 | 3.461 | 0.71x |

结论：在当前 910C / torch_npu 2.6 / Qwen3 shape 下，PFA core 本身未转正；prefill 不应把 official PFA 单独设为默认。

### Prefill full-layer：HF vs official vs official+fusion

单位：ms / layer，48 层中位数。

| Batch | Seq | HF | official | official+RMSNorm/RoPE | best vs HF |
|---:|---:|---:|---:|---:|---:|
| 1 | 128 | 0.756 | 1.020 | 0.896 | 0.84x |
| 1 | 512 | 0.765 | 1.019 | 0.933 | 0.82x |
| 1 | 1024 | 0.894 | 1.057 | 0.946 | 0.95x |
| 8 | 128 | 0.800 | 1.028 | 0.910 | 0.88x |
| 8 | 512 | 1.783 | 1.769 | 1.509 | 1.18x |
| 8 | 1024 | 3.523 | 3.662 | 3.024 | 1.17x |
| 32 | 128 | 1.672 | 1.769 | 1.692 | 0.99x |
| 32 | 512 | 7.597 | 8.082 | 6.566 | 1.16x |
| 32 | 1024 | 17.755 | 18.930 | 14.733 | 1.21x |

结论：full-layer 中 fusion 的收益能覆盖 raw PFA 的劣势，尤其是中/大 seq；但小 shape 仍不适合默认切 official。

### Decode core：HF/SDPA vs official IFA

单位：ms / layer，48 层中位数。

| Batch | Cache | HF/SDPA | official IFA | speedup |
|---:|---:|---:|---:|---:|
| 1 | 128 | 0.049 | 0.046 | 1.07x |
| 1 | 512 | 0.048 | 0.045 | 1.07x |
| 1 | 1024 | 0.052 | 0.047 | 1.11x |
| 8 | 128 | 0.071 | 0.046 | 1.54x |
| 8 | 512 | 0.088 | 0.047 | 1.87x |
| 8 | 1024 | 0.114 | 0.050 | 2.28x |
| 32 | 128 | 0.137 | 0.058 | 2.36x |
| 32 | 512 | 0.217 | 0.065 | 3.34x |
| 32 | 1024 | 0.431 | 0.078 | 5.53x |

结论：IFA core 是 decode Attention kernel 层面的明确正收益，且 batch/cache 越大越明显。

### Decode full-layer：HF vs official vs official+fusion

单位：ms / layer，48 层中位数。

| Batch | Cache | HF | official | official+RMSNorm/RoPE | best vs HF |
|---:|---:|---:|---:|---:|---:|
| 1 | 128 | 0.737 | 0.779 | 0.646 | 1.14x |
| 1 | 512 | 0.716 | 0.752 | 0.644 | 1.11x |
| 1 | 1024 | 0.723 | 0.769 | 0.659 | 1.10x |
| 8 | 128 | 0.728 | 0.760 | 0.646 | 1.13x |
| 8 | 512 | 0.727 | 0.765 | 0.644 | 1.13x |
| 8 | 1024 | 0.752 | 0.746 | 0.632 | 1.19x |
| 32 | 128 | 0.741 | 0.748 | 0.645 | 1.15x |
| 32 | 512 | 0.772 | 0.752 | 0.642 | 1.20x |
| 32 | 1024 | 1.051 | 0.758 | 0.649 | 1.62x |

结论：decode full-layer 中 `official + fused RMSNorm/RoPE` 是当前单层最佳 Attention 路径。

## 真实 decode-DBO TPOT 与 pipeline 图

同一 3x3 矩阵在 Host2 以 `--attn-size 1 --ffn-size 1 --ffn-tp-size 1`、MB2、`--comm-timing-mode completion`、`--attn-kernel npu-official --attn-fused-rmsnorm --attn-fused-rope` 跑 3 次取中位数。命令请求 `--tokens 10`，timing JSON 中有效 `decode_steps=9`，`decode_tpot_ms = decode_loop_ms / 9`。

| Batch | Seq | runs | Decode TPOT | Prefill | Decode loop | TBT p99 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 128 | 3 | 147.851 ms | 1505.818 ms | 1330.658 ms | 190.906 ms |
| 1 | 512 | 3 | 147.642 ms | 1566.023 ms | 1328.780 ms | 190.748 ms |
| 1 | 1024 | 3 | 145.796 ms | 1629.519 ms | 1312.162 ms | 191.482 ms |
| 8 | 128 | 3 | 329.983 ms | 1676.454 ms | 2969.844 ms | 367.659 ms |
| 8 | 512 | 3 | 325.665 ms | 2130.335 ms | 2930.982 ms | 390.316 ms |
| 8 | 1024 | 3 | 363.003 ms | 2571.169 ms | 3267.024 ms | 424.517 ms |
| 32 | 128 | 3 | 750.708 ms | 2151.859 ms | 6756.370 ms | 967.178 ms |
| 32 | 512 | 3 | 755.041 ms | 3480.649 ms | 6795.370 ms | 855.561 ms |
| 32 | 1024 | 3 | 700.748 ms | 6606.341 ms | 6306.731 ms | 843.208 ms |

Pipeline 图索引：

| Batch | Seq | plot |
|---:|---:|---|
| 1 | 128/512/1024 | 无图：batch=1 时 MB2 退化为 1 个 microbatch，timing JSON 无 per-layer `events`。 |
| 8 | 128 | `results_npu/attention_scope_aligned/decode_dbo/plots/pipeline_b8_s128_t10_rep3.png` |
| 8 | 512 | `results_npu/attention_scope_aligned/decode_dbo/plots/pipeline_b8_s512_t10_rep3.png` |
| 8 | 1024 | `results_npu/attention_scope_aligned/decode_dbo/plots/pipeline_b8_s1024_t10_rep3.png` |
| 32 | 128 | `results_npu/attention_scope_aligned/decode_dbo/plots/pipeline_b32_s128_t10_rep1.png` |
| 32 | 512 | `results_npu/attention_scope_aligned/decode_dbo/plots/pipeline_b32_s512_t10_rep1.png` |
| 32 | 1024 | `results_npu/attention_scope_aligned/decode_dbo/plots/pipeline_b32_s1024_t10_rep1.png` |

## 单层 bench 与 pipeline Attention 时间是否一致

对齐脚本：`scripts/analyze_attention_scope_alignment.py`。

比较对象：

- `single_layer_global_ms`：单层 bench 直接用全局 batch B 跑一次完整 decode full-layer。
- `single_layer_mb_sum_ms`：按 MB2 实际 microbatch size 跑单层 bench 后求和，例如 B=8 对应 `B_eff=4 + 4`。
- `pipeline_attn_per_layer_sum_median_ms`：真实 decode-DBO timing JSON 中同一 layer 的 MB `attn_compute` 事件求和，排除 layer0。

| Batch | Cache | Pipeline per-layer MB-sum | Single global | Single MB-sum | Pipeline / MB-sum |
|---:|---:|---:|---:|---:|---:|
| 1 | 128 | N/A | 0.646 ms | 0.627 ms | N/A |
| 1 | 512 | N/A | 0.644 ms | 0.621 ms | N/A |
| 1 | 1024 | N/A | 0.659 ms | 0.626 ms | N/A |
| 8 | 128 | 1.798 ms | 0.646 ms | 1.265 ms | 1.42x |
| 8 | 512 | 1.844 ms | 0.644 ms | 1.261 ms | 1.46x |
| 8 | 1024 | 1.899 ms | 0.632 ms | 1.263 ms | 1.50x |
| 32 | 128 | 1.799 ms | 0.645 ms | 1.257 ms | 1.43x |
| 32 | 512 | 1.796 ms | 0.642 ms | 1.248 ms | 1.44x |
| 32 | 1024 | 1.900 ms | 0.649 ms | 1.258 ms | 1.51x |

结论：

- 单层 bench 的全局 batch 时间不能直接拿来解释 pipeline 逐层时间，因为 decode-DBO 按 microbatch 切分；更合理的是用 MB-equivalent 单层 bench 求和。
- 即便用 MB-equivalent 求和，真实 pipeline `attn_compute` 仍高 `1.42-1.51x`。这部分差异来自 scheduler 事件边界、KV cache slice/merge、真实生成路径状态、同步/计时开销和通信重叠时的流调度，不应简单归因为 Attention kernel 本身。
- b1 没有 pipeline event 是预期行为：`DecodeDBOScheduler._compute_mb_sizes()` 会把 microbatch 数限制到不超过 batch size，B=1 时实际就是 1 个 MB。

## Batch / Seq scaling 解读

- Prefill core 更接近 attention kernel，但也不严格按 `B*S^2` 线性放大，因为内核选择、tiling、launch overhead 和硬件利用率会改变斜率。
- Prefill full-layer 包含 QKV/OProj、RMSNorm、RoPE、layout、mask 等大量非 attention-core 开销；其中许多项近似 `B*S` 或固定开销，不能用 `B*S^2` 单独解释。
- Decode core 主要随 `B*cache` 增长；IFA 对大 batch/cache 的收益最清晰。
- Decode full-layer 在本次矩阵里更接近固定开销主导，fusion 后约 `0.63-0.66ms/layer`，但真实 TPOT 仍由 FFN/通信/pipeline 主导。

## 历史补充：旧单层 benchmark（保留参考，不作为主口径）

Benchmark 脚本：`scripts/bench_attention_layer_npu.py`。

本轮已扩展能力：

- `--layer-idxs 0,23,47` 或 `--layer-idxs all`。
- `--batches 1,2,4,8,16,32,64,128`。
- `--seqs 1,16,32,64,128,256,512,1024`。
- `--decode-cache-lens 1,16,32,64,128,256,512,1024`。
- 输出 per-layer raw rows、48 层 aggregate、CSV、Markdown。
- 表格字段已拆分为性能差异 `aggregate_speedup_vs_hf` / `aggregate_delta_ms_vs_hf` 和数值误差 `max_abs_diff` / `mean_abs_diff`，不再把“相比 HF 的差异”混写成一个含糊字段。
- 新增 `--include-prefill-core`，单独测 SDPA core vs PFA core。
- 新增 `--include-decode-full-layer`，单独测完整 decode AttentionLayer 的 fusion ablation。

主要产物：

- `results_npu/attention_layer_bench/shape_matrix_layers_0_23_47.csv`
- `results_npu/attention_layer_bench/shape_matrix_layers_0_23_47.md`
- `results_npu/attention_layer_bench/all48_representative_matrix.csv`
- `results_npu/attention_layer_bench/all48_representative_matrix.json`
- `results_npu/attention_layer_bench/all48_representative_matrix.md`
- `results_npu/attention_layer_bench/precopy_cross_device_layers_0_23_47.csv`
- `results_npu/attention_layer_bench/precopy_cross_device_layers_0_23_47.md`
- `results_npu/attention_layer_bench/scope_matrix_layers_0_23_47.csv`
- `results_npu/attention_layer_bench/scope_matrix_layers_0_23_47.md`
- `results_npu/attention_layer_bench/scope_all48_b1_s128_c128.csv`
- `results_npu/attention_layer_bench/scope_all48_b1_s128_c128.md`

实验环境：

- Host1 `afd-npu-test`
- 模型：`/models/Qwen3-30B-A3B`
- dtype：`float16`
- sampled shape matrix：layers `0,23,47`，batch `1..128`，prefill seq / decode cache `1..1024`
- 48 层 representative matrix：layers `all`，batch `1,8,32,128`，prefill seq / decode cache `1,128,1024`

示例命令：

```bash
ASCEND_VISIBLE_DEVICES=0 python scripts/bench_attention_layer_npu.py \
  --model-name /models/Qwen3-30B-A3B \
  --device npu:0 \
  --dtype float16 \
  --layer-idxs all \
  --batches 1,8,32,128 \
  --seqs 1,128,1024 \
  --decode-cache-lens 1,128,1024 \
  --warmup 1 \
  --iters 3 \
  --repeats 3 \
  --include-precopy \
  --no-print-json \
  --output results_npu/attention_layer_bench/all48_representative_matrix.json \
  --summary-csv results_npu/attention_layer_bench/all48_representative_matrix.csv \
  --summary-md results_npu/attention_layer_bench/all48_representative_matrix.md
```

### 单层 benchmark 口径补测

用户指出 prefill 和 decode 在真实推理时都应该代表 Attention 侧完整计算。代码核对后，原 benchmark 的两个主要 phase 口径确实不一致：

| phase | 当前计时对象 | 包含 | 不包含 |
|---|---|---|---|
| `prefill_full_layer` | 完整 `AttentionLayer.forward()` | input RMSNorm、QKV、q/k norm、RoPE、PFA/SDPA、OProj、必要 device copy | FFN、A/F 通信 |
| `decode_core` | 已准备好 q/k/v/cache 后的 core attention kernel | `scaled_dot_product_attention` 或 `npu_incre_flash_attention` | input RMSNorm、QKV、q/k norm、RoPE、真实 cache update、OProj |
| `prefill_core` | 新增：已准备 q/k/v 后的 prefill core kernel | SDPA core 或 PFA core | input RMSNorm、QKV、RoPE、OProj |
| `decode_full_layer` | 新增：完整 decode `AttentionLayer.forward()` | input RMSNorm、QKV、q/k norm、RoPE、cache update/read、IFA/SDPA、OProj | FFN、A/F 通信 |

新增命令：

```bash
ASCEND_VISIBLE_DEVICES=0 python scripts/bench_attention_layer_npu.py \
  --model-name /models/Qwen3-30B-A3B \
  --device npu:0 \
  --dtype bfloat16 \
  --layer-idxs all \
  --batches 1 \
  --seqs 128 \
  --decode-cache-lens 128 \
  --warmup 1 \
  --iters 5 \
  --repeats 3 \
  --include-prefill-core \
  --include-decode-full-layer \
  --no-print-json \
  --output results_npu/attention_layer_bench/scope_all48_b1_s128_c128.json \
  --summary-csv results_npu/attention_layer_bench/scope_all48_b1_s128_c128.csv \
  --summary-md results_npu/attention_layer_bench/scope_all48_b1_s128_c128.md
```

#### all-48 b1/s128/cache128 新口径结果

| phase | 路径 | 48 层中位延迟 | speedup vs HF | max abs diff | 结论 |
|---|---|---:|---:|---:|---|
| `prefill_core` | HF/SDPA | 0.077 ms | 1.00 | N/A | prefill core 基线 |
| `prefill_core` | official PFA | 0.195 ms | 0.392 | 0 | PFA core 在该小形状下慢于 SDPA core |
| `prefill_full_layer` | HF | 0.868 ms | 1.00 | N/A | full-layer 基线 |
| `prefill_full_layer` | official + 两个 fusion | 0.990 ms | 0.877 | 2.78e-2 | full-layer 仍慢 |
| `decode_core` | HF/SDPA | 0.047 ms | 1.00 | N/A | decode core 基线 |
| `decode_core` | official IFA | 0.047 ms | 1.00 | 0 | b1/cache128 下 core 收益不明显 |
| `decode_full_layer` | HF | 0.850 ms | 1.00 | N/A | decode full-layer 基线 |
| `decode_full_layer` | official | 0.887 ms | 0.958 | 0 | raw official full-layer 略慢 |
| `decode_full_layer` | official + RMSNorm fusion | 0.815 ms | 1.044 | 8.30e-3 | 小幅更快 |
| `decode_full_layer` | official + RoPE fusion | 0.786 ms | 1.082 | 2.93e-3 | 更快 |
| `decode_full_layer` | official + 两个 fusion | 0.721 ms | 1.179 | 7.81e-3 | 当前该形状最佳 |

#### sampled layers 0/23/47 新口径结果

| phase | Batch | Seq/cache | 路径 | 3 层中位延迟 | speedup vs HF | 结论 |
|---|---:|---:|---|---:|---:|---|
| `prefill_core` | 1 | seq 16 | official PFA | 0.171 ms | 0.563 | 慢于 SDPA core |
| `prefill_core` | 1 | seq 128 | official PFA | 0.161 ms | 0.442 | 慢于 SDPA core |
| `prefill_core` | 8 | seq 16 | official PFA | 0.207 ms | 0.519 | 慢于 SDPA core |
| `prefill_core` | 8 | seq 128 | official PFA | 0.165 ms | 0.664 | 慢于 SDPA core |
| `decode_full_layer` | 1 | cache 16 | official + 两个 fusion | 0.629 ms | 1.137 | 快于 HF full-layer |
| `decode_full_layer` | 1 | cache 128 | official + 两个 fusion | 0.624 ms | 1.109 | 快于 HF full-layer |
| `decode_full_layer` | 8 | cache 16 | official + 两个 fusion | 0.680 ms | 1.022 | 略快于 HF full-layer |
| `decode_full_layer` | 8 | cache 128 | official + 两个 fusion | 0.687 ms | 1.115 | 快于 HF full-layer |

解读：

- Prefill 小/中形状下，PFA core 本身就慢于 SDPA core；full-layer 慢不只是 QKV/OProj/RoPE/layout 的问题。
- 但旧矩阵显示大 batch / 长 seq 下 `official + fusion` full-layer 能反超，因此还需要补大形状 `prefill_core`，判断 PFA core 是否在大形状转正。
- Decode 必须区分 core 与 full-layer。`decode_core` 只说明 IFA kernel；新增 `decode_full_layer` 显示 fusion 对完整 decode Attention 侧确实有正收益。
- 端到端 TPOT 仍需单独看，因为它还包含 FFN、A/F 通信、pipeline bubble，不等同于单层 Attention 侧时间。

### 48 层 prefill full-layer 代表结果

`speedup_vs_hf = HF latency / 当前 latency`，大于 1 表示当前路径更快。

| Batch | Seq | 路径 | 48 层中位延迟 | speedup vs HF | max abs diff | 结论 |
|---:|---:|---|---:|---:|---:|---|
| 1 | 128 | HF | 0.726 ms | 1.00 | N/A | 基线 |
| 1 | 128 | official | 0.986 ms | 0.736 | 2.87e-3 | 更慢 |
| 1 | 128 | official + 两个 fusion | 0.870 ms | 0.834 | 3.91e-3 | 仍慢于 HF |
| 1 | 128 | official + 两个 fusion + precopy | 0.875 ms | 0.829 | 3.91e-3 | 仍慢于 HF |
| 8 | 128 | HF | 0.783 ms | 1.00 | N/A | 基线 |
| 8 | 128 | official | 0.992 ms | 0.789 | 3.91e-3 | 更慢 |
| 8 | 128 | official + 两个 fusion | 0.882 ms | 0.888 | 4.88e-3 | 仍慢于 HF |
| 32 | 1024 | HF | 17.255 ms | 1.00 | N/A | 基线 |
| 32 | 1024 | official | 17.912 ms | 0.963 | 7.57e-3 | 略慢 |
| 32 | 1024 | official + 两个 fusion | 14.038 ms | 1.229 | 9.52e-3 | 更快 |
| 128 | 1024 | HF | 69.357 ms | 1.00 | N/A | 基线 |
| 128 | 1024 | official | 71.563 ms | 0.969 | 7.81e-3 | 略慢 |
| 128 | 1024 | official + 两个 fusion | 55.393 ms | 1.252 | 9.98e-3 | 更快 |

解读：

- 小 seq / 小 batch 下，official prompt FA 的 launch、layout、mask、手动拆解 QKV/RoPE 等开销会抵消 kernel 收益。
- 大 batch / 长 seq 下，`official + RMSNorm/RoPE fusion` 开始显著优于 HF。
- official 不带 fusion 仍普遍不够好，说明收益不只来自 PFA 本身，还依赖 RMSNorm/RoPE 与 layout 开销控制。

### 48 层 decode core 代表结果

| Batch | Cache len | 路径 | 48 层中位延迟 | speedup vs HF/SDPA | max abs diff | 结论 |
|---:|---:|---|---:|---:|---:|---|
| 1 | 1 | HF/SDPA | 0.071 ms | 1.00 | N/A | 基线 |
| 1 | 1 | official IFA | 0.043 ms | 1.638 | 0 | 更快 |
| 8 | 128 | HF/SDPA | 0.067 ms | 1.00 | N/A | 基线 |
| 8 | 128 | official IFA | 0.046 ms | 1.468 | 0 | 更快 |
| 32 | 1024 | HF/SDPA | 0.434 ms | 1.00 | N/A | 基线 |
| 32 | 1024 | official IFA | 0.076 ms | 5.688 | 6.10e-5 | 显著更快 |
| 128 | 1024 | HF/SDPA | 1.585 ms | 1.00 | N/A | 基线 |
| 128 | 1024 | official IFA | 0.246 ms | 6.451 | 1.22e-4 | 显著更快 |

解读：

- `npu_incre_flash_attention` 是当前最明确的 Attention 单层正收益路径。
- layer0 与 48 层 aggregate 接近但不完全相同；部分 shape 的 sampled-layer / all48 比值超过 10%，因此最终报告不应只引用第 0 层。

### 社区 flash-attention-npu 对照

在 sampled shape matrix 中，社区 v2/v3 继续作为 decode kvcache 对照：

| Batch | Cache len | official IFA | community v2 | community v3 | 结论 |
|---:|---:|---:|---:|---:|---|
| 1 | 1 | 0.049 ms | 3.864 ms | 4.134 ms | 社区慢约 79x/85x |
| 8 | 128 | 0.053 ms | 3.669 ms | 4.007 ms | 社区慢约 70x/76x |
| 32 | 1024 | 0.081 ms | 3.882 ms | 3.958 ms | 社区慢约 48x/49x |
| 128 | 1024 | 0.250 ms | 3.882 ms | 3.916 ms | 社区慢约 15.5x/15.7x |

社区实现仍有两个部署问题：

- `flash_attn_with_kvcache` 可运行，但当前形状下慢于官方 IFA。
- RoPE 参数路径报 `NPU FlashAttention does not support rotary embedding`，不能满足本项目的 RoPE 融合诉求。

## Layer 输入预拷贝效果

用户关心的 precopy 已补专项 A/B：layer 在 `npu:0`，layer-invariant inputs 起始在 `npu:1`，对比 `precopy_inputs=False` 与 `True`。这比默认 1A1F 单 attention device 更能体现减少重复 `.to()` 的效果。

产物：

- `results_npu/attention_layer_bench/precopy_cross_device_layers_0_23_47.csv`
- `results_npu/attention_layer_bench/precopy_cross_device_layers_0_23_47.md`

| Batch | Seq | 路径 | 非 precopy | precopy | 同路径收益 | 结论 |
|---:|---:|---|---:|---:|---:|---|
| 1 | 128 | HF | 1.009 ms | 0.830 ms | 17.7% | 小形状收益明显 |
| 1 | 128 | official + 两个 fusion | 1.140 ms | 0.976 ms | 14.4% | 小形状收益明显 |
| 32 | 1024 | HF | 18.273 ms | 18.104 ms | 0.9% | 大形状收益较小 |
| 32 | 1024 | official + 两个 fusion | 15.207 ms | 14.924 ms | 1.9% | 大形状仍有小幅收益 |
| 128 | 1024 | HF | 72.844 ms | 72.227 ms | 0.8% | 大形状收益较小 |
| 128 | 1024 | official + 两个 fusion | 59.225 ms | 58.666 ms | 0.9% | 大形状仍有小幅收益 |

结论：precopy 是低风险优化。它对小 shape 的重复跨设备迁移更敏感；对大 shape，主耗时转移到 MatMul/PFA/layout，precopy 不是主要瓶颈，但仍没有观察到负收益。

## prompt_flash_attention 与 incre_flash_attention 的区别

`npu_prompt_flash_attention` / `prompt_flash_attention` 面向 prefill：

- Q/K/V 都来自 prompt token，通常 `q_len > 1`。
- 需要处理 causal mask 或 prompt mask。
- 不依赖历史 KV cache。
- 本项目当前环境没有暴露公开 `torch_npu.npu_flash_attention`，实际使用可用的 `torch_npu.npu_prompt_flash_attention`。

`npu_incre_flash_attention` / `incre_flash_attention` 面向 decode：

- 通常每步 `q_len = 1`。
- K/V 来自已存在的 KV cache。
- 算子目标是小 query 对长 KV cache 的增量 attention。
- 当前 910C 上它是单层 decode core 最明确的正收益来源。

为什么 prompt FA 在小 shape 比 HF 慢：

- official 路径在项目内手动拆解 QKV、q/k norm、RoPE、transpose、mask、PFA、o_proj；HF 路径可能已经触发 PyTorch/torch_npu 的优化组合。
- 新增 `prefill_core` 表明在 b1/b8、seq16/128 这些小/中形状下，PFA kernel core 本身也慢于 SDPA core；因此不能只把小 shape 慢归因于 full-layer 外围开销。
- b1/s128 这类 shape 太小，PFA launch、layout 转换、mask 构造等固定开销占比高。
- 当前 official prefill 需要显式 causal bool mask；mask 构造和格式转换会进入 trace。
- BNSD layout、`.contiguous()`、transpose/cast 在小 shape 下占比高。

## msprof 分析

本轮尝试了两类 profiling：

- `msprof op`：只抓到了单个 Cast 算子，不足以解释完整 Attention 路径。
- full profiler：`msprof --runtime-api=on --task-time=on --sys-hardware-mem=on` 可以导出 `op_summary_*.csv`，用于本轮分析。

full profiler 产物在远端：

- `results_npu/attention_msprof_full/prefill_b1_s128_l0/.../op_summary_*.csv`
- `results_npu/attention_msprof_full/prefill_b32_s1024_l0/.../op_summary_*.csv`
- `results_npu/attention_msprof_full/decode_b128_c1024_l0/.../op_summary_*.csv`
- 本地精简摘要：`results_npu/attention_msprof_full/summary_attention_msprof.csv`

注意：benchmark 程序在 msprof application 内生成随机输入，因此 `DSARandomNormal` 也被采集。下面的 op_summary 用于定位算子类别占比，不能直接等同于纯 forward 的精确分解；后续若要严格归因，应把输入预生成并从 profiling 区间排除。

| Profiling 点 | benchmark 结果 | op_summary 关键观察 |
|---|---|---|
| prefill b1/s128/layer0 | HF 1.320 ms；official 1.238 ms；official+both 1.017 ms | PFA/FlashAttention 类 op 合计约 0.954 ms；MatMul 约 1.077 ms；RoPE/RMSNorm/相关 elementwise 约 3.158 ms；layout/cast/transpose 约 2.479 ms。小 shape 下非 attention kernel 开销很高。 |
| prefill b32/s1024/layer0 | HF 17.383 ms；official 18.030 ms；official+both 14.226 ms | PFA/FlashAttention 类 op 合计约 55.9 ms；MatMul 约 87.7 ms；layout/cast/transpose 约 84.0 ms。大 shape 下 fusion 反超，但 layout 与投影仍是主要优化对象。 |
| decode b128/cache1024/layer0 | HF/SDPA 1.679 ms；official IFA 0.338 ms | IFA 单 op 约 0.223 ms/次，而 FlashAttentionScore/SDPA 相关 op 约 1.56~1.75 ms/次；这解释了 IFA 在大 batch/cache decode 上的明显优势。 |

结论：

- prompt FA 小 shape 慢不是单一 PFA kernel 问题，而是 QKV/O projection、RoPE/RMSNorm、layout/cast/transpose、mask 等固定开销共同造成。
- 新增 `prefill_core` 后可进一步确认：在 b1/b8、seq16/128 上 PFA core 本身也慢，需要补大形状 core 才能判断 PFA kernel 的适用区间。
- decode IFA 的 kernel 本身在大 batch/cache 上确实快；新增 `decode_full_layer` 显示 fusion 能改善完整 decode Attention 侧，但端到端 decode-DBO TPOT 未转正，说明瓶颈可能继续在 FFN、A/F 通信或 pipeline 气泡。

## MB3 / MB4 切分位置与效果

MB3/MB4 的切分实现不在 Attention kernel 内，而在 pipeline 调度层：

| 路径 | 实现位置 | 说明 |
|---|---|---|
| CLI / launch | `src/main.py:121`、`scripts/run_npu.sh:73`、`scripts/run_experiment_matrix_npu.sh:132` | 解析 `--num-micro-batches` 并透传。 |
| Prefill DBO | `src/pipeline/async_scheduler.py:87-113`、`src/pipeline/async_scheduler.py:259-260`、`src/pipeline/micro_batch.py:101-158` | `MicroBatchManager.split_batch()` 按 batch 维均分，余数给前几个 MB。 |
| Decode DBO | `src/pipeline/decode_scheduler.py:132-137`、`src/pipeline/decode_scheduler.py:431-465`、`src/pipeline/decode_scheduler.py:483-520` | `_compute_mb_sizes()` 计算 MB size，并用 `mb_offsets` 切 input、mask、position embeddings 和 KV cache slice。 |
| FFN / EP 侧 | `src/pipeline/decode_scheduler.py:717-730`、`src/pipeline/decode_scheduler.py:875-884`、`src/pipeline/decode_scheduler.py:999-1008` | FFN/coordinator/expert 侧按同一 `mb_sizes` 接收或构造 tensor。 |

因此，MB3/MB4 是端到端 pipeline overlap / scheduling 优化，不应该混入单层 Attention kernel benchmark。

本轮补了 Host1 2-rank 1A1F、`--attn-stream-overlap`、b12/s128/t10、每个配置 3 次中位数：

本地精简摘要：`results_npu/attention_mb_bench/summary_b12_s128_t10_aggregate.csv`。

| 模式 | MB | total_time_ms 中位数 | decode_tpot_ms 中位数 | 相对 MB2 结论 |
|---|---:|---:|---:|---|
| prefill DBO | 2 | 746.3 | N/A | 基线 |
| prefill DBO | 3 | 797.7 | N/A | 慢 6.9% |
| prefill DBO | 4 | 1002.4 | N/A | 慢 34.3% |
| decode DBO | 2 | 379.6 | 277.7 | TPOT 基线 |
| decode DBO | 3 | 339.9 | 294.0 | total 更低，但 TPOT 慢 5.9% |
| decode DBO | 4 | 464.5 | 391.9 | TPOT 慢 41.1% |

结论：MB3/MB4 已经验证“能跑”，但在当前 b12/s128/t10 下没有成为更好的默认值。decode MB3 的 total_time 看起来更低，但最终 decode 性能应以 `decode_tpot_ms` 为准，因此 MB2 仍是当前默认选择。

## Host1 单机端到端 decode-DBO 检查

早期 gate 配置：Host1 单机 2 rank，`--attn-size 1 --ffn-size 1 --ffn-tp-size 1`，batch 8，seq 128，tokens 20。

| 路径 | Prefill | Decode TPOT | Total | 解释 |
|---|---:|---:|---:|---|
| HF | 1248.0 ms | 319.3 ms | 325.6 ms | 基线 |
| official | 1350.5 ms | 374.6 ms | 312.7 ms | total 字段更低，但 TPOT 更慢 |
| official + fused RMSNorm/RoPE | 1673.7 ms | 348.6 ms | 310.8 ms | TPOT 仍慢于 HF |

单层 decode core 的收益暂时没有稳定传导到真实 decode-DBO 的 `decode_tpot_ms` 指标。因此，多机大 batch decode-DBO 继续等待，直到 Host1 单机端到端 TPOT 先转正。

## Async MoE communicator 协议

当前 coordinator communicator 接口已经扩展为：

- `dispatch_async(...)`
- `wait_dispatch(handle)`
- `combine_async(...)`
- `wait_combine(handle)`

对于 `FallbackMoECommunicator`：

- route counts 的交换仍是同步进行，因为后续 payload 大小依赖这些 counts；
- hidden-state 和 weight 的 payload all-to-all 改为 `async_op=True`；
- 返回的 handle 会保留 send buffer、count tensor、receive buffer 和 `dist.Work` 引用，直到 `wait_dispatch()` 完成。

`combine_async()` 同样将 reverse all-to-all 改为 `async_op=True`；`wait_combine()` 在等待完成后执行 inverse permutation 和 top-k weighting。DeepEP wrapper 目前提供兼容层形式的 async handle，方便后续真实异步接入而不改变上层 API。coordinator skeleton attention worker 已经改用 async dispatch / combine 调用。

验证情况：

- Host1 容器内对 `src/coordinator_arch/comm/*`、coordinator attention worker 和 communicator test 文件的 compile 检查已通过。
- Host1 mocked-collective smoke 已验证 fallback `dispatch_async -> wait_dispatch -> combine_async -> wait_combine` 能返回预期 tensor，并且确实等待 payload work handles。

## FFN EP 通信路径与官方 MoE distribute API 探测

### 当前真实 decode 路径里的 EPFFNLayer 通信

`src/model/ep_moe.py` 的 `EPFFNLayer` 是当前静态 A/F 真实 decode EP overlap 路径的一部分，接入条件是 `--ffn-ep-backend broadcast_reduce_overlap` 且 `num_micro_batches >= 2`。

其通信语义如下：

| 阶段 | 当前实现 | 说明 |
|---|---|---|
| dispatch | `dist.broadcast(fused_buf, src=ffn_coordinator_rank, group=ffn_ep_dispatch_group)` | coordinator 将 `hidden_2d + selected_experts + routing_weights` 打包为 `uint8` 后广播给所有 EP ranks。 |
| local compute | `ShardedExperts.forward_local()` | 每个 rank 只计算本 rank 拥有的 experts，但输入仍是完整 hidden/routing。 |
| combine/reduce | `dist.reduce(partial, dst=ffn_coordinator_rank, op=SUM, group=ffn_ep_reduce_group)` | 每个 rank 返回 dense partial `[tokens, hidden]`，coordinator 聚合后加 residual。 |

因此当前真实 NPU EP overlap 不是 token-aware sparse dispatch/combine，而是 full hidden broadcast + dense reduce。

### coordinator 的 FallbackMoECommunicator 是否在真实 decode path

`src/coordinator_arch/comm/fallback_a2a.py` 的 `FallbackMoECommunicator` 是 coordinator 架构的默认 communicator：

- dispatch/combine 使用 `torch.distributed.all_to_all_single`。
- 语义是 token-aware all-to-all dispatch/combine。
- skeleton attention worker 已经调用 async dispatch/combine API。

但它当前没有替换 `src.main` 静态 decode path 的 FFN 计算。`src/model/disaggregated.py` 里的 coordinator bridge 主要用于 routing table / expert ownership validation；真实 Qwen FFN compute 仍在 `src/model/ffn_worker.py` 和 `src/model/ep_moe.py`。

### 昇腾官方 MoE distribute API 探测

新增探测脚本：

```bash
python scripts/probe_moe_distribute_npu.py --mode schema
```

Host1 `afd-npu-test` 上的 torch_npu 2.6.0 暴露了以下 API：

| API | 是否存在 | 当前 schema 关键点 |
|---|---|---|
| `npu_moe_distribute_dispatch` | 存在 | 需要 `group_ep`、`ep_world_size`、`ep_rank_id`、`moe_expert_num`；返回 7 个 tensor。 |
| `npu_moe_distribute_combine` | 存在 | 需要 `expand_x`、`expert_ids`、`expand_idx`、`ep_send_counts`、`expert_scales`、`group_ep` 等。 |
| `npu_moe_distribute_dispatch_v2` | 存在 | 当前 Python schema 没有公开 `comm_alg` 参数。 |
| `npu_moe_distribute_combine_v2` | 存在 | 当前 Python schema 没有公开 `comm_alg` 参数。 |

2-rank / 8-rank smoke 结果：

| 配置 | 结果 | 含义 |
|---|---|---|
| v2, EP=2, H=64 | 失败：`Attr commAlg is invalid ... only support fullmesh_v1 and fullmesh_v2, but got commAlg = 0` | 910C/A3 需要 `comm_alg`，但当前 torch_npu 2.6 schema 不接受 `comm_alg` keyword。 |
| base, EP=2, H=64 | 失败：`epWorldSize should be divisible by 8` | base 版不支持 2-rank EP，对当前小 EP smoke 不适配。 |
| base, EP=8, H=64 | 失败：`xShape dims1(H) only supports 7168, but got 64` | base 版 hidden size 有强约束。 |
| base, EP=8, H=7168 | dispatch/combine smoke 通过，identity combine diff 为 0 | 说明 hcomm 获取方式和 base API 基本可用，但 shape 不是本项目 Qwen3 hidden size。 |

关键结论：

- 本项目 Qwen3-30B-A3B 的 `hidden_size=2048`，而当前 base dispatch 在 910C 上报错只支持 H=7168；不能直接替换 `EPFFNLayer` 的 Qwen3 FFN 通信。
- v2 看起来更接近通用接口，但当前 torch_npu 2.6 暴露的 Python schema 缺少 `comm_alg`，导致 910C 下无法选择 `fullmesh_v1/fullmesh_v2`，smoke 失败。
- 因此当前阶段不应把官方 MoE distribute 直接接入真实 decode path；下一步应先确认是否有更新 torch_npu/op-plugin 版本或底层 custom op 能暴露 `comm_alg`，否则只能继续优化当前 broadcast/reduce 或 coordinator fallback all-to-all。

### MoE dispatch/combine microbenchmark

新增 benchmark 脚本：

```bash
torchrun --nproc_per_node=8 scripts/bench_moe_dispatch_npu.py \
  --backend broadcast_reduce \
  --tokens 4,16,64 \
  --hidden 2048 \
  --top-k 8 \
  --experts 128 \
  --output results_npu/ffn_moe_dispatch_bench/broadcast_reduce_ep8_h2048.json
```

产物：

- `results_npu/ffn_moe_dispatch_bench/broadcast_reduce_ep8_h2048.json`
- `results_npu/ffn_moe_dispatch_bench/official_base_ep8_h2048_fail.json`
- `results_npu/ffn_moe_dispatch_bench/official_v2_ep8_h2048_fail.json`
- `results_npu/ffn_moe_dispatch_bench/official_base_ep8_h7168.json`
- `results_npu/ffn_moe_dispatch_bench/summary.csv`
- `results_npu/ffn_moe_dispatch_bench/summary.md`

Host1 EP8 / HCCL / Qwen3 hidden=2048 对比结果：

| Backend | Tokens | Hidden | Top-k | dispatch 中位数 | reduce/combine 中位数 | total 中位数 | 结论 |
|---|---:|---:|---:|---:|---:|---:|---|
| current broadcast/reduce | 4 | 2048 | 8 | 0.212 ms | reduce 0.152 ms | 0.364 ms | 当前真实 EPFFNLayer 等价通信可测。 |
| current broadcast/reduce | 16 | 2048 | 8 | 0.155 ms | reduce 0.147 ms | 0.302 ms | 小 token 下固定 HCCL/launch 开销占主导。 |
| current broadcast/reduce | 64 | 2048 | 8 | 0.149 ms | reduce 0.145 ms | 0.294 ms | payload 增大到 64 tokens 后仍主要受固定开销影响。 |
| official base | 4 | 2048 | 8 | FAIL | FAIL | FAIL | `xShape dims1(H) only supports 7168, but got 2048`。 |
| official v2 | 4 | 2048 | 8 | FAIL | FAIL | FAIL | `commAlg = 0` 无效；当前 schema 又无法传 `comm_alg`。 |
| official base | 4 | 7168 | 2 | 0.184 ms | combine 0.094 ms | 0.279 ms | 支持形状下可跑通且 diff=0，但不是本项目 Qwen3 形状。 |

这个 benchmark 将 `bench-moe-dispatch-comm` 的结论落地为：当前官方 MoE distribute 不能在 Qwen3 hidden=2048 上形成 apples-to-apples 性能对比；只能证明官方 base 在 H=7168 的特定形状可运行。

`bench-moe-dispatch-real-routing` 也因此暂时阻塞：真实 Qwen router 输出的 hidden 维度仍是 2048，官方 base dispatch 在 dispatch 阶段即失败，v2 也卡在 `comm_alg` 暴露问题；即使用真实 routing 也无法进入 official combine。

### 隔离容器版本验证：CANN 8.5.1 + torch_npu 2.9

根据用户要求，版本排查没有修改现有 `afd-npu-test` / `afd-npu-test-h2`，而是在 Host2 新建本任务专用容器：

| 项 | 值 |
|---|---|
| 容器 | `afd-npu-version-probe-torch29-npu` |
| 镜像 | `quay.io/ascend/vllm-ascend:v0.18.0rc1-a3` |
| CANN | `8.5.1`，inner `V100R001C25SPC002B220` |
| torch | `2.9.0+cpu` |
| torch_npu | `2.9.0.post1+gitee7ba04` |
| 产物 | `results_npu/moe_version_probe/` |

关键 schema 变化：

```text
npu_moe_distribute_dispatch_v2(..., str comm_alg="", ...)
npu_moe_distribute_combine_v2(..., str comm_alg="", ...)
```

隔离 probe 结果：

| Case | 结果 | 结论 |
|---|---:|---|
| base H=7168, top_k=2 | total `0.906 ms`，diff=0 | sanity 通过，但不是 Qwen3 shape。 |
| base H=2048, top_k=8 | FAIL：`xShape dims1(H) only supports 7168, but got 2048` | base 版仍不适合 Qwen3 hidden size。 |
| v2 H=2048, default `comm_alg` | total `2336 ms`，diff=0 | 默认能过但极慢，不可作为候选。 |
| v2 H=2048, `comm_alg=fullmesh_v1` | total `1.080 ms`，diff=0 | 可用。 |
| v2 H=2048, `comm_alg=fullmesh_v2` | total `0.482 ms`，diff=0 | 当前最佳 isolated probe 结果。 |
| 更新后的 `bench_moe_dispatch_npu.py`，v2 H=2048, `fullmesh_v2` | warm 后 total `0.860 ms`，diff=0 | 仓库脚本已能条件化传 `--comm-alg` 并跑通。 |

解释：

- 这说明前一次 v2 失败确实和版本/API 配套有关：torch_npu 2.6 schema 没有 `comm_alg`，torch_npu 2.9 schema 已暴露该参数。
- 仅升级到 CANN 8.5.1/torch_npu 2.9 并不修复 base 版 H=2048 限制；真正可用的是 `dispatch_v2/combine_v2 + comm_alg=fullmesh_v2`。
- 这个验证没有改变当前生产/验证容器；如果要接入真实 decode path，需要先在新环境中跑 real routing correctness/perf，再设计 official v2 backend。
- 该镜像未安装 `torchair`，所以没有验证用户提到的静态图尾节点强校验场景；本项目当前 probe 和真实路径是 eager/torchrun 口径。

当前 backend 决策：

- 当前 `afd-npu-test*` 环境仍不实现 `EPFFNLayer` 的 official MoE distribute backend。
- 新环境候选路径是 `npu_moe_distribute_dispatch_v2/combine_v2 + comm_alg=fullmesh_v2`，不是 base 版。
- 短期 FFN 优化仍继续围绕当前 `EPFFNLayer` 的 broadcast/reduce、`ShardedExperts.forward_local()`、post-attn RMSNorm、expert grouped compute、以及 coordinator fallback all-to-all；official v2 backend 需要单独设计和 real routing 验证。

## 下一步计划

1. 保持当前 Attention 侧结论不变：不把 official prefill 单独设为默认；decode 侧继续以 `official IFA + fused RMSNorm/RoPE` 作为优先单层路径。
2. 下一阶段重点转到 FFN：优先分析/优化 `src/model/ffn_worker.py`、`src/model/ep_moe.py`、`src/pipeline/decode_scheduler.py` 中的 post-attention layernorm、dispatch/combine、local expert compute 和 pipeline bubble。
3. 若继续做归因，优先补“单机端到端 TPOT 为什么没吃到 Attention 收益”的 msprof/时间线分析，而不是继续扩大单层 Attention benchmark 覆盖面。
4. MB2 继续作为默认 microbatch 数；MB3/MB4 保留为实验开关，不默认启用。
5. `attention-tp` 在完成拓扑设计前保持 blocked，不做局部硬接实现。
6. 官方 MoE distribute 的可行候选已收敛为新环境下的 v2 + `comm_alg=fullmesh_v2`；在 real routing 与真实 decode path 验证完成前，仍不接入当前生产路径。
7. 只有 Host1 单机端到端 decode TPOT 转正后，才继续跑跨机大 batch decode-DBO，并使用 fresh `MASTER_PORT` / `HCCL_IF_BASE_PORT`。
