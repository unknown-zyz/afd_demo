# Prefill DBO 可视化中 L1 异常耗时：根因分析与修复

## 1. 现象

在 `results_npu/prefill-dbo/` 生成的 4 泳道甘特图中，**Layer 1 的 attention 与 FFN 柱远远长于其他层**，导致：

- DBO 每一步总时长被拉到 ~2.4 s（真实值应在 0.5–0.6 s 量级）
- 与 serial baseline 对比的 speedup 被错算为 **0.17×**（应 ≥ 1×）
- 整张图里 L1 占据了 > 80% 的宽度，L2–L4 被压成一条细线

以 `pipeline_prefill-dbo_b8_s512_t20.png` 为例（修复前）：

```
L0 (skipped in title but present in data):
  attn mb0 = 316.1 ms   attn mb1 = 1.5 ms
  ffn  mb0 = 366.4 ms   ffn  mb1 = 8.4 ms
L1:
  attn mb0 = 463.1 ms   attn mb1 = 2.9 ms     ← 异常
  ffn  mb0 = 125.7 ms   ffn  mb1 = 6.2 ms     ← 异常
L2 起：
  attn mb0 ≈ 2.4 ms     ffn  mb0 ≈ 11 ms     ← 正常
```

同配置在 GPU 上（`results/prefill-dbo/`）L0/L1 没有任何异常，speedup 稳定在 1× 附近。
所以 GPU 看不见这个问题 —— **是 NPU 特有**。

## 2. 根因：Ascend 910C 的 per-shape 图编译

Ascend CANN / torch_npu 的执行模型是：每当遇到一个从未见过的 **(op, 输入 shape, dtype, format)** 组合，就会在运行时 JIT 一段 Ascend-C kernel / TBE 算子图并缓存。第一次遇到要编译，之后命中缓存就只跑 kernel。

`src/pipeline/async_scheduler.py::AsyncPipelineScheduler.run()` 在实验里只被 `src/main.py:306` 调用**一次**，并且在这一次内部 `TimingTracker` 已经开始记录。所以所有 JIT 编译开销都落在 **第一个 step 的前几层的 mb0 里**：

| 层 | op 首次遇到的原因 |
|----|-------------------|
| L0 mb0 attn / FFN | `[B, S, H]` 输入的 QKV 投影、RoPE、mask、MoE router、grouped-matmul、residual 第一次出现 |
| L0 mb1 attn / FFN | 同 shape，命中 L0 mb0 的缓存，1.5 / 8 ms |
| L1 mb0 attn / FFN | hidden state 来自 L0 输出而非 embedding，NPU 把它当作新的 tensor 格式（stride / layout / storage format 不同），**再编一次** |
| L1 mb1 | 命中 L1 mb0 缓存，2.9 / 6 ms |
| L2 mb0+ | 完全命中，≈ 2.4 / 11 ms |

从 L2 开始每层的 hidden state 都来自"前一个 DecoderLayer 的输出"，layout 和 L1 输入一致，因此不再触发新编译。

> 这里"格式差异"在 Ascend 上是真实存在的：torch_npu 会把 tensor 打成 ND / NZ / FRACTAL_NZ 等内部 layout；embedding 输出和 DecoderLayer 输出在某些路径下会走不同的内部存储格式，从而被算子编译器当作不同的 shape key。

### 为什么 GPU 上没有

- CUDA 大部分 kernel 是 ahead-of-time 编译好的，相当于 `cudaLaunchKernel` 直接拉 PTX
- torch.compile / Inductor 在 GPU 上默认 eager 执行，不走这类 per-shape JIT
- 少量首次 cuBLAS algo search 开销几十毫秒，远小于 NPU 的数百毫秒，且不会在 L1 又复发

## 3. 修复

采取"双保险"：根治 + 兜底。

### 3.1 根治：scheduler 增加 warmup pass（`src/main.py`）

新增 CLI：

```
--prefill-warmup-rounds N    # 默认：NPU=1，CUDA/CPU=0
```

实现逻辑（`src/main.py` 内）：

```python
warmup_rounds = args.prefill_warmup_rounds
if warmup_rounds is None:
    warmup_rounds = 1 if args.backend == 'npu' else 0
if warmup_rounds > 0:
    saved_timing = scheduler.enable_timing
    scheduler.enable_timing = False           # 关掉 tracker
    for _ in range(warmup_rounds):
        run_with_scheduler(scheduler)         # 吃掉 JIT
    scheduler.enable_timing = saved_timing    # 恢复
    ctx.barrier(); devmod.synchronize()       # 两节点对齐
output, elapsed = run_with_scheduler(scheduler)  # 真正计时
```

关键点：

- warmup 必须用**完全相同的 input shape**（代码里复用同一个 `input_ids` / `attention_mask`），否则 JIT 会再编一次
- warmup 期间 `enable_timing=False`，`AsyncPipelineScheduler.run()` 就不会构造 `TimingTracker`，也不会污染 `self.stats`（每次 run 重置）
- 节点间必须再 barrier + sync 一次，保证两侧的 `TimingTracker` 基准时间对齐
- **只 1 轮就够**：实测 L0/L1 都在第 1 轮里完成编译，第 2 轮所有层都命中缓存

### 3.2 兜底：可视化器自动跳过 warmup 层（`scripts/visualize_dbo_pipeline.py`）

默认开启 `--auto-skip-warmup`（可用 `--no-auto-skip-warmup` 关闭）。逻辑：

1. 读取 attention timing JSON，取所有层的 `attn_compute` mb0 时长
2. 以第 4 层起的中位数为参考值 `m`
3. 从 `start_layer` 起，若该层 mb0 > 5 × m，则判定为 warmup，`start_layer += 1`
4. 直到遇到第一个正常层为止

这样 **历史 JSON 也能出干净图**，不必重跑。修复后的 JSON 因为 L0/L1 都已正常，auto-skip 不会生效，等价于 `start_layer = 0`（但 `plot_all_pipelines.py` 里 prefill-dbo 默认仍给 `start_layer=1`，保留一个 L0 缓冲）。

title 里也会显示真实跳过的区间，例如：

- 修复前 + auto-skip：`L2–L4 (L0–L1 skipped), 2 Micro-batches`
- 修复后：`L1–L4 (L0 skipped), 2 Micro-batches`（plot_all_pipelines 传入的 start_layer=1）

## 4. 修复前后对比

### 单配置数值（`b8_s512_t20`）

| 指标 | 修复前 | 修复后 | 变化 |
|---|---|---|---|
| L0 mb0 attn | 316.1 ms | 1.4 ms | **−225×** |
| L0 mb0 FFN | 366.4 ms | ~2 ms | **−180×** |
| L1 mb0 attn | 463.1 ms | 2.5 ms | **−185×** |
| L1 mb0 FFN | 125.7 ms | ~5 ms | **−25×** |
| 端到端 step | 2432 ms | 565 ms | **−4.3×** |
| 与 serial 的 speedup | 0.17× | ≥ 1× | 恢复正常 |

### 可视化图

- 修复前：`results_npu/prefill-dbo/pipeline_prefill-dbo_b8_s512_t20.png` 的 L1 绿色巨块
- 修复后：同名 PNG 重新生成后，L1–L4 四层等宽、pipeline 重叠清晰可见
- 兜底：老 JSON 用 `--auto-skip-warmup` 重画后自动跳到 L2，也能出干净图

## 5. 复现

### 5.1 NPU 上跑带 warmup 的 prefill-dbo

```bash
# 在 afd-npu-fp8 / zhangyz-npu-1 容器里：
cd /workspace/afd_demo
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 单 config smoke
./scripts/run_npu.sh --attn-size 1 --ffn-size 1 --ffn-tp-size 1 \
    --batch 8 --seq 512 --tokens 20 \
    --model-name /models/Qwen3-30B-A3B --no-generate

# 全 prefill-dbo 矩阵
./scripts/run_experiment_matrix_npu.sh --modes serial,prefill-dbo
```

日志里会看到：

```
AsyncPipelineScheduler initialized: num_mb=2, use_cuda_streams=True, timing=True
Running 1 prefill warmup round(s) to absorb JIT compile cost
  warmup 1/1: 3206.2 ms
[DBO] prefill_time=583.93ms
```

### 5.2 只重新画图（不跑新实验）

```bash
# 单张图，明确跳过 warmup
python scripts/visualize_dbo_pipeline.py \
  --attn-timing results_npu/prefill-dbo/timing_attention_prefill-dbo_b8_s512_t20.json \
  --ffn-timing  results_npu/prefill-dbo/timing_ffn_prefill-dbo_b8_s512_t20.json \
  --output /tmp/clean.png \
  --num-layers 5

# 批量重绘
python scripts/plot_all_pipelines.py --root results_npu
```

### 5.3 关闭自动跳过（用于调试）

```bash
python scripts/visualize_dbo_pipeline.py --no-auto-skip-warmup ...
```

## 6. 兼容性 / 回退

- CUDA backend：`--prefill-warmup-rounds` 默认 0，行为和修复前一致，GPU 结果不受影响
- 历史 NPU JSON：可视化器 auto-skip 默认开启，不改 JSON 也能画干净图
- Serial scheduler：warmup 逻辑对 `SimplePipelineScheduler` 同样生效（`hasattr(scheduler, 'enable_timing')` 保护）；serial 模式一般不开 `--timing` 所以影响有限

## 7. 其它可能受影响的路径

- **Decode DBO**：`DecodePipelineScheduler` 已经有 `_timing_step = 1`（`src/pipeline/decode_scheduler.py:103-104`）的内置 warmup（丢掉第 0 个 decode step），所以 decode 图不受此问题影响。本次修复不涉及 decode。
- **Cross-layer prefetch**：warmup 也会预热 irecv 路径；若用户在 prefill 加了 crosslayer，建议把 `--prefill-warmup-rounds` 提到 2，第一轮跑通 L0 的 HCCL，第二轮再缓存新 shape 的 irecv/isend，再走 timed。

## 8. 不做的事

- 不改 `TimingTracker` 本身（问题不在记录方式，而在被记录的第一 step 里混有 JIT）
- 不改 serial 数据或 serial scheduler（serial 没有层间对比图）
- 不把 warmup 默认值在 CUDA 上打开（会让 e2e 延迟 +1 轮，得不偿失）
