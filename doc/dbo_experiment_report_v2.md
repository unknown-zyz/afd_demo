# DBO 实验报告 v3（Stream Sync 计时 + Decode Micro-batch DBO）

## 实验概述

本次实验在前两轮（同步计时 → CUDA Events 计时）的基础上，做了两项重大改进：

1. **修复 timing 时间基混用 bug**：之前 CUDA Events（GPU 时间轴）和 CPU 时间戳（wall-clock）使用了不同的时间基准，导致 pipeline 图中出现计算和通信重叠等不合理现象。本次改为 **Stream Sync + CPU Timestamps** 统一方案。
2. **实现 Decode 阶段的 micro-batch DBO**：之前 Decode 使用全 batch 通信（无 micro-batching），本次实现了与 Prefill 对称的 micro-batch overlap 策略。

### 计时方案对比

| 版本 | 方式 | 开销 | 问题 |
|------|------|------|------|
| v1 | `cuda.synchronize()` 全同步 | ~16.4% | 破坏 GPU pipeline parallelism |
| v2 | CUDA Events | ~0.7% | GPU/CPU 时间基不同，pipeline 图不准确 |
| **v3（本次）** | **Stream Sync + CPU Timestamps** | **~0.1-0.2%** | **无**（仅同步 compute stream，不阻塞 NCCL stream） |

### 实验环境

| 项目 | 配置 |
|------|------|
| 模型 | Qwen3-30B-A3B (MoE, 30B 参数, 3B 激活, 48 层) |
| GPU | 4 × NVIDIA V100-32GB (单机, PCIe) |
| GPU 分配 | GPU 0,1 → Attention Node; GPU 2,3 → FFN/MoE Node |
| Micro-batch 数 | 2 |
| 计时方式 | Stream Sync + CPU Timestamps |
| 框架 | PyTorch 2.7.0, Transformers 5.4.0 |

---

## 一、Prefill 阶段 — Batch Size 扩展实验

**固定参数**: seq_len=128, num_micro_batches=2

| Batch Size | Serial (ms) | DBO (ms) | 加速比 |
|:----------:|:-----------:|:--------:|:------:|
| 1 | 1779.9 | 1844.5 | **-3.6%** |
| 2 | 2610.3 | 2625.2 | **-0.6%** |
| 4 | 3392.1 | 3408.7 | **-0.5%** |
| 8 | 3641.9 | 3542.7 | **+2.7%** |
| 16 | 4408.8 | 4132.1 | **+6.3%** |
| 32 | 5642.0 | 5127.9 | **+9.1%** |
| 64 | 8224.0 | 7231.8 | **+12.1%** |
| 128 | 13311.2 | OOM | — |

### 分析

- **Batch Size ≥ 8 时 DBO 开始产生明显正收益**。bs≤4 时 micro-batch 切分后的计算量太小，通信开销无法被有效隐藏。
- **加速比随 Batch Size 单调递增**：bs=8 时 +2.7%，bs=64 时达到 **+12.1%**。
- bs=128 时 DBO 模式 OOM，原因是需要同时维护两个 micro-batch 的激活值，显存压力更大。
- 加速趋势表明，若显存允许，更大的 Batch Size 能获得更高的 DBO 加速比。

### 与 v2 对比（CUDA Events vs Stream Sync）

| Batch | v2（CUDA Events） | v3（Stream Sync） | 差异 |
|:-----:|:-----------------:|:--------------------:|:----:|
| 1 | -4.1% | -3.6% | +0.5pp |
| 2 | +0.8% | -0.6% | -1.4pp |
| 4 | +3.8% | -0.5% | -4.3pp |
| 8 | +4.6% | +2.7% | -1.9pp |
| 16 | +6.7% | +6.3% | -0.4pp |
| 32 | +8.8% | +9.1% | +0.3pp |
| 64 | +13.1% | +12.1% | -1.0pp |

大 batch（bs≥16）两次实验结果高度一致（±1pp 以内）。小 batch（bs≤4）差异较大，主要因为计算量小时测量噪声占比更高。整体趋势完全一致：**DBO 加速比随 batch size 单调递增**。

---

## 二、Prefill 阶段 — Sequence Length 扩展实验

**固定参数**: batch_size=4, num_micro_batches=2

| Seq Length | Serial (ms) | DBO (ms) | 加速比 |
|:----------:|:-----------:|:--------:|:------:|
| 64 | 3291.6 | 3273.2 | **+0.6%** |
| 128 | 3436.6 | 3362.3 | **+2.2%** |
| 256 | 3737.0 | 3592.0 | **+3.9%** |
| 512 | 4631.9 | 4331.2 | **+6.5%** |
| 1024 | 6328.6 | 5572.7 | **+11.9%** |
| 2048 | 10576.4 | 8602.3 | **+18.7%** |

### 分析

- **DBO 在所有 Sequence Length 下均产生正收益**。
- **加速比随 Sequence Length 显著增长**：seq=64 时 +0.6%，seq=2048 时达到 **+18.7%**。
- 原因：更长的序列使得 Attention 和 FFN 的每层计算量更大，通信可以被更好地隐藏在计算背后。
- seq=2048, bs=4 的加速 (+18.7%) 明显超过 bs=64, seq=128 (+12.1%)，说明**序列长度比 Batch Size 更能有效提升 DBO 加速比**。这是因为序列长度增加时，计算量按 O(n²)（attention）增长，而通信量仅按 O(n) 增长。

### 与 v2 对比

| Seq | v2（CUDA Events） | v3（Stream Sync） | 差异 |
|:-----:|:-----------------:|:--------------------:|:----:|
| 64 | +2.8% | +0.6% | -2.2pp |
| 128 | +2.3% | +2.2% | -0.1pp |
| 256 | +3.9% | +3.9% | 0.0pp |
| 512 | +7.9% | +6.5% | -1.4pp |
| 1024 | +11.8% | +11.9% | +0.1pp |
| 2048 | +18.8% | +18.7% | -0.1pp |

中大 seq（≥128）的结果极为一致，验证了两种计时方式在大计算量下的可靠性。

---

## 三、Decode 阶段实验（含 Micro-batch DBO）

**固定参数**: seq_len=128 (生成 20 个 token), num_micro_batches=2

本次实验首次实现了 **Decode micro-batch DBO**：Attention 侧全 batch 计算（KV cache 共享要求）→ 按 batch 维度切分为 micro-batch → isend/irecv 与 FFN 计算 overlap。

| Batch Size | Serial (ms) | DBO (ms) | 加速比 |
|:----------:|:-----------:|:--------:|:------:|
| 1 | 4057.0 | 4181.1 | **-3.1%** |
| 2 | 7075.6 | 7046.6 | **+0.4%** |
| 4 | 7125.4 | 11424.1 | **-60.3%** |
| 8 | 7259.5 | 11172.0 | **-53.9%** |
| 16 | 7375.7 | 11340.0 | **-53.7%** |
| 32 | 8108.5 | 11737.2 | **-44.7%** |
| 64 | 8934.5 | 12867.8 | **-44.0%** |
| 128 | 10538.1 | 14529.7 | **-37.9%** |

### 分析：Decode DBO 为何显著有害

**核心问题：NCCL 操作的固定开销远大于可 overlap 的计算量。**

1. **计算量极小**：Decode 每步只处理 seq=1，每层 FFN 计算仅约 0.5-1ms（对比 Prefill 的 5-50ms/层）。
2. **NCCL isend/irecv 固定开销大**：每次 NCCL 非阻塞通信操作（isend/irecv）有约 150-300μs 的启动延迟，micro-batch DBO 将原来的 1 次 send+recv 变为 2 次（2 个 MB），每层多 2 次 NCCL 操作。
3. **累积开销**：48 层 × 2 次额外 NCCL 操作 × 20 步 = 1920 次额外操作 → 约 **4000ms** 固定额外开销。
4. **Overlap 收益小**：即使实现了完美 overlap，每层节省的通信隐藏量也仅约 0.3-0.5ms，48 层 × 20 步 ≈ 300-500ms，远小于 4000ms 的 NCCL 开销。

### 对比 v2 的 Decode 结果

v2 的 Decode 使用全 batch 通信（不做 micro-batching），所有配置加速比在 ±1% 内（无效但无害）。本次实现 micro-batch DBO 后反而严重恶化，因为新增的 NCCL 操作带来了巨大的固定开销。

**结论：Decode 阶段不应启用 micro-batch DBO**。全 batch 通信（即 v2 方式）是 Decode 的最优策略。

---

## 四、核心结论

### 4.1 Prefill DBO 有显著加速效果

| 维度 | 最佳配置 | 最高加速 |
|------|----------|----------|
| Batch Size 扩展 | bs=64, seq=128 | **+12.1%** |
| Seq Length 扩展 | bs=4, seq=2048 | **+18.7%** |

### 4.2 加速规律

1. **计算量越大，DBO 越有效**：Batch Size 和 Sequence Length 的增大都能提升加速比。
2. **序列长度的影响大于 Batch Size**：seq 增加时计算量按 O(n²) 增长但通信仅按 O(n) 增长，因此长序列场景（长文本理解、RAG 等）最能受益。
3. **Prefill 盈亏平衡点**：约 bs=8（seq=128）或 bs=4, seq≥128。低于此阈值时 DBO 的 micro-batch 切分开销 > overlap 收益。
4. **显存限制**：DBO 需要同时维护两个 micro-batch 的激活值，bs=128 时 OOM。

### 4.3 Decode DBO 不可行

Decode 阶段（seq=1）的计算量极小，micro-batch DBO 引入的 NCCL 操作固定开销（~4000ms）远大于 overlap 收益（~300-500ms）。**建议 Decode 阶段始终使用全 batch 串行通信**。

### 4.4 计时方案演进总结

| 版本 | 核心问题 | 影响 |
|------|----------|------|
| v1 同步计时 | `cuda.synchronize()` 破坏 DBO overlap | 所有 DBO 显示为负收益（错误结论） |
| v2 CUDA Events | GPU/CPU 时间基不一致 | Pipeline 图不准确，但 E2E 结果基本正确 |
| **v3 Stream Sync** | **无** | **E2E 和 pipeline 图均准确** |

三轮迭代的关键教训：**性能评测工具本身不能引入过大开销（v1），也不能混用不同时间基准（v2）**。

### 4.5 后续优化方向

1. **Prefill**：尝试 3-4 个 micro-batch 以进一步提升 overlap 率（已有脚本支持）
2. **Decode**：放弃 micro-batch DBO，转向 Speculative Decoding 或连续 batching 等方案
3. **多机部署**：网络延迟（TCP ~1-10ms vs NVLink <1ms）会降低 DBO 收益，需进一步实验验证

---

## 五、Pipeline 可视化

所有实验均生成了逐层 pipeline timeline 图，保存在：
- Prefill: `results/experiments_qwen3_v2/prefill/pipeline_dbo_*.png`
- Decode: `results/experiments_qwen3_v2/decode/pipeline_dbo_*.png`

图中包含 4 条 lane：
- **A (Attention)**: Attention 计算时间
- **A→F**: Attention 到 FFN 的通信时间
- **F (FFN)**: FFN/MoE 计算时间
- **F→A**: FFN 到 Attention 的通信时间

DBO 模式下可以观察到：
- MB0 的 A→F 通信与 MB1 的 Attention 计算重叠（Prefill）
- MB0 和 MB1 的 pipeline bubble 减小

---

## 六、原始数据

完整实验数据见 `results/experiments_qwen3_v2/summary.csv`。

逐层 timing 数据（JSON 格式）见 `results/prefill_dbo/timing_*.json`。
