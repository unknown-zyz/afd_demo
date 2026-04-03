# DBO 实验报告（CUDA Events 计时版）

## 实验概述

本次实验使用 **CUDA Events** 计时方式（替代之前的 `cuda.synchronize()` 同步计时），重新评估 Dual Batch Overlap (DBO) 在 Attention-FFN 分离架构下的性能表现。

### 关键改进

之前的实验使用同步计时（`cuda.synchronize()`），会在每个计时点强制 CPU-GPU 同步，破坏 GPU pipeline parallelism，导致 DBO 模式下额外产生约 **16.4%** 的开销，完全掩盖了 DBO 的真实加速效果。本次实验使用 CUDA Events 计时，开销仅为 **0.7%**，能真实反映 DBO 的性能。

### 实验环境

| 项目 | 配置 |
|------|------|
| 模型 | Qwen3-30B-A3B (MoE, 30B 参数, 3B 激活) |
| GPU | 4 × NVIDIA V100-32GB (单机) |
| GPU 分配 | GPU 0,1 → Attention Node; GPU 2,3 → FFN/MoE Node |
| Micro-batch 数 | 2 |
| 计时方式 | CUDA Events（近零开销） |
| 框架 | PyTorch 2.7.0, Transformers 5.4.0 |

---

## 一、Prefill 阶段 — Batch Size 扩展实验

**固定参数**: seq_len=128, num_micro_batches=2

| Batch Size | Serial (ms) | DBO (ms) | 加速比 |
|:----------:|:-----------:|:--------:|:------:|
| 1 | 1774.9 | 1846.8 | **-4.1%** |
| 2 | 2617.7 | 2596.0 | **+0.8%** |
| 4 | 3444.6 | 3313.3 | **+3.8%** |
| 8 | 3657.2 | 3489.4 | **+4.6%** |
| 16 | 4399.1 | 4104.8 | **+6.7%** |
| 32 | 5639.3 | 5144.4 | **+8.8%** |
| 64 | 8263.9 | 7182.6 | **+13.1%** |
| 128 | 13385.1 | OOM | — |

### 分析

- **Batch Size ≥ 2 时 DBO 即开始产生正收益**，bs=1 由于 micro-batch 切分后计算量太小，无法掩盖通信开销。
- **加速比随 Batch Size 单调递增**：bs=4 时 +3.8%，bs=64 时达到 **+13.1%**。
- bs=128 时 DBO 模式 OOM，因为 DBO 需要同时维护两个 micro-batch 的激活值，显存占用更高。
- 加速趋势表明，若显存允许，更大的 Batch Size 能获得更高的 DBO 加速比。

### 与旧实验对比（同步计时 vs CUDA Events）

| Batch | 旧实验（同步计时） | 新实验（CUDA Events） | 差异 |
|:-----:|:-----------------:|:--------------------:|:----:|
| 1 | -5.1% | -4.1% | +1.0pp |
| 4 | -3.3% | **+3.8%** | +7.1pp |
| 8 | -2.7% | **+4.6%** | +7.3pp |
| 16 | -1.6% | **+6.7%** | +8.3pp |
| 32 | -0.8% | **+8.8%** | +9.6pp |
| 64 | -0.2% | **+13.1%** | +13.3pp |

旧实验中所有配置均显示为负加速（DBO 无效），实际上是同步计时 +16.4% 的开销完全掩盖了 DBO 的真实收益。

---

## 二、Prefill 阶段 — Sequence Length 扩展实验

**固定参数**: batch_size=4, num_micro_batches=2

| Seq Length | Serial (ms) | DBO (ms) | 加速比 |
|:----------:|:-----------:|:--------:|:------:|
| 64 | 3299.8 | 3207.3 | **+2.8%** |
| 128 | 3398.9 | 3321.4 | **+2.3%** |
| 256 | 3709.3 | 3565.9 | **+3.9%** |
| 512 | 4637.4 | 4271.9 | **+7.9%** |
| 1024 | 6326.7 | 5582.0 | **+11.8%** |
| 2048 | 10553.0 | 8573.1 | **+18.8%** |

### 分析

- **DBO 在所有 Sequence Length 下均产生正收益**。
- **加速比随 Sequence Length 显著增长**：seq=64 时 +2.8%，seq=2048 时达到 **+18.8%**。
- 原因：更长的序列使得 Attention 和 FFN 的计算量更大，通信/计算比下降，DBO 的 overlap 空间更大。
- seq=2048, bs=4 的加速 (+18.8%) 甚至超过了 bs=64, seq=128 (+13.1%)，说明**序列长度比 Batch Size 更能有效提升 DBO 加速比**。

---

## 三、Decode 阶段实验

**固定参数**: seq_len=128 (生成 20 个 token), num_micro_batches=2

| Batch Size | Serial (ms) | DBO (ms) | 加速比 |
|:----------:|:-----------:|:--------:|:------:|
| 1 | 3930.2 | 3857.4 | +1.9% |
| 2 | 7106.2 | 7130.2 | -0.3% |
| 4 | 7157.1 | 7162.6 | -0.1% |
| 8 | 7295.3 | 7262.9 | +0.4% |
| 16 | 7399.5 | 7447.7 | -0.6% |
| 32 | 8066.7 | 8068.9 | 0.0% |
| 64 | 8926.7 | 8961.6 | -0.4% |
| 128 | 10571.2 | 10566.1 | 0.0% |

### 分析

- **Decode 阶段 DBO 几乎无效**，所有配置的加速比在 ±1% 以内，属于测量噪声。
- 原因分析：
  1. Decode 每步只处理 1 个 token（seq=1），**计算量极小**，通信/计算比过高。
  2. 当前 Decode DBO 使用全 batch 通信（不做 micro-batching），无法实现有效 overlap。
  3. cuBLAS 对小矩阵（seq=1）的 kernel launch overhead 使得 2×FFN(b/2) ≈ FFN(b)，切分无收益。

---

## 四、核心结论

### 4.1 DBO 在 Prefill 阶段有显著加速效果

| 维度 | 最佳配置 | 最高加速 |
|------|----------|----------|
| Batch Size 扩展 | bs=64, seq=128 | **+13.1%** |
| Seq Length 扩展 | bs=4, seq=2048 | **+18.8%** |

### 4.2 加速规律

1. **计算量越大，DBO 越有效**：Batch Size 和 Sequence Length 的增大都能提升加速比。
2. **序列长度的影响大于 Batch Size**：seq=2048 时加速 18.8%，说明长序列场景（如长文本理解、RAG）最能受益。
3. **存在下限**：bs=1, seq=128 时 DBO 有 -4.1% 的负收益（micro-batch 切分开销 > overlap 收益）。

### 4.3 计时方式的重要性

同步计时引入 +16.4% 额外开销，完全掩盖了 DBO 的真实加速效果。使用 CUDA Events（0.7% 开销）后：
- 旧结论「DBO 在所有配置下均无效」→ 新结论「DBO 在大多数 Prefill 配置下有效，最高加速 18.8%」。
- 这提醒我们：**性能评测工具本身不能引入过大开销，否则会得出错误结论**。

### 4.4 Decode 阶段的优化方向

当前 Decode DBO 无显著效果。可能的改进方向：
- 增大 Decode 的计算量（如 Speculative Decoding、更大的 batch 聚合）
- 在 Decode 阶段使用异步通信（NCCL async send/recv）替代当前的阻塞通信
- 探索 Decode 阶段的 micro-batching 策略

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

---

## 六、原始数据

完整实验数据见 `results/experiments_qwen3_v2/summary.csv`。

逐层 timing 数据（JSON 格式）见 `results/prefill_dbo/timing_*.json`。
