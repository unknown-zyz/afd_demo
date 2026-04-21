# DBO 实验报告 v4

## 一、实验环境

| 项目 | 配置 |
|------|------|
| 硬件 | 4 × NVIDIA V100-32GB, PCIe, 单机 |
| 模型 | Qwen3-30B-A3B (MoE, 30B 参数, 3B 激活, 48 层, 128 experts, top-8) |
| GPU 分配 | GPU 0,1 → Attention Node; GPU 2,3 → FFN/MoE Node |
| Micro-batch 数 | 2 |
| Timing | CUDA Events mode（零开销） |
| 框架 | PyTorch 2.7.0, Transformers 5.4.0 |

---

## 二、Prefill DBO 实验

### 2.1 Batch 扩展实验（seq=128）

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

**分析**：

- Batch Size ≥ 8 时 DBO 开始产生明显正收益。bs ≤ 4 时 micro-batch 切分后的计算量太小，通信开销无法被有效隐藏。
- 加速比随 Batch Size 单调递增：b8(+2.7%) → b16(+6.3%) → b32(+9.1%) → b64(+12.1%)。
- bs=128 时 DBO 模式 OOM，原因是 micro-batch 的激活值，显存压力更大。

### 2.2 Seq 扩展实验（batch=4）

**固定参数**: batch_size=4, num_micro_batches=2

| Seq Length | Serial (ms) | DBO (ms) | 加速比 |
|:----------:|:-----------:|:--------:|:------:|
| 64 | 3291.6 | 3273.2 | **+0.6%** |
| 128 | 3436.6 | 3362.3 | **+2.2%** |
| 256 | 3737.0 | 3592.0 | **+3.9%** |
| 512 | 4631.9 | 4331.2 | **+6.5%** |
| 1024 | 6328.6 | 5572.7 | **+11.9%** |
| 2048 | 10576.4 | 8602.3 | **+18.7%** |

**分析**：

- DBO 在所有 Sequence Length 下均产生正收益。
- 加速比随 Sequence Length 显著增长：s64(+0.6%) → s256(+3.9%) → s512(+6.5%) → s1024(+11.9%) → s2048(+18.7%)。
- 序列长度增加时，Attention 计算量按 O(n²) 增长而通信量仅按 O(n) 增长，因此长序列场景最能受益于 DBO。
- seq=2048, bs=4 的加速(+18.7%) 明显超过 bs=64, seq=128(+12.1%)，说明**序列长度比 Batch Size 更能有效提升 DBO 加速比**。

### 2.3 Prefill 结论

- **DBO 收益与计算量正相关**：Batch Size 和 Sequence Length 的增大都能提升 overlap 收益。
- **最高加速**：bs=4, seq=2048 时达到 **+18.7%**。
- **盈亏平衡点**：约 bs=8（seq=128）或 bs=4, seq≥128。
- **显存限制**：DBO 需同时维护两个 micro-batch 的激活值，bs=128 时 OOM。

---

## 三、Decode DBO 实验

### 3.1 全 Batch DBO（旧实现）

**方案**：Decode 阶段不做 micro-batch 切分，全 batch 通信 → 全 batch FFN 计算。

| Batch Size | Serial (ms) | DBO (ms) | 变化 |
|:----------:|:-----------:|:--------:|:------:|
| 1 | 3856.3 | 3868.5 | +0.3% |
| 2 | 7107.3 | 7100.2 | -0.1% |
| 4 | 7134.0 | 7138.9 | +0.1% |
| 8 | 7248.9 | 7267.4 | +0.3% |
| 16 | 7400.9 | 7379.3 | -0.3% |
| 32 | 8120.6 | 8092.6 | -0.3% |
| 64 | 8966.9 | 8946.0 | -0.2% |
| 128 | 10597.1 | 10526.4 | -0.7% |



### 3.2 per-MB Attention DBO（新实现）

**方案**：将 KV cache 按 batch 维度切分为 micro-batch，实现 Decode 阶段与 Prefill 对称的 micro-batch overlap 策略。

| Batch Size | Serial (ms) | DBO (ms) | 变化 |
|:----------:|:-----------:|:--------:|:------:|
| 1 | 4202.5 | 3892.6 | **-7.4%** (1 MB, 无重叠) |
| 2 | 7111.3 | 7140.4 | +0.4% |
| 4 | 7150.6 | 11093.2 | **+55.1%** |
| 8 | 7259.5 | 11242.0 | **+54.9%** |
| 16 | 7406.4 | 11460.8 | **+54.7%** |
| 32 | 8095.4 | 12031.1 | **+48.6%** |
| 64 | 8928.3 | 12825.7 | **+43.7%** |
| 128 | 10541.1 | 14564.2 | **+38.2%** |

**分析**：

- **bs=1 时 -7.4%**：只有 1 个 micro-batch，无重叠开销但减少了同步点，略有收益。
- **bs≥4 时严重恶化（+38%~+55%）**：DBO 变成显著的负优化。

**根因分析**：Decode attention 计算极轻（~2.2ms/MB），而 NCCL 通信开销（~2.6ms/MB）无法被有效重叠。具体来说：
1. **计算量极小**：Decode 每步只 seq=1，每层 Attention 计算仅约 2.2ms。
2. **通信开销大于计算**：每次 NCCL isend/irecv 通信约 2.6ms/MB，compute/comm 比 < 1，无法实现有效 overlap。
3. **额外 NCCL 操作**：micro-batch 切分将 1 次 send+recv 变为 2 次，每层多 2 次 NCCL 操作，48 层 × 20 步 = 1920 次额外操作。
4. **累积开销**：约 **3500-4000ms** 额外固定开销，远大于理v4 overlap 收益。

### 3.3 Decode 结论

- **Decode 阶段 DBO 不适用**。全 batch DBO 无效（±0.7%），per-MB DBO 是负优化（+38%~+55%）。
- **根因**：decode 每 token attention ~2.2ms，通信 ~2.6ms，compute/comm 比 < 1，overlap 无法产生收益。
- **建议**：Decode 阶段始终使用全 batch 串行通信。

---

## 四、Pipeline 空泡分析

 timing 数据的 pipeline 空泡分析（`feat/analyze-pipeline-bubbles` 分支）。

### 4.1 空泡占比

- **空泡占层时间 42%~81%**，是当前 pipeline 效率的主要瓶颈。
- 空 FFN/MoE 节点：FFN 比 Attention 慢约 **3.5×**（86ms vs 25ms/layer）。
- Attention 节点在等待 FFN 完成期间产生大量空闲时间。

### 4.2 跨层流水线分析

- MB0 等待 MB1 仅 **0.2ms/层**，几乎可忽略。
- 跨层流水线（MB0 的 Layer L+1 不等待 MB1 的 Layer L recv 完成）理论收益仅 **0.2%**。
lsAttention 侧的微小优化对端到端影响极小。

### 4.3 优化方向

- **短期**：跨层流水线收益有限（0.2%），不值得优先投入。
- **关键路径**：**FFN/MoE 算子优化**是减少空泡的核心。将 FFN/Attention 时间比从 3.5× 降低到接近 1× 后，跨层流水线和 DBO 的收益将显著提升。

---

## 五、优化实现总结

### 5.1 跨层 MB 流水线（`feat/layer-pipelined-dbo`）

- **状态**：已实现
- **原理**：MB0 的 Layer L+1 不等待 MB1 的 Layer L recv 完成，减少跨层等待。
- **当前收益**：理论仅 0.2%（受限于 FFN/Attention 3.5× 比率）。
   ~1×），收益将显著提升。

### 5.2 Decode per-MB Attention（`feat/decode-mb-attention`）

- **状态**：已实现，实验证明无效
- **原理**：将 KV cache 按 batch 维度切分为 micro-batch，实现 decode 阶段的 compute-communication overlap。
- **实验结果**：bs≥4 时性能恶化 +38%~+55%，是负优化。
- **结论**：decode attention 计算太轻，无法覆盖 NCCL 通信开销。

---

## 六、结论与展望

### 6.1 核心结论

| 实验 | 结果 | 关键数据 |
|------|------|----------|
| Prefill DBO | ✅ 有效 | +2.7% ~ +18.7%（随计算量增大） |
| Decode 全 Batch DBO | ➖ 无效 | ±0.7%（无害但无益） |
| Decode per-MB DBO | ❌ 负优化 | +38% ~ +55% 性能恶化 |
| Pipeline 空泡 | 🔍 已分析 | 空泡占 42%~81%，FFN 是瓶颈 |
| 跨层流水线 | ⏳ 待优化 | 当前理论收益仅 0.2% |

### 6.2 下一步方向

1. **MoE 算子优化**（最高优先级）：减少 FFN/Attention 时间比率（ 3.5× → 目标 ~1×），这是解锁 pipeline 效率和跨层流水线收益的关键。
2. **Prefill 多 micro-batch**：尝试 3-4 个 micro-batch 以进一步提升 overlap 率。
3. **跨层流水线测试**：FFN 优化后重新评估跨层流水线收益。
4. **Decode 优化方向**：放弃 DBO 路线，转向 Speculative Decoding 或连续 batching 等方案。

---

## 七、原始数据

ls`results/experiments_qwen3_v2/summary.csv`
- 全 Batch Decode 基线数据：`results/experiments_qwen3/summary.csv`
- 逐层 timing 数据：`results/prefill_dbo/timing_*.json`
- Pipeline 可视化：`results/experiments_qwen3_v2/prefill/pipeline_dbo_*.png`
