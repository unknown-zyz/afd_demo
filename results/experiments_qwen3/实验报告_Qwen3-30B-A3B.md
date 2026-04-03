# Qwen3-30B-A3B AFD + DBO 实验报告

## 实验概述

本报告对 Qwen3-30B-A3B (MoE) 模型在 AF 分离 (Attention-FFN Disaggregation) 架构下，评估 DBO (Dual Batch Overlap) 流水线调度的效果。

### 实验环境
- **GPU**: 4× Tesla V100-SXM2-32GB (NVLink 互联)
- **节点分配**: GPU 0,1 → Attention 节点；GPU 2,3 → FFN/MoE 节点
- **部署方式**: 单机 4 卡
- **模型**: Qwen3-30B-A3B — 48 层, hidden=2048, 128 experts, 8 active/token
- **模型大小**: ~57GB (磁盘), ~39GB (bf16 GPU 显存)
- **PyTorch**: 2.7.0+cu126, Transformers 5.4.0

### 实验设计
1. **Decode Batch 扩展**: batch=1,2,4,8,16,32,64,128, seq=128, 20 tokens
2. **Prefill Batch 扩展**: batch=1,2,4,8,16,32,64,128, seq=128
3. **Prefill Seq 扩展**: seq=64,128,256,512,1024,2048, batch=4

每个配置分别测试 Serial (同步) 和 DBO (流水线) 模式。

---

## 实验结果

### 1. Decode 阶段 — Batch 扩展

| Batch | Serial (ms) | DBO (ms) | DBO 加速比 | tok/s (Serial) | tok/s (DBO) |
|------:|------------:|---------:|----------:|---------------:|------------:|
| 1     | 3856        | 3868     | -0.32%    | 5.2            | 5.2         |
| 2     | 7107        | 7100     | +0.10%    | 2.8            | 2.8         |
| 4     | 7134        | 7139     | -0.07%    | 2.8            | 2.8         |
| 8     | 7249        | 7267     | -0.26%    | 2.8            | 2.8         |
| 16    | 7401        | 7379     | +0.29%    | 2.7            | 2.7         |
| 32    | 8121        | 8093     | +0.34%    | 2.5            | 2.5         |
| 64    | 8967        | 8946     | +0.23%    | 2.2            | 2.2         |
| 128   | 10597       | 10526    | +0.67%    | 1.9            | 1.9         |
| 256   | 13978       | 13969    | +0.06%    | 1.4            | 1.4         |
| 512   | OOM         | —        | —         | —              | —           |

**分析**:
- Decode 阶段 DBO 效果基本持平（±0.7%），无显著减速
- 大 batch (≥16) 时 DBO 有微弱正向加速（0.2~0.7%），趋势随 batch 增大而增强
- batch=256 成功运行（DBO +0.06%），batch=512 OOM（KV cache 显存不足）
- 这是因为 Decode 阶段已采用全批次通信策略（不做 micro-batching），避免了 cuBLAS 小矩阵性能退化

### 2. Prefill 阶段 — Batch 扩展

| Batch | Serial (ms) | DBO (ms) | DBO 加速比 |
|------:|------------:|---------:|----------:|
| 1     | 1776        | 1866     | -5.1%     |
| 2     | 2597        | 2792     | -7.5%     |
| 4     | 3399        | 3560     | -4.7%     |
| 8     | 3696        | 3768     | -1.9%     |
| 16    | 4360        | 4518     | -3.6%     |
| 32    | 5695        | 5751     | -1.0%     |
| 64    | 8287        | 8305     | -0.2%     |
| 128   | 13364       | OOM      | —         |

**分析**:
- Prefill DBO 加速比为 -0.2%~-7.5%（轻微减速），随 batch 增大减速幅度递减
- **batch=64 时减速仅 0.2%**，接近零损耗
- batch=128 DBO OOM（DBO 需要额外的 micro-batch 缓冲区内存）
- MoE 模型的 DBO 减速幅度显著低于 Dense 模型（Qwen2-1.5B 为 1~10%），因为 MoE 的 FFN 计算量更大，更容易被通信掩盖

### 3. Prefill 阶段 — Seq 扩展 (batch=4)

| Seq   | Serial (ms) | DBO (ms) | DBO 加速比 |
|------:|------------:|---------:|----------:|
| 64    | 3286        | 3440     | -4.7%     |
| 128   | 3391        | 3557     | -4.9%     |
| 256   | 3704        | 3844     | -3.8%     |
| 512   | 4588        | 4691     | -2.3%     |
| 1024  | 6338        | 6381     | -0.7%     |
| 2048  | 10572       | 10610    | -0.4%     |

**分析**:
- DBO 减速幅度随 seq 增大显著递减：从 4.9% (seq=128) 降至 0.4% (seq=2048)
- 长序列时计算量增大，通信的相对占比降低，DBO 的流水线掩盖效果更好
- **seq≥1024 时减速 <1%**，趋近于零

---

### 4. Prefill 阶段 — 多 Micro-batch 对比

测试不同 micro-batch 数量对 DBO 性能的影响（seq=128）：

| Batch | Serial (ms) | MB=2 (ms) | MB=3 (ms) | MB=4 (ms) | MB=2 加速比 | MB=3 加速比 | MB=4 加速比 |
|------:|------------:|----------:|----------:|----------:|-----------:|-----------:|-----------:|
| 32    | 5695        | 5691      | 6765      | 8047      | +0.1%      | -18.8%     | -41.3%     |
| 64    | 8287        | 8322      | 9376      | 10426     | -0.4%      | -13.1%     | -25.8%     |

**分析**:
- **2 micro-batch 是当前最优策略**：加速比接近零损耗
- 3 micro-batch 引入 13~19% 的减速，4 micro-batch 引入 26~41% 的减速
- 更多 micro-batch 导致每个 MB 更小，计算效率下降（cuBLAS 小矩阵 kernel 不高效）
- 额外的通信轮次（每多一个 MB 就多一对 send/recv）进一步增加延迟
- **结论**: 在当前 NVLink 单机环境下，2 micro-batch 已是最佳平衡点

---

## 核心发现

### 1. DBO 在大规模配置下趋近零损耗
- Prefill batch=64: 仅 -0.2% (几乎无损)
- Prefill seq=2048: 仅 -0.4%
- Decode batch=128: **+0.67% 正向加速**

### 2. MoE 模型比 Dense 模型更适合 DBO
Qwen3-30B-A3B 是 MoE 模型（128 experts, 8 active），每个 token 的 FFN 计算量远大于 Dense 模型。较大的计算/通信比使得 DBO 流水线更容易掩盖通信延迟。

### 3. DBO 损耗来源
当前 DBO 的轻微减速来自：
- **Micro-batch 拆分/合并**: 将 batch 拆成 2 个 micro-batch 的额外操作
- **流水线同步点**: micro-batch 之间的依赖等待
- **额外显存**: micro-batch 需要双缓冲区

随着计算量增大（大 batch/长 seq），这些固定损耗的占比趋于零。

### 4. Scaling 趋势
- Decode: 基本持平，大 batch (≥16) 有微弱正向加速（最高 +0.67%），batch=256 可正常运行
- Prefill Batch: 从 -5% (b1) 逐步趋近 0% (b64)
- Prefill Seq: 从 -5% (s64) 逐步趋近 0% (s2048)

### 5. 多 Micro-batch 并非越多越好
- 2 micro-batch 是当前最优：接近零损耗
- 3/4 micro-batch 导致 13~41% 的严重减速
- 原因：cuBLAS 小矩阵 kernel 效率低 + 额外通信轮次

---

## Pipeline 可视化

每个 DBO 配置均生成了 pipeline Gantt 图，展示 Attention/FFN 计算与 A→F/F→A 通信的时序关系：
- `results/experiments_qwen3/prefill/pipeline_dbo_b{N}_s{M}.png`

图上标注了基于实测 Serial 数据计算的 Speedup。

---

## 结论

1. **DBO 在 Qwen3-30B-A3B 上是安全的**：最大减速不超过 7.5%（仅在小 batch/短 seq 时出现），大规模配置下趋近零。

2. **推荐使用场景**：
   - Prefill batch ≥ 32 或 seq ≥ 512 时，DBO 减速 <2%
   - Decode 阶段始终可以安全开启

3. **局限性**：
   - DBO 需要额外显存（Prefill batch=128 DBO OOM，Decode batch=512 OOM）
   - 当前单机 NVLink 通信延迟很低，DBO 的流水线掩盖优势不明显；多机场景（PCIe/网络通信）下 DBO 收益可能更显著

4. **Micro-batch 策略**：
   - 2 micro-batch 是最优选择，3/4 MB 反而大幅退化
   - 不建议增加 micro-batch 数量，应从其他方向优化（如跨层流水线）

5. **后续方向**：
   - 多机实验验证 DBO 在高通信延迟场景下的效果
   - 跨层流水线（cross-layer pipelining）替代多 micro-batch
   - 更大模型的扩展测试
   - 更大模型（A14B 等 MoE 变体）的扩展测试
