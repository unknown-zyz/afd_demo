# DBO 实验报告：Qwen2-1.5B 单机测试

## 1. 实验环境

| 项目 | 配置 |
|------|------|
| GPU | 4× Tesla V100-SXM2-32GB，NVLink 互联 |
| 模型 | Qwen2-1.5B（Dense，28 层，hidden=1536） |
| 精度 | bfloat16 |
| 框架 | PyTorch 2.7.0 (cu126)，Transformers 5.4.0 |
| 部署 | AF 分离：GPU 0,1 → Attention；GPU 2,3 → FFN |
| DBO | Prefill: micro-batch=2 + async isend/irecv；Decode: full-batch dist.send/recv |

## 2. 实验设计

本次实验旨在系统测量 DBO（Dual Batch Overlap）在 Prefill 和 Decode 两阶段对推理延迟的影响。

### 2.1 实验维度

| 阶段 | 变量 | 范围 | 固定参数 |
|------|------|------|----------|
| Prefill Batch 扩展 | batch_size | 1, 2, 4, 8, 16, 32, 64, 128 | seq=128 |
| Prefill Seq 扩展 | seq_len | 32, 64, 128, 256, 512, 1024, 2048 | batch=8 |
| Decode Batch 扩展 | batch_size | 1, 2, 4, 8, 16, 32, 64 | seq=128, tokens=20 |

### 2.2 对比方式

每个配置跑两次：
- **DBO ON**：Prefill 使用 `AsyncPipelineScheduler`（2 micro-batch + 异步通信），Decode 使用 `DecodeDBOScheduler`
- **DBO OFF（串行）**：Prefill 使用 `SimplePipelineScheduler`（逐层同步），Decode 直接 `forward_decode()`

## 3. 实验结果

### 3.1 Prefill 阶段：Batch 扩展（seq=128）

| Batch Size | DBO ON (ms) | Serial (ms) | 加速比 | 开销 |
|:----------:|:-----------:|:-----------:|:------:|:----:|
| 1 | 876 | 855 | 0.976x | +2.5% |
| 2 | 1042 | 942 | 0.904x | +10.6% |
| 4 | 1089 | 1018 | 0.935x | +7.0% |
| 8 | 1205 | 1131 | 0.939x | +6.5% |
| 16 | 1517 | 1465 | 0.966x | +3.5% |
| 32 | 2094 | 2071 | 0.989x | +1.1% |
| 64 | 3303 | 3256 | 0.986x | +1.4% |
| 128 | 5823 | 5746 | 0.987x | +1.3% |

**趋势分析**：DBO 在小 batch 时开销较大（batch=2 最高达 10.6%），随 batch 增大开销显著下降。在 batch≥32 时开销收敛到 ~1%，接近串行水平。

### 3.2 Prefill 阶段：Seq 扩展（batch=8）

| Seq Length | DBO ON (ms) | Serial (ms) | 加速比 | 开销 |
|:----------:|:-----------:|:-----------:|:------:|:----:|
| 32 | 1013 | 970 | 0.958x | +4.4% |
| 64 | 1097 | 1018 | 0.928x | +7.8% |
| 128 | 1172 | 1125 | 0.960x | +4.2% |
| 256 | 1544 | 1461 | 0.946x | +5.7% |
| 512 | 2138 | 2097 | 0.981x | +2.0% |
| 1024 | 3536 | 3441 | 0.973x | +2.8% |
| 2048 | 6685 | 6602 | 0.988x | +1.2% |

**趋势分析**：与 batch 扩展类似，seq 越长 DBO 开销越低。seq=2048 时开销仅 1.2%，说明计算量增大后 DBO 管线化的通信开销占比更小。

### 3.3 Decode 阶段：Batch 扩展（seq=128, tokens=20）

| Batch Size | DBO ON (ms) | Serial (ms) | 加速比 | tok/s (DBO) | tok/s (Serial) |
|:----------:|:-----------:|:-----------:|:------:|:-----------:|:--------------:|
| 1 | 2019 | 1986 | 0.984x | 9.9 | 10.1 |
| 2 | 2517 | 2548 | 1.012x | 7.9 | 7.9 |
| 4 | 2534 | 2519 | 0.994x | 7.9 | 7.9 |
| 8 | 2606 | 2564 | 0.984x | 7.7 | 7.8 |
| 16 | 2611 | 2606 | 0.998x | 7.7 | 7.7 |
| 32 | 2978 | 2921 | 0.981x | 6.7 | 6.8 |
| 64 | 3294 | 3270 | 0.993x | 6.1 | 6.1 |

**趋势分析**：Decode DBO 与串行性能基本持平（±2%），这符合预期：
- Decode DBO 已重写为 full-batch 通信（无 micro-batch），开销极小
- 之前的 micro-batch 版本有 -44% 的 cuBLAS kernel 回归，修复后已消除

## 4. 分析与讨论

### 4.1 DBO 开销来源

在 Qwen2-1.5B 上 DBO 未能实现加速，根本原因是**通信延迟无法被计算掩盖**：

| 因素 | 说明 |
|------|------|
| **模型太小** | Qwen2-1.5B 是 dense 模型，每层 FFN 只有一个 MLP（hidden=1536），单层计算仅 ~100-500μs |
| **NVLink 延迟固定** | 单次 NCCL P2P 通信约 150-300μs，与计算量相当甚至更大 |
| **Micro-batch 拆分开销** | 2 micro-batch 的 FFN 计算总量比 1 full-batch 多 ~5-10%（cuBLAS kernel 效率差异） |
| **Python 调度开销** | DBO 管线多了 async 调度、状态机管理等开销 |
| **`cuda.synchronize()` 影响** | timing 模式下的同步点减少了实际的 overlap 效果 |

### 4.2 开销随 Scale 递减的原因

DBO 开销从小 batch 的 ~10% 降至大 batch/seq 的 ~1%，原因是：

```
DBO 开销比 = 固定开销(通信 + 调度) / 计算时间

当 batch/seq 增大 → 计算时间线性增长 → 固定开销占比降低
```

这意味着在更大的模型（如 Qwen3-30B-A3B MoE）上，每层 FFN 包含 MoE 路由 + 多个 Expert 计算，单层计算时间远大于通信延迟，DBO 的 overlap 收益有望超过开销。

### 4.3 Decode DBO 的特殊性

Decode 阶段每个 token 的 seq=1，计算量极小。之前的 micro-batch 方案（拆成 2 个小 batch 交错执行）因 cuBLAS kernel 选择异常导致严重性能回归。改为 full-batch 通信后 DBO 与串行持平，这是合理的结果。

### 4.4 单机 vs 多机

| 场景 | 推荐 | 原因 |
|------|------|------|
| **小模型 (≤7B)** | 单机，关 DBO | 模型放得下单卡/双卡，无需 AF 分离 |
| **中等 MoE 模型** | 单机 AF 分离 + DBO | NVLink 延迟低，DBO 可掩盖通信 |
| **大型 MoE 模型 (>30B)** | 多机 AF 分离 + DBO | 需要更多显存；DBO 在高 FFN 计算量下收益更大 |
| **多机 Dense 模型** | 多机 AF 分离，关 DBO | 跨机通信延迟高，DBO 难以掩盖 |

## 5. 可视化

### 5.1 对比图

- `results/experiments/dbo_summary.png` — 三维度 DBO 加速比综合对比
- `results/experiments/prefill_batch_scaling.png` — Prefill batch 扩展详细对比
- `results/experiments/prefill_seq_scaling.png` — Prefill seq 扩展详细对比
- `results/experiments/decode_batch_scaling.png` — Decode batch 扩展详细对比

### 5.2 Pipeline 时序图

- `results/experiments/prefill/pipeline_dbo_b*_s*.png` — 各配置的 DBO pipeline Gantt 图
  - 4 lane：F→A 接收、FFN 计算、A→F 发送、Attention 计算
  - 跳过 L0（启动开销异常高）

## 6. 结论

1. **Qwen2-1.5B 上 DBO 不能加速推理**，因为该模型太小、计算/通信比不利于 overlap。
2. **DBO 开销与 scale 负相关**：batch/seq 越大，开销越低（从 10% 降至 1%）。
3. **Decode DBO 修复后不引入回归**，与串行性能持平（±2%）。
4. **DBO 的设计目标是大型 MoE 模型**，在 FFN 计算量远大于通信延迟时才能体现 pipeline overlap 收益。
5. 建议在目标模型 Qwen3-30B-A3B 上进行进一步验证。

## 7. 原始数据

完整数据见 `results/experiments/summary.csv`。
