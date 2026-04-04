# Prefill DBO Pipeline 空泡分析

## 一、核心发现

**Pipeline 空泡的主要来源不是 MB0 等待 MB1（仅 0.2ms），而是 Attention 节点等待 FFN 节点完成计算。**

### 数据概览（b32_s128 为例）

| 指标 | 数值 |
|------|------|
| Attention 每层耗时 | ~25ms (2×12ms attn + 2×0.4ms send + 2×0.2ms recv) |
| FFN 每层耗时 | ~86ms (2×40ms FFN/MoE + overhead) |
| FFN/Attention 比率 | **~3.5×** |
| Attention 空闲时间 | ~60ms/层 |
| MB0 等 MB1 的时间 | **仅 0.2ms/层** |
| 空泡占层时间比例 | 65-81% |

### 时序示例（Layer 1, b32_s128）

```
Attention 侧:
  863.8ms  [A(MB0) 12ms][A(MB1) 12ms][send 1ms][recv 0.4ms]  888.6ms
                                                                ↓
                                                        ===== 68ms IDLE =====
                                                                ↓
  957.1ms  [A_L2(MB0) 12ms]...

FFN 侧:
  864.4ms  [recv]  875.9ms  [F(MB0) 40ms][send][recv][F(MB1) 40ms][send]  957.1ms
```

**Attention 在 888.6ms 完成当前层后，必须等到 957.1ms（FFN 处理完两个 MB 后发回结果）才能开始下一层。这 68ms 的空闲是因为 FFN 计算量是 Attention 的 ~3.5 倍。**

---

## 二、详细数据

### 各配置的空泡分析

| 配置 | 层时间(ms) | 空泡(ms) | 等MB1(ms) | 空泡比 | 等MB1比 | E2E(ms) | 理论省时 | 理论加速 |
|:----:|:---------:|:-------:|:--------:|:-----:|:------:|:-------:|:-------:|:-------:|
| b32_s128 | 85.8 | 60.5 | 0.20 | 70.5% | 0.2% | 4662 | 9.1ms | 0.2% |
| b64_s128 | 124.4 | 81.0 | 0.21 | 65.1% | 0.2% | 6506 | 9.8ms | 0.2% |
| b4_s256 | 57.7 | 46.9 | 0.19 | 81.3% | 0.3% | 3305 | 8.7ms | 0.3% |
| b4_s512 | 70.8 | 52.8 | 0.20 | 74.6% | 0.3% | 3979 | 9.0ms | 0.2% |
| b4_s1024 | 94.4 | 59.3 | 0.20 | 62.8% | 0.2% | 5092 | 9.2ms | 0.2% |
| b4_s2048 | 152.5 | 65.2 | 0.20 | 42.8% | 0.1% | 7877 | 9.0ms | 0.1% |

### 关键观察

1. **等 MB1 的时间始终约 0.2ms**，与配置无关。MB1 的 FFN 结果在 MB0 之后几乎立即到达。
2. **空泡比例随计算量增大而降低**：b4_s256(81%) → b4_s2048(42%)。这是因为更大的输入使 FFN/Attention 时间比更均衡。
3. **跨层 MB 流水线（消除等 MB1）的理论收益仅 ~0.2%**，几乎可以忽略。

---

## 三、为什么 MB0 不需要等 MB1

查看代码 (`async_scheduler.py:451-460`)：

```python
for mb_idx, mb in enumerate(micro_batches):
    recv_handle, recv_tensor = self._recv_async(...)
    recv_handle.wait()  # MB0 等 ~0.2ms, MB1 等 ~0.2ms
```

实际数据显示：
- MB0 recv 在 888.2ms 开始，888.4ms 结束（FFN 结果已经在等待了）
- MB1 recv 在 888.4ms 开始，888.6ms 结束（FFN 结果也已经到了）

**原因**：FFN 侧在发送 MB0 结果后立即开始 MB1 计算，等到 Attention 侧完成所有发送并开始接收时，FFN 两个 MB 的结果都已经发送完毕（FFN 每 MB 耗时 40ms，Attention 从发送到接收大约 24ms，而 FFN 在 ~80ms 内处理完两个 MB，远早于 Attention 需要的时间）。

---

## 四、空泡的真正来源

空泡 = Attention 完成所有 recv 到下一层 attn 开始的时间 ≈ **FFN 计算完成时间 - Attention 计算完成时间**

| 组件 | 每层耗时 | 说明 |
|------|---------|------|
| Attention Compute (×2 MB) | ~24ms | 两个 MB 的 attention 计算 |
| A→F Communication | ~1ms | isend 几乎瞬时完成 |
| **FFN/MoE Compute (×2 MB)** | **~80ms** | **瓶颈：MoE expert 计算** |
| F→A Communication | ~1ms | 结果发回 |
| Attention 空闲等待 | ~60ms | **FFN 计算 - Attention 计算 ≈ 80 - 24 = 56ms** |

MoE (Mixture of Experts) 的 expert 计算（每 MB ~30ms）是 FFN 侧耗时的主体。

---

## 五、跨层 MB 流水线的评估

### 5.1 消除等 MB1 的收益

| 优化 | 理论省时/层 | 总省时 (48层) | 占 E2E |
|------|-----------|-------------|--------|
| 不等 MB1 直接开始下一层 | 0.2ms | ~9ms | **0.2%** |

**结论：跨层 MB 流水线对消除等 MB1 的收益可以忽略。**

### 5.2 真正有效的优化方向

| 方向 | 预期收益 | 可行性 |
|------|---------|--------|
| **增加 micro-batch 数** (3→4 MB) | FFN 侧更多 overlap，隐藏通信 | ✅ 已有代码支持 |
| **FFN/MoE 计算优化** | 降低 FFN 每层 80ms 的耗时 | 需要算法优化 |
| **重新平衡负载** | 将部分计算从 FFN 移到 Attention | 架构变更大 |
| **异步跨层流水线** | Attention 提前发送下一层 | 复杂，需重构 |

### 5.3 增加 micro-batch 的潜在收益分析

当前 2-MB DBO 在 FFN 侧的 overlap：

```
FFN:  [recv MB0] → [F(MB0) 40ms] → [isend(MB0) || recv MB1] → [F(MB1) 40ms] → [isend(MB1)]
       ↑ overlap: isend(MB0) 与 F(MB1) 计算重叠，节省 ~0.4ms 通信时间
```

4-MB DBO 可以实现更多 overlap：

```
FFN:  [F(MB0)] → [isend(MB0) || F(MB1)] → [isend(MB1) || F(MB2)] → [isend(MB2) || F(MB3)] → [isend(MB3)]
       ↑ 3 次 overlap vs 当前 1 次
```

但由于每次通信仅 ~0.4ms（PCIe 带宽足够），增加 MB 数对 overlap 的边际收益也有限。

---

## 六、结论

1. **跨层 MB 流水线不值得实现**：等 MB1 仅占 0.2ms，优化后 E2E 收益 < 0.2%。
2. **空泡来自 FFN/Attention 计算不均衡**：FFN (MoE) 耗时是 Attention 的 ~3.5×，Attention 节点大部分时间空闲。
3. **DBO 的真正收益**来自 FFN 侧的 micro-batch overlap（isend || compute），而非 Attention 侧。
4. **最有效的优化**：降低 FFN/MoE 的计算开销、或在 Attention 空闲期间做有用工作（如预取、KV cache 管理等）。
