# 通信延迟分析

## 问题概述

在单机 DBO 测试中发现：
- **A2F 传输时间**: ~5.95ms (平均)
- **F2A 传输时间**: ~0.55ms (平均)
- **时间比例**: 10.8x 差距

## GPU 拓扑

```
GPU0 ↔ GPU1: NV2 (2 条 NVLink)
GPU0 ↔ GPU2: NV2
GPU0 ↔ GPU3: NV2
...所有 GPU 间都是 NV2
```

理论上所有 GPU 间的链路是对称的，不应该有如此大的差异。

## 数据传输大小

| 方向 | Shape | 大小 | 理论时间 (100GB/s) |
|------|-------|------|--------------------|
| A2F | [4, 1, 4096] | 32 KB | 0.0003 ms |
| F2A | [4, 1, 2048] | 16 KB | 0.0002 ms |

数据量比例仅为 2:1，但传输时间比例达到 10.8:1。

## 根本原因

### 1. 小数据传输效率低

**实际带宽**：
- A2F: 0.01 GB/s (理论的 0.01%)
- F2A: 0.03 GB/s (理论的 0.03%)

**原因**：
- NCCL 协议开销：每次 `isend`/`irecv` 都有初始化、同步等开销
- 小数据 (16-32 KB) 无法充分利用 100 GB/s 的 NVLink 带宽
- 类似"发一封信的时间不在于信的内容多少，而在于邮递过程"

### 2. 为什么 A2F 比 F2A 慢得多？

**时序分析** (Layer 1, MB0 为例):

#### A2F 传输 (Attention → FFN)
```
763.46ms: Attn 发起 isend          ← 发送方准备好
765.99ms: FFN 发起 irecv           ← 接收方延迟 2.53ms 才准备好
766.53ms: FFN irecv 完成           ← 接收完成 (耗时 0.54ms)
768.86ms: Attn send_transfer 完成  ← 发送端监控线程检测到完成 (耗时 5.40ms)
```

**关键问题**：FFN 节点在 Attn 发起 isend 之后 **2.53ms** 才发起 irecv
- 发送方必须等待接收方准备好才能开始传输
- Attn 的 send_transfer 时间 (5.40ms) 包含：
  - 等待 FFN 准备好接收的时间 (~2.5ms)
  - NCCL 协议开销
  - 真实数据传输时间 (~0.5ms，与 FFN recv 时间相近)

#### F2A 传输 (FFN → Attention)
```
771.12ms: Attn 发起 irecv          ← 接收方已准备好
771.31ms: FFN 发起 isend           ← 发送方延迟 0.18ms 才准备好
771.33ms: Attn irecv 完成          ← 接收完成 (耗时 0.20ms)
771.91ms: FFN send_transfer 完成   ← 发送完成 (耗时 0.61ms)
```

**为什么快**：Attn 节点在 FFN 发起 isend **之前** 0.18ms 就已经发起 irecv
- 接收方已就绪，发送方可以立即传输
- FFN 的 send_transfer 时间 (0.61ms) 是真实传输时间
- Attn 的 recv 时间 (0.20ms) 只是检查完成的开销

#### 根本原因

**A2F 慢的本质**：发送方等待接收方 (Producer-Consumer 不匹配)
- Attention 计算完成后立即 isend (763.46ms)
- 但 FFN 要等到 765.99ms 才发起 irecv
- 这 2.53ms 的延迟是 FFN 节点的调度开销

**F2A 快的本质**：接收方提前就绪 (良好的 Pipeline 重叠)
- Attention 在等待 FFN 结果时已经提前发起 irecv (771.12ms)
- FFN 计算完成后立即 isend (771.31ms)
- 几乎无等待延迟

### 3. recv_wait 时间很短的原因

查看 Attention 节点的 `recv_wait`:
```python
recv_start = time.perf_counter()
tag = self._get_tag(layer_idx, mb_idx, "ffn_to_attn")
recv_handle, recv_tensor = self._recv_async(...)  # 发起 irecv
recv_handle.wait()                                  # 等待完成
recv_end = time.perf_counter()
```

**recv_wait 只有 0.2ms 的原因**：
- `irecv` 发起后，可能立即返回 (异步)
- 在 `wait()` 调用时，数据已经在 FFN 计算期间传输完成
- 所以 recv_wait 测得的是 "irecv 发起 + 检查完成" 的时间，不是真实传输时间

## Pipeline 时序验证

### 问题：Layer 2 是否在 Layer 1 完成前开始？

**检查结果**：✅ **时序正确**

```
Layer 1 recv 最后完成: 771.51ms
Layer 2 attn 最早开始: 771.59ms
时间差: 0.07ms
```

Layer 2 的 Attention 计算在 Layer 1 的所有 recv 完成后才开始，符合预期。

**可视化中的"重叠"假象**：
- 可能是因为 Layer 1 的 F2A `send_transfer` 还在进行 (771.91ms 才完成)
- 但 Attention 节点的 `recv_wait` 已经完成 (771.51ms)
- 这是正常的：`send_transfer` 是 FFN 节点的监控线程记录的，Attention 节点不需要等它

## 结论

1. **A2F 延迟高的根本原因**：
   - 小数据传输 (32KB) 的 NCCL 协议开销占主导
   - 实际带宽仅为理论的 0.01%

2. **F2A 延迟低的原因**：
   - FFN 计算期间数据已经传输完成
   - `recv_wait` 测得的只是检查完成的时间

3. **Pipeline 时序正确**：
   - 每层都正确等待上一层的 recv 完成
   - 可视化中的"重叠"是正常的异步行为

## 优化建议

### 1. ✅ 提前发起 irecv (最有效)
**问题**：A2F 传输中，FFN 节点延迟 2.53ms 才发起 irecv，导致发送方等待

**解决方案**：
```python
# 在 FFN 节点处理上一个 micro-batch 时，提前发起下一个 micro-batch 的 irecv
# 类似 F2A 的成功模式
```

**预期效果**：A2F 延迟从 5.4ms 降低到 0.6ms (与 F2A 相当)

### 2. 增加 Batch Size 或 Sequence Length
- 当前：batch=4, seq=1 → 32KB
- 如果 batch=32, seq=128 → 32MB
- 更大的数据量可以更好地利用带宽

### 3. 双缓冲优化
- FFN 节点使用双缓冲区
- 在处理当前 MB 时，同时接收下一个 MB
- 减少等待时间

### 4. 考虑使用 RDMA 或 GPUDirect
- 如果是多机环境，使用 InfiniBand + GPUDirect RDMA
- 可以绕过 CPU，直接 GPU-to-GPU 传输

## 当前测试配置的限制

本测试使用：
- batch=4, seq=1 (prefill 第一个 token)
- 数据量极小 (16-32 KB)
- 主要测量 DBO 的 pipeline 逻辑，不是通信性能

**对于真实推理场景**：
- Prefill: seq=128-2048，数据量更大
- Decode: batch=32-64，虽然 seq=1 但 batch 更大
- 通信开销占比会明显下降
