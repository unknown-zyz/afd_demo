# AFD Demo 设计文档

## 1. 概述

### 1.1 项目背景

实现 **Attention-FFN Disaggregation (AFD)** + **DBO (Dual Batch Overlap)** 的概念验证。

参考：
- [vLLM AFD #22799](https://github.com/vllm-project/vllm/issues/22799)
- [vLLM DBO #23693](https://github.com/vllm-project/vllm/pull/23693)

### 1.2 设计目标

1. 验证 Attention-FFN 分离架构
2. 实现 2-micro-batch 流水线重叠
3. 支持 KV Cache 和自回归文本生成

### 1.3 当前状态

- ✅ Phase 1: DBO 流水线 (Prefill)
- ✅ Phase 2: KV Cache + 文本生成
- ⏳ Phase 3: MoE 支持

---

## 2. 架构

```
┌─────────────────────┐   NCCL P2P   ┌─────────────────────┐
│   Attention Node    │◄────────────►│     FFN Node        │
│  - Embedding        │              │  - LayerNorm        │
│  - Self-Attention   │              │  - MLP              │
│  - LM Head          │              │                     │
│  - ★ KV Cache ★    │              │                     │
└─────────────────────┘              └─────────────────────┘
```

**每层数据流：**
1. Attention 计算 → Pack [attn_output, residual]
2. Send to FFN (isend)
3. FFN 计算 → output
4. Send to Attention (isend)

**KV Cache**: 仅存储在 Attention 节点，不参与跨节点通信。

---

## 3. 核心组件

| 组件 | 说明 |
|------|------|
| `DistributedContext` | 管理 rank、device、角色 |
| `AFDCommunicator` | 双缓冲 NCCL 通信 |
| `AttentionWorker` | Embedding + Attention + LM Head |
| `FFNWorker` | LayerNorm + MLP |
| `SimplePipelineScheduler` | 同步串行调度 |
| `AsyncPipelineScheduler` | DBO 异步重叠调度 |

---

## 4. DBO 实现

### 4.1 同步 vs 异步

```
同步（SimplePipelineScheduler）:
[Attn_MB0] → [Send] → [FFN_MB0] → [Send] → [Attn_MB1] ...
                ↑ 等待 ↑

异步（AsyncPipelineScheduler）:
[Attn_MB0][Attn_MB1]...
    [isend0]   [isend1]...  ← 非阻塞，与下一个计算重叠
        [FFN_MB0]   [FFN_MB1]...
```

### 4.2 关键技术

1. **CUDA Streams**: compute_stream + comm_stream 分离
2. **isend/irecv**: 非阻塞通信
3. **tensor.clone()**: 避免发送时数据被覆盖
4. **Tag 方案**: `layer_idx * 1000 + mb_idx * 10 + direction`

### 4.3 性能验证

```
环境: 2x GPU, Qwen2-1.5B, batch=4, micro_batches=2

Sync:  166ms
Async: 145ms
Speedup: 1.15x ✅
```

---

## 5. 时间测量

启用 `--timing` 后，记录每个 micro-batch 的：
- Attention 计算时间
- FFN 计算时间
- Send 等待时间
- Recv 等待时间

输出 JSON 到 `results/timing_*.json`，可用 `visualize_dbo.py` 可视化。

---

## 6. KV Cache 与生成

### 6.1 设计决策

- **存储位置**: KV Cache 存储在 Attention 节点
- **理由**: KV 只在 Attention 计算时使用，传输会增加通信量
- **实现**: 使用 HuggingFace DynamicCache

### 6.2 两阶段前向

| 阶段 | 输入 | KV Cache | 说明 |
|------|------|----------|------|
| Prefill | `[B, S]` | 初始化 | 处理完整 prompt |
| Decode | `[B, 1]` | 追加 | 逐 token 生成 |

### 6.3 生成流程

```python
# Prefill
logits, cache = forward_prefill(input_ids)
next_token = sample(logits[:, -1, :])

# Decode loop
for step in range(max_new_tokens):
    logits = forward_decode(next_token)
    next_token = sample(logits[:, -1, :])
```

### 6.4 采样策略

- Temperature scaling
- Top-k filtering
- Top-p (nucleus) sampling

---

## 7. 文件结构

```
src/
├── distributed/
│   ├── context.py      # DistributedContext
│   └── communicator.py # AFDCommunicator
├── model/
│   ├── attention_worker.py
│   ├── ffn_worker.py
│   ├── disaggregated.py
│   └── kv_cache.py     # KVCacheManager (备用)
├── pipeline/
│   ├── scheduler.py      # SimplePipelineScheduler
│   ├── async_scheduler.py # AsyncPipelineScheduler
│   └── micro_batch.py
├── utils/
│   ├── timing.py         # TimingTracker
│   └── sampling.py       # 采样工具
└── main.py
```

---

## 8. 后续计划

- Phase 3: MoE 支持 (Qwen3-30B-A3B)
- Decode 阶段 DBO
- Continuous batching
