# AFD Demo 架构设计文档

## 1. 概述

### 1.1 项目背景

实现 **Attention-FFN Disaggregation (AFD)** + **Dual Batch Overlap (DBO)** 的概念验证。

**参考资料**:
- [vLLM AFD #22799](https://github.com/vllm-project/vllm/issues/22799)
- [vLLM DBO #23693](https://github.com/vllm-project/vllm/pull/23693)

### 1.2 设计目标

1. 验证 Attention-FFN 分离架构的可行性
2. 实现 2-micro-batch 流水线重叠优化
3. 支持 KV Cache 和自回归文本生成
4. 适配 MoE 模型（Qwen3-30B-A3B）

### 1.3 实现状态

- ✅ Phase 1: DBO 流水线 (Prefill 阶段)
- ✅ Phase 2: KV Cache + 自回归文本生成
- ✅ Phase 3: MoE 支持 (Qwen3-30B-A3B)
- ✅ Phase 4: Decode DBO 实现

---

## 2. 系统架构

### 2.1 节点分离架构

```
┌─────────────────────┐   NCCL P2P   ┌─────────────────────┐
│   Attention Node    │◄────────────►│     FFN Node        │
│  - Embedding        │              │  - LayerNorm        │
│  - Self-Attention   │              │  - MLP (FFN)        │
│  - LM Head          │              │  - MoE Router       │
│  - ★ KV Cache ★    │              │  - Experts          │
└─────────────────────┘              └─────────────────────┘
```

### 2.2 每层数据流

**前向计算流程**:
1. Attention 节点：`Embedding → Self-Attention → Pack [attn_output, residual]`
2. 通信：`isend(packed_tensor)` → FFN 节点
3. FFN 节点：`Unpack → LayerNorm → MLP/MoE → hidden_states`
4. 通信：`isend(hidden_states)` → Attention 节点
5. 重复所有层，最后 Attention 节点：`LM Head → logits`

**KV Cache 管理**:
- KV Cache 仅存储在 Attention 节点
- 不参与跨节点通信
- 每个 decode step 增量更新

---

## 3. 核心组件

### 3.1 组件概览

| 组件 | 文件 | 说明 |
|------|------|------|
| `DistributedContext` | `src/distributed/context.py` | 管理 rank、device、节点角色 |
| `AFDCommunicator` | `src/distributed/communicator.py` | 双缓冲 NCCL 通信 |
| `AttentionWorker` | `src/model/attention_worker.py` | Embedding + Attention + LM Head |
| `FFNWorker` | `src/model/ffn_worker.py` | LayerNorm + MLP/MoE |
| `KVCacheManager` | `src/model/kv_cache.py` | KV Cache 管理 |
| `SimplePipelineScheduler` | `src/pipeline/scheduler.py` | 同步串行调度（无 DBO） |
| `AsyncPipelineScheduler` | `src/pipeline/async_scheduler.py` | Prefill DBO 调度 |
| `DecodeDBOScheduler` | `src/pipeline/decode_scheduler.py` | Decode DBO 调度 |

### 3.2 Worker 模式

Workers 接收完整的 HuggingFace 模型，提取所需的子模块：

**AttentionWorker 提取**:
- `embed_tokens` (Embedding 层)
- `layers[i].self_attn` (所有层的 Self-Attention)
- `lm_head` (Language Model Head)

**FFNWorker 提取**:
- `layers[i].post_attention_layernorm` (所有层的 LayerNorm)
- `layers[i].mlp` (所有层的 MLP)
- `layers[i].moe` (MoE 模型：Router + Experts)

---

## 4. DBO 流水线实现

### 4.1 同步 vs 异步对比

**同步串行（SimplePipelineScheduler）**:
```
[Attn_MB0] → [Send] → [FFN_MB0] → [Recv] → [Attn_MB1] → [Send] → [FFN_MB1] ...
                ↑ 阻塞等待 ↑
```

**异步重叠（AsyncPipelineScheduler）**:
```
[Attn_MB0][Attn_MB1]...
    [isend0]   [isend1]...  ← 非阻塞，与下一个计算重叠
         [FFN_MB0]   [FFN_MB1]...
```

### 4.2 关键技术

**1. CUDA Streams 分离**:
- `compute_stream`: 计算任务
- `comm_stream`: 通信任务
- 允许计算和通信并发执行

**2. 非阻塞通信**:
- `isend()` / `irecv()`: 异步发送接收
- `wait()`: 在需要时同步

**3. 数据防覆盖**:
- 发送前 `tensor.clone()` 避免数据被后续计算覆盖

**4. 消息 Tag 方案**:
```python
tag = layer_idx * 1000 + mb_idx * 10 + direction
# direction: 0=Attn→FFN, 1=FFN→Attn
```

### 4.3 Micro-batch 状态机

```
WAITING → IN_ATTENTION → SENDING_TO_FFN → IN_FFN → SENDING_TO_ATTN → COMPLETE
```

**状态转换**:
- `WAITING`: 等待开始处理
- `IN_ATTENTION`: Attention 节点计算中
- `SENDING_TO_FFN`: 发送到 FFN 节点
- `IN_FFN`: FFN 节点计算中
- `SENDING_TO_ATTN`: 发送回 Attention 节点
- `COMPLETE`: 该层处理完成

---

## 5. Prefill vs Decode 阶段

### 5.1 Prefill 阶段

**特点**:
- 输入序列长（128-512 tokens）
- 矩阵运算密集（Attention 计算量大）
- 通信占比小，DBO 重叠效果好

**DBO 效果** (单机 Qwen2-1.5B):
- Attention 节点效率：54.8%
- FFN 节点效率：71.2%
- 收益明显，推荐使用

### 5.2 Decode 阶段

**特点**:
- 每次生成 1 个 token
- 计算时间短（毫秒级）
- 对 Python 对象创建开销敏感

**DBO 效果** (单机 Qwen2-1.5B):
- Batch=2: -4% ~ -12% (轻度倒退)
- Batch>=4: -44% ~ -46% (严重倒退)
- **根因**: KV Cache 切片每次创建新对象（2800+ 次/50 tokens）

**优化方向**:
- 缓存 KV Cache 切片对象
- 动态启用策略（仅在高延迟多机环境启用）

---

## 6. MoE 支持

### 6.1 架构适配

**Qwen3-30B-A3B** (Mixture-of-Experts):
- 48 层，每层 128 个专家
- 每 token 激活 8 个专家
- 专家全部放在 FFN 节点

**资源分配**:
- Attention 部分：~2.86GB (6.87%)
- FFN 部分（含 MoE）：~38.68GB (93.13%)
- 每个节点需要 2 × 32GB GPU

### 6.2 FFN Worker 处理

```python
# FFN Worker 自动检测是否为 MoE
if hasattr(layer, 'moe'):
    # MoE 路径：Router + Experts
    router_output = layer.moe.router(hidden_states)
    expert_outputs = layer.moe.experts(...)
else:
    # 标准 MLP 路径
    output = layer.mlp(hidden_states)
```

---

## 7. 通信延迟测量

### 7.1 单机环境

**测试工具**: `scripts/measure_comm_latency.py`

**实测结果** (4 GPU, NVLink/PCIe):
- Round-trip: 1.16ms
- One-way: **0.58ms**
- P95: 0.16ms, P99: 0.18ms

### 7.2 多机环境

**预期延迟**:
- 局域网（千兆以太网）: 10-50ms
- 跨机柜（InfiniBand）: 1-5ms
- 跨数据中心: 50-200ms

**DBO 收益预期**:
- 单机（<1ms）: Prefill 有效，Decode 效果差
- 多机（10-100ms）: Prefill 和 Decode 均有明显收益

---

## 8. 性能特征总结

### 8.1 Prefill DBO

| 指标 | 单机 | 多机（预期） |
|------|------|-------------|
| 通信延迟 | 0.58ms | 10-100ms |
| Attention 效率 | 54.8% | 预期 >70% |
| FFN 效率 | 71.2% | 预期 >80% |
| **推荐** | ✅ 推荐 | ✅ 强烈推荐 |

### 8.2 Decode DBO

| 指标 | 单机 | 多机（预期） |
|------|------|-------------|
| 通信延迟 | 0.58ms | 10-100ms |
| 性能影响 | -44% (严重倒退) | 需优化后测试 |
| **推荐** | ❌ 不推荐 | ⚠️ 优化后再验证 |

### 8.3 建议配置

**生产环境**:
- Prefill: 启用 DBO ✅
- Decode: 禁用 DBO ❌ (当前实现)
- 多机: 优先优化后再部署

**开发测试**:
- 使用 `--no-dbo` 禁用 DBO 进行对比
- 使用 `--timing` 启用详细计时
- 单机测试优先验证功能正确性

---

## 9. 扩展性

### 9.1 支持的模型

当前已验证：
- ✅ Qwen2-1.5B (标准 Transformer)
- ✅ Qwen3-30B-A3B (MoE)

理论支持：
- 任何 HuggingFace Transformers 架构
- 需要 `self_attn`, `mlp`, `post_attention_layernorm` 结构

### 9.2 节点扩展

当前：2 节点（Attention + FFN）

未来扩展：
- 多 FFN 节点（专家并行）
- 多 Attention 节点（Tensor 并行）
- Pipeline 并行（多层分布）

---

## 10. 相关文档

- [使用指南](02-usage.md) - 命令行参数和运行示例
- [API 参考](03-api-reference.md) - 代码接口说明
- [部署指南](04-deployment.md) - 环境配置和资源要求
