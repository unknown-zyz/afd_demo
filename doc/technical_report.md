# Attention-FFN Disaggregation (AFD) + DBO 流水线优化：技术汇报

## 一、项目概述

### 1.1 研究背景

随着大语言模型（LLM）规模持续增长，传统的 Tensor Parallelism 和 Pipeline Parallelism 难以高效利用异构计算资源。**Attention-FFN Disaggregation (AFD)** 是一种新型分布式推理架构，将 Transformer 的两大核心组件——Self-Attention 和 Feed-Forward Network (FFN/MoE)——分离到不同的计算节点上，使其能够**异步并行执行**，从而通过计算-通信重叠提升推理吞吐。

**Dual Batch Overlap (DBO)** 是基于 AFD 架构的流水线优化策略：将一个 batch 拆分为多个 micro-batch，当一个 micro-batch 在 FFN 节点计算时，Attention 节点可以同时处理下一个 micro-batch，实现**跨节点的计算-通信重叠**。

本项目参考 vLLM 社区的 AFD 提案（[#22799](https://github.com/vllm-project/vllm/issues/22799)）和 DBO 实现（[#23693](https://github.com/vllm-project/vllm/pull/23693)），在 PyTorch 原生框架上实现了完整的概念验证系统，并在 Qwen3-30B-A3B（MoE 模型）上进行了系统性的性能评估。

### 1.2 技术栈

| 组件 | 版本/规格 |
|------|----------|
| PyTorch | 2.7.0 (CUDA 12.6) |
| NCCL | 2.26.2 |
| Transformers | 5.4.0 |
| 模型 | Qwen3-30B-A3B (48层, MoE, bfloat16) |
| 硬件 | 4× NVIDIA V100-32GB (2 Attention + 2 FFN) |
| 通信 | NCCL P2P isend/irecv |

### 1.3 实现里程碑

| 阶段 | 内容 | 状态 |
|------|------|------|
| Phase 1 | AFD 分离架构 + Prefill DBO 流水线 | ✅ 完成 |
| Phase 2 | KV Cache + 自回归文本生成 | ✅ 完成 |
| Phase 3 | MoE 模型适配 (Qwen3-30B-A3B) | ✅ 完成 |
| Phase 4 | Decode DBO + 跨层流水线 | ✅ 完成 |
| Phase 5 | NCCL P2P 预热 + 保活优化 | ✅ 完成 |
| Phase 6 | Pipeline 可视化 + 时钟对齐 | ✅ 完成 |

---

## 二、系统架构

### 2.1 整体架构

系统采用双节点分离架构，每个节点可以使用多 GPU 进行层间分片：

```
┌──────────────────────────────────┐          ┌──────────────────────────────────┐
│       Attention Node (GPU 0,1)   │  NCCL    │         FFN Node (GPU 2,3)       │
│                                  │  P2P     │                                  │
│  ┌────────────────────────────┐  │ isend/   │  ┌────────────────────────────┐  │
│  │  Embedding Layer           │  │ irecv    │  │  Post-Attention LayerNorm  │  │
│  │  Self-Attention × 48 层   │◄─┼──────────┼─►│  MoE/MLP × 48 层          │  │
│  │  Final LayerNorm           │  │          │  │    ├─ Gate (Router)        │  │
│  │  LM Head                   │  │          │  │    ├─ 128 Experts         │  │
│  │  ★ KV Cache               │  │          │  │    └─ Shared Expert       │  │
│  └────────────────────────────┘  │          │  └────────────────────────────┘  │
└──────────────────────────────────┘          └──────────────────────────────────┘
```

### 2.2 模型拆分策略

Transformer 的每一层被**纵向拆分**为 Attention 部分和 FFN 部分：

**Attention Node 持有的组件**：
- `embed_tokens`: Token 嵌入层
- `input_layernorm` × 48: 每层的前置 LayerNorm
- `self_attn` × 48: Self-Attention 模块（含 Q/K/V 投影、RoPE、输出投影）
- `rotary_emb`: 旋转位置编码
- `norm`: 最终 LayerNorm
- `lm_head`: 语言模型头（投影到词表）
- **KV Cache**: 所有层的 Key-Value 缓存（仅存在于 Attention 节点）

**FFN Node 持有的组件**：
- `post_attention_layernorm` × 48: 每层的后 Attention LayerNorm
- `mlp` × 48: MoE 模块
  - `gate`: Router（投影到 128 个专家的 logits）
  - `experts`: 128 个稀疏专家
  - `shared_expert`: 共享稠密专家
  - `shared_expert_gate`: 共享专家门控

**拆分时机**：在 `DisaggregatedQwenModel.load_weights()` 中，加载完整模型后，根据当前节点角色（`is_attention_node`）仅实例化对应的 Worker，释放不需要的参数以节省显存。

### 2.3 多 GPU 层分片

每个节点内部支持多 GPU 分片，将 48 层分配到 2 块 GPU 上：

```python
# 策略：为 NCCL 通信预留 ~6GB 显存开销（约 3 层的参数量）
layers_on_gpu0 = max(1, num_layers // num_devices - 3)  # 21 层
layers_on_gpu1 = num_layers - layers_on_gpu0             # 27 层
```

GPU 0 承载更少的层是因为 NCCL 通信缓冲区（NCCL_BUFFSIZE=32MB）和 P2P 传输的内存开销集中在 rank 0 的 GPU 上。

### 2.4 每层数据流

```
Attention Node                              FFN Node
─────────────                              ─────────
1. input_layernorm(hidden_states)
2. self_attn(normed, position_emb, kv_cache)
   → attn_output, present_kv
3. packed = attn_output + residual          ← 残差预合并，带宽减半
   ─── isend(packed) ──────────────────►    4. irecv(packed)
                                            5. post_attn_layernorm(packed)
                                            6. MoE: gate → route → experts → combine
                                            7. hidden = packed + ffn_output
   ◄─── irecv(hidden) ────────────────     8. isend(hidden)
   → 下一层的 hidden_states
```

**关键优化：残差预合并**

传统做法需要发送 `attn_output` 和 `residual` 两个张量（2×H 带宽）。本系统在 Attention 侧将两者相加后发送 `packed = attn_output + residual`，FFN 侧直接以此作为输入和残差基准，**将 A→F 通信量减半**。

---

## 三、通信子系统

### 3.1 NCCL P2P 异步通信

#### 串行路径：`AFDCommunicator`

`AFDCommunicator`（`src/distributed/communicator.py`）封装了同步通信路径，使用独立 CUDA Stream + 双缓冲实现基本的计算-通信重叠，供 Serial 模式使用。

#### DBO 快速路径：原生 `dist.isend/irecv`

DBO 调度器（`async_scheduler.py` / `decode_scheduler.py`）**直接调用** `dist.isend()` / `dist.irecv()`，绕过 `AFDCommunicator`，以获得更精确的时序控制和更低的开销：

```python
# DBO 调度器中的实际通信代码
handle = dist.isend(tensor.contiguous(), dst=self.ctx.peer_rank, tag=tag)
# ...
handle = dist.irecv(tensor, src=self.ctx.peer_rank, tag=tag)
```

**Tag 编码**：每次通信使用唯一 tag 标识。Prefill 使用 `layer × 1000 + micro_batch × 10 + direction`；Decode 使用 `10000 + layer × (num_mb × 2) + mb × 2 + direction`。

### 3.2 NCCL 冷启动问题与解决方案 `[feat/nccl-warmup 分支]`

> 以下功能仅在 `feat/nccl-warmup` 分支中实现，尚未合入 `main`。

**问题**：NCCL P2P 首次通信存在 20-40ms 的冷启动延迟，原因是 Proxy 线程休眠 + 通道懒加载。这在 DBO 流水线中会导致 Layer 0 的 A→F 通信特别慢。

**解决方案 1：P2P 预热**（`src/distributed/warmup.py`）

```python
def warmup_p2p(rank, peer_rank, device, num_rounds=3):
    """在正式推理前，对所有 GPU 对进行小数据量的双向通信"""
    for i in range(num_rounds):
        if rank < peer_rank:
            handle = dist.isend(small_tensor, peer_rank)
            dist.recv(small_tensor, peer_rank)
        else:
            dist.recv(small_tensor, peer_rank)
            handle = dist.isend(small_tensor, peer_rank)
        handle.wait()
    # 测量冷/热延迟: cold ~25ms → warm ~0.2ms
```

**解决方案 2：内联 Proxy 保活**（`P2PKeepalive` 类）

Decode 阶段 step 间隔可能超过 1 秒，导致 NCCL Proxy 线程再次休眠。保活机制通过后台心跳维持活跃：

```python
class P2PKeepalive(threading.Thread):
    def run(self):
        while not self._stop_event.is_set():
            if time.monotonic() - self._last_comm > self.interval_s:
                # 发送心跳保持 Proxy 活跃
                dist.isend(heartbeat_tensor, self.peer_rank)
                dist.irecv(heartbeat_tensor, self.peer_rank)
```

### 3.3 NCCL 调优参数

| 参数 | 值 | 说明 |
|------|---|------|
| `NCCL_BUFFSIZE` | 33554432 (32MB) | 必须 ≥ 单次传输张量大小，否则触发流控阻塞 |
| `NCCL_P2P_LEVEL` | NVL | 使用 NVLink 直传 |
| `NCCL_SOCKET_TIMEOUT` | 600000 | 长超时避免多机启动失败 |

---

## 四、DBO 流水线实现

### 4.1 串行基线 (Serial)

无 DBO 时，每层的执行严格串行：

```
层 L:  [Attn] → [A→F Send] → [FFN] → [F→A Send] → 层 L+1
       |←──────── 不可重叠，通信阻塞计算 ──────────→|
```

### 4.2 Prefill DBO

将 batch 拆分为 2 个 micro-batch (MB0, MB1)，实现层间流水线：

```
Attention Node:                                FFN Node:
  L1 MB0: [Attn ██████] [send]                  L1 MB0:      [recv] [FFN ██████████████] [send]
  L1 MB1:    [Attn ██████] [send]               L1 MB1:         [recv] [FFN ██████████████] [send]
  L2 MB0:                [wait F→A] [Attn]      L2 MB0:                                   [recv] ...
  L2 MB1:                      [wait F→A] [Attn]
```

**核心思想**：Attention 处理 MB1 的时间窗口内，FFN 可以并行处理 MB0；理想情况下 FFN 处理时间 ≤ Attention 处理时间 + 通信时间，即可实现完全重叠。

**实现细节**（`AsyncPipelineScheduler`）：

1. **Attention 侧**：对每层的每个 MB 依次执行 Attention → isend(A→F)，然后 irecv(F→A) 并等待
2. **FFN 侧**：irecv(A→F) → 等待数据到达 → FFN 计算 → isend(F→A)
3. **计时**：使用 `TimingTracker` 记录每个 MB 每层的 compute、send_transfer、recv_wait 时间
4. **dist.barrier()**：在两侧创建 TimingTracker 前同步，确保时钟基准一致

### 4.3 Decode DBO

Decode 阶段每步仅处理 1 个 token（per batch item），计算量远小于 Prefill。采用**跨层流水线**：

```
Attention Node:
  L0: [MB0 attn][send]  [MB1 attn][send]  [wait MB0 F→A]
  L1: [MB0 attn][send]  [wait MB1 F→A]    [MB1 attn][send]  [wait MB0 F→A]
  L2: [MB0 attn][send]  ...
```

**关键区别**：MB0 可以不等 MB1 完成就进入下一层（跨层独立推进），但最终每层所有 MB 的 F→A 结果都必须收集完才能合并输出。

**KV Cache 分片**：DynamicCache 存储张量为 `[batch, heads, seq, dim]`，DBO 模式下按 batch 维度（第 0 维）切片，每个 MB 只访问自己的 batch 范围：

```python
# batch_size=8, num_mb=2 → MB0: batch[0:4], MB1: batch[4:8]
# 对每层的 keys/values 做 batch 维度切片
layer_cache.keys = orig_keys[mb_start:mb_end]    # [mb_size, heads, seq, dim]
layer_cache.values = orig_values[mb_start:mb_end]
```

---

## 五、性能分析基础设施（Profiling）

### 5.1 TimingTracker

核心计时类，记录每个 micro-batch 在每层的各阶段耗时：

```python
class TimingTracker:
    def __init__(self):
        self.start_time = time.perf_counter()  # 全局基准
        self.events = []  # [(type, layer, mb, start, end, duration_ms), ...]
    
    def record_event(self, event_type, layer, mb, start, end):
        self.events.append({
            "type": event_type,      # "attn_compute" | "ffn_compute" | "recv_wait" | "send_transfer"
            "layer": layer,
            "mb": mb,
            "start": start - self.start_time,  # 相对时间
            "end": end - self.start_time,
            "duration_ms": (end - start) * 1000
        })
```

**两种计时模式**：

| 模式 | 实现 | 开销 | 适用场景 |
|------|------|------|---------|
| `cuda_events` (默认) | `stream.synchronize()` + `perf_counter()` | ~0 | 生产环境 |
| `sync` (传统) | `torch.cuda.synchronize()` | +16.4% | 调试验证 |

### 5.2 Send Transfer 计时

**历史演进**：

1. **v1-v4**：`SendTransferMonitor` 后台线程轮询 `handle.is_completed()`，每 100μs 检测一次
   - 问题：Python GIL 竞争导致测量值膨胀到 20-30ms（实际传输 ~0.2ms）
2. **v5+**：改为直接在 `isend()` 返回时记录时间戳
   - `isend()` 返回代表数据已入队 NCCL 内部缓冲区（~0.2ms）
   - 准确反映了 ATT 侧的实际阻塞时间

### 5.3 输出格式

每次实验生成两个 JSON 文件（Attention 侧 + FFN 侧）：

```json
{
  "node": "attention",
  "num_layers": 48,
  "num_micro_batches": 2,
  "total_time_ms": 3432.08,
  "total_compute_ms": 2891.5,
  "total_recv_wait_ms": 412.3,
  "compute_ratio": 0.842,
  "events": [
    {
      "type": "attn_compute",
      "layer": 1, "mb": 0,
      "start": 0.0312, "end": 0.0534,
      "duration_ms": 22.17,
      "node": "attention"
    },
    ...
  ]
}
```

### 5.4 Pipeline 可视化

`visualize_dbo_pipeline.py` 生成 4 泳道甘特图，直观展示流水线重叠效果：

```
 泳道             时间轴 →
┌─────────┬──────────────────────────────────────────┐
│ A (ATT) │ ██MB0██ ██MB1██    ██MB0██ ██MB1██      │
│ A→F     │    ▮send▮  ▮send▮     ▮send▮  ▮send▮   │
│ F (FFN) │      ████MB0████  ████MB1████  ████...  │
│ F→A     │                ▮▮         ▮▮            │
└─────────┴──────────────────────────────────────────┘
```

**跨进程时钟对齐** `[feat/nccl-warmup 分支]`：ATT 和 FFN 运行在不同进程，虽然 Linux 上 `perf_counter()` 共享 `CLOCK_MONOTONIC`，但进程初始化时间差导致 TimingTracker 的 `start_time` 不同。使用 A→F 通信边界作为锚点，计算中位数偏移量对齐 FFN 时间轴：

```python
# 对齐算法
offsets = []
for layer in visible_layers:
    for mb in micro_batches:
        attn_send_end = attn_events['send_transfer'].end_time
        ffn_recv_start = ffn_events['recv_wait'].start_time
        if ffn_recv_wait_duration < 5ms:  # 可靠锚点
            offsets.append(ffn_recv_start - attn_send_end)
clock_offset = median(offsets)
# 将所有 FFN 事件的时间戳减去 clock_offset
```

---

## 六、实验结果

### 6.1 实验配置

- **模型**：Qwen3-30B-A3B（48 层，128 稀疏专家 + 共享专家，bfloat16）
- **硬件**：4× V100-32GB（GPU 0,1 → Attention，GPU 2,3 → FFN）
- **通信**：NCCL P2P（NCCL_BUFFSIZE=32MB）
- **DBO 参数**：2 micro-batches
- **NCCL 预热**：3 轮（仅 `feat/nccl-warmup` 分支）
- **结果版本**：V6（`feat/nccl-warmup` 分支，含 NCCL 预热 + 时钟对齐 + isend-return 计时）

### 6.2 Prefill 实验结果

#### Sequence Length 扩展（batch=4, seq=128/256/512/1024）

| 配置 | Serial (ms) | DBO (ms) | 加速比 | 延迟降低 % |
|------|------------|----------|--------|-----------|
| b4 s128 | 3568 | 3432 | 1.04x | 3.8% |
| b4 s256 | 3893 | 3524 | 1.10x | 9.5% |
| b4 s512 | 4780 | 4104 | 1.16x | 14.1% |
| b4 s1024 | 6492 | 4988 | **1.30x** | **23.2%** |

#### Batch Size 扩展（seq=128, batch=4/8/64）

| 配置 | Serial (ms) | DBO (ms) | 加速比 | 延迟降低 % |
|------|------------|----------|--------|-----------|
| b4 s128 | 3568 | 3432 | 1.04x | 3.8% |
| b8 s128 | 3833 | 3533 | 1.08x | 7.8% |
| b64 s128 | 8452 | 6445 | **1.31x** | **23.7%** |

**分析**：

- **DBO 加速随计算量增大而显著提升**：seq=1024 或 batch=64 时达到 1.3x
- 底层机制：MoE FFN 计算量随输入 token 数线性增长，更长的 FFN 计算意味着更多可被 Attention 计算"遮盖"的时间
- 小配置（b4s128）加速有限，因为每层 FFN ~25ms，Attention ~15ms，重叠窗口仅能覆盖部分 FFN 时间

### 6.3 Decode 实验结果

> 注：Decode 逐层计时数据取自第 1 个生成 step（`_timing_step=1`），而非所有 step 的平均值。

| 配置 | Serial (ms) | DBO (ms) | 加速比 | tok/s (Serial) | tok/s (DBO) |
|------|------------|----------|--------|---------------|-------------|
| b2 s128 | 7251 | 6348 | **1.14x** | 2.8 | **3.2** |
| b4 s128 | 7324 | 10081 | 0.73x | 2.7 | 2.0 |
| b8 s128 | 7387 | 10191 | 0.73x | 2.7 | 2.0 |
| b16 s128 | 7600 | 10172 | 0.75x | 2.6 | 2.0 |
| b32 s128 | 8326 | 10475 | 0.79x | 2.4 | 1.9 |

**分析**：

Decode DBO 仅在 b=2 时有正收益，b≥4 时性能**显著退化**。根因分析：

| batch | attn/MB (ms) | ffn/MB (ms) | FFN/ATT 比 | ATT 空等比例 |
|-------|-------------|-------------|------------|-------------|
| 2 | 1.96 | 2.14 | **1.09x** | 3% ✅ |
| 4 | 2.51 | 4.60 | **1.83x** | 41% ❌ |
| 8+ | ~2.5 | ~4.6 | ~1.83x | ~41% ❌ |

- **b=2**：每个 MB 仅 1 token，MoE routing 开销主导 → FFN ≈ ATT → 重叠窗口覆盖 95% FFN 时间
- **b≥4**：FFN 计算量随 token 数线性增长（专家计算），ATT（single-position decode query）几乎不变 → FFN 1.83× ATT → 每层 ATT 空等 ~2.1ms → 48 层累积 ~200ms 空转

### 6.4 性能演进对比

| 版本 | Prefill b4s1024 | Decode b2 | 主要改进 |
|------|-----------------|-----------|---------|
| V3 | — | — | 初始实现，基线 |
| V4 | ~1.46x | -44% | 修复 A2F 流控阻塞 |
| V5 | 1.70x | +17.6% | 跨层流水线 + 正确计时 |
| V6 | **1.30x** | **+12.5%** | NCCL 预热 + 精确计时 |

> V6 的 Prefill 加速比低于 V5 是因为 V5 的 serial 基线计时有偏差（串行模式的开销未与 DBO 一致），V6 修正了这一问题，结果更加可信。

### 6.5 Pipeline 可视化示例

以 Prefill b4 s512 为例，pipeline 甘特图展示了有效的计算-通信重叠：

- **Layer 1-4**：MB0 和 MB1 交替在 ATT/FFN 间流水
- **A→F 通信**：~0.3ms（isend 入队时间，非阻塞）
- **FFN 计算**：~35ms/MB（MoE routing + expert dispatch）
- **ATT 计算**：~20ms/MB（Self-Attention + RoPE）
- **时钟偏移**：~24ms（已对齐，反映真实的流水线延迟）

---

## 七、代码结构

```
afd_demo/
├── src/
│   ├── main.py                      # 入口：CLI 参数解析、推理调度
│   ├── model/
│   │   ├── disaggregated.py         # 模型拆分主类，协调 ATT/FFN Worker
│   │   ├── attention_worker.py      # Attention 节点：Embedding + SelfAttn + LMHead
│   │   ├── ffn_worker.py            # FFN 节点：LayerNorm + MoE/MLP + Residual
│   │   └── kv_cache.py              # KV Cache 管理（预分配、分片、重置）
│   ├── distributed/
│   │   ├── communicator.py          # NCCL P2P 异步通信（双缓冲、独立 Stream）
│   │   └── warmup.py                # P2P 预热 + 保活心跳 [feat/nccl-warmup]
│   ├── pipeline/
│   │   ├── scheduler.py             # 串行基线调度器
│   │   ├── async_scheduler.py       # Prefill DBO 调度器
│   │   ├── decode_scheduler.py      # Decode DBO 调度器（跨层流水线）
│   │   └── micro_batch.py           # Micro-batch 拆分与状态管理
│   └── utils/
│       ├── timing.py                # TimingTracker (cuda_events / sync 模式)
│       ├── profiler.py              # GPU 显存和性能分析
│       ├── sampling.py              # Token 采样（greedy / top-p / temperature）
│       └── validation.py            # 输入验证
├── scripts/
│   ├── run_single.sh                # 单次实验运行器
│   ├── run_experiments.sh           # 自动化实验套件 (V3-V6)
│   ├── visualize_dbo_pipeline.py    # 4 泳道 Pipeline 甘特图
│   ├── plot_experiment_results.py   # 实验结果对比图
│   └── measure_transfer_time.py     # P2P 传输延迟测量工具
├── config/
│   └── qwen3_30b.yaml              # 模型和推理配置
├── doc/                             # 设计文档、实验报告
├── results/                         # 实验结果 (V1-V6)
│   ├── experiments_qwen3_v5/        # 主分支最新结果
│   └── experiments_qwen3_v6/        # warmup 分支结果 [feat/nccl-warmup]
└── tests/                           # 单元测试
```

---

## 八、关键设计决策

### 8.1 为什么用 2 个 Micro-batch

- **最小复杂度**：2-MB 是实现流水线重叠的最小单位
- **内存开销小**：仅需 2× 通信缓冲区
- **延迟低**：更多 MB 意味着更多通信开销，对 Decode（计算量极小）不利
- **参考 vLLM DBO**：社区实现也采用 2-MB 方案

### 8.2 为什么采用残差预合并

传统 Transformer：FFN 需要 `residual`（前一层输出）和 `attn_output` 两个输入。分离架构下需要发送两次。

本系统在 Attention 侧直接计算 `packed = attn_output + residual`，一次发送。FFN 侧以 `packed` 同时作为输入和残差基准。**通信量减半**，但要求 FFN 的第二个残差连接使用收到的 `packed` 值。

### 8.3 为什么 KV Cache 仅在 Attention 节点

KV Cache 仅被 Self-Attention 读写，FFN 节点无需访问。将 KV Cache 留在 Attention 节点：
- 避免跨节点传输 Cache（带宽巨大：48层 × batch × seq × heads × head_dim × 2）
- 简化 Cache 一致性管理
- Decode DBO 时对 Cache 按 batch 维度切片即可支持 MB 并行

### 8.4 为什么 Decode DBO 采用跨层流水线

传统层内 DBO（同一层内等两个 MB 都完成再进下一层）会引入同步开销。跨层流水线允许 MB0 在 L+1 层开始时 MB1 仍在 L 层处理，最大化重叠。代价是需要更复杂的 F→A recv 管理。

---

## 九、已知问题与限制

### 9.1 Decode DBO 大 batch 性能退化

**根因**：MoE FFN 计算量随 token 数线性增长，Decode Attention（单位置查询）几乎不变，导致 FFN/ATT 比达到 1.83×，Attention 侧有 41% 时间空等 FFN 返回。

### 9.2 NCCL P2P FIFO 序列化

NCCL 的所有 P2P 操作在内部 stream 上严格 FIFO 执行。尝试为未来层预发 `irecv` 会因 head-of-line blocking 导致死锁（已验证并回退：commit `432329f`）。

### 9.3 V100 计算能力限制

V100 不支持 bfloat16 原生加速（无 Tensor Core BF16 支持），实际以 float32 累加执行，限制了 MoE 专家的计算吞吐。A100/H100 上预期 DBO 收益更大。

---

## 十、后续工作

### 10.1 短期优化

| 方向 | 说明 | 预期收益 |
|------|------|---------|
| **自适应 DBO 开关** | Decode 时根据 FFN/ATT ratio 动态关闭 DBO | 避免 b≥4 decode 退化 |
| **非对称 MB 分割** | 按计算比例做 70/30 而非 50/50 分割 | 平衡 ATT/FFN 时间窗口 |
| **Prefill 多 MB** | 3-4 个 micro-batch 更深流水线 | Prefill 进一步加速 |
| **合并 warmup 到 main** | 将 feat/nccl-warmup 分支合入主分支 | 代码整合 |

### 10.2 中期探索

| 方向 | 说明 |
|------|------|
| **跨 Step 流水线** | Decode 阶段当前 step FFN 和下一 step ATT 重叠 |
| **NVSHMEM 替代方案** | 使用 GPU-initiated 单边 put/get 替代 NCCL P2P，消除 CPU-side 开销 |
| **SHM 通信** | 单机场景使用共享内存（cudaIPC）替代 NCCL，降低延迟 |
| **Chunked Prefill** | 将长序列 Prefill 分 chunk 处理，平衡显存和流水线效率 |

### 10.3 长期方向

| 方向 | 说明 |
|------|------|
| **集成到 vLLM** | 将 AFD 架构集成到 vLLM 推理引擎 |
| **多模型并行** | 支持多个模型实例共享 ATT/FFN 资源池 |
| **A100/H100 适配** | 利用更高带宽和 BF16 Tensor Core |
| **Continuous Batching** | 支持动态 batch 的 AFD 调度 |

---

## 附录

### A. 快速运行

```bash
# 环境准备
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# --- main 分支 ---
# 单机 Serial 基线 (batch=8, seq=128)
./scripts/run_single.sh local 8 128 --tokens 5 --no-dbo

# 单机 Prefill DBO
./scripts/run_single.sh local 8 128 --tokens 5

# --- feat/nccl-warmup 分支 (额外支持 --warmup-p2p) ---
# 单机 Prefill DBO + NCCL 预热
./scripts/run_single.sh local 8 128 --tokens 5 --warmup-p2p --warmup-rounds 3

# 生成 Pipeline 可视化
python scripts/visualize_dbo_pipeline.py \
  --attn-timing results/prefill_dbo/timing_attention_qwen3_dbo_b4_s512.json \
  --ffn-timing results/prefill_dbo/timing_ffn_qwen3_dbo_b4_s512.json \
  -o pipeline.png --start-layer 1 --num-layers 4
```

### B. Git 分支说明

| 分支 | 内容 |
|------|------|
| `main` | 稳定版本，含 V5 实验结果 |
| `feat/nccl-warmup` | NCCL 预热 + 保活 + V6 实验结果 + Pipeline 可视化修复 |
| `feat/nvshmem` | NVSHMEM 通信方案探索（实验性） |

### C. 参考文献

1. vLLM AFD Proposal: https://github.com/vllm-project/vllm/issues/22799
2. vLLM DBO Implementation: https://github.com/vllm-project/vllm/pull/23693
3. Qwen3 Technical Report: https://arxiv.org/abs/2505.09388
4. NCCL Documentation: https://docs.nvidia.com/deeplearning/nccl/
