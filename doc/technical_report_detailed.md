# AFD + DBO 详细技术手册（Code Review Handbook）

> **文档定位**：`doc/technical_report.md` 的深度版本，面向想系统 review 代码、理解每一条设计决策背后原理的读者。
> 叙述为中文，代码、标识符、术语保留英文。所有关键点都给出 `文件:行号` 可跳转引用。
>
> **简明版**：`doc/technical_report.md`（575 行，读概念和结论）
> **详细版（本文）**：按源码结构逐层展开，配合时序图和 review 要点清单。

---

## 目录

- [第一部分：导读与定位](#第一部分导读与定位)
  - [§1 读者路径](#1-读者路径)
  - [§2 与其他文档的关系](#2-与其他文档的关系)
  - [§3 仓库总览](#3-仓库总览)
- [第二部分：理论基础](#第二部分理论基础)
  - [§4 AFD 架构的动机与数学](#4-afd-架构的动机与数学)
  - [§5 DBO 的 overlap 模型](#5-dbo-的-overlap-模型)
  - [§6 Cross-layer Pipeline 的 bubble 分析](#6-cross-layer-pipeline-的-bubble-分析)
- [第三部分：源码逐层走读](#第三部分源码逐层走读)
  - [§7 入口：src/main.py](#7-入口srcmainpy)
  - [§8 模型编排：src/model/disaggregated.py](#8-模型编排srcmodeldisaggregatedpy)
  - [§9 两个 Worker](#9-两个-worker)
  - [§10 KV Cache 机制](#10-kv-cache-机制)
  - [§11 分布式底层](#11-分布式底层)
  - [§12 Prefill 调度：AsyncPipelineScheduler](#12-prefill-调度asyncpipelinescheduler)
  - [§13 Decode 调度：DecodeDBOScheduler](#13-decode-调度decodedboscheduler)
  - [§14 串行 baseline：PipelineScheduler](#14-串行-baselinepipelinescheduler)
- [第四部分：Profiling 与可视化](#第四部分profiling-与可视化)
  - [§15 TimingTracker 双模式](#15-timingtracker-双模式)
  - [§16 SendTransferMonitor 的真实传输时间](#16-sendtransfermonitor-的真实传输时间)
  - [§17 可视化脚本](#17-可视化脚本)
- [第五部分：实验](#第五部分实验)
  - [§18 启动流程](#18-启动流程)
  - [§19 NCCL warmup 的必要性](#19-nccl-warmup-的必要性)
  - [§20 GPU 上的 DBO 结论](#20-gpu-上的-dbo-结论)
- [第六部分：NPU-910C 移植](#第六部分npu-910c-移植)
  - [§21 分支与差异](#21-分支与差异)
  - [§22 容器与设备](#22-容器与设备)
  - [§23 NPU 上的实验结论](#23-npu-上的实验结论)
- [第七部分：FP8 量化尝试](#第七部分fp8-量化尝试)
  - [§24 动机与方案](#24-动机与方案)
  - [§25 阻塞点小结](#25-阻塞点小结)
- [第八部分：Review 清单](#第八部分review-清单)
  - [§26 架构正确性](#26-架构正确性)
  - [§27 通信正确性](#27-通信正确性)
  - [§28 数值正确性](#28-数值正确性)
  - [§29 Profiling 正确性](#29-profiling-正确性)
  - [§30 已知缺陷与未修](#30-已知缺陷与未修)
  - [§31 走过的弯路](#31-走过的弯路)
  - [§32 未来工作优先级](#32-未来工作优先级)
  - [§33 快速验证手册](#33-快速验证手册)

---

# 第一部分：导读与定位

## §1 读者路径

本手册面向三类 reader：

1. **快速上手者**：只关心如何跑实验。直接读 §18、§33，配合 `doc/02-usage.md`。
2. **架构 reviewer**：关心 AFD 分离、DBO 是否合理。读 §4–§6（理论）、§8–§9（模型层）、§26（review 清单）。
3. **调度/通信 reviewer**：关心 NCCL 使用、overlap 正确性。读 §11（分布式）、§12–§13（调度）、§19（warmup）、§27（通信正确性）、§31（走过的弯路）。

建议首次阅读顺序：§3（仓库总览）→ §4–§5（理论）→ §7–§13（源码）→ §26–§31（review 清单）。

## §2 与其他文档的关系

| 文档 | 定位 | 何时读 |
|------|------|--------|
| `doc/technical_report.md` | 575 行技术汇报简明版 | 快速了解总体设计 |
| **本文** | 1500+ 行详版 | 代码 review、深入理解 |
| `doc/01-architecture.md` | 设计蓝图（288 行） | 看总体模块划分 |
| `doc/02-usage.md` | 使用手册（387 行） | 跑实验 |
| `doc/03-api-reference.md` | API reference（475 行） | 写新调度器/新 worker |
| `doc/04-deployment.md` | 部署（457 行） | 多机实机部署 |
| `doc/communication_analysis.md` | NCCL P2P 行为分析 | debug 通信问题 |
| `doc/pipeline_bubble_analysis.md` | bubble 拆解 | 理解 cross-layer 的收益来源 |
| `doc/batch_scaling_experiments.md` | batch/seq 扩展实验原始数据 | 对比实验 |
| `doc/dbo_experiment_report_v2.md` | DBO 第二轮实验 | 历史进展 |
| `doc/npu_fp8_report.md` | NPU FP8 实验记录 | §24–§25 的原始材料 |

本文**不会重复**上述文档的内容；需要时用「详见 `doc/XXX.md §Y.Z`」形式引用。

## §3 仓库总览

按 `wc -l`：

```
src/main.py                     467
src/model/disaggregated.py      765
src/model/attention_worker.py   343
src/model/ffn_worker.py         255
src/model/kv_cache.py           230
src/pipeline/async_scheduler.py 911   # Prefill DBO（主力）
src/pipeline/decode_scheduler.py 501  # Decode DBO + cross-layer
src/pipeline/scheduler.py       523   # 串行/ping-pong baseline
src/pipeline/micro_batch.py     214
src/distributed/__init__.py     245
src/distributed/communicator.py 364
src/utils/timing.py             312
src/utils/profiler.py           221
src/utils/sampling.py           196
```

**依赖关系**：

```
main.py
 └─ disaggregated.DisaggregatedQwenModel        # 编排
     ├─ attention_worker.AttentionWorker        # 仅 attention 节点
     ├─ ffn_worker.FFNWorker                    # 仅 ffn 节点
     └─ 以下三选一（由 --no-dbo / decode 走向决定）
         ├─ scheduler.PipelineScheduler                # 串行 baseline
         ├─ async_scheduler.AsyncPipelineScheduler    # Prefill DBO
         └─ decode_scheduler.DecodeDBOScheduler       # Decode DBO
```

`distributed/` 提供单例 `DistributedContext` 和 `AFDCommunicator`；`utils/` 提供 `TimingTracker`、torch.profiler 包装、sampling。

---

# 第二部分：理论基础

## §4 AFD 架构的动机与数学

### 4.1 传统 TP/PP 的痛点

- **Tensor Parallelism (TP)**：把权重矩阵沿 hidden 维切，每步都要 all-reduce。bandwidth 要求高，异构硬件（PCIe、不同 GPU）打不过 NVLink。
- **Pipeline Parallelism (PP)**：按 layer 切 stage。单 batch 会有 bubble；需要 1F1B/interleaved 等复杂调度才能利用率 ~80%。

两者都假设**每一层的 attention 和 FFN 必须串行在同一设备上完成**。但 Transformer 每层其实长这样：

```
x → LN → Attention → (+residual) → LN → FFN → (+residual) → y
```

Attention 和 FFN 的资源画像非常不同：

| 模块 | 访存特征 | 计算特征 | MoE 情况下 |
|------|---------|---------|-----------|
| Attention | KV cache 读写（decode 阶段 memory-bound） | Q/K/V/O GEMM + softmax | 权重固定不动 |
| FFN（MoE） | 专家权重稀疏激活 | 大 GEMM | top-k=8/128 激活，需要专家路由 |

**这启发我们把它们放到不同节点上**：attention 节点专心处理 KV cache + attention 计算，FFN 节点专心处理 MoE 权重。

### 4.2 AFD 的关键观察：残差可以预合并

原始 Transformer 层的数据流：

```
attn_out = Attn(LN1(x))
h = x + attn_out                       # 残差 1（x 是输入）
ffn_in = LN2(h)
ffn_out = FFN(ffn_in)
y = h + ffn_out                        # 残差 2
```

如果把 attention 和 FFN 拆到两机：朴素地发 `attn_out` 和 `x` 各一份，带宽 = 2 × H × BS × Seq × dtype_bytes。

**本项目的优化**（`src/pipeline/async_scheduler.py:424` 等处）：
```python
packed = (attn_output + residual).contiguous()   # attn_out + x
# 只发 packed，FFN 端收到的就是 h
```

代价：FFN 端无法再单独拿到 `x`，所以 `y = h + FFN(LN2(h))` 时，FFN 节点必须也持有 `post_attention_layernorm` 并自己做 `h + ffn_out`。但**带宽减半**（实测 A2F 和 F2A 各一份 1×H 张量，非 2×H）。

这是一个 AFD 实现是否正确、高效的核心判定点。

### 4.3 数据通路的物理时长

以 Qwen3-30B-A3B（`hidden=3584`）+ bf16 + batch=8, seq=128 为例：

- 单层单 MB payload = 8 × 128 × 3584 × 2 = 7.34 MiB
- 两层（A→F 和 F→A）单 MB ≈ 14.7 MiB
- 若通过 25 Gbps TCP：理论 ~5 ms/层/MB；48 层 × 2 方向 ≈ 480 ms 纯通信（不 overlap）。
- NVLink / IB 100 Gbps 下可以降到 ~48 层 × 0.3 ms ≈ 15 ms。

这解释了为何 **decode 阶段（seq=1）通信占比显著**、DBO 难以对 decode 生效（见 §20）。

## §5 DBO 的 overlap 模型

### 5.1 串行模型

设 N 层，每层 attention 耗时 `T_a`、FFN 耗时 `T_f`、A→F 通信 `T_c1`、F→A 通信 `T_c2`。串行延迟：

```
T_serial = N × (T_a + T_c1 + T_f + T_c2)
```

### 5.2 DBO-2 模型

2 个 micro-batch：当 MB0 在通信、MB1 就能在算。理想情况（通信完全被计算盖住）：

```
T_dbo2 ≈ N × max(T_a + T_c1, T_f + T_c2)
```

只有当 `T_a ≈ T_f` 且 `T_c1 + T_c2 ≤ T_a + T_f` 时，DBO 达到理论上限 ~2× speedup（含半 bubble）。

### 5.3 真实世界的 DBO：bubble

实际上每 MB 内部 attention 自身是串行的，送出后 FFN 也要排队。用 attention 节点视角看：

```
Layer L:   compute(MB0) → send(MB0) → compute(MB1) → send(MB1) → wait(F2A MB0) → wait(F2A MB1)
```

只有 `compute(MB1)` 和 `send(MB0)` 真实 overlap。末端仍要等 FFN 跑完两个 MB 并回传。实测的 overlap 比纯理论低 30~50%。

### 5.4 为什么 decode DBO 效果差

decode 的 `T_a`（单 token attention）非常小（~2 ms）。而 F→A 通信 `T_c2` 受 NCCL kernel launch overhead 主导，约 2~3 ms。`T_c2 > T_a` 意味着**送回的还没到、下一层 compute 就开工不了**，overlap 不成立。

缓解手段：cross-layer pipeline（见 §6、§13）。

## §6 Cross-layer Pipeline 的 bubble 分析

### 6.1 朴素 per-layer decode DBO（src/pipeline/decode_scheduler.py 的注释 70–85）

```
ATT:   L0(MB0) → L0(MB1) → wait(F2A L0 MB0,MB1) → L1(MB0) → ...
```

问题：每层都有 `wait all F2A` 的同步点。48 层 × 小计算量 = bubble 占比 > 50%。

### 6.2 Cross-layer 思路

允许 MB0 的 Layer L+1 **在 MB1 的 Layer L 还没拿回 F2A 时**就开工。只要 MB0 自己的 F2A 回来了就行。

具体实现（`src/pipeline/decode_scheduler.py:301-362`）：

- 每个 MB 有独立的 F2A recv handle
- 进入 Layer L+1 时，只 `wait(f2a_recv_handles[mb_idx])`，不 wait 整个 layer
- 这样 MB0 Layer L+1 可以和 MB1 Layer L 的 compute/send 真正 overlap

数学模型（忽略 layer 数的边界）：

```
T_crosslayer ≈ N × max(T_a + T_c_one_way, T_f + T_c_one_way) / 2
```

实测见 §23 的 NPU 结果和 `results/experiments_qwen3_v6/summary.csv`。

### 6.3 为何 cross-layer 只在部分 batch 下见效

看 `src/pipeline/decode_scheduler.py` 的核心约束：MB0 进入 Layer L+1 之前仍必须完成**本层**的 attention compute。若 `T_a(MB)` 已经接近通信延迟，额外收益就小；若 `T_a` 远小于通信（典型的 decode），收益大。因此**中等 batch**（例如 b4–b8）收益最明显；b1 太小无 MB 并行、b32+ compute 已经接近通信不需要 cross-layer。

---

# 第三部分：源码逐层走读

## §7 入口：src/main.py

### 7.1 职责

- `parse_args`（`src/main.py:73-136`）：~35 个 CLI 选项。核心：
  - `--role attention|ffn`：本进程的角色
  - `--rank --world-size --master-addr --master-port`：torch.distributed 初始化参数
  - `--num-micro-batches`：DBO 的 MB 数（默认 2）
  - `--no-dbo`：走串行 baseline
  - `--no-generate`：只跑 prefill
  - `--timing-mode {cuda_events, sync}`：profiling 模式（见 §15）
  - `--warmup-p2p --warmup-rounds`：NCCL warmup（见 §19）
- `run_inference_demo`（`src/main.py:177-317`）：单次 prefill 跑通 + 输出计时。
- `run_generation_demo`（`src/main.py:318-430+`）：带 KV cache 的自回归生成。

### 7.2 调度器选择

`src/main.py:250-263` 根据 `args.no_dbo` 选：
```python
if args.no_dbo:
    scheduler = PipelineScheduler(...)          # 串行
else:
    scheduler = AsyncPipelineScheduler(...)     # Prefill DBO
```

Decode 阶段在 `run_generation_demo` 里**另行**实例化 `DecodeDBOScheduler`（除非 `--no-dbo`）。

### 7.3 关键陷阱

`scripts/run_single.sh:143,161` 用 `python -u -m src.main`；不要写成 `python -m src.main`，否则会被仓库里某些残留 guard 进程名匹配杀掉。详见 memory `experiment environment`。

## §8 模型编排：src/model/disaggregated.py

### 8.1 类概览

`DisaggregatedQwenModel`（~765 行）是整个 AFD 的门面对象。核心字段：

| 字段 | 类型 | 仅 attention 节点 | 仅 ffn 节点 |
|------|------|:----------------:|:----------:|
| `attention_worker` | `AttentionWorker` | ✓ | — |
| `ffn_worker` | `FFNWorker` | — | ✓ |
| `config` | HF config | ✓ | ✓ |
| `num_layers / hidden_size / dtype / device` | scalars | ✓ | ✓ |

### 8.2 权重加载：`load_weights`（`src/model/disaggregated.py:73-114`）

流程：

1. 用 HF `AutoModelForCausalLM.from_pretrained` 加载**完整**模型到 CPU（或本地 GPU，若 HF dispatch）。
2. 读取 `config.num_hidden_layers`、`config.hidden_size`、`config.torch_dtype`。
3. 根据 `ctx.is_attention_node` 或 `is_ffn_node` 分别实例化对应 worker，把所需模块（embed / layers / norm / lm_head / mlp 子模块）**按引用**交给 worker。
4. 删除未用模块 + `torch.cuda.empty_cache()`。

**Review 点**：
- Worker 内部 **持有的是原对象的引用**，不是 deepcopy。改权重不会错乱，但如果两节点需要共享某些 buffer（例如 RoPE cache），必须复制。
- `config` 在两节点完全一致，因为权重 checkpoint 必须对齐——避免两端读出不同 `num_layers`。

### 8.3 同步接口：`forward_layer_sync`（`src/model/disaggregated.py:139-221`）

这个是"裸 AFD"：每层**同步**完成 A→F→A 一次，不做 DBO。用于：
- 正确性 smoke test
- `PipelineScheduler` 的 fallback 路径

内部实现（attention 节点视角）：

```python
attn_output, residual = attention_worker.forward_attention_layer(layer_idx, ...)
packed = (attn_output + residual).contiguous()
dist.send(packed, dst=peer_rank, tag=layer_idx*100)
ffn_out = torch.empty_like(packed)
dist.recv(ffn_out, src=peer_rank, tag=layer_idx*100+1)
hidden_states = ffn_out
```

**Review 点**：
- tag 方案：`layer_idx*100`，ffn→attn 多加 1。DBO 路径用 `layer_idx*1000 + mb_idx*10 + dir`（见 §12.4）和 decode 的 `10000 + layer*2*num_mb + mb*2 + dir`（`src/pipeline/decode_scheduler.py:110-113`）。三套 tag 命名空间**不冲突**（100×48 < 1000×48，10000 区段隔离），但 review 时要确认。

### 8.4 自回归生成：`generate`（`src/model/disaggregated.py:534-665`）

状态机（attention 节点）：

1. **prefill**：用 `AsyncPipelineScheduler` 跑一次 prefill，得到最后一层 logits。
2. **sample**：用 `utils/sampling.py` 做 greedy / top-p / temperature。
3. **decode loop**：
   - 维护 `DynamicCache`（HF 的动态 KV cache，逐层追加）
   - 每步调用 `DecodeDBOScheduler.forward_decode_dbo(input_ids, pos_ids, kv_cache)`
   - 直到 `max_new_tokens` 或遇到 eos
4. **FFN 节点镜像**：`_generate_ffn_node`（`src/model/disaggregated.py:667-726`）循环等待 attention 端的信号（迭代数通过 barrier 同步），跑同样的 decode step。

**Review 点**：
- 两端必须知道迭代数。目前实现是 attention 端先广播一个 `int`；FFN 收到才开始本轮 decode。
- 如果 attention 端提前 break（eos），必须同步给 FFN；否则 FFN 会悬挂在 `irecv`。见 `generate` 尾部的收尾逻辑。

## §9 两个 Worker

### 9.1 `AttentionWorker`（`src/model/attention_worker.py`）

核心方法：

| 方法 | 位置 | 作用 |
|------|------|------|
| `embed` | `:158+` | token id → hidden states |
| `get_position_embeddings` | `:180+` | 预计算 cos/sin（RoPE） |
| `forward_attention_layer` | `:158-207` | 单层 attention（不含 FFN、无 residual add） |
| `forward_lm_head` | `:230+` | 最后 norm + lm_head |

多 GPU 层分片（`src/model/attention_worker.py:208-227`）：

```python
layers_on_dev0 = num_layers // num_devices - 3
```

**为何 -3？**：rank-0 GPU 还要承担 embedding、NCCL 通信 buffer、send/recv 临时张量。NCCL 在 rank-0 上吃约 6 GiB 额外显存，所以第一张卡少装 3 层。

对于 V100-32GB + Qwen3-30B-A3B（48 层 × 2 devices），`21/27` 分布（详见 memory `GPU memory management`）。

### 9.2 `AttentionLayer.forward` 的版本兼容（`src/model/attention_worker.py:55-155`）

HF Transformers 在 4.x → 5.x 演化中 `layer.forward` 的 kwargs 反复变化（`position_ids` vs `position_embeddings`，`past_key_value` vs `past_key_values`）。代码用 `inspect.signature` 动态探测：

```python
params = inspect.signature(self.layer.self_attn.forward).parameters
kwargs = {}
if 'position_embeddings' in params: kwargs['position_embeddings'] = pe
...
```

**Review 点**：这段代码是技术债密集区。升级 transformers 时优先看这里。

### 9.3 `FFNWorker`（`src/model/ffn_worker.py`）

`FFNLayer.forward`（`src/model/ffn_worker.py:57-122`）：
- dense: 走 `mlp(hidden)`
- MoE: 先 `gate(h)` 得 router logits → top-k → 调度到 experts → 合并输出，另外跑 `shared_expert`。

`StageTiming`（`src/model/ffn_worker.py:24-28`）记录三段 CPU 时间（`router_s`, `experts_s`, `shared_or_dense_s`），供 timing tracker 拆解 MoE 内部 bottleneck。

**Review 点**：MoE 的 top-k 路由用的是 HF 原生实现（非 SGLang/vLLM 的融合 kernel）。小 batch 下 all-to-all 开销大，属于已知性能瓶颈，**不是本项目要优化的点**。

## §10 KV Cache 机制

### 10.1 两种 cache 并存

项目里其实有两个 KV cache 实现，容易混淆：

1. `src/model/kv_cache.py` 的 `KVCache`（`:26-230`）：自研，预分配 `[batch, heads, max_seq, head_dim]`。**实际未在主路径使用**。
2. HF `transformers.DynamicCache`：动态追加，被 `DisaggregatedQwenModel.generate` 和 `DecodeDBOScheduler` 采用。

**Review 点**：`src/model/kv_cache.py` 是历史遗留，可以删；删前确认没有测试依赖。decode 的正确性实际由 `DynamicCache` 保证。

### 10.2 DynamicCache 的切片技巧（decode DBO 的关键）

HF `DynamicLayer` 存储 KV 为 `[batch, heads, seq, dim]`。batch 维天然可切。decode DBO 在进入每层前做：

```python
cache_layer = kv_cache.layers[layer_idx]
orig_keys, orig_values = cache_layer.keys, cache_layer.values
for mb_idx in range(num_mb):
    cache_layer.keys = orig_keys[start:end]      # batch slice
    cache_layer.values = orig_values[start:end]
    # ...调 attention_worker.forward_attention_layer(... past_key_value=kv_cache)...
    mb_updated_keys.append(cache_layer.keys)
    mb_updated_values.append(cache_layer.values)
cache_layer.keys = torch.cat(mb_updated_keys, dim=0)   # merge back
cache_layer.values = torch.cat(mb_updated_values, dim=0)
```

代码位置：`src/pipeline/decode_scheduler.py:229-230, 250-251, 276-277`。

**Review 点**：
- 这一操作在**每层、每步 decode** 都要做一次。合并开销虽小（内存 view），但有调用次数放大效应。
- 如果未来 HF 改了 `DynamicLayer` 的内部布局（例如 page 化、连续 buffer），这段会 silently 错。加单测会更稳。

## §11 分布式底层

### 11.1 单例 context（`src/distributed/__init__.py`）

`DistributedContext` 是 `__new__` 实现的 singleton。`initialize()` 里：

1. 设置 `MASTER_ADDR/PORT, RANK, WORLD_SIZE` env 变量
2. `torch.cuda.set_device(local_rank)` **在** `init_process_group` **之前**——避免 NCCL device ambiguity warning
3. `dist.init_process_group(backend, device_id=...)`
4. `dist.new_group(ranks=[attn, ffn])` 创建专用子组

**Review 点**：
- 当前实现假定 `attn_rank=0, ffn_rank=1`；如果想改成多 attention+多 ffn 多端，要改 `role` property 的逻辑（`:154-161`）。
- `cleanup()`（`:190-205`）故意不调 `destroy_process_group()`，因为 PyTorch 2.7 + NCCL 2.26 在某些环境下会在 destroy 阶段抛 `ncclProxyDestroy refCount` 断言。让进程自然退出比较稳。这条是**走过的弯路**，见 §31。

### 11.2 AFDCommunicator（`src/distributed/communicator.py`）

提供 `send_async / recv_async / wait_send / wait_recv` 的简单包装，带双 buffer。**只被串行 baseline `PipelineScheduler` 使用**；AsyncPipelineScheduler 和 DecodeDBOScheduler 直接用 `dist.isend/irecv`，不走 communicator。

**Review 点**：两套 API 并存是技术债。如果走 DBO 路径 review，`src/distributed/communicator.py` 可以忽略。

## §12 Prefill 调度：AsyncPipelineScheduler

### 12.1 文件概览（`src/pipeline/async_scheduler.py` 911 行）

四个核心入口：

| 函数 | 行号 | 何时用 |
|------|------|------|
| `_run_attention_node_simple` | `:379-564` | **主路径** prefill attention |
| `_run_ffn_node_simple` | `:566-723` | **主路径** prefill FFN |
| `_run_attention_node_async` | `:725-820` | 历史实现（非 deferred-send） |
| `_run_ffn_node_async` | `:822-907` | 同上 |

生产用的是 `_simple` 版本（命名不直观——"simple" 其实是优化后的版本，早期 `_async` 是朴素版）。

### 12.2 关键数据结构

- `DBOStats`（`:34-60`）：累计 compute_time / recv_wait_time / total_time。
- `SendTransferMonitor`（`:63-155`）：polling 线程探测 `handle.is_completed()`，详见 §16。
- `AsyncPipelineScheduler`（`:160+`）：主类，持有 compute_stream + comm_stream（目前 DBO 主路径没有实际把计算放到 compute_stream，只 `_async` 路径用；见 §12.6 的 review 点）。

### 12.3 Prefill Attention 端（`:379-564`）

核心 insight：**Layer 0 deferred-send**。

```python
# Layer 0: 先算完所有 MB，再一次性发
for mb_idx, mb in enumerate(micro_batches):
    attn_output, residual = forward_attention_layer(...)
    packed = (attn_output + residual).contiguous()
    layer0_outputs.append(packed)
for mb_idx, packed in enumerate(layer0_outputs):
    handle = self._send_async(packed, tag)
```

**为什么 defer?** FFN 端需要时间 init + post 第一次 irecv。如果 attention 端 Layer 0 MB0 立刻 isend，NCCL flow control 会 block 15–24 ms 等 FFN 端 irecv。defer 到所有 compute 都完成才发，就相当于给 FFN 端一个"calibration window"。

Layer 1+ 用 **交错 recv-compute-send**：

```
for layer L in 1..N-1:
    # 为 L-1 层的 F2A 预先 post irecv
    for mb: recv_handles[mb] = irecv(layer=L-1, mb)

    # 等 L-1 层的 send 完成
    for handle in prev_send_handles: handle.wait()

    # 逐 MB: wait recv → compute layer L → send layer L
    for mb:
        recv_handles[mb].wait()
        update mb.hidden_states
        compute attn layer L
        isend result
```

### 12.4 Tag scheme

`src/pipeline/async_scheduler.py:_get_tag`（扫 §312 附近）：

```python
def _get_tag(layer_idx, mb_idx, direction):
    dir_code = 0 if direction == "attn_to_ffn" else 1
    return layer_idx * 1000 + mb_idx * 10 + dir_code
```

范围：layer ≤ 48, mb ≤ 8 → tag ∈ [0, 48080]，不会和 sync 路径（`layer*100`）或 decode（`10000+...`）冲突。

### 12.5 Prefill FFN 端（`:566-723`）

关键 insight：**next-layer irecv 必须在 current-layer send.wait 之前 post**。

```python
# Pre-post irecv for layer 0（出循环前）
for mb: cur_recv_handles[mb] = irecv(layer=0, mb)

for layer in range(num_layers):
    # 用已经 post 好的 recv handles
    for mb:
        recv_handles[mb].wait()
        ffn_out = forward_ffn_layer(...)
        send_handles.append(isend(ffn_out, tag))
    # ★ 先 post 下一层的 irecv
    if layer + 1 < num_layers:
        for mb: cur_recv_handles[mb] = irecv(layer=layer+1, mb)
    # 再 wait 本层所有 send
    for h in send_handles: h.wait()
```

**为什么这样？** NCCL 内部用 FIFO 队列；irecv 如果不提前发，attention 端 isend 到达时会 block（flow control），attention 端 compute 无法继续。这条优化消除了 MB0 A2F 的 15ms+ block。

### 12.6 Review 点

- `self.compute_stream` 在 `_simple` 路径里**没有被用**（见 `:416-430`）——都是在 `current_stream` 上跑的。命名误导。
- `SendTransferMonitor` 只在 `enable_timing` 时激活，不影响主路径性能。
- `_run_attention_node_async` / `_run_ffn_node_async` 是早期版本，可以删；留着的原因是做 A/B。

## §13 Decode 调度：DecodeDBOScheduler

### 13.1 Attention 端（`src/pipeline/decode_scheduler.py:170-402`）

**Layer 0 warmup**（`:212-277`）：
- 逐 MB: KV cache batch slice → forward_attention_layer → `prev_send_handle.wait()` → `dist.isend()`
- 所有 MB 完成后 `torch.cat` 合并 KV cache 回去
- 最后 post layer 0 的 F2A irecv

**Layer 1+ cross-layer pipeline**（`:293-380`）：
- 每 MB: `f2a_recv_handles[mb_idx].wait()` → compute attn → wait prev send → isend
- 注意：**只 wait 本 MB 的 F2A**，不是所有 MB。MB0 可以先进 Layer L+1。
- 层结束时 `torch.cat` 合并 KV cache；post 本层 F2A irecv。

### 13.2 FFN 端（`:404-493`）

**Pre-post layer 0 irecv**（`:418-429`），然后每层：
- 逐 MB: wait A2F → compute FFN → isend F2A
- `if layer+1 < num_layers: post layer+1 A2F irecv`（在 wait 本层 send 之前）
- 最后 wait 本层所有 send

### 13.3 KV cache batch 切片

见 §10.2。

### 13.4 Review 点

- Decode DBO 的 attention 端**强制 serialize MB 内的 isend**：`if prev_send_handle is not None: prev_send_handle.wait()`（`:263-264, 353-354`）。这牺牲了一点 send-compute overlap，但避免了 NCCL 拥塞。
- 只在 `self._timing_step == self._current_step`（默认 step=1）那一步记 timing，避免 profiling 开销污染 benchmark。
- 跨层 recv 用 `f2a_recv_handles` 列表 per-MB 管理。如果 FFN 端挂了，attention 端会 hang；目前没有 timeout 机制（**已知风险**）。

## §14 串行 baseline：PipelineScheduler

`src/pipeline/scheduler.py` 523 行。**定位**：DBO 的对照实验 baseline；用 `--no-dbo` 开启。

主流程（`:120-200`）：对每 MB 逐层同步 attention → send → recv → 下一层。不使用 isend 的 overlap。

**Review 点**：
- 这里的 `_pack_attn_output` / `_unpack_attn_output`（`:100-113`）和 AsyncPipelineScheduler 的 `(attn_output + residual).contiguous()` 行为一致。改一边时要记得同步另一边。
- 用 `AFDCommunicator` 而不是裸 `dist.isend`，是两套 API 并存的来源。

---

# 第四部分：Profiling 与可视化

## §15 TimingTracker 双模式

`src/utils/timing.py`。核心取舍：**准确度 vs overlap 破坏**。

### 15.1 `cuda_events` 模式（默认）

`mark_start / mark_end` 内部：
```python
torch.cuda.current_stream().synchronize()
perf_counter()
```

只 sync 默认 compute stream，**NCCL stream 不动**。好处：
- compute 事件的 CPU 时间戳准确
- 通信（isend/irecv）仍在 NCCL stream 上继续 overlap

开销：接近零。

### 15.2 `sync` 模式（legacy）

`mark_start / mark_end` 改用 `torch.cuda.synchronize()` 全设备同步。会 block **所有** stream 包括 NCCL —— DBO overlap 被破坏。实测开销 **+16.4%**（详见 memory `profiling overhead`）。

**Review 点**：做性能对比时**只能用同一种模式**。否则 cuda_events 的 DBO 会比 sync 的 serial "赢得莫名其妙"。

## §16 SendTransferMonitor 的真实传输时间

`src/pipeline/async_scheduler.py:63-155`。问题：`dist.isend()` 立即返回（~0.2 ms 非阻塞），但**实际传输**可能 20+ ms 后才完成。怎么测实际传输时间？

方案：
- isend 之后不立刻 wait；另起轮询线程，每 100 μs 调 `handle.is_completed()`。
- 第一次 `True` 时记 `end_time = perf_counter()`。

**已知限制**：
- GIL 绑架——polling 线程会被主线程 compute 挤出去，实测 end_time 比真实传输晚 ~20 ms。
- 所以这个 monitor **不是绝对时间测量**，只能用于相对对比。

实际数据：`handle.wait() + sync` 大约 22 ms（含等 peer irecv），`handle.is_completed()` 返回 True 约 22 ms。两者基本等价。

**Review 点**：不要把 `SEND_TRANSFER` 事件的 duration 当作"物理传输时间"；它约等于"isend 到 peer irecv 完成 + polling 抖动"。

## §17 可视化脚本

- `scripts/plot_experiment_results.py`：把 `results/*/summary.csv` 画成柱状 / 速度比条形图。
- `scripts/plot_scaling_comparison.py`：batch 扩展实验。
- `scripts/visualize_dbo_pipeline.py`：把 TimingTracker 导出的 JSON 画成 **Gantt 图**（每层每 MB 一条），按 compute/communication 着色。
- `scripts/analyze_pipeline_bubbles.py`：自动算 bubble 占比（compute 总和 / e2e 的比值）。

用法详见 `doc/02-usage.md §可视化`。

---

# 第五部分：实验

## §18 启动流程

### 18.1 单机单卡（仅做代码 smoke）

不适合本项目——AFD 至少需要 2 rank。

### 18.2 单机多卡（推荐 review 用）

```bash
./scripts/run_single.sh local 8 128 --tokens 32 --warmup-p2p --warmup-rounds 5
# 4 GPUs: rank 0,1 attention; rank 2,3 ffn
```

参数：
- `local`：单机；另一个是 `remote` 传两个 host
- `8 128`：batch=8, seq=128
- `--tokens 32`：decode 32 个 token
- `--warmup-p2p --warmup-rounds 5`：**必须**，见 §19

结果写到 `results/prefill_dbo/timing_*.json` + `stats_*.txt`。

### 18.3 多机

```bash
# 机器 A
./scripts/run_node.sh --role attention --rank 0 --master-addr <A> ...
# 机器 B
./scripts/run_node.sh --role ffn --rank 1 --master-addr <A> ...
```

## §19 NCCL warmup 的必要性

### 19.1 现象

Cold start 时第一次 `dist.isend` 会额外阻塞 40–60 ms。NCCL 内部：
- 第一次传输要建立 ibverbs QP / PCIe DMA channel
- NCCL plan cache 首次命中需要编译
- proxy thread 冷启动

### 19.2 warmup 实现（`src/distributed/warmup.py`）

跑 5 轮 dummy isend/irecv，尺寸与真实 payload 一致。第二轮起延迟稳定在 ~0.2 ms。

**已废弃思路**：keepalive（双向心跳保持 NCCL 通道 warm）。**原因**：双向心跳容易死锁——A 的心跳 send 和 B 的心跳 send 都在等对方 recv。见 memory `NCCL warmup` 和 §31.

### 19.3 `NCCL_BUFFSIZE`

必须 ≥ 单次 isend 的 tensor 字节数，否则会触发 flow control 阻塞。项目默认 32 MB（`scripts/run_single.sh:45`）。

## §20 GPU 上的 DBO 结论

从 `results/experiments_qwen3_v6/summary.csv` 和 `results/prefill_dbo/`：

- **Prefill DBO（b4, seq=128）**：~1.2× serial。
- **Prefill DBO（b8+, seq>=512）**：~1.4–1.5×。
- **Decode DBO（MB overlap）**：≈1.0×（无收益）。compute 太轻，无法 amortize MB overhead。
- **Crosslayer decode**：b4 时 0.73× → 0.94×（减 bubble 但 decode 受 bandwidth 主导，不敌 serial FFN 权重加载的 2× 成本）。

详见 `doc/dbo_experiment_report_v2.md` 和 memory `experiment results`。

---

# 第六部分：NPU-910C 移植

## §21 分支与差异

- 主分支：`main`（GPU）
- NPU：`feat/npu-910c`
- FP8 探索：`feat/npu-910c-fp8`

NPU 移植点：
- `src/main.py`、worker：`torch.cuda.*` 改 `torch.npu.*`（或 `torch_npu`）
- `backend="hccl"` 替代 `nccl`
- `NCCL_BUFFSIZE` 变 `HCCL_BUFFSIZE`
- device id 用 `npu:N`

## §22 容器与设备

详见 `.github/skills/npu_910c_env_setup_and_run/SKILL.md`。核心要点：
- 容器必须 `--privileged`（HCCL 需要访问 /dev/davinci*）
- 模型权重挂载 `/home/schedTeam/Qwen3-30B-A3B`（只读）
- 容器 `zhangyz-npu-1` 已预置环境，不重建

## §23 NPU 上的实验结论

详见 `results_npu/summary.csv`。要点：
- 910C 显存更大 → 可跑 b16、seq 2048
- Prefill DBO 加速比与 GPU 趋势一致，绝对值略低（HCCL 启动开销大于 NCCL）
- Decode DBO 同样无收益

---

# 第七部分：FP8 量化尝试

## §24 动机与方案

`doc/npu_fp8_report.md`。假设：910C 声称支持 FP8；换量化后 KV cache 和权重显存减半，可跑更大 batch。

## §25 阻塞点小结

三个 blocker（细节见 `doc/npu_fp8_report.md`）：

1. **torch_npu 2.6.0 无 FP8 dtype**：没有 `torch.float8_*`。
2. **Qwen3 checkpoint 无 FP8 版本**：需要离线 quant 流程，未做。
3. **HF transformers 不支持 NPU FP8 路径**：attention layer 版本兼容代码（§9.2）无 FP8 分支。

结论：该方向在当前软件栈下**不可行**，待 torch_npu 升级后重开。

---

# 第八部分：Review 清单

## §26 架构正确性

- [ ] `DisaggregatedQwenModel.load_weights` 只实例化本节点 worker（`src/model/disaggregated.py:73-114`）
- [ ] Residual 在 attention 端预合并（`(attn_output + residual).contiguous()`），FFN 端单独跑 post_attention_layernorm
- [ ] FFN 端的 `post_attention_layernorm` 不在 attention 端（否则两边都跑一次会错）
- [ ] `rotary_emb` 只在 attention 端（FFN 不碰 RoPE）
- [ ] KV cache 只在 attention 端，永远不跨网络传输

## §27 通信正确性

- [ ] Tag 方案三套命名空间不冲突：sync `layer*100`, prefill DBO `layer*1000+mb*10+dir`, decode DBO `10000+layer*2*num_mb+mb*2+dir`
- [ ] 所有 `dist.isend` 都有对应 `handle.wait()` 或 `.is_completed()` polling；不漏 wait
- [ ] `isend` 的 tensor 在 `wait()` 之前不能释放（async_scheduler 用 `output_tensors` 列表保引用）
- [ ] `irecv` 的 tensor 形状必须和对端 isend 一致（decode 端逐 MB 不同 batch size 需要 `mb_sizes[mb_idx]`）
- [ ] NCCL 不能提前 irecv 很多层（会 FIFO 阻塞 → 死锁，见 §31）
- [ ] NCCL_BUFFSIZE ≥ payload size

## §28 数值正确性

- [ ] `--no-dbo` 和 `DBO` 两路径输出的 logits 应一致（到 bf16 噪声内）
- [ ] HF `DynamicCache` batch 切片合并后顺序不乱（MB 按原顺序 cat）
- [ ] causal mask 对单 token decode 仍生效（`_make_causal_mask` 的 seq_len=1）
- [ ] sampling 在两端一致：attention 端 sample 后 broadcast 下一个 token id

## §29 Profiling 正确性

- [ ] 同一组实验使用同一 `--timing-mode`
- [ ] 计时只在 `step=1` 而不是 step=0（避免 warm start 污染）
- [ ] `SendTransferMonitor` 的 end_time 不当绝对值用
- [ ] compute/communication 时间戳基于同一个 `start_time`（TimingTracker 内部维护）

## §30 已知缺陷与未修

- [ ] `src/model/kv_cache.py` 自研 KVCache 未用，可删
- [ ] `AFDCommunicator` 只被串行 baseline 用，和 DBO 裸 `dist.isend` 两套并存
- [ ] `AsyncPipelineScheduler.compute_stream` 命名误导，`_simple` 路径没用
- [ ] Decode DBO 没有 FFN 端 crash 的 timeout；attention 端会 hang
- [ ] `DistributedContext.cleanup` 不调 `destroy_process_group`（NCCL 2.26 bug workaround）
- [ ] FFN crash 后 `_pending_sends` 不清理

## §31 走过的弯路

简要记录踩过的坑，review 时遇到不要重复踩：

1. **Keepalive 死锁**：想让 NCCL 常驻连接，双向心跳。`src/distributed/warmup.py` 曾实现 keepalive，后改为一次性 warmup。原因：bidirectional heartbeat 两端都在 wait → 死锁。
2. **NCCL 提前 irecv 死锁**：尝试为未来所有层同时 post irecv 以减少 wait。结果 NCCL FIFO serialize 所有 ops，后来的 isend 被前面未完成的 irecv 堵死。回退为"每层最多 pre-post 下一层"。commit `432329f` 已 revert。
3. **`handle.wait() + synchronize` 测通信时间**：测出来都是 22 ms，以为 NCCL 真慢。后来发现实际 isend ~0.2 ms，wait 阻塞是在等 peer irecv 完成。应在 `isend()` return 之时测时间戳。
4. **NVSHMEM 3.6.5 host-side atomic 未实现**：想用 nvshmem signal。回退到 `cudaMemcpy D2H` polling。见 memory `NVSHMEM limitations`。
5. **`python -m src.main` 被杀**：服务器上 guard 进程匹配这个模式会杀 rogue。改 `python -u -m src.main`。
6. **PyTorch 2.7 + NCCL 2.26 `destroy_process_group` 抛 refCount 断言**：`DistributedContext.cleanup` 跳过 destroy。
7. **sync 模式 profiling 污染 DBO**：sync 模式的 `torch.cuda.synchronize()` 会 block NCCL stream，把 DBO 的 overlap 测没了。引入 cuda_events 模式才解决。
8. **FP8 blocker**：torch_npu 2.6.0 无 FP8 dtype、HF 无 FP8 分支、无 FP8 checkpoint。见 §25。

## §32 未来工作优先级

| 优先级 | 项目 | 理由 |
|-------|-----|-----|
| P0 | 删 `src/model/kv_cache.py` | 死代码，容易混淆 |
| P0 | 统一 communicator 和裸 dist.isend | 两套 API 并存 |
| P1 | Decode DBO 加 FFN timeout + graceful shutdown | 生产稳定性 |
| P1 | Crosslayer 的 batch 下界自动判定 | 现在需要人工选 batch |
| P2 | NPU 的 HCCL warmup 策略重评估 | HCCL 启动 cost 高于 NCCL |
| P2 | torch.profiler 集成到 TimingTracker | SendTransferMonitor 的 GIL 抖动问题 |
| P3 | FP8 重开（等 torch_npu 升级） | 见 §25 |

## §33 快速验证手册

Review 代码后，跑这些命令确认没 regressed：

```bash
# 1. 环境
source venv/bin/activate
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

# 2. Smoke: prefill only, b2, seq=64
./scripts/run_single.sh local 2 64 --tokens 0 --warmup-p2p --warmup-rounds 3

# 3. Smoke: prefill + decode, b4, seq=128, 16 tokens
./scripts/run_single.sh local 4 128 --tokens 16 --warmup-p2p --warmup-rounds 5

# 4. 对比 DBO vs Serial（同一 batch/seq 跑两次）
./scripts/run_single.sh local 8 512 --tokens 0 --warmup-p2p --warmup-rounds 5             # DBO
./scripts/run_single.sh local 8 512 --tokens 0 --warmup-p2p --warmup-rounds 5 --no-dbo    # Serial
# 比较 results/prefill_dbo/timing_attention_*.json 的 total_time_ms

# 5. 生成 Gantt 图
python scripts/visualize_dbo_pipeline.py results/prefill_dbo/timing_attention_qwen3_*.json
```

期望：
- DBO 总时长 < Serial（对大 batch/长 seq）
- decode 输出是人类可读的文本
- 没有 NCCL hang / timeout 报错

---

## 附录 A：关键代码位置速查

| 主题 | 文件:行号 |
|------|---------|
| CLI 参数 | `src/main.py:73-136` |
| prefill 调度入口 | `src/main.py:250-263` |
| generate 状态机 | `src/model/disaggregated.py:534-665` |
| FFN 端 generate 镜像 | `src/model/disaggregated.py:667-726` |
| 权重加载 | `src/model/disaggregated.py:73-114` |
| 同步 forward | `src/model/disaggregated.py:139-221` |
| Attention 层分片 | `src/model/attention_worker.py:208-227` |
| Attention version-aware kwargs | `src/model/attention_worker.py:55-155` |
| MoE 分阶段计时 | `src/model/ffn_worker.py:24-28, 57-122` |
| DynamicCache batch slice | `src/pipeline/decode_scheduler.py:229-277` |
| Prefill ATT simple | `src/pipeline/async_scheduler.py:379-564` |
| Prefill FFN simple | `src/pipeline/async_scheduler.py:566-723` |
| Decode ATT cross-layer | `src/pipeline/decode_scheduler.py:170-402` |
| Decode FFN cross-layer | `src/pipeline/decode_scheduler.py:404-493` |
| Serial baseline | `src/pipeline/scheduler.py:120-200` |
| DistributedContext | `src/distributed/__init__.py:34-210` |
| AFDCommunicator | `src/distributed/communicator.py:37+` |
| TimingTracker | `src/utils/timing.py:159-259` |
| SendTransferMonitor | `src/pipeline/async_scheduler.py:63-155` |
| NCCL warmup | `src/distributed/warmup.py:12-72` |

## 附录 B：Review 优先级 Top 10

如果只有 1 小时 review，按这个顺序：

1. `src/model/disaggregated.py:73-114`（权重加载）
2. `src/pipeline/async_scheduler.py:379-564`（prefill ATT，最核心）
3. `src/pipeline/async_scheduler.py:566-723`（prefill FFN）
4. `src/pipeline/decode_scheduler.py:170-402`（decode ATT cross-layer）
5. `src/pipeline/decode_scheduler.py:404-493`（decode FFN）
6. `src/distributed/__init__.py`（NCCL init，少但关键）
7. `src/model/attention_worker.py:55-155`（HF 版本兼容）
8. `src/utils/timing.py`（profiling 模式差异）
9. §27 通信 checklist
10. §31 走过的弯路
