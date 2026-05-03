# 01. 架构设计

本文说明当前 AFD Demo 的模型拆分、MoE/EP 执行位置、DBO 流水线、
token-aware dispatch/combine 设计方向，以及 NPU EP 探索过程。

## 1. 总体目标

AFD Demo 将 Transformer 推理中的 Attention 子图和 FFN / MoE 子图拆成不同角色：

```text
输入 token
   │
   ▼
Attention role：embedding、self-attention、KV cache、lm_head、sampling
   │ A2F：hidden states
   ▼
FFN role：post-attention norm、MoE gate、experts、combine
   │ F2A：hidden states
   ▼
Attention role：下一层 attention 或最终 logits
```

这种拆分允许 Attention 与 FFN 在不同设备、不同 rank 或不同节点上运行。DBO
在此基础上把 batch 切成 micro-batch，使 Attention 与 FFN 在层间流水重叠。

Serial baseline 不是“未拆分模型”，而是关闭 DBO 的 A/F 分离串行路径：

```text
layer i attention -> A2F -> layer i FFN -> F2A -> layer i+1 attention
```

## 2. 主要代码结构

| 模块 | 职责 |
|---|---|
| `src/main.py` | CLI、分布式初始化、模型加载、scheduler 选择、timing 输出。 |
| `src/model/disaggregated.py` | Qwen3 A/F 拆分模型封装、自回归生成、KV cache 维护。 |
| `src/model/attention_worker.py` | embedding、attention layer、norm、lm_head、采样。 |
| `src/model/ffn_worker.py` | FFN / MoE layer 计算；在 NPU EP 下构造 `EPFFNLayer`。 |
| `src/model/ep_moe.py` | NPU EP MoE helper：专家分片、broadcast/reduce、overlap work item。 |
| `src/pipeline/scheduler.py` | `SimplePipelineScheduler`，serial AF baseline。 |
| `src/pipeline/async_scheduler.py` | `AsyncPipelineScheduler`，prefill DBO。 |
| `src/pipeline/decode_scheduler.py` | `DecodeDBOScheduler`，decode DBO、crosslayer、EP overlap decode path。 |
| `src/distributed/__init__.py` | rank、role、process group、P2P group 管理。 |
| `src/utils/device.py` | CUDA / NPU / CPU 设备与分布式 backend 抽象。 |
| `src/utils/timing.py` | timing event、layer timing、JSON 输出结构。 |

## 3. 执行模式

| 模式 | 触发方式 | Scheduler | 主要指标 |
|---|---|---|---|
| Serial baseline | `--no-dbo --generate` | `SimplePipelineScheduler` + generation path | `prefill_ms`、`decode_tpot_ms` |
| Prefill DBO | 默认单次 DBO；不加 `--generate` | `AsyncPipelineScheduler` | 模型侧 TTFT-path |
| Decode DBO | `--generate` | `DecodeDBOScheduler` | 准确 TPOT |
| Decode crosslayer | `--generate --crosslayer` | `DecodeDBOScheduler(use_crosslayer=True)` | 准确 TPOT |
| NPU EP sync | `--ffn-ep-backend broadcast_reduce_sync` | `DecodeDBOScheduler` + `EPFFNLayer.forward()` | 准确 TPOT |
| NPU EP overlap | `--ffn-ep-backend broadcast_reduce_overlap` | `DecodeDBOScheduler._run_ffn_ep_overlap_decode()` | 准确 TPOT |

Speedup 统一为 `serial / DBO`，大于 `1.0x` 才表示 DBO 更快。

## 4. 角色与 rank 拆分

### 4.1 Attention rank

Attention rank 负责：

1. token embedding；
2. 每层 self-attention；
3. 持有和更新 KV cache；
4. 向 FFN coordinator 发送 A2F hidden states；
5. 从 FFN coordinator 接收 F2A hidden states；
6. 最后一层之后执行 final norm、lm_head 和 sampling。

KV cache 只在 Attention rank 上。Decode 时每一步只把当前 token 的 hidden states
送到 FFN 侧，FFN 侧不持有 KV cache。

### 4.2 FFN coordinator

FFN coordinator 是 FFN/EP group 中负责对接 Attention rank 的 rank。它负责：

1. 从 Attention rank 接收 A2F hidden states；
2. 执行 post-attention residual 合并；
3. 执行 post-attention layernorm；
4. 执行 MoE gate / router；
5. 将 hidden/router metadata dispatch 给 EP ranks；
6. 收集或 reduce EP ranks 的 partial output；
7. 完成 residual + MoE output；
8. 将 FFN 输出通过 F2A 发回 Attention rank。

在当前 EP 实现中，Gate 在 FFN coordinator 上执行，不在 Attention rank 上执行。

### 4.3 FFN expert-only ranks

FFN expert-only ranks 不直接与 Attention rank 通信，只参与 FFN EP group 内部 collective。
它们负责：

1. 接收 coordinator dispatch 的 hidden/router metadata；
2. 只计算自己持有的 local experts；
3. 生成 dense partial output；
4. 通过 reduce 把 partial output 汇总到 coordinator。

当前专家分片由 `ExpertShardPlan` 控制，默认 `round_robin`：

```text
owner_rank = expert_id % ep_size
```

Qwen3-30B-A3B 有 128 个 routed experts：

| 拓扑 | 含义 | 每个 FFN EP rank routed experts |
|---|---|---:|
| EP4 | 1 Attention + 4 FFN EP ranks | 32 |
| EP7 | 1 Attention + 7 FFN EP ranks | 18 或 19 |

EP7 当前是 8 张 910C NPU 环境下的主候选：1 张做 Attention，7 张做 FFN EP。

## 5. 单层操作位置

| 操作 | 当前执行位置 | 说明 |
|---|---|---|
| embedding | Attention rank | `AttentionWorker.embed()` |
| self-attention | Attention rank | `forward_attention_layer()`，同时更新 KV cache |
| A2F send | Attention rank -> FFN coordinator | 发送 `attn_output + residual` |
| post-attention residual 合并 | FFN coordinator | EP coordinator 在 `create_work_item()` / `forward()` 内做 `residual + hidden_states` |
| post-attention layernorm | FFN coordinator | `post_attention_layernorm` 只在 coordinator 上执行 |
| Gate / Router | FFN coordinator | `self.gate(hidden_2d)` 产生 `selected_experts` 和 `routing_weights` |
| Dispatch | FFN coordinator -> FFN EP ranks | 当前是 broadcast 完整 `hidden_2d`、`selected_experts`、`routing_weights` |
| Local experts | 每个 FFN EP rank | 每 rank 只持有自己的 expert shard；当前只遍历本 MB 实际命中的本地 experts |
| Combine | 当前由 dense reduce 隐式完成 | 每 rank 产生 dense partial output，reduce SUM 到 coordinator |
| Residual + MoE output | FFN coordinator | coordinator reshape reduced partial 后加 residual |
| F2A send | FFN coordinator -> Attention rank | 将 FFN 输出发回 Attention rank |
| final norm / lm_head / sampling | Attention rank | 只在最后一层之后执行 |

这里的 Combine 不是 token-aware combine。当前实现仍是 dense partial reduce。

## 6. 非 EP A/F 单层数据流

非 EP FFN 路径中，FFN rank 拥有完整 FFN/MoE layer：

```text
Attention rank:
  attention(layer i, MB j)
  isend(A2F hidden)

FFN rank:
  irecv(A2F hidden)
  full FFN / MoE
  isend(F2A hidden)

Attention rank:
  irecv(F2A hidden)
  attention(layer i+1, MB j)
```

这个路径适用于 serial AF baseline、GPU/NPU 普通 DBO，以及非 EP FFN。

## 7. 当前 EP 后端

### 7.1 `broadcast_reduce_sync`

同步 EP 后端是 correctness-first 原型：

```text
FFN coordinator:
  norm + gate
  broadcast full hidden/router metadata

all FFN EP ranks:
  local experts -> dense partial output

EP group:
  reduce SUM dense partial output to coordinator

FFN coordinator:
  residual + reduced output
  F2A send to Attention
```

它的优点是简单、结果容易对齐；缺点是每个 micro-batch 都严格串行：

```text
dispatch MB0 -> compute MB0 -> reduce MB0 -> dispatch MB1 -> compute MB1 -> reduce MB1
```

EP4 sync 结果已保留在 `results_npu/ep4_broadcast_reduce_sync/`。这是负结果证据，
不是待删除临时目录。

### 7.2 `broadcast_reduce_overlap`

overlap 后端不改变数据语义，仍然是 full hidden broadcast + dense reduce；它改变的是
调度顺序，让上一 MB 的 reduce 尽量被下一 MB 的 local expert compute 隐藏：

```text
dispatch MB0
dispatch MB1
compute MB0
reduce MB0 async || compute MB1
wait MB0 reduce
reduce MB1 async
wait MB1 reduce
```

实现位置：

| 组件 | 职责 |
|---|---|
| `EPWorkItem` | 保存一个 micro-batch 的 hidden/router tensors、partial output、async handles 和 timing。 |
| `create_work_item()` | coordinator 执行 residual、layernorm、gate；expert-only ranks 创建接收 buffer。 |
| `dispatch_async()` | 异步 broadcast hidden/router metadata。 |
| `finish_dispatch()` | 等待 dispatch 完成，保证 local experts 输入就绪。 |
| `compute_local()` | 执行本 rank local experts。 |
| `reduce_async()` | 异步 reduce dense partial output。 |
| `finish_reduce()` | 等待 reduce 完成，并记录 hidden overlap / wait。 |
| `finish_output()` | coordinator 生成最终 FFN output；expert-only ranks 返回 dummy tensor。 |

所有 FFN EP ranks 必须以完全相同的 layer-major、MB-major 顺序进入 dispatch、
compute、reduce。HCCL collective 顺序不一致会死锁。

## 8. Decode DBO 流水线

Decode DBO 在每个 decode step 内把 batch 切成 micro-batch。外层 A/F 流水线为：

```text
Layer 0:
  A: compute MB0 -> send A2F MB0
  A: compute MB1 -> send A2F MB1
  F: recv MB0 -> FFN MB0 -> send F2A MB0
  F: recv MB1 -> FFN MB1 -> send F2A MB1

Layer 1+:
  A: wait F2A previous layer MBj -> attention -> send A2F current layer MBj
  F: wait A2F current layer MBj -> FFN -> send F2A current layer MBj
```

普通 decode DBO 会在一层的 A2F sends drain 之后再 post F2A irecv。`--crosslayer`
启用后，A2F 和 F2A 使用独立方向通信组：

| 通信组 | 方向 | 作用 |
|---|---|---|
| `a2f_group` | Attention -> FFN | Attention `isend`，FFN `irecv` |
| `f2a_group` | FFN -> Attention | FFN `isend`，Attention `irecv` |

这样 Attention 可以更早 post F2A irecv，减少单一 communicator FIFO 顺序带来的层间
气泡。但 crosslayer 只改变 A/F P2P 调度，不能自动修复 FFN 内部过慢的问题。

## 9. 两级流水线：A/F 外层 + EP 内层

NPU EP overlap 叠加了两级流水：

1. 外层 A/F DBO：Attention rank 与 FFN coordinator 在 layer/MB 维度交错执行。
2. 内层 EP overlap：FFN group 内部将 dispatch、local experts、reduce 拆成 work item，
   在同一层内跨 MB 重叠 reduce 与下一 MB compute。

目标是让 FFN wall time 接近 Attention compute time，并让通信 wait 被计算覆盖：

```text
Attention rank:      ATTN MB0 ── A2F ── wait F2A ── ATTN next
FFN coordinator:          recv ─ dispatch ─ local/reduce overlap ─ F2A
FFN expert ranks:                 dispatch ─ local/reduce overlap
```

如果 FFN local experts 仍远慢于 Attention，Attention 侧会继续长时间等待 F2A，
DBO 仍可能是负收益。

## 10. token-aware dispatch/combine 是什么

当前 `broadcast_reduce_overlap` 仍然有两个浪费：

1. coordinator 把完整 `hidden_2d`、`selected_experts`、`routing_weights` broadcast
   给所有 EP ranks，即使某个 rank 只负责少量命中的 experts。
2. 每个 rank 返回完整 dense partial output，然后 reduce SUM 到 coordinator，通信量与
   `tokens * hidden_size` 绑定，而不是与该 rank 实际 token assignment 绑定。

token-aware dispatch/combine 的目标是按实际 expert ownership 传输 token：

```text
coordinator gate 后构造:
  rank -> token_idx, topk_idx, local_expert_id, routing_weight

dispatch:
  rank k 只收到自己需要计算的 token rows

local experts:
  rank k 只对收到的 token rows 执行本地 experts

combine:
  rank k 返回 token_idx + partial_output
  coordinator 用 index_add_ 合并到 dense output
```

NPU/HCCL 上 variable-size all-to-all 风险较高，因此 MVP 不直接做完全动态 all-to-all，
而是 fixed-capacity padded dispatch：

```text
每个 EP rank:
  token_buffer[capacity, hidden]
  token_idx[capacity]
  expert_id[capacity]
  topk_idx[capacity]
  routing_weight[capacity]
  count
```

expert rank 只计算 `count` 以内的 token；返回 padded partial + token_idx；coordinator
再 `index_add_` 到 dense output。这样先验证通信量和 compute 是否随真实 assignment
下降，再考虑更复杂的 variable-size collective。

## 11. 探索尝试过程

### 11.1 EP4 sync：先跑通正确性

第一阶段实现 `1 Attention + 4 FFN EP ranks`，后端为 `broadcast_reduce_sync`。
Qwen3-30B-A3B 的 128 个 routed experts 按 round-robin 分到 4 个 FFN EP ranks，
每个 rank 32 个 experts。

这一步证明：

- Attention rank 只需要和 FFN coordinator 通信。
- Gate、dispatch、local experts、reduce、combine 都可以放在 FFN 侧完成。
- expert-only ranks 不应等待 Attention P2P，只参与 EP collectives。

但性能明显负优化：

| 配置 | Serial TPOT | 旧 2-rank DBO | EP4 sync | EP4 sync vs Serial |
|---|---:|---:|---:|---:|
| b4/s128/t20 | 252.722 ms | 273.469 ms | 783.681 ms | 0.322x |
| b8/s512/t20 | 351.484 ms | 332.727 ms | 931.040 ms | 0.378x |

原因是 FFN wall compute 仍是 Attention compute 的 `7x~9x`，Attention 大量时间等待 F2A。

### 11.2 EP4 overlap：让 reduce 被下一 MB compute 覆盖

第二阶段实现 `broadcast_reduce_overlap`，并修复两个问题：

1. overlap path 只能在 FFN coordinator/expert-only ranks 进入，Attention rank 不能参与 EP collective。
2. EP hot path 默认不能插入 `torch.npu.synchronize()`，否则细粒度 timing 会破坏真实 overlap。

同时 `ShardedExperts.forward_local()` 改为 active-only：只遍历当前 MB 实际命中的本地
experts，避免 decode 小 batch 下大量 inactive experts 仍发 `where` kernel。

EP4 overlap 将 b8/s512/t20 从 EP4 sync 的 931.040 ms 降到 464.009 ms，但仍低于
serial 和旧 2-rank DBO。

### 11.3 EP7 overlap：找到首个正收益配置

第三阶段在当前 8 张 910C 可见环境中测试 `1 Attention + 7 FFN EP ranks`。
EP7 减少了每 rank local experts 数量，和 overlap/active-only 结合后更接近
Attention/FFN 平衡点。

当前首个正收益配置：

| 配置 | Serial TPOT | 旧 2-rank DBO TPOT | EP overlap TPOT | vs Serial | vs 旧 DBO |
|---|---:|---:|---:|---:|---:|
| EP7 b16/s512/t20 | 502.899 ms | 546.922 ms | 463.440 ms | 1.085x | 1.180x |

结论：DBO 负优化不是不可逆；但小 batch 仍不稳定，下一步应优先做 EP7 矩阵和
token-aware dispatch/combine，而不是盲目继续加 EP degree。

## 12. 结果目录命名

| 目录 | 含义 |
|---|---|
| `results_npu/ep4_broadcast_reduce_sync/` | EP4 + `broadcast_reduce_sync` 同步版负结果；保留用于解释探索过程和瓶颈。 |
| `results_npu/ep_overlap/` | `broadcast_reduce_overlap` 修复结果，包含 EP4/EP7 对比和首个正收益配置。 |

`ep4_broadcast_reduce_sync` 不删除、不归档；它是探索链路的一部分。

## 13. KV cache 与自回归生成

当前实现使用 HuggingFace `DynamicCache`。生成流程是：

1. prefill 处理完整 prompt，初始化 KV cache；
2. 采样得到第一个 token；
3. decode loop 每次只处理最新 token；
4. Attention role 更新 KV cache；
5. FFN role 只处理 hidden states，不持有 cache。

若单次输出只有一个 token，通常是因为未开启 `--generate`、`--tokens` 设置过小，
或运行的是 prefill-only DBO。

## 14. 后端抽象

`src/utils/device.py` 统一处理设备与 backend：

| 后端 | 设备 API | 分布式 backend |
|---|---|---|
| CUDA | `torch.cuda` | NCCL |
| Ascend NPU | `torch.npu` / `torch_npu` | HCCL |
| CPU | `torch.device("cpu")` | Gloo |

NPU 分支使用 `torch_npu.contrib.transfer_to_npu` 兼容部分 CUDA API 表面，但
NPU/HCCL 仍有独立脚本和环境变量。

## 15. 计时与加速比口径

Timing JSON 中常见字段：

| 字段 | 含义 |
|---|---|
| `total_time_ms` | 当前 scheduler 记录的总时间；prefill DBO speedup 使用它。 |
| `prefill_ms` | Serial cache 中的 prefill-only 时间。 |
| `decode_loop_ms` | 自回归 decode loop 总时间，不含 prefill 首 token 路径。 |
| `decode_steps` | decode loop 的 token step 数，通常是 `max_new_tokens - 1`。 |
| `decode_tpot_ms` | 准确 TPOT：`decode_loop_ms / decode_steps`。 |
| `prefill_seq_len` | 实验请求的 prefill 长度。 |
| `actual_prompt_len` | tokenizer 后实际输入长度；用于审计 `s<seq>` 标签是否真实生效。 |
| `events` | 用于 Gantt 图的 decode step 1 layer / micro-batch 事件。 |

EP overlap 额外关注：

| 字段 | 含义 |
|---|---|
| `total_ep_dispatch_wait_ms` | 等待 dispatch 输入就绪的累计时间。 |
| `total_ep_reduce_wait_ms` | 真正等待 reduce handle 完成的累计时间。 |
| `total_ep_overlap_hidden_ms` | reduce enqueue 后到实际 wait 之间，被后续 compute 覆盖的窗口。 |

Speedup 统一为：

```text
speedup = serial / DBO
```

| 模式 | 指标 | 公式 |
|---|---|---|
| Prefill DBO | 模型侧 TTFT-path | `serial_prefill_ms / dbo_total_time_ms` |
| Decode DBO | 准确 TPOT | `serial_decode_tpot_ms / dbo_decode_tpot_ms` |
| Crosslayer decode | 准确 TPOT | `serial_decode_tpot_ms / dbo_decode_tpot_ms` |
| NPU EP decode | 准确 TPOT | `serial_decode_tpot_ms / ep_decode_tpot_ms` |

`events` 和 decode step 1 timing 可以解释 overlap，但不能作为最终 speedup 分母。

## 16. 结果解读边界

- 启动耗时很大一部分来自 Qwen3-30B-A3B 权重加载、进程启动和 warmup，不等于
  scheduler timing。
- OOM 行是容量边界，不是缺失数据。
- 旧 Qwen2 时代或旧 fallback 口径的性能结论不能作为当前结论。
- 当前 `broadcast_reduce_overlap` 还不是 token-aware；不要把它描述成按 token 稀疏通信。
- t3/t5 只用于 smoke，不能作为 speedup 结论。
- 当前 GPU/NPU 实验结论见 [08-gpu-npu-experiment-summary.md](08-gpu-npu-experiment-summary.md)，
  NPU EP overlap 结论见 `results_npu/ep_overlap/README.md`。
