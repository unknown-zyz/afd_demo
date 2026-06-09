# 项目内通信架构对比

本文对比当前 AFD Demo 中已经实现、已接入实验、或已完成可行性验证的通信架构，重点回答三个问题：

1. 每种架构理论上在传什么、同步几次、能 overlap 到什么程度。
2. 现有 NPU 实测显示瓶颈在哪里。
3. 当前应该默认使用哪条路径，哪些路径适合作为下一步优化方向。

## 结论摘要

当前真实端到端 decode DBO 路径仍建议默认使用 `broadcast_reduce_overlap`。它不是理论最优通信模式，但在现有 Ascend 910C/HCCL 环境里是最稳定、实测最快的 EPFFN 路径。Host1 单机 `1A8F` 小配置复测也保持该结论：`b2/s32/t3` 下 broadcast/reduce crosslayer TPOT `280.855ms`，`all_to_all_single` TPOT `483.774ms`，同拓扑 serial TPOT `160.657ms`。

`all_to_all_single` 已经接入真实 `EPFFNLayer`，并在跨机 `1A16F / EP16` 上跑通 smoke 与代表配置；但当前 HCCL all-to-all dispatch 延迟高于 full broadcast，端到端 TPOT 反而更慢。因此它应保留为实验 backend，而不是默认 backend。

DeepEP 与 `torch_npu.npu_moe_distribute_dispatch_v2/combine_v2` 理论上更接近 MoE sparse dispatch/combine 的目标形态，但当前在本项目真实 Qwen 路径中仍是 blocked/experimental：DeepEP 还不是可用生产路径，official MoE v2 在隔离 probe 中可跑通部分接口，但真实 nonzero expert output combine 仍有 timeout 风险。

## 架构总览

| 架构 | 当前入口 | 通信粒度 | 理论优势 | 当前实测/状态 | 建议 |
|---|---|---|---|---|---|
| 静态 A/F disaggregation P2P | `src/model/disaggregated.py`, `src/distributed/` | Attention 与 FFN rank 间发送 hidden states | 简单、稳定，是所有 DBO/EP 路径的基础 | 可用；serial baseline 也是 A/F disaggregated，不是 monolithic | 保留为基线和基础设施 |
| `broadcast_reduce_sync` | `--ffn-ep-backend broadcast_reduce_sync` | full hidden/router broadcast + dense partial reduce | 实现简单，collective 顺序清晰 | 功能路径；overlap 少 | 仅作正确性/对照 |
| `broadcast_reduce_overlap` | `--ffn-ep-backend broadcast_reduce_overlap` | full hidden/router broadcast + dense partial reduce，调度侧 overlap | 当前最成熟；能部分隐藏 FFN/通信 | EP16 sweep 最佳 DBO speedup 仍只有 `0.916x`，但比 all-to-all 快 | 当前默认推荐 |
| 真实 EPFFN `all_to_all_single` | `--ffn-ep-backend all_to_all_single` | token-expert assignment all-to-all + all-to-all combine | 理论上利用 MoE 稀疏性，避免 full broadcast/dense reduce | EP16 可跑通；`b32/s256` TPOT `664.915ms`，慢于 broadcast/reduce | 保留实验；先 profile/优化 HCCL dispatch |
| Coordinator sparse P2P | `--ffn-ep-backend sparse_p2p_overlap` | coordinator 单源 P2P count/hidden/expert/output | 理论上避免 all-to-allv 固定开销与 dense reduce | CPU/Gloo reference 通过；Host1 NPU EP7 smoke 出现 HCCL P2P payload/tag 错配 | 继续视为 blocked，不作为性能路径 |
| Coordinator fallback A2A | `src/coordinator_arch/comm/fallback_a2a.py` | routing-table 驱动的 PyTorch `all_to_all_single` | 动态路由/EPLB 的通用 fallback | 单元/RT bench 可用；已复用到真实 EPFFN backend | 作为 coordinator 正确性基线 |
| DeepEP normal/low_latency | `--use-deepep`, coordinator comm | 专用 MoE dispatch/combine | 理论上减少 PyTorch collective overhead，更适合 MoE | 安装/导入有记录，但端到端仍 experimental | 中长期候选，不作默认 |
| Official torch_npu MoE v2 | `npu_moe_distribute_dispatch_v2/combine_v2` probe | CANN MoE distribute ops | 理论上最贴近 Ascend 官方优化路径，可使用 `fullmesh_v2` | CANN 8.5.1 + torch_npu 2.9 隔离环境 probe 有进展；真实 Qwen combine_v2 曾 timeout | 继续隔离验证，暂不接生产 |

## 理论性能模型

以下记号用于粗略比较通信形态：

- `T = batch * seq`：prefill token 数；decode 单步可近似为当前 batch token 数。
- `H`：hidden size，本项目 Qwen3 path 为 `2048`。
- `K`：top-k experts，Qwen3 MoE 常见为 top-8。
- `E`：EP rank 数。
- `dtype`：常用 BF16，即每元素 2 bytes。

### 1. 静态 A/F disaggregation P2P

静态 A/F 拆分是项目的基础路径：Attention rank 持有 embedding、attention、KV cache、lm head 和 sampling；FFN rank 持有 post-attention norm、gate、experts 和 combine。层间 hidden states 在 Attention 与 FFN 之间发送。

理论特征：

| 项 | 分析 |
|---|---|
| Payload | 每层 A2F/F2A 都与 `T * H` 成正比。 |
| Collective | 主要是 rank 间 point-to-point 或小组通信，不涉及 EP 内专家稀疏 dispatch。 |
| 同步点 | 每层 Attention 和 FFN 之间天然有依赖。 |
| Overlap 空间 | DBO 通过 microbatch 把下一个 microbatch 的 Attention 与当前 microbatch 的 FFN/通信重叠。 |

这条路径的重点不是减少 MoE 内部通信，而是为 pipeline overlap 提供稳定分层边界。所有 serial/DBO speedup 都应使用同拓扑 A/F serial 作为 baseline。

### 2. `broadcast_reduce_sync`

该路径是最直接的 FFN EP 实现：

1. FFN coordinator rank 计算 layernorm/router/top-k。
2. coordinator 将完整 `hidden_2d + selected_experts + routing_weights` broadcast 到所有 EP ranks。
3. 每个 EP rank 只计算自己拥有专家的 dense partial output。
4. EP ranks 对完整 `[T, H]` partial 做 dense reduce/sum 回 coordinator。

理论特征：

| 项 | 分析 |
|---|---|
| Dispatch payload | `hidden_2d` 是 `T * H`，还要带 router/weights；broadcast 到 `E` 个 rank。 |
| Combine payload | 每个 rank 输出 dense `[T, H]` partial，reduce 仍与 `E * T * H` 相关。 |
| 稀疏性利用 | 低。虽然每个 rank 只算本地专家，但通信仍传所有 token 的 hidden 和 dense partial。 |
| 优点 | collective 顺序简单，容易 debug，HCCL 稳定性好。 |
| 缺点 | EP 越大，同步和 dense reduce 成本越容易吞掉 local compute 降低带来的收益。 |

该路径适合作为正确性对照，不适合作为最终性能目标。

### 3. `broadcast_reduce_overlap`

`broadcast_reduce_overlap` 与 sync 版本的通信形态相同，但在 decode scheduler 中把 EP dispatch/local/reduce 拆成 work item，允许 Attention 侧继续计算后续 microbatch 或后续层，减少等待气泡。

理论特征：

| 项 | 分析 |
|---|---|
| Payload | 与 `broadcast_reduce_sync` 相同，没有减少字节数。 |
| 收益来源 | 调度 overlap，而不是通信量下降。 |
| 主要瓶颈 | 当 dispatch/reduce 的绝对时间大于可重叠 compute 窗口时，Attention recv wait 仍会出现。 |
| 当前优势 | collective 形态简单，HCCL 实测比当前 all-to-all 更快。 |

EP16 sweep 的关键结论是：大 batch 下 local FFN 已接近 Attention，但通信/等待仍主导。例如 `decode-dbo b256/s1024`：

| 指标 | 数值 |
|---|---:|
| 同拓扑 serial TPOT | `1041.612ms` |
| DBO TPOT | `1137.203ms` |
| Speedup | `0.916x` |
| Attention avg/layer | `1.537ms` |
| FFN local avg/layer | `1.560ms` |
| Dispatch avg/layer | `1.916ms` |
| Reduce avg/layer | `1.017ms` |
| Attention recv wait avg/layer | `1.518ms` |

这说明继续单纯增大 EP 并不能解决问题：local FFN 已对齐，瓶颈转移到了通信和 pipeline bubble。

## `all_to_all_single` sparse dispatch/combine

### 设计目标

`all_to_all_single` backend 的目标是把 EPFFN 从 full broadcast + dense reduce 改为 token-aware sparse dispatch/combine：

1. coordinator 根据 top-k routing 生成 `(token, expert)` assignment。
2. 按 expert ownership 将 assignment 发往目标 EP rank。
3. 每个 rank 只对收到的 assignment 运行本地 expert。
4. combine 阶段将 assignment output gather 回 coordinator，再按原 token/top-k 顺序加权求和。

理论上，dispatch payload 从 full broadcast 的 `E * T * H` 形态变为约 `T * K * H` 的 sparse assignment 形态；当 `K << E` 且 HCCL all-to-all 足够高效时，它应该优于 broadcast/reduce。

### 当前实现状态

当前项目已经完成真实路径接入：

- CLI：`--ffn-ep-backend all_to_all_single`
- 通信复用：`src/coordinator_arch/comm/fallback_a2a.py`
- EPFFN 接入：`src/model/ep_moe.py`
- Decode overlap 接入：`src/pipeline/decode_scheduler.py`

实现细节上有三个阶段：

| 阶段 | 变化 | 结论 |
|---|---|---|
| 初版 | hidden + weights + expert ids 都通过 all-to-all | 功能跑通，但 dispatch 慢。 |
| no-weight | 不发送 routing weights，combine 时使用 coordinator 原始 top-k weights | payload 减少，但实测没有稳定收益。 |
| metadata experts | 不发送 expert ids；按 per-destination/per-expert counts 重构 `recv_experts` | 避免 expert-id payload，但 metadata broadcast/变长 all-to-all 仍慢。 |

一个重要修复是：PyTorch `dist.broadcast(src=..., group=...)` 的 `src` 必须是全局 rank，而不是 group 内 rank。因此 metadata source 使用 `ctx.ffn_coordinator_rank`。

### 实测结果

| 配置 | Backend | TPOT | Dispatch | Local experts | Combine/Reduce | 结论 |
|---|---|---:|---:|---:|---:|---|
| EP16 `b2/s32/t3`, debug 2 layers | `all_to_all_single` metadata | `483.923ms` | N/A | N/A | N/A | HCCL smoke 通过。 |
| Host1 EP8 `b2/s32/t3`, debug 2 layers | same-topology serial | `160.657ms` | N/A | N/A | N/A | 小配置 serial 仍明显更快。 |
| Host1 EP8 `b2/s32/t3`, debug 2 layers | `broadcast_reduce_overlap + crosslayer + early_recv` | `280.855ms` | N/A | N/A | N/A | 功能通过；仍慢于 serial。 |
| Host1 EP8 `b2/s32/t3`, debug 2 layers | `all_to_all_single + crosslayer + early_recv` | `483.774ms` | N/A | N/A | N/A | 功能通过；慢于 broadcast/reduce。 |
| EP16 `b16/s128/t20` | `all_to_all_single` metadata | `596.726ms` | `4.972ms` | `0.946ms` | `1.154ms` | 慢于 broadcast/reduce `315.297ms`。 |
| EP16 `b32/s256/t20` | `broadcast_reduce_overlap` | `384.361ms` | `1.943ms` | `1.467ms` | `0.917ms` | 当前真实路径较优。 |
| EP16 `b32/s256/t20` | `all_to_all_single` 初版 | `636.266ms` | `4.370ms` | `1.055ms` | `1.052ms` | sparse 语义跑通但 dispatch 偏慢。 |
| EP16 `b32/s256/t20` | `all_to_all_single` no-weight | `611.773ms` | `4.761ms` | `1.432ms` | `1.312ms` | 减少 weight payload 未解决主瓶颈。 |
| EP16 `b32/s256/t20` | `all_to_all_single` metadata | `664.915ms` | `5.131ms` | `1.090ms` | `1.211ms` | metadata/count exchange 开销仍高。 |
| EP16 `b256/s1024/t20` | `all_to_all_single` no-weight | `1325.661ms` | `4.025ms` | `1.345ms` | `1.200ms` | local expert 低，但 TPOT 慢于 broadcast/reduce `1137.203ms`。 |

结论：当前 all-to-all 的理论 payload 优势没有转化为端到端收益，主要因为 HCCL `all_to_all_single` dispatch 固定成本、变长 split sizes、count exchange、排序/恢复等开销超过了 broadcast/reduce 的简单 collective 成本。

## Coordinator sparse P2P overlap

`sparse_p2p_overlap` 的目标是避免 EP 全体 all-to-all：真实 source 只有 FFN coordinator，所以 coordinator 可以按 expert ownership 把 assignment 直接发给对应 EP rank，expert rank 只返回 packed assignment outputs。

当前验证结果：

| 项 | 结果 |
|---|---|
| CPU/Gloo reference | `tests/test_ep_moe_reference.py` 通过，assignment combine 与 reference output 一致。 |
| Host1 NPU EP7 smoke | `b2/s32/t3` 未通过；不同 expert rank 收到非本地 expert id，另有 count tensor 读出异常大值。 |
| 已尝试修复 | P2P group 从 `WORLD` 改为 `ctx.ffn_ep_group`，peer rank 使用全局 rank；P2P tag 改为低位、按 layer/seq/peer/slot 分段；增加 count 上限校验，避免协议错配时按异常 count 分配大 tensor。 |
| 当前判断 | HCCL NPU P2P 在多 peer、多 payload、同源 coordinator 模式下仍有 tag/payload 匹配风险，短期不适合作为性能主线。 |

因此，短期不要把 `sparse_p2p_overlap` 纳入矩阵性能实验；它只保留为实验 backend 和后续 HCCL P2P 最小复现入口。

## Coordinator fallback communicator

Coordinator 架构的目标不是只替换一个 collective，而是提供动态 MoE 调度能力：

- routing table 可由 coordinator 维护；
- worker 可按 routing table 做 expert ownership；
- fallback communicator 使用 `torch.distributed.all_to_all_single` 提供可移植正确性基线；
- DeepEP 或 official MoE op 可以作为更高性能 communicator 插件。

`FallbackMoECommunicator` 的价值在于：

| 维度 | 说明 |
|---|---|
| 正确性 | 能表达 token/expert assignment 级 sparse dispatch/combine。 |
| 可移植性 | 基于 PyTorch distributed，CPU/Gloo 和 NPU/HCCL 都可验证。 |
| 可扩展性 | 可接 routing table、EPLB、动态专家副本。 |
| 性能限制 | 仍依赖通用 `all_to_all_single`，当前 HCCL 变长 all-to-all 成本偏高。 |

因此 fallback communicator 适合作为 coordinator 路线的功能基线，而不是当前性能上限。

## DeepEP normal / low_latency

DeepEP 的理论定位是 MoE 专用通信库，比通用 PyTorch all-to-all 更接近生产 MoE dispatch/combine：

| 项 | 理论优势 |
|---|---|
| Dispatch/combine | 面向 MoE token dispatch，减少通用 collective 的布局和同步开销。 |
| Low latency | 更适合 decode 小 token/batch 场景。 |
| Normal mode | 更适合 prefill 或较大 token payload。 |
| 与 coordinator | 可以作为 `MoECommunicator` 后端，隐藏在 routing table 之后。 |

当前状态仍是 experimental。已有文档记录了 Ascend 环境中的安装、导入与测试问题；在真实端到端路径稳定前，不应作为默认通信后端。下一步应先跑最小 RT bench，再进入 EPFFN end-to-end。

## Official torch_npu MoE distribute v2

Ascend 官方 MoE distribute ops 是理论上最值得继续投入的方向之一：

- `torch_npu.npu_moe_distribute_dispatch_v2`
- `torch_npu.npu_moe_distribute_combine_v2`
- `comm_alg=fullmesh_v2`

理论优势：

| 项 | 分析 |
|---|---|
| 官方 CANN 路径 | 更可能使用 Ascend 优化过的 fullmesh 通信和 layout。 |
| MoE 语义 | API 直接表达 dispatch/combine，比手写 PyTorch all-to-all 更接近目标。 |
| 潜在融合 | dispatch/combine/count/layout 可能减少 Python 和多 collective 开销。 |

当前已知状态：

| 环境/场景 | 结果 |
|---|---|
| torch_npu 2.6 | 对 Qwen3 hidden=2048 不直接可用：base API 要求 EP multiple of 8 且 H=7168，v2 缺少 `comm_alg` kwarg。 |
| 隔离 Host2 容器 CANN 8.5.1 + torch_npu 2.9 | `dispatch_v2` 暴露 `comm_alg`，H=2048 probe 可用，`comm_alg=fullmesh_v2` 可跑。 |
| 真实 Qwen nonzero expert output | 曾在 `combine_v2` 触发 aicore timeout；zero-experts/synthetic 路径不代表真实可用。 |

因此 official v2 目前不能直接替换 EPFFN 生产路径。它的下一步不是再做端到端大矩阵，而是先在隔离脚本里把真实 Qwen expert output 的 nonzero combine 稳定性问题复现、定位并修复。

## 选择建议

| 时间尺度 | 推荐动作 | 原因 |
|---|---|---|
| 当前默认 | 保持 `broadcast_reduce_overlap` | 实测最快、稳定、可解释，适合作为后续所有通信优化 baseline。 |
| 短期优化 | 继续优化 `broadcast_reduce_overlap` 调度，并只对 `all_to_all_single` 做 targeted profile | 当前 NPU 结果显示 broadcast/reduce 仍快于 all-to-all；sparse P2P 还没过 NPU smoke。 |
| 中期路线 | 继续完善 coordinator fallback + routing table | 为 EPLB、动态专家副本和可插拔 communicator 打基础。 |
| 中长期路线 | official MoE v2 / DeepEP | 理论性能最好，但必须先解决真实 Qwen combine/RT bench 稳定性。 |

## 后续实验清单

1. `msprof` 对比 `broadcast_reduce_overlap` 与 `all_to_all_single`：
   - 重点看 Communication Time、AllToAll、Broadcast、Reduce、Overlap Analysis。
   - 代表配置：`b16/s128`, `b32/s256`, `b256/s1024`。
2. 做 padded/equal-split all-to-all 原型：
   - 避免每层变长 split metadata；
   - 对比浪费 padding payload 与降低 collective 固定成本之间的权衡。
3. 继续 official v2 隔离复现：
   - 固定 CANN 8.5.1 + torch_npu 2.9；
   - 使用真实 Qwen hidden=2048 和真实 expert output；
   - 优先解决 `combine_v2` nonzero output timeout。
4. DeepEP 最小 RT bench：
   - normal 与 low_latency 分别验证；
   - 先单机，再跨机；
   - 通过后再接入真实 EPFFN。

## 相关文件

- `src/model/ep_moe.py`：真实 EPFFN backend、`ShardedExperts`、all-to-all 接入点。
- `src/coordinator_arch/comm/fallback_a2a.py`：coordinator fallback `all_to_all_single` communicator。
- `src/coordinator_arch/comm/moe_communicator.py`：communicator 抽象接口。
- `src/pipeline/decode_scheduler.py`：decode DBO overlap 调度。
- `doc/12-coordinator-arch.md`：coordinator 架构状态。
- `doc/13-deepep-install-test-error-guide.md`：DeepEP 安装与测试问题记录。
- `doc/14-communication-modes.md`：通信模式背景。
- `doc/20-attention-worker-npu-optimization-progress.md`：Attention/NPU 优化与 MoE op 调研上下文。
- `doc/22-crosshost-static-ep-overlap.md`：cross-host static EP overlap 说明。
- `crosshost_static_ep16_sweep/experiment_report.md`：EP16 sweep、负收益分析、all-to-all 对比结果。
