# Profile / Pipeline Visualization Fix

> **术语**：MoE 语义里的 **combine** 与底层 op `dist.reduce(...)` (event 名 `ep_reduce*`) 是同一操作。文档与新版图统一称 *combine*；JSON event/code 仍叫 `ep_reduce*`。

本目录保存 `scripts/visualize_dbo_pipeline.py` 修复后的新旧 pipeline 图对比，用于回答：

1. 为什么旧图里 FFN MB0/MB1 看起来重叠严重。
2. `router / dispatch / local_experts / combine` 这种拆法是否更直观。
3. 哪些图只能说明“画法差异”，哪些图能看到 EP 子阶段。

## 图的命名

| 前缀 | 含义 |
|---|---|
| `old_legacy_*` | 旧 composite F lane 画法：一个 F 大框包住 router/dispatch、local_experts、reduce。 |
| `new_staged_*` | round-3 staged 画法：F 被拆成 `F/router`、`F/dispatch`、`F/local_experts`、`F/combine`。 |
| `new4lane_*` | **round-4 默认画法**：4 泳道 `Attention / A2F / FFN / F2A`，目标是直观展示 GPU compute 与通信通道的重叠。 |

新图默认使用 4-lane（`--ffn-view fourlane`）：

```bash
python3 scripts/visualize_dbo_pipeline.py ...    # 默认 fourlane
python3 scripts/visualize_dbo_pipeline.py ... --ffn-view staged   # 想看 EP 子阶段拆解
python3 scripts/visualize_dbo_pipeline.py ... --ffn-view legacy   # 旧 composite F
```

## 如何解读 4-lane 图（round-4，默认）

核心目标：让"compute 行 vs comm 行"的占满程度直接说明 NPU 资源利用率，让"通信泳道的 bar"直接对应"端到端通信代价"。

| 泳道 | bar 起点 | bar 终点 | 包含 stage | 用途 |
|---|---|---|---|---|
| **Attention** | `attn_compute.start` | `attn_compute.end` | ATT compute | ATT 占用计算单元的时段 |
| **A2F** | `ATT.send_transfer.start` | `FFN.ep_local_experts.start` | send + recv_wait + moe_router + ep_dispatch + ep_dispatch_wait | "传输 + FFN 端 GEMM 之前的所有准备"全部计入通信侧；bar 末端 = FFN 真正进入 GEMM 时刻 |
| **FFN** | `ep_local_experts.start` | `ep_local_experts.end`（fallback `moe_experts` / `ffn_compute`） | FFN expert GEMM 主体 | FFN 占用计算单元的时段 |
| **F2A** | `ep_reduce.start`（fallback `FFN.send_transfer.start`） | `ATT.recv_wait.end` | combine + reduce_wait + send + ATT recv_wait | 含 ATT 端串行 recv 排队；bar 长度 = 端到端"通信 + 接收侧 ready"时长 |

**重要**：A2F / F2A 终点都用接收方 wait_end，**不替换为网络到达时刻**。这样：

- bar 真实反映"通信 + 接收侧排队"总占用，是 pipeline 优化的指示器；
- 例：`new4lane_ep7_decode_dbo_b16_s512_t20.png` L2 的 MB1 F2A bar 长达 ~3.9 ms，原因不是网络慢，而是 ATT 串行 recv 阻塞（先 wait MB0 → 才 post MB1 irecv）。完整解释见 `doc/QA.md` §3.4.4。

`hidden in-flight`（round-3 staged 图里的 xx 斜线）在 4-lane 图里**不再单画**，因为 combine 整段并入 F2A 通信泳道，hidden 自然包含在 F2A bar 里；`ep_overlap_hidden` event 仍保留在 timing JSON 中供分析（见 §3.4.5）。

## 如何解读 staged 图（round-3 备选）

staged 图保留 `A / A2F / F2A` 三类链路，同时把 FFN 拆成四条子泳道：

| 泳道 | 含义 | 主要用途 |
|---|---|---|
| `F/router` | FFN coordinator 上的 MoE gate / top-k routing | 看轻量 routing 是否能和其它阶段流水。 |
| `F/dispatch` | EP dispatch，默认包含 enqueue + wait span | 看 A2F 后 FFN 内部数据分发耗时。 |
| `F/local_experts` | 本地 expert GEMM 主体 | 判断 MB0/MB1 FFN 实际计算是否重叠的核心依据。 |
| `F/combine` | EP reduce / combine。深色实心 bar = effective combine（= total − overlap_hidden），两个 MB 在这个度量下时间接近 ≈ 0.1ms。浅色 xx 斜线 = `ep_overlap_hidden`，表示在后续 MB compute 之下被掩盖的 in-flight 时间。 | 看 combine 真正"非掩盖"耗时（应小且对称）；看通信被 compute 掩盖的程度。不要把 hidden 部分理解成阻塞。 |

默认不再把 dispatch enqueue、dispatch wait、reduce enqueue、reduce wait 拆成更多泳道，因为那会更像 profiler trace。当前 review 重点是避免把“轻阶段/通信等待”误读成“FFN GEMM 主体并行”，所以 `F/local_experts` 单独成泳道最直观。

## A2F / F2A bar 语义（重要，新版已修改）

`A2F` 与 `F2A` lane 的 bar 跨度为 **sender enqueue start → receiver recv_wait end**，bar 末端就是 *接收方拿到数据* 的时刻。这样可以直观看出"FFN/ATT 何时能开始下一个计算阶段"：

| 形态 | 含义 |
|---|---|
| 实心 bar + 起点三角 marker | 起点 = sender `isend()` 返回（enqueue），终点 = receiver `recv_wait` end（数据到达接收方） |
| bar 长度 | 端到端可见传输延迟，包含 backend 排队、sender/receiver 进度差。**不是**纯硬件链路传输时间。 |

旧版 (round 2) 把 ATT-side `send_transfer` 与 FFN-side `recv_wait` 分别叠加显示在 A2F lane，新版去掉了 recv_wait 空心 overlay 与 mb 标签，统一改用上述合成 bar。

如果 bar 很短（比如 L1 MB0 0.1 ms），说明 sender enqueue 时 receiver 已经在 wait，数据立刻到位；如果 bar 很长（比如 L3 MB0 2.4 ms），说明 receiver 那一侧还没来得及 post irecv，A2F 在等 receiver 进度——这是 non-crosslayer 的层间空泡，正是 crosslayer 优化的目标。

## Serial baseline / Speedup 自动推断

新版脚本会自动从 `--attn-timing` 文件名解析 `b{B}_s{S}_t{T}`，到下面优先级路径里查找匹配的 serial cache JSON：

```text
results_npu/serial/timing_attention_serial_b{B}_s{S}_t{T}.json
results/serial/timing_attention_serial_b{B}_s{S}_t{T}.json
```

找到后自动作为 `--serial-timing` 输入，title 显示 `Serial TPOT/TTFT: ... | Speedup: ... x`。找不到才显示 N/A 并打印警告。手动指定 `--serial-timing` 仍优先。

## 已生成对比图

### EP7 decode（推荐重点看）

这组来自已有 EP7 overlap timing JSON，包含完整 FFN 子阶段事件：

- `old_legacy_ep7_decode_dbo_b16_s512_t20.png`
- `new_staged_ep7_decode_dbo_b16_s512_t20.png`

输入文件：

```text
results_npu/ep7_matrix_v2/decode-dbo/timing_attention_decode-dbo_npu_ep7_broadcast_reduce_overlap_b16_s512_t20.json
results_npu/ep7_matrix_v2/decode-dbo/timing_ffn_coordinator_decode-dbo_npu_ep7_broadcast_reduce_overlap_b16_s512_t20.json
```

这组最适合判断新图是否直观：`F/router`、`F/dispatch`、`F/local_experts`、`F/combine` 四条泳道都有事件。

注意：这组文件名和路径是 `decode-dbo`，不是 `decode-dbo-crosslayer`，所以这张图 **没有开启 crosslayer**。标题里也会显示 `no crosslayer`。它体现的是 layer-synchronous decode DBO，不是跨层提前 post 下一层 recv 的版本。

### 普通 decode / crosslayer 历史图

这两组历史 JSON 只有 `ffn_compute / recv_wait / send_transfer`，没有 EP 子阶段事件；staged 图会把 `ffn_compute` fallback 到 `F/local_experts`，只能用于比较新旧布局，不能用于判断真实 EP 子阶段。

- `old_legacy_decode_dbo_b8_s512_t20.png`
- `new_staged_decode_dbo_b8_s512_t20.png`
- `old_legacy_decode_crosslayer_b4_s512_t20.png`
- `new_staged_decode_crosslayer_b4_s512_t20.png`

### prefill 历史图

这组有 `moe_router / moe_experts`，但不是 EP7 dispatch/reduce 口径，因此 `F/dispatch` 和 `F/combine` 为空：

- `old_legacy_prefill_dbo_b8_s512_t20.png`
- `new_staged_prefill_dbo_b8_s512_t20.png`

## 可信边界

本次脚本修复解决的是“图如何表达得更直观”。对于已有历史 JSON：

- 如果 JSON 已经包含 `moe_router / ep_dispatch / ep_local_experts / ep_reduce`，新图能拆分显示这些阶段。
- 如果 JSON 只有 `ffn_compute`，新图无法恢复当时没有记录的真实子阶段，只能 fallback。
- 如果旧 JSON 的 EP 子阶段 start/end 是由旧逻辑重建出来的，新图仍会使用这些历史 event；它能减少 composite bar 的视觉误导，但不能把历史时序变成真实时序。

代码已补充真实 stage timestamp 记录：后续新跑数据时，`decode_scheduler.py` 会优先使用 `EPStageTiming` 里的真实 start/end 写入 timing event。只有新跑出来的 `new_staged` 图，才适合作为 `F/local_experts` 是否真实重叠的依据。

另外，`TimingTracker` 已改为通过 `src/utils/device.py` 调用当前 CUDA/NPU backend 的 stream/device synchronize，不再在 timing 核心路径里硬编码 `torch.cuda.*`。这使 NPU profile 的同步语义和项目 backend abstraction 保持一致。

## 对后两个现象的详细解释

### 为什么 MB1 的 combine 比 MB0 短很多

这不是实验 bug。它主要是 **真实 async reduce overlap 现象 + 旧图语义不够清晰**。

当前代码里的 overlap 调度顺序大致是：

```text
MB0 local_experts
MB0 reduce_async        # reduce 开始 in-flight
MB1 local_experts       # 用 MB1 计算掩盖 MB0 reduce
MB1 reduce_async
finish_reduce(MB0)
finish_reduce(MB1)
```

因此 `MB0 ep_reduce` 的总 span 会很长，因为它从 `reduce_start` 一直跨到 `reduce_wait_end`，中间大段是 `ep_overlap_hidden`：通信已经发起，但被 MB1 的计算掩盖了。以 L3 为例：

```text
L3 MB0 ep_reduce:         20.503–22.909 ms, dur 2.406 ms
L3 MB0 ep_overlap_hidden: 20.571–22.897 ms, dur 2.326 ms

L3 MB1 ep_reduce:         22.763–23.078 ms, dur 0.315 ms
L3 MB1 ep_overlap_hidden: 22.851–23.068 ms, dur 0.217 ms
```

所以 MB0 combine 长，不代表 MB0 阻塞了 2.4ms；真正阻塞应看 `ep_reduce_wait`，这里只有约 0.012ms。MB1 是该层最后一个 MB，没有后续 MB 的 compute 用来隐藏它的 reduce，所以它的总 span 反而短。

判断性质：

| 层面 | 结论 |
|---|---|
| 实验是否错 | 不是。async reduce 被后续 MB compute 掩盖是 overlap 设计目标。 |
| 是否性能问题 | 不一定。`ep_overlap_hidden` 大通常是好事，说明通信被盖住；真正要警惕的是 `ep_reduce_wait` 大。 |
| 是否绘图问题 | 是。旧 staged 图把整个 `ep_reduce` 画成一条实心 combine，容易误读成阻塞。 |

因此新图在 `F/combine` lane 内区分总 span、hidden in-flight 和真正 wait。

### 为什么 L3 A2F 和 router 之间有很大空泡

这也不是单一 bug，而是 **真实 layer-synchronous 空泡 + send enqueue 语义 + 绘图缺少接收/预处理阶段** 叠加。

从原始事件看，L3 MB0：

```text
ATT L3 MB0 send_transfer: 14.236–14.299 ms  # enqueue，不是到达
FFN L3 MB0 recv_wait:     16.584–16.623 ms
FFN L3 MB0 router:        16.936–17.054 ms
```

第一段大空泡来自非 crosslayer 的 layer-synchronous 调度：FFN coordinator 要等上一层 F2A sends drain 后，才 post 下一层 A2F irecv。与此同时，Attention 侧的 A2F `send_transfer` 在 enqueue 模式下很快返回，但这不代表接收端已经 ready 或数据已经到达。

第二段小空泡是 `recv_wait` 结束到 `router` 开始，中间包括 residual/hidden preparation、post-attention layernorm、reshape/contiguous、Python 调度与同步。当前 `router` event 只覆盖 gate/top-k，不覆盖这些 pre-router 工作。

判断性质：

| 部分 | 性质 |
|---|---|
| A2F send 很早结束但 FFN 很晚 recv/router | 真实的 non-crosslayer 性能空泡 + enqueue 语义误读，不是正确性 bug。 |
| recv_wait 到 router 的小间隔 | 图粒度问题 + 少量真实 pre-router 开销。 |
| 是否 crosslayer 能改善 | 能。crosslayer 的目标就是提前 post 下一层 irecv，减少这种层间 recv matching 空泡。 |
