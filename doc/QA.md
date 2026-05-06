# full-matrix-v2 QA / Review 说明

本文解释 `results_npu/full_matrix_v2/` 这轮 NPU-EP7 full matrix 实验的口径、图中现象、已知 bug、双 stream 负结果，以及下一步方案 4/5。目标是之后 review 时能快速回答“数据怎么来的、图怎么看、代码入口在哪、下一步为什么这么做”。

相关入口：

- 实验总览：`results_npu/full_matrix_v2/README.md`
- 聚合脚本：`scripts/aggregate_full_matrix_v2.py`
- 单点报告生成：`scripts/gen_experiment_report.py`
- pipeline 图：`scripts/visualize_dbo_pipeline.py`
- NPU 矩阵 runner：`scripts/run_experiment_matrix_npu.sh`
- NPU per-rank 启动：`scripts/run_npu.sh`
- decode DBO 调度：`src/pipeline/decode_scheduler.py`
- timing/profile 记录：`src/utils/timing.py`
- P2P warmup：`src/distributed/warmup.py`

---

## 1. 为什么 prefill 用 `t=8`，decode 用 `t=20`

**结论**：prefill 的 `t=8` 不是“跑 8 次 prefill”，也不是“生成 8 个 token”。`prefill-dbo` 带 `--no-generate`，性能上只跑一次 prefill forward / TTFT path；`t=8` 只是统一 runner 的 `--max-new-tokens` 字段，并影响 correctness-check 最多记录多少条样本。decode 用 `t=20` 是因为 TPOT 需要足够多 decode step 才能摊薄 cold-start 和 pipeline warmup。

### 1.1 prefill 为什么可以用 t=8

`prefill-dbo` 在矩阵脚本里对应：

```bash
mode=prefill-dbo -> extra="--no-generate"
```

见 `scripts/run_experiment_matrix_npu.sh::run_one()`。`--no-generate` 后只跑 prefill 前向，不进入 autoregressive decode loop。因此 prefill 的性能指标不是 TPOT，而是 TTFT / prefill path latency：

- serial baseline 使用 `prefill_ms`
- DBO prefill 使用 `total_time_ms` / TTFT path
- 报告里叫 `Prefill / TTFT-path` 或 `TTFT speedup`

所以 `prefill t=8` 的准确含义是：

```text
--max-new-tokens 8 + --no-generate
```

性能路径：

```text
只跑 1 次 scheduler.run(input_ids, attention_mask)
```

它不代表：

- 重复跑 8 次 prefill；
- 生成 8 个 autoregressive token；
- 对 8 个 decode step 求平均。

如果启用 `--correctness-check N`，prefill-only path 记录的是：

```python
predicted_ids[:n, -1]
```

也就是 batch 中前 N 条样本的 last-position next-token argmax。这里的 N 来自 correctness-check 上限，不是时间维上的 N 个生成 token。

### 1.2 decode 为什么必须用 t=20

decode TPOT 的定义来自 `src/model/disaggregated.py`：

```text
decode_tpot_ms = decode_loop_ms / decode_steps
decode_steps = max_new_tokens - 1
```

`t=8` 只有 7 个 decode step。对 DBO 来说，前几步会包含：

- HCCL/NCCL 通信通道 cold-start
- graph capture / backend lazy init
- pipeline 从空到满的 warmup
- layer-0/早期 layer 的 shape/kernel 首次开销

这些固定成本在 7 步里占比过高，会把 DBO TPOT 拉得很离谱。已有 b16/s256 例子：

| 配置 | decode TPOT 结论 |
|---|---|
| t=8 | decode-dbo TPOT 约 209.7 ms，speedup 0.62×，是假回退 |
| t=20 | decode-dbo TPOT 约 341.4 ms，speedup 1.47×，是可信值 |

因此这轮 full_matrix_v2 的 decode 主结果全部使用 `t=20`，图和报告里的 decode speedup 都应以 `decode_tpot_ms` 为准，而不是旧的单 step timing。

---

## 2. decode 测试时开 crosslayer 了吗

**结论**：`full_matrix_v2` 主表和主图里的 `decode-dbo` 没有开 `--crosslayer`；当前目录下也没有 `decode-dbo-crosslayer/` 结果文件。代码和脚本支持 crosslayer，但这轮 full matrix 的 decode heatmap/curves 口径是 layer-synchronous `decode-dbo`。

### 2.1 脚本 mode 映射

`scripts/run_experiment_matrix_npu.sh::run_one()` 的 mode 映射是：

| mode | 传给 `run_npu.sh` / `src.main` 的额外参数 | 含义 |
|---|---|---|
| `serial` | `--no-dbo` | AF 分离仍在，但不做 DBO micro-batch overlap |
| `prefill-dbo` | `--no-generate` | 只测 prefill / TTFT path |
| `decode-dbo` | 无额外 DBO 开关 | 默认 decode DBO，`use_crosslayer=False` |
| `decode-dbo-crosslayer` | `--crosslayer` | decode DBO + cross-layer micro-batch pipeline |

`scripts/run_npu.sh` 会把 `--crosslayer` 透传到 `src.main`。`src/main.py` 中参数定义为：

```text
--crosslayer: Enable cross-layer micro-batch pipelining in decode DBO
```

并在生成时传给：

```python
model.generate(..., decode_use_crosslayer=args.crosslayer)
```

### 2.2 crosslayer 在调度器里的实际行为

`src/pipeline/decode_scheduler.py` 中关键逻辑是：

- `use_crosslayer=True`：当前层 F2A sends 还没全部 drain 前，先 post 下一层 A2F `irecv`
- `use_crosslayer=False`：先等当前层 sends drain，再 post 下一层 A2F `irecv`

代码注释说明：

```text
use_crosslayer=True: post BEFORE draining sends
use_crosslayer=False: post AFTER draining sends
```

也就是说，crosslayer 的核心不是改变单层内 MB0/MB1 的顺序，而是让“下一层 A2F recv matching”提前发生，从而减少层与层之间的通信匹配空泡。

### 2.3 本目录数据口径

当前 `results_npu/full_matrix_v2/` 下：

- `decode-dbo/`：有 t=20 主结果
- `prefill-dbo/`：有 t=8 主结果
- `serial/`：有 serial cache/baseline
- `decode-dbo-crosslayer/`：当前没有结果文件

因此 review 图表时不要把 `fig_decode_speedup_heatmap.png` 理解成 crosslayer 结果；它是普通 `decode-dbo`。

---

## 3. 图中 FFN MB0/MB1 的 router、GEMM、dispatch 看起来重叠，是否合理

**结论**：router / dispatch 阶段出现轻微重叠是合理的；旧实现里的 `ep_local_experts` GEMM 主体不应大面积重叠，因为它跑在单条 NPU compute stream 上，mb0 和 mb1 基本串行。旧图的主要问题有两个：一是曾经漏画 MoE 主体阶段，二是后续 composite F lane 虽然补了内部 segment，但外层大框和旧 duration 重建仍容易让人误读成“整个 FFN 主体在重叠”。

### 3.1 先解释几个概念

| 概念 | 在本项目里的含义 |
|---|---|
| FFN | Transformer layer 的 feed-forward / MoE 部分。AF 分离后由 FFN 侧执行。 |
| Router / Gate | MoE 路由器。对每个 token 计算 top-k expert id 和 routing weight。 |
| Dispatch | 把 token 按 expert 分发到拥有该 expert 的 EP rank。EP7 下是 FFN ranks 之间的通信/重排。 |
| `ep_local_experts` / GEMM | 每个 FFN rank 对自己本地 expert 执行 gate/up/down GEMM，是 MoE 主计算开销。 |
| Combine / Reduce | 把各 EP rank 的 partial output 合并回 coordinator。当前实现用 reduce / broadcast_reduce_overlap 后端。 |
| A2F | Attention 侧把 hidden states 发送给 FFN 侧。 |
| F2A | FFN 侧把 FFN output 发送回 Attention 侧进入下一层。 |

一次 EP MoE FFN 大致流程：

```text
A2F recv
  -> router/gate
  -> EP dispatch
  -> local experts GEMM
  -> EP reduce/combine
  -> F2A send
```

### 3.2 为什么 router / dispatch 可以重叠

router 是 NPU compute，dispatch/reduce 是 HCCL 通信。它们走的资源和 stream 不完全相同，且调度器对 mb0/mb1 是 pipelined 的，因此图上看到：

- MB0 dispatch 与 MB1 router 有一点重叠
- MB0 reduce 与 MB1 某些前后处理有一点重叠
- send enqueue 很短，可能夹在其它阶段之间

这些都是合理现象，不代表主 GEMM 已经并行。

### 3.3 为什么 GEMM 主体原本不应重叠

关键瓶颈是 `ep_local_experts`。在 full_matrix_v2 对 b16/s256 layer-1 的诊断里：

```text
mb0 ep_local_experts: 2.864–4.193 ms
mb1 ep_local_experts: 4.298–5.525 ms
```

两段几乎完全串行。原因是当前 `forward_local` 内部按 active expert 循环执行 stacked GEMM，默认在单条 NPU compute stream 上排队。

所以如果图上看起来 MB0/MB1 的“FFN 大块”有重叠，需要先看它到底画的是：

1. router/dispatch 复合阶段；
2. local_experts GEMM；
3. reduce/wait；
4. 还是外层 composite FFN bar。

只有第 2 项才是真正的 expert GEMM 主体。full_matrix_v2 的诊断结论是：router/dispatch 可重叠，`ep_local_experts` 主体串行。

### 3.4 旧图和 composite F lane 为什么容易误导

第一版旧图里，`scripts/visualize_dbo_pipeline.py` 的 F lane 只画了 `ffn_compute`，而当时 `ffn_compute` 实际更接近“router + dispatch enqueue”，约 1.5 ms 就结束。真正耗时的：

- `ep_dispatch_wait`
- `ep_local_experts`
- `ep_reduce`

没有被画进 F lane。于是视觉上会出现：

```text
F lane 很早结束
F2A send 过了很久才出现
```

看起来像 FFN 完成后 F2A 等了很久。修复后的绘图逻辑在 `scripts/visualize_dbo_pipeline.py`：

- `--ffn-view staged` 默认使用分段泳道：
  - `F/router`
  - `F/dispatch`
  - `F/local_experts`
  - `F/combine`
- `--ffn-view legacy` 保留旧 composite F lane，用于前后对比。

为什么默认选 staged，而不是继续用一个大 F bar：

1. 大 F bar 会把 router、dispatch wait、GEMM、reduce wait 全包在一起，读者第一眼看到的是“FFN 大块重叠”。
2. 用户真正关心的是 GEMM 主体是否重叠，因此 `F/local_experts` 应单独成泳道。
3. dispatch 默认合并 enqueue+wait，比进一步拆成 6-8 条泳道更直观；combine 则在同一 lane 内区分总 span、hidden in-flight 和真正 wait，避免把被掩盖的通信误读成阻塞。

还有一个更细的 profile 修复：旧 `ep_local_experts` 等 stage 的 start/end 在 overlap path 中主要由 `event_start + duration` 重建。现在代码在 `EPStageTiming` 中记录真实 start/end，绘图优先使用真实 stage event；历史 JSON 重绘只能改善表达方式，不能恢复当时没有记录的真实时间戳。

因此修复后可以看到 F2A 基本紧跟 reduce，不存在图上那种“大空泡”。同时，A2F lane 会叠加 FFN 侧 `recv_wait`，避免把 Attention 侧 `send_transfer(enqueue)` 误读成“数据已经到 FFN”。

### 3.4.1 MB0 (2.4ms) vs MB1 (0.3ms) 哪个 combine 时间准确

(combine 与 `ep_reduce` 是同一操作；EP MoE 语义里的 "combine" 在代码里用 `dist.reduce(...)` 实现，事件名为 `ep_reduce*`。文档与新版图统一用 **combine**。)

以 `new_staged_ep7_decode_dbo_b16_s512_t20.png` 对应的 L3 原始事件为例：

```text
L3 MB0 ep_reduce:         31.738–34.144 ms  span 2.406 ms
L3 MB0 ep_overlap_hidden: 31.806–34.132 ms  span 2.326 ms
L3 MB0 ep_reduce_wait:    34.132–34.144 ms  span 0.012 ms

L3 MB1 ep_reduce:         33.998–34.313 ms  span 0.315 ms
L3 MB1 ep_overlap_hidden: 34.086–34.303 ms  span 0.217 ms
L3 MB1 ep_reduce_wait:    34.303–34.313 ms  span 0.010 ms
```

哪个"准确"取决于定义：

| 定义 | MB0 | MB1 |
|---|---|---|
| Wall-clock 总 span (`ep_reduce`) | 2.406 ms | 0.315 ms |
| 真正 *阻塞 CPU* 的时间 (`ep_reduce_wait`) | 0.012 ms | 0.010 ms |
| 真正 *非掩盖* 的有效时间 (total − overlap_hidden) | 0.080 ms | 0.098 ms |

**结论**：
- MB0 的 2.4 ms 不是 combine 自身耗时，主要是被 MB1 的 dispatch/local_experts 掩盖的 in-flight 时间（`ep_overlap_hidden = 2.326 ms`）。
- 两个 MB 真正"非掩盖"的 combine 时间都 ≈ 0.1 ms，**接近**；这才是绘图上应该让两个 MB 看起来一样的"准确"时间。
- 真正阻塞调度的时间是 `ep_reduce_wait`，两 MB 都 ~0.01 ms。

调度上 MB0 长是 overlap 设计目标（async reduce 被后续 MB compute 掩盖），不是实验 bug。

性质判断：

| 层面 | 结论 |
|---|---|
| 实验是否错 | 不是。async reduce 被后续 MB compute 掩盖是 overlap 设计目标。 |
| 是否性能问题 | 不一定。`ep_overlap_hidden` 大通常是好事；真正需要关注 `ep_reduce_wait` 是否大。 |
| 是否绘图问题 | 是（旧图）。把整个 `ep_reduce` 画成一条实心 combine 会误导成"MB0 combine 阻塞 2.4 ms"。 |

新图在 `F/combine` 内：
- 主体（实心深色）= **effective combine = total − overlap_hidden**，两 MB 都 ≈ 0.1 ms，看起来对称。
- 浅色斜线（xx 纹）= `ep_overlap_hidden`，表示在 next MB compute 下的 in-flight 时间。
- 这样既能看到 combine 真实有效耗时接近，也能看到通信被计算掩盖的程度。

### 3.4.2 为什么"L1 没有空泡，L3 A2F→router 有空泡"

注意：A2F bar 在新版图中已改为 **sender enqueue start → receiver recv_wait end**（接收方拿到数据时刻）。早期版本是 enqueue 时长（~50 µs）。

从 `decode-dbo b16 s512 t20` 原始 JSON 对齐后看：

```text
L1 MB0  ATT send_transfer: 1.093–1.158 ms
L1 MB0  FFN recv_wait:     0.225–1.209 ms   # FFN 早早 post irecv 等了 0.98 ms 才拿到
L1 MB0  FFN router:        1.520–1.640 ms

L3 MB0  ATT send_transfer: 14.236–14.299 ms
L3 MB0  FFN recv_wait:     16.584–16.623 ms # FFN 16.584 ms 才 post irecv，立刻就收到
L3 MB0  FFN router:        16.936–17.054 ms
```

L1 与 L3 的差别**不**是 combine 引起，而是 **FFN coordinator 何时能 post 下一层 A2F irecv**：

- L1 时 FFN 还没工作（冷启动后 coordinator idle），提前 post 了 irecv —— 表现为 "FFN 在等 ATT"。
- L3 时 FFN coordinator 还在跑 L2 的 `ep_reduce_wait` / F2A send，没机会提前 post L3 irecv；ATT 早就发完，必须等 FFN 走到这一层才匹配上。

这是 non-crosslayer / layer-synchronous 路径的真实层间空泡，正是 `decode-dbo-crosslayer` 的优化对象（提前 post 下一层 irecv）。这张图来自 `decode-dbo`（非 crosslayer），所以才会看到。

性质判断：

| 部分 | 性质 |
|---|---|
| L1/L3 A2F bar 长度差异 | 真实空泡：FFN 侧 recv posting 时机随调度堆积。crosslayer 可缓解。 |
| recv_wait 到 router 的 ~0.3 ms 间隔 | 绘图粒度缺失（pre-router layernorm/reshape/Python 调度未单独成 lane） + 少量真实开销 |
| 是否 bug | 不是正确性 bug；是 non-crosslayer 性能特征。 |

### 3.4.3 send 时间到底代表什么

默认 `comm_timing_mode=enqueue`：
- `send_transfer` event 的 `start/duration` ≈ `dist.isend()` 返回耗时 (~50 µs)。
- 不是数据到达接收方时间。
- "数据到达接收方" 的真实时刻 = 接收方 `recv_wait` 的 end。

新版图修正：A2F bar 跨度 = `[ATT.send_transfer.start, FFN.recv_wait.end]`；F2A 同理 `[FFN.send_transfer.start, ATT.recv_wait.end]`。bar 末端就是接收方拿到数据时刻，下一个计算阶段可立即开始。bar 长度 = 端到端可见传输延迟（含调度等待，不是纯硬件链路时间）。

### 3.5 最终判断

| 现象 | 是否合理 | 解释 |
|---|---|---|
| MB0 router 与 MB1 dispatch/其它轻阶段重叠 | 合理 | compute 与 HCCL/host enqueue 可流水 |
| MB0/MB1 `ep_local_experts` GEMM 大面积重叠 | 旧实现下不合理 | 单 NPU compute stream 串行；若看见重叠，多半是 composite bar 或旧 timestamp 重建误导 |
| MB0 FFN 完成后 F2A 没立刻开始 | 旧图不可信 | 旧图/旧 composite 可能漏画或混画 local_experts/reduce；staged 图应看 `F/combine -> F2A`，并区分 hidden vs wait |
| L3 A2F bar 比 L1 长很多 | 真实层间空泡 | 非 crosslayer 下 FFN 来不及提前 post 下一层 irecv；新版 A2F bar 反映"数据到达 FFN"时刻，长度即可见 |
| F2A 与下一层 attention 是否能重叠 | crosslayer 模式才强化 | 普通 decode-dbo 是 layer-synchronous；`--crosslayer` 会提前 post 下一层 irecv |

---

## 4. prefill-dbo 推理 bug

**结论**：`prefill-dbo` 当前有真实输出 bug，不能用于真实文本生成；但 TTFT 性能数据仍可用于分析 prefill 前向时延趋势。

### 4.1 现象

`results_npu/full_matrix_v2/README.md` 记录的正确性审计：

```text
serial b8/s512:      [334, 16141, 25, 56177]
prefill-dbo b8/s512: [33975, 33975, 33975, 33975]
```

多个 prefill-dbo 配置都输出固定 token id `33975`，不是普通浮点误差。

### 4.2 为什么这是 bug，而不是正常非确定性

decode-dbo 中有少数 token 分歧，例如第 3 个 token 因 EP reduce 顺序改变导致 logits 边界翻转。这种现象在 MoE/EP 系统里可以接受，因为：

- 前几个 token 大多一致；
- 分歧不是常量；
- 不同 batch/seq 下 token 仍像正常文本分布。

prefill-dbo 的问题不同：

- 所有配置趋向固定 `33975`
- serial 同配置正常
- token 分布常量化

这说明 hidden state 的 last-position、lm_head 输入或 micro-batch combine 结果很可能错位/被覆盖。

### 4.3 可能根因

下一步应优先排查：

1. **last-token slice 错位**：prefill-dbo 只需要每条样本最后一个有效 token 的 logits；如果从错误位置取 hidden，会得到固定 token。
2. **micro-batch 合并错误**：MB0/MB1 的 batch 维还原可能错位，导致 row0 取到无效 hidden。
3. **lm_head 输入错误**：只对某个 micro-batch 做了 lm_head，或把 coordinator 的 partial output 当最终 output。
4. **combine/reduce 覆盖**：FFN reduce 后 residual add / reshape / output_device 转移时覆盖了部分 hidden。
5. **attention_mask / position_ids 不匹配**：prefill DBO 的分块与原始 batch/seq 对齐不一致。

### 4.4 对本轮数据的影响

| 数据类型 | 是否可信 | 说明 |
|---|---|---|
| prefill-dbo TTFT / latency | 基本可信 | 前向路径确实跑完，可用于性能趋势 |
| prefill-dbo 文本生成正确性 | 不可信 | 输出固定 token，必须修 |
| decode-dbo TPOT | 可信 | 主结果来自 t=20，正确性多数正常，少量浮点分歧已记录 |

---

## 5. 数据缺陷：t=8 为什么不能测 decode TPOT

**结论**：t=8 可以做 smoke / correctness，但不能做 decode TPOT 结论；原因是 decode TPOT 需要足够多 step 摊薄 warmup，而 t=8 只有 7 个 decode loop step。

### 5.1 TPOT 计算口径

`scripts/gen_experiment_report.py` 明确：

```text
Decode speedup uses exact decode_tpot_ms, averaged over all decode-loop steps.
```

`decode_tpot_ms` 来自完整 decode loop：

```text
decode_tpot_ms = decode_loop_ms / decode_steps
```

而不是 pipeline 图里展示的某一个 step。

### 5.2 t=8 的问题

t=8 时：

- decode_steps = 7
- step0/early steps 里包含 backend cold-start
- DBO 相比 serial 多一些 pipeline 建立和通信图捕获成本
- 分母太小，固定开销无法摊薄

所以 t=8 得到的 TPOT 更像“warmup + 少量 decode”的混合指标，不适合作为 steady-state decode TPOT。

### 5.3 本轮处理方式

- prefill-dbo：保留 t=8，因为不看 decode TPOT。
- decode-dbo：全部用 t=20 重跑。
- 报告与热力图：使用 `decode_tpot_ms`，不是单 step timing，也不是旧的 `decode_step_ms`。

---

## 6. 当前分支下 profile / 串行 / 并行 / warmup 代码在哪，原理是什么

这一节用于 code review。重点是从“矩阵命令”一路追到“每个 rank 记录 timing JSON”的路径。

### 6.1 矩阵入口：`scripts/run_experiment_matrix_npu.sh`

核心函数是 `run_one(mode, batch, seq, tokens, outdir)`。

它负责：

1. 把 mode 转成 `src.main` 参数；
2. 启动 `scripts/run_npu.sh`；
3. 把 `results/prefill_dbo/timing_*.json` 移到 `results_npu/full_matrix_v2/{mode}/`；
4. 对 serial baseline 做 cache；
5. 调 `scripts/gen_experiment_report.py` 生成单点 markdown report；
6. 写 `experiment_matrix_summary.csv`。

mode 映射：

```bash
serial                -> --no-dbo
prefill-dbo           -> --no-generate
decode-dbo            -> <默认 DBO decode>
decode-dbo-crosslayer -> --crosslayer
```

profile/timing 相关参数：

| 参数 | 作用 |
|---|---|
| `--comm-timing-mode enqueue` | send bar 表示 `isend()` enqueue/return 开销，默认模式 |
| `--comm-timing-mode completion` | send bar 表示 Work 完成跨度，包含真实传输、排队和接收端 readiness |
| `--no-timing` | 关闭详细 timing，用于评估 profile 开销 |
| `--correctness-tokens N` | 透传 `--correctness-check N`，把 greedy 前 N token 写入 timing JSON |

### 6.2 per-rank 启动：`scripts/run_npu.sh`

`run_npu.sh` 做的事：

- 根据 preset 设置拓扑：
  - `npu-ep7`: `ATTN_SIZE=1`，`FFN_SIZE=7`，`FFN_EP_SIZE=7`
- 设置 HCCL 环境：
  - `HCCL_BUFFSIZE`
  - `HCCL_CONNECT_TIMEOUT`
  - `HCCL_EXEC_TIMEOUT`
- 为每个 rank 启动一个进程：

```bash
python -u -m src.main \
  --backend npu \
  --role attention|ffn \
  --world-size ... \
  --rank ... \
  --ffn-ep-size ... \
  --batch-size ... \
  --prefill-seq-len ... \
  --max-new-tokens ... \
  --timing \
  --timing-suffix ...
```

默认带 `--timing`；传 `--no-timing` 时，`TIMING_ARGS=()`，用于对比 profile 开销。

文件命名由 `MODE_TAG` 和 `SUFFIX` 控制，例如：

```text
decode-dbo_npu_ep7_broadcast_reduce_overlap_b16_s256_t20
```

每个 rank 日志在：

```text
results/logs/npu_${SUFFIX}_r${RANK}.log
```

### 6.3 `src.main` 中的 profile / timing 参数

主要参数在 `src/main.py`：

| 参数 | 作用 |
|---|---|
| `--timing` | 开启详细 per-MB timing，保存到 `results/prefill_dbo/timing_*.json` |
| `--timing-mode cuda_events` | 默认；通过 `src/utils/device.py` 调用当前 CUDA/NPU compute stream 同步 + `perf_counter`，尽量保留 DBO overlap |
| `--timing-mode sync` | legacy；device-level sync，会破坏 overlap，只适合 debug |
| `--timing-suffix` | 控制 timing JSON 文件名 |
| `--comm-timing-mode enqueue/completion` | 控制 send event 的含义 |
| `--correctness-check N` | 强制 greedy；decode 记录生成 token，prefill-only 记录 batch 前 N 行 last-position next-token |

`src/utils/timing.py` 中 `TimingTracker` 定义了事件：

```text
attn_compute
ffn_compute
moe_router
ep_dispatch
ep_local_experts
ep_reduce
ep_dispatch_wait
ep_reduce_wait
send_transfer
recv_wait
```

其中 `send_transfer` 的语义取决于 `comm_timing_mode`：

- `enqueue`：只代表 `isend()` 返回/排队开销，不是真实传输完成时间；
- `completion`：代表 Work 完成跨度，但包含 backend 排队、接收端 readiness、通信流调度，不是纯硬件链路时间。

这也是为什么 pipeline 图里通信 bar 不能简单理解成“线缆传输耗时”。

### 6.3.1 profile 方式是否准确，speedup 是否可信

结论分两层：

1. **TTFT / TPOT 加速比是主指标，整体可信。**
   - prefill 使用 `prefill_ms` / `total_time_ms`。
   - decode 使用 `decode_tpot_ms = decode_loop_ms / decode_steps`。
   - `scripts/experiment_baselines.py` 会按 mode 匹配 serial baseline；decode 不再接受旧的 `decode_step_ms` 作为精确 TPOT。

2. **pipeline 子事件图是诊断图，不是 speedup 来源。**
   - 图里只展示选定 layer 范围和一个 timed decode step。
   - A/F 是不同进程，脚本需要用 A2F send/recv anchor 对齐时钟，存在近似。
   - `send_transfer` 的意义取决于 `comm_timing_mode`，不能直接当纯链路传输时间。
   - 历史 JSON 中 EP 子阶段时间线曾经由 duration 重建，因此只能用于趋势解释。
   - `TimingTracker` 已改为使用 `src/utils/device.py` 的 CUDA/NPU 统一同步封装，避免 NPU profile 仍依赖硬编码 `torch.cuda.*`。

以 `new_staged_ep7_decode_dbo_b16_s512_t20.png` 对应的 JSON 为例：

```text
decode_steps      = 19
decode_tpot_ms    ≈ 351.15 ms
timed_decode_step = 1 (0-based, 跳过 step0 cold-start)
comm_timing_mode  = enqueue
```

这说明：

- TPOT 是 19 个 decode loop step 的平均，不是图中单个 timed step 的子事件相加；
- 图中 send bar 是 enqueue 语义，不是传输完成；
- 该图来自已有历史 JSON，能看 EP 阶段和画法，但不能作为修复后真实 timestamp 的最终证据。

本次修复后：

- `EPStageTiming` 新增 router、dispatch、local_experts、reduce/combine 的真实 start/end timestamp。
- `decode_scheduler.py` 写 timing event 时优先使用真实 timestamp；旧 JSON 没有这些字段时仍 fallback 到旧 duration 重建逻辑。
- `visualize_dbo_pipeline.py --ffn-view staged` 默认把 F 拆成 `F/router`、`F/dispatch`、`F/local_experts`、`F/combine`，避免用一个 composite F bar 误导读者。

因此，判断口径应是：

| 问题 | 可信口径 |
|---|---|
| DBO 到底快不快 | 看 `prefill_ms` / `decode_tpot_ms` 与 serial baseline 的比值 |
| FFN GEMM 是否真的重叠 | 看新跑数据里的 `F/local_experts` 泳道 |
| 历史图为什么看着重叠 | 多半是 composite F bar 或旧 stage 时间线重建造成的视觉误导 |
| 通信是否被掩盖 | 结合 `F/dispatch`、`F/combine`、A2F/F2A 与 compute overlap 看趋势，不要把 send bar 当纯硬件传输；combine 中要区分 `ep_overlap_hidden` 和 `ep_reduce_wait` |

### 6.4 串行 vs 并行的代码路径

#### serial

命令口径：

```bash
--no-dbo
```

含义：

- AF 分离仍存在；
- Attention rank 和 FFN rank 仍通信；
- 不做 DBO micro-batch pipeline；
- 用作 serial baseline cache。

#### prefill-dbo

命令口径：

```bash
--no-generate
```

含义：

- 只跑 prefill；
- 使用 DBO scheduler 做 prefill micro-batch overlap；
- 不进入 autoregressive decode loop；
- 性能指标看 TTFT，而不是 TPOT。

#### decode-dbo

命令口径：

```bash
<默认>  # no --no-dbo, no --no-generate, no --crosslayer
```

含义：

- prefill 后进入 decode loop；
- 每个 decode step 用两个 micro-batch 做 layer 内 DBO；
- `use_crosslayer=False`，普通 layer-synchronous。

#### decode-dbo-crosslayer

命令口径：

```bash
--crosslayer
```

含义：

- 在 decode DBO 基础上启用跨层 pipeline；
- 当前层 send drain 前先 post 下一层 A2F irecv；
- 目标是减少层间 irecv match latency。

### 6.5 warmup 代码和原理

#### P2P warmup

入口参数：

```text
src/main.py:
  --warmup-p2p
  --warmup-rounds
```

调用路径：

```text
src/main.py -> ctx.warmup(...) -> src/distributed/warmup.py::warmup_p2p(...)
```

`warmup_p2p()` 原理：

1. 创建小 tensor；
2. 对 peer rank 做多轮双向 `dist.isend` / `dist.irecv`；
3. `handle.wait()` 等待完成；
4. 记录第一轮 cold latency 和后续 warm latency；
5. 如果传了 `extra_groups`，也对 directional groups（如 A2F/F2A）做同样预热。

目的：

- 启动 HCCL/NCCL proxy thread；
- 建立 P2P 通道；
- 触发 communicator / graph 的 lazy init；
- 避免第一个真实 decode step 被几十 ms cold-start 污染。

#### prefill warmup

入口参数：

```text
--prefill-warmup-rounds
```

位置：`src/main.py` prefill-only path。默认：

```python
warmup_rounds = 1 if DEVICE_TYPE == 'npu' else 0
```

原理：

- 在 timed run 之前先跑若干次 untimed prefill；
- 暂时关闭 scheduler timing；
- 吸收 NPU 后端 JIT / graph compile / kernel lazy init；
- warmup 后 barrier + device synchronize，再开始正式计时。

### 6.6 Review 建议路径

如果要 review 当前分支 profile/串并行/warmup，建议按这个顺序读：

1. `scripts/run_experiment_matrix_npu.sh::run_one()`：确认 mode 到参数的映射。
2. `scripts/run_npu.sh`：确认 preset、rank、环境变量、`--timing`、日志和 suffix。
3. `src/main.py`：确认 `--no-dbo`、`--no-generate`、`--crosslayer`、`--timing`、`--warmup-*` 如何进入模型。
4. `src/utils/timing.py`：确认每个 event 的语义，特别是 `send_transfer`。
5. `src/pipeline/decode_scheduler.py`：确认 DBO / crosslayer 调度顺序。
6. `src/distributed/warmup.py`：确认 P2P warmup 原理。
7. `scripts/gen_experiment_report.py` 和 `scripts/visualize_dbo_pipeline.py`：确认 report 和图如何解释 timing JSON。

---

## 7. `ep_local_experts` 双 stream 并行

**结论**：双 stream 方案已经在后续分支验证过，能修正 stream 同步语义，但没有性能收益；根因是 910C 上 MoE GEMM 受 HBM 带宽限制，两个 compute stream 竞争同一 HBM，无法带来真正并行。

### 7.1 方案详细介绍

目标：让 MB0 / MB1 的 `ep_local_experts` 不再排在同一条默认 NPU compute stream 上，而是：

```text
MB0 -> compute_streams[0]
MB1 -> compute_streams[1]
```

核心改动思路：

1. 在 decode scheduler 中懒初始化两条 `torch.npu.Stream()`；
2. 在 FFN per-MB loop 中按 `mb_idx % 2` 选择 stream；
3. `compute_local(item, stream=s)` 中：
   - `s.wait_stream(default_stream)`：保证 side stream 读到 dispatch/router 生产的数据；
   - 对输入/输出 tensor 调 `record_stream(s)`：避免 caching allocator 提前回收；
   - `with torch.npu.stream(s): forward_local(...)`：把 GEMM enqueue 到侧流；
4. `reduce_async(item)` 在同一 side stream context 下 enqueue reduce，保证 collective 看到 producer ordering；
5. `finish_reduce(item)` 在 `reduce_handle.wait()` 后让 default stream 等 side stream，保证后续 residual add / F2A send 读到正确数据。

### 7.2 实施中发现的同步 bug

#### bug 1：side stream 读未完成的 dispatch 结果

现象：出现类似：

```text
KeyError: 4440336306335333802
```

这是 `selected_experts` 读到了未初始化/错误内存，expert id 变成巨大随机数。根因是 side stream 没有等待 default stream 上的 producer op。

修复：

```python
stream.wait_stream(default_stream)
```

#### bug 2：错误 back-sync 导致 MB1 被串行化

如果在 `compute_local` 末尾立即：

```python
default_stream.wait_stream(side_stream)
```

会形成：

```text
default -> side0 -> default -> side1
```

这会让 MB1 的 side stream 间接等待 MB0 完成，完全抵消双流并行。正确做法是把 default 等 side 的动作推迟到 `finish_reduce()`，也就是真正消费 reduce 后 output 的位置。

### 7.3 为什么无效果

实测 b16/s256/t20 decode-dbo：

| 配置 | TPOT | 结论 |
|---|---:|---|
| dual-stream on | 约 376.36 ms | 无收益 |
| dual-stream off | 约 376.13 ms | 与 on 持平 |

on/off 在同一分支、同一环境下几乎一致，说明双 stream 没有带来可观硬件并行。

根因：

1. `ep_local_experts` 是 MoE stacked GEMM，主要受 HBM 读权重/激活带宽限制；
2. 两条 NPU stream 并不能复制 HBM 带宽，只会抢同一内存子系统；
3. `record_stream` 会延迟 allocator 回收，可能提高 HBM 峰值；
4. HCCL reduce ordering 更复杂，收益却没有出现。

### 7.4 当前建议

- 双 stream 代码可以保留为实验开关；
- 默认应关闭；
- 只有在方案 4（grouped/fused MoE）降低 HBM 往返后，才值得重新复测双 stream。

---

## 8. 方案 4：`npu_grouped_matmul` / fused-MoE

**结论**：方案 4 是下一步最值得做的优化，因为它不是“让两个 HBM-bound GEMM 抢带宽”，而是减少 Python loop、kernel launch 和不必要的 HBM 往返。

### 8.1 当前瓶颈

当前 `forward_local` 大致逻辑：

1. 找出 routed 到本 rank 的 token/expert assignment；
2. 按 expert id 排序；
3. `torch.unique_consecutive` 得到每个 expert 的 token 数；
4. Python loop 遍历 active local expert：
   - `F.linear(seg, gate_up_stack[local_idx])`
   - `gate, up = gu.chunk(2)`
   - `silu(gate) * up`
   - `F.linear(hidden, down_stack[local_idx])`
   - copy 回 output slice
5. `index_add_` 合并到 partial。

问题：

- active expert 多时，每层每 MB 有多次小 GEMM；
- Python loop + kernel launch 开销高；
- 每个 expert 分段访问权重，HBM 访问不够连续；
- 双 stream 无法解决 HBM-bound 的本质。

### 8.2 方案设计

使用 torch_npu 已暴露的 API：

```text
torch_npu.npu_grouped_matmul
torch_npu.npu_grouped_matmul_swiglu_quant
torch_npu.npu_moe_init_routing
torch_npu.npu_moe_compute_expert_tokens
torch_npu.npu_moe_finalize_routing
torch_npu.npu_moe_token_permute / npu_moe_token_unpermute
```

优先做最小可控版本：

1. 保留现有 routing / sort / counts 逻辑；
2. 构造 `group_list` 表示每个 active expert 的 token 数；
3. 把 active expert 的 gate/up weights 组成 grouped weight；
4. 用 `npu_grouped_matmul` 一次完成所有 active expert 的 gate/up GEMM；
5. 执行 SiLU * up；
6. 再用 `npu_grouped_matmul` 一次完成 down GEMM；
7. 保持原有 `index_add_` combine，先确保 correctness；
8. 再尝试 fused `npu_grouped_matmul_swiglu_quant` 或 MoE routing/finalize API。

### 8.3 预期收益

收益来源：

- 减少 Python loop；
- 减少每 expert 单独 kernel launch；
- grouped GEMM 对 active experts 一次调度，硬件利用率更高；
- 访问模式更连续，有机会降低 HBM 压力；
- 为之后重新测试双 stream 创造条件。

预期收益等级：**中高**。如果 `ep_local_experts` 目前是每层 1-3 ms 级别，grouped path 只要降低 15%-30%，decode TPOT 就可能有可见收益。

### 8.4 风险

| 风险 | 说明 |
|---|---|
| weight layout | PyTorch `F.linear` 使用 `[out, in]`，grouped matmul 可能要求 `[k, n]` 或特定转置 |
| group_list 语义 | `group_list_type=0/1`、cumsum vs counts 必须严格对齐 |
| dtype / shape 限制 | bf16、2D/3D weight、inner dim 限制需要逐项验证 |
| correctness | MoE routing 权重、token order、unpermute/index_add 必须与旧路径一致 |
| fallback | API 在某些 shape 下可能报错，需要保留旧路径开关 |

### 8.5 验证指标

先做单点：

- b16/s256/t20
- b64/s512/t20
- b128/s512/t20

对比：

- `ep_local_experts avg/layer`
- FFN avg/layer
- decode TPOT
- correctness tokens
- HBM 峰值 / OOM 边界

---

## 9. 方案 5：Token-aware combine / reduce-F2A overlap

**结论**：方案 5 是更激进的通信/协议优化，收益可能中等，但复杂度高；建议排在方案 4 之后。

### 9.1 当前问题

当前 EP combine/reduce 更接近“每个 MB 的 local experts 做完后再整体 reduce，然后 F2A send”。这导致：

- F2A 需要等 reduce 完成；
- attention 侧下一层 A2F/attn 启动可能被 F2A recv wait 卡住；
- 大 batch/seq 下 reduce/F2A 等待会放大。

### 9.2 token-aware combine 的想法

不要等整个 MB 所有 expert 都完成后再整体 combine，而是按 token chunk / expert chunk 推进：

```text
expert chunk done
  -> partial combine/reduce
  -> coordinator 得到部分 token output
  -> 尽早 F2A send 或为下一层准备
```

如果与 crosslayer 配合，目标是让：

```text
当前层 MB0 reduce/F2A
```

更早与：

```text
下一层 MB1 attn / A2F
```

重叠。

### 9.3 预期收益

收益来源：

- 缩短 attention 侧 F2A recv wait；
- 减少层间空泡；
- 对长 seq / 大 batch 的通信等待更有帮助。

预期收益等级：**中等**，但依赖方案 4 之后 FFN 主体是否仍是瓶颈。如果 FFN compute 仍远大于通信，方案 5 收益有限；如果方案 4 把 FFN 主体压下来，方案 5 的相对价值会上升。

### 9.4 风险

| 风险 | 说明 |
|---|---|
| 协议复杂 | partial token output 的 tag、buffer、ordering 都要重新设计 |
| HCCL ordering | 不能提前 post 会阻塞后续 op 的 recv，避免 NCCL/HCCL stream 死锁 |
| 生命周期 | partial tensor 必须保留到 async Work 完成 |
| correctness | token chunk 级别 combine 容易错位 |
| 图表解释 | pipeline 图需要支持 chunk-level event，否则难以判断收益 |

### 9.5 建议执行顺序

1. 先做方案 4，降低 FFN 主体；
2. 复测 F2A recv-wait 是否成为主瓶颈；
3. 如果 recv-wait / reduce 仍占比高，再做方案 5；
4. 每一步都必须有 correctness tokens 和 pipeline 图。

---

## 10. 本轮 full_matrix_v2 的核心结论

1. `decode-dbo` 主结果是 t=20、非 crosslayer，42 个 ok 配置，speedup 约 1.17×-5.87×。
2. `prefill-dbo` t=8 TTFT 结果显示 35 个 ok 配置，speedup 约 2.93×-8.94×，但有真实推理输出 bug。
3. 旧 pipeline 图里 F2A 大空泡主要是绘图漏画 MoE 主体；修复后 F lane 应看 composite bar。
4. MB0/MB1 router/dispatch 轻微重叠合理；`ep_local_experts` GEMM 主体旧实现基本串行。
5. t=8 不适合 decode TPOT，原因是 warmup/cold-start 被 7 个 decode step 放大。
6. 当前分支 review profile 要从 `run_experiment_matrix_npu.sh -> run_npu.sh -> src.main -> timing.py / decode_scheduler.py / warmup.py` 顺着看。
7. 双 stream 已验证无收益，下一步应优先做方案 4 grouped/fused MoE，再考虑方案 5 token-aware combine。
