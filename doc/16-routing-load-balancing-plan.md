# Coordinator Routing / 负载均衡现状与后续计划

本文聚焦 **Coordinator 控制面下的 routing / 负载均衡**，解释当前仓库里已经实现到哪一步、真实 Qwen3 路径里哪些能力已经接通、哪些地方仍然只是骨架，以及后续实验为什么先以 **synthetic skew / trace-replay** 为主，而不是直接把真实数据集回放设为第一阶段主路径。

---

## 1. 当前结论

先给结论：

1. 当前真实 Qwen3 decode/prefill 路径已经支持 `--routing-backend coordinator`，但**已验证的是 one-shot routing**，不是动态负载均衡。
2. 当前 `expert_to_rank` 在真实 EP 路径里的语义更接近 **“expert ownership / shard placement”**，而不是“每步都可自由改写的 dispatch 目标”。
3. 虽然代码里已经有 `LoadAwareRouter`、`UpdateMetrics`、`routing-update-mode=poll`，但 **real path 还没有形成闭环的 load-aware rebalancing**：
   - 真实 Qwen3 路径当前只会在初始化时拉表；poll 只是在 safe point 拉一次最新表。
   - 如果新表改变了 EP ownership，decode 路径会**拒绝切换**，因为 live expert migration 还没实现。
   - router 需要的 metrics 目前主要只在 skeleton worker 路径里上报，真实 `src.main` 路径还没有同等级的 queue / per-expert load 上报。
4. 因此，当前最合理的推进顺序不是直接跑真实数据集，而是：
   - **第一阶段**：用 synthetic skew / trace-replay 验证 routing 机制是否真的会重排、safe point 是否稳定、指标是否可观测；
   - **第二阶段**：再用真实数据或真实 trace 回放验证“这个机制在实际 workload 上是否仍有收益”。

---

## 2. 当前代码里 routing 是怎么接进去的

### 2.1 控制面已有的组件

当前仓库里，Coordinator 控制面相关实现已经具备以下组件：

| 组件 | 位置 | 当前作用 |
|---|---|---|
| gRPC server | `src/coordinator_arch/coordinator_server.py` | worker 注册、返回 routing table、接收 metrics、触发 rebalance |
| gRPC client | `src/coordinator_arch/coordinator_client.py` | worker 侧拉表 / 订阅 / 上报 metrics |
| rebalance 算法 | `src/coordinator_arch/router.py` | 根据 `queue_len_avg + dispatch_rate_tps * weight` 做贪心装箱和限步平滑 |
| routing CLI | `src/main.py` | 暴露 `--routing-backend/--coord-addr/--routing-update-mode/--routing-poll-interval-steps/--routing-rpc-timeout-s` |
| real-path 接线 | `src/model/disaggregated.py` | 初始化时取表、decode safe point poll、把 `expert_to_rank` 下沉到 FFN worker 初始化 |

也就是说，**“控制面能发表、业务路径能取表”** 这部分已经成立。

### 2.2 真实 Qwen3 路径里 `expert_to_rank` 的当前语义

当前真实路径里，`expert_to_rank` 最重要的用途不是“每个 token dispatch 时临时决定去哪台机器”，而是：

1. 模型初始化时取到 coordinator table；
2. 在 EP 模式下把 table 传给 `FFNWorker(..., expert_to_rank=...)`；
3. `FFNWorker` 基于这张表构造 `ExpertShardPlan`，决定**当前 rank 实际加载/持有哪组 experts**。

也就是说，在当前实现中：

- `expert_to_rank` 决定的是 **哪个 FFN EP rank 持有哪个 expert 的权重**；
- 它不是一个仅影响 dispatch 路由、却不影响权重布局的“轻量表”。

这点非常关键，因为它直接决定了后续“动态负载均衡”不能只靠多拉几次表就成立。

### 2.3 `poll` 为什么现在还不等于“动态负载均衡”

真实 decode 路径里的 safe-point poll 逻辑在 `src/model/disaggregated.py`：

- 只有 `routing_update_mode == "poll"` 时才会按步数间隔拉表；
- 如果拉到的新表版本号没变，就直接跳过；
- 如果在 EP 模式下发现 **新表改变了 `expert_to_rank` ownership**，当前代码会记录 warning，然后继续沿用旧表。

原因也很直接：当前实现没有 live expert migration。  
如果某个 decode step 中途把 expert 3 从 rank1 改到 rank5，但 rank5 并没有那份权重，业务路径就会立刻不一致。

所以当前 `poll` 模式的真实能力更接近：

- 可以在 decode safe point 拉取 coordinator table；
- 可以接受**不改变 ownership** 的元数据更新；
- **不能**在真实 EP decode 过程中完成 ownership 迁移式重平衡。

---

## 3. 当前 router 为什么还没有在真实路径里形成闭环

### 3.1 server/router 侧已有 rebalance 触发条件

`CoordinatorServicer.UpdateMetrics()` 收到 worker metrics 后会尝试触发 `_maybe_rebalance()`，内部调用 `LoadAwareRouter.rebalance(...)`。

router 当前使用的输入主要是：

- `queue_len_avg`
- `dispatch_rate_tps`
- `per_expert_load`（如果有）

如果负载不均衡超过阈值，就会给出一张新的 `expert_to_rank`。

### 3.2 但真实 Qwen3 路径并没有把这些 metrics 真正喂满

目前可见的 `coord.update_metrics(...)` 调用点主要在：

- `src/coordinator_arch/workers/attention_worker.py`
- `src/coordinator_arch/workers/ffn_worker.py`

这两处属于早期 skeleton worker 路径，更多是为了：

- 保持 worker 不被 stale-sweep 回收；
- 给 coordinator 一个基本 heartbeat；
- FFN 侧带一个非常简化的 `queue_len_avg`。

而当前真实 Qwen3 NPU 实验主路径走的是 `src.main -> src/model/disaggregated.py`，并不是这套 skeleton worker 服务循环。  
因此，**router 虽然存在，但 real path 还没有稳定、持续、可解释的 load metrics 上报链路**。

这意味着当前状态更准确的表述应该是：

- **one-shot control-plane plumbing 已经通了**
- **real-path dynamic load balancing 还没闭环**

而不是“我们已经实现了动态负载均衡，只差跑实验”

---

## 4. 当前 one-shot 结果到底证明了什么

截至目前，单机 1A7F / EP7 real decode 路径已经证明：

1. `coordinator -> routing table -> explicit expert ownership -> real FFN shard init` 这条链是通的；
2. coordinator one-shot 与 static EP7 可以在同一真实路径下做 apples-to-apples 对比；
3. 当前 one-shot coordinator 在部分配置上可以优于 static，但它本质上仍接近 static round-robin ownership，不应被解读成动态路由收益。

它**没有**证明的事情包括：

1. poll 模式在真实 decode loop 中稳定；
2. 路由表会因为负载变化而发生真实有效的重排；
3. ownership 变化后的 live 切换可行；
4. 真实业务 workload 下 dynamic routing 比 static 有确定收益。

---

## 5. 为什么第一阶段不把真实数据集作为主路径

### 5.1 因为当前还在验证“机制是否存在”

在真实数据集上直接跑 routing 实验，看起来更“真实”，但对于当前阶段并不是最高效的选择。  
原因是当前还有多个机制层问题没有 isolate：

1. router 的 metrics 闭环还没接到 real path；
2. poll 能否稳定拉表与计时还没验证；
3. ownership change 当前会被拒绝；
4. 跨机实验还叠加了 HCCL / TBE JIT / warm cache / network 波动。

如果现在直接上真实数据集，最后即使结果“不好”，也很难判断到底是：

- 路由机制没生效；
- 指标口径错了；
- 数据集负载太平；
- 还是 compile / network 噪声盖掉了收益。

### 5.2 synthetic skew / trace-replay 更适合第一阶段

第一阶段应该优先选择 **可控、可复现、可放大不均衡** 的 workload：

| 工作负载 | 作用 | 为什么适合第一阶段 |
|---|---|---|
| synthetic skew | 主动制造热点 expert / 热点 rank | 能验证 router 是否真的想重排、是否过度抖动 |
| trace-replay | 重放真实请求长度/到达节奏，但不必依赖完整业务数据集 | 更接近线上负载，同时仍保留可复现性 |
| 真实数据集 | 验证最终外部有效性 | 适合第二阶段，不适合作为机制 bring-up 的唯一入口 |

因此，本轮设计上的明确选择是：

- **第一阶段主路径：synthetic skew + trace-replay**
- **第二阶段验证：真实数据集或真实 trace 回放**

---

## 6. 后续路线图

### Phase A — 文档与观测先行

目标：先把“能不能测明白”解决。

1. 补 routing 设计文档（本文档）。
2. 明确当前 real path 缺哪些 metrics：
   - FFN queue/backlog
   - per-expert load
   - dispatch tokens/sec
   - poll 次数与耗时
3. 明确跨机实验还需要的系统级指标：
   - MFU
   - 网络 payload / 带宽利用率

### Phase B — 让 real path 至少具备可观测的 poll 闭环

目标：不是马上追求收益，而是先回答“动态路径是否真的在动”。

建议最小工作包括：

1. 在真实 Qwen3 路径里补 metrics 上报，而不是只依赖 skeleton worker heartbeat。
2. 让 decode summary / timing JSON 明确记录：
   - `routing_backend`
   - `routing_update_mode`
   - `routing_table_version`
   - `routing_poll_count`
   - `routing_poll_ms`
   - 后续新增的 queue / load 指标
3. 先验证 `poll` 模式在**不改变 ownership** 的情况下不会引入明显回退。

### Phase C — 明确“动态路由”到底选哪种语义

这是后续设计里最关键的分叉点。

#### 路线 C1：epoch-level / stop-the-world ownership rebalance

思路：

- 只在请求边界、实验轮次边界或显式同步点重排 `expert_to_rank`；
- 重排后重新构建/重启 FFN shard；
- 不追求 decode loop 中的 live migration。

优点：

- 最贴合当前代码语义；
- 实现风险低；
- 适合先验证“换一种 ownership 会不会改善长时负载均衡”。

缺点：

- 不能做到在线细粒度调度；
- 更像 coarse-grained repartition，而不是真正 per-step dynamic routing。

#### 路线 C2：live ownership migration

思路：

- 允许运行中把 expert ownership 从 rank A 迁到 rank B；
- 需要权重迁移、双写、barrier、版本一致性保障。

优点：

- 语义最完整，真正支持在线重平衡。

缺点：

- 实现复杂度高；
- 当前代码完全没有这层迁移基础设施；
- 在跨机 1A7F + TBE cache 背景下，调试成本极高。

#### 路线 C3：保持 ownership 静态，仅优化 dispatch 级路由

思路：

- 不变更专家权重放置；
- 只在 token dispatch 策略上做更灵活的分发。

但对当前实现来说，这条路**暂时不可直接落地**，因为当前 explicit sharding 直接把 `expert_to_rank` 绑定成 ownership/source of truth。

综合来看，现阶段最现实的顺序是：

1. 先把 **C1** 做出来，验证 coarse-grained rebalance 是否值得；
2. 如果收益明确，再讨论是否投资 **C2**。

---

## 7. 实验计划

### 7.1 第一阶段：synthetic skew

目标：证明 routing 机制“会动、没抖坏、方向正确”。

建议配置：

1. 固定少数 hotspot experts，占据大部分 token；
2. 人工制造长短请求混部，让某些 FFN rank backlog 明显升高；
3. 比较：
   - static
   - coordinator oneshot
   - coordinator poll（若当前 ownership 不变）

观察指标：

- routing table version 演进
- poll count / poll ms
- queue/backlog 分布
- per-expert load 分布
- TPOT / throughput / overlap-hidden

### 7.2 第二阶段：trace-replay

目标：用更接近真实 workload 的节奏验证第一阶段观察到的趋势。

建议做法：

1. 记录或构造一组请求 trace：
   - prompt len
   - decode len
   - 到达时间 / batch 形成节奏
2. 重放 trace，但不要求一开始就接入完整业务数据集。

### 7.3 第三阶段：真实数据集 / 真实 trace 回放

目标：做外部有效性验证，而不是机制 bring-up。

这里才适合回答：

- dynamic routing 在真实 workload 下是否仍优于 static？
- 收益是否稳定覆盖 compile / communication / scheduling 噪声？

---

## 8. 与跨机 1A7F 实验的关系

routing 与跨机 1A7F 不是两条独立线。

当前跨机实验至少要同时回答三个问题：

1. control-plane + fallback real decode path 是否稳定；
2. 当前 compile 过久问题是否已经通过 warm cache / shape 覆盖被 isolate；
3. 除了 TPOT 之外，MFU 和网络带宽利用率的观测是否成立。

因此，跨机 1A7F 在 routing 视角下的合理定位应该是：

- **先验证观测与机制**
- 再谈“routing 带来的收益”

而不是一上来直接把跨机结果解释成“动态负载均衡有效/无效”。

---

## 9. 当前建议

按优先级排序：

1. 先把 real path 的 metrics / poll 可观测性补齐；
2. 先用 synthetic skew / trace-replay 验证 routing 机制；
3. 并行推进跨机 1A7F，但先把 compile isolate 和 MFU / bandwidth 口径固定；
4. 在 ownership live migration 没实现之前，不把当前 poll 模式包装成“已支持在线动态负载均衡”；
5. 真实数据集验证保留到第二阶段，不作为当前 routing 文档与第一轮实验的硬前置。

---

## 10. 关联文档

- [12-coordinator-arch.md](12-coordinator-arch.md)：Coordinator 总体设计与控制/数据面背景
- [14-communication-modes.md](14-communication-modes.md)：Fallback / DeepEP 通信方式说明
- [15-cross-host-communication-diagnosis.md](15-cross-host-communication-diagnosis.md)：跨机 HCCL / fallback / TBE JIT 现状
- [`results_npu/coordinator_arch/singlehost_ep7/README.md`](../results_npu/coordinator_arch/singlehost_ep7/README.md)：单机 1A7F / EP7 one-shot 对照结果
