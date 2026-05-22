# 12. Coordinator-based Dynamic MoE Architecture

> 状态：**Phase α+**（骨架已落地；`src.main` 已支持 Host1 单机 1A1F coordinator smoke，真实多 FFN / EP decode 路径仍未接入）
> 关联代码：`src/coordinator_arch/`
> 关联文档：`doc/01-architecture.md`（旧静态 A↔F）、`doc/deepep_ascend_install_report.md`

## 1. 背景与动机

现有 `src/model/{attention_worker,ffn_worker,disaggregated}.py` 实现了 ATTN/FFN 角色
分离 + DBO 流水线，但通信拓扑是 **静态** 的：

- 启动时定死 `ATTN_SIZE / FFN_SIZE / EP`；运行时无法改 expert→FFN-rank 映射
- 单机内 HCCS 验证 OK，但跨机 RoCE 路径未抽象出来
- 没有控制面，FFN 间负载不均只能靠静态切分硬扛
- 不支持 worker 动态加入/退出

为支持双机 32×910C → 未来 DeepSeek-V3 多机 EP，需要一个 **Coordinator 控制面 +
DeepEP-Ascend 数据面** 的新子系统。本子系统位于 `src/coordinator_arch/`，
**不替换** 旧 prefill-DBO/decode-DBO，二者作为性能对照保留。

当前进展补充：

- `src.main` 已新增 `--routing-backend coordinator --coord-addr ...`，可在 **Host1 单机 1A1F**
  上验证“真实 Qwen3 decode/prefill 路径 + coordinator 控制面”的最小 smoke
- 这一桥接目前只覆盖 **注册 + 一次性拉表**；真实多 FFN / EP / 动态路由更新仍待后续接入

## 2. 目标拓扑

### 2.1 物理拓扑

```
┌────────────────────────── Host1 (1.95.114.229) ──────────────────────────┐
│  16 × 910C  (ATTN_DP rank 0..15)        Coordinator (gRPC, :50051)      │
│    │                                          ▲                          │
│    └─ HCCS intra-host all-reduce ─┐           │ control plane (gRPC)     │
└───────────────────────────────────┼───────────┼──────────────────────────┘
                                    │ RoCE      │
                                    │ A↔F       │
                                    ▼           ▼
┌────────────────────────── Host2 (192.168.0.192) ─────────────────────────┐
│  16 × 910C  (FFN_EP rank 16..31)                                         │
│    └─ HCCS intra-host all-reduce                                         │
└──────────────────────────────────────────────────────────────────────────┘
```

### 2.2 切分理由

| 维度 | 选择 | 说明 |
|---|---|---|
| ATTN 并行 | DP=16 (Host1) | Qwen3-30B dense ≈ 8 GB / rank，64 GB HBM 足够 |
| FFN 并行 | EP=16 (Host2) | 128 experts ÷ 16 = 8 expert/rank ≈ 16 GB |
| A↔F | RoCE | 跨机，DeepEP-Ascend low_latency 模式 |
| ATTN 内部 | HCCS | 同机 all-reduce |
| FFN 内部 | HCCS | 同机 expert 间 all-to-all |

清晰的机器边界让 RoCE 只承担 A↔F 一条流，便于带宽/时延建模。
未来 DeepSeek-V3 (256 experts) → EP=64+ 仅需加节点，本架构不变。

## 3. 组件架构

### 3.1 三个组件

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Coordinator (gRPC server)                    │
│   ┌──────────────┐  ┌──────────────────┐  ┌──────────────────────┐  │
│   │ WorkerRegistry│  │  RoutingTable    │  │  LoadAwareRouter     │  │
│   │  rank→info    │  │  version monotonic│  │  metrics→expert→rank │  │
│   └──────────────┘  └────────┬─────────┘  └──────────▲───────────┘  │
└─────────────────────────────┬┼─────────────────────────┼─────────────┘
                              ││ stream RoutingTable     │ UpdateMetrics
              ┌───────────────┘└──────┐                  │
              ▼                       ▼                  │
   ┌────────────────────┐   ┌──────────────────────┐    │
   │ AttentionWorker × N│   │   FFNWorker × M       │────┘
   │  - attn forward    │   │  - FFNQueue           │
   │  - ContinuousBatcher  │  - grouped MoE        │
   │  - MoECommunicator │◀──┤  - MoECommunicator   │
   │      .dispatch()   │   │  .combine()-side     │
   └────────────────────┘   └──────────────────────┘
              ▲                       ▲
              └───── DeepEP-Ascend data plane (RoCE/HCCS) ─────┘
```

### 3.2 控制面 vs 数据面

| 平面 | 协议 | 频率 | 内容 |
|---|---|---|---|
| 控制 | gRPC over TCP | 0.1-10 Hz | 注册、metrics、路由表更新 |
| 数据 | DeepEP-Ascend (RoCE+HCCS) | per-step (≥100 Hz) | token hidden states + topk |

控制面用 gRPC 是因为低频、需要可靠 RPC + 流式推送，对延迟不敏感。
数据面用 DeepEP-Ascend 是因为高频、单次小（low_latency 模式下 batch ≤ 64）、
对端到端 RTT 极敏感（目标 <300 μs）。

### 3.3 启动时序

```
Coord                  Attn rank i              FFN rank j
  │                         │                       │
  ├─ bind :50051            │                       │
  │                         ├─ init HCCL/EP group   │
  │                         │                       ├─ init HCCL/EP group
  │◀─── RegisterWorker(attn,i) ──┤                  │
  │─── RegisterAck ──────────────▶                  │
  │                                  ◀── RegisterWorker(ffn,j) ──┤
  │                                  ─── RegisterAck ────────────▶
  │ (wait until N_attn + M_ffn 都注册)                            │
  │── SubscribeRoutingTable ──────────▶                            │
  │── SubscribeRoutingTable ─────────────────────────────────────▶│
  │── stream RoutingTable(v=1) ──────▶── stream RoutingTable(v=1)▶│
  │                         │                       │
  │                         │  build MoECommunicator using table │
  │                         │                       │
  │                         ├──── decode loop ──────│              │
```

### 3.4 单步时序（decode）

```
Attn i                                          FFN j
  │ forward_attention(hidden_in)                  │
  │ ► topk_idx, topk_w                            │
  │ batcher.split(hidden, idx, w, expert→rank)   │
  │   → micro_batches                             │
  │                                               │
  │ for mb in mbs:                                │
  │   h = comm.dispatch(mb)  ──── RoCE ──────────▶│
  │     (returns handle, non-blocking)            │ FFNQueue.put
  │                                               │
  │   # 下一 MB 的 attention 可以在这里并行         │
  │   forward_attention(next_batch) ...           │ batch = FFNQueue.pop
  │                                               │ grouped_moe(batch) ► ffn_out
  │ for h in handles:                             │
  │   out = comm.combine(ffn_out, h) ◀── RoCE ───│ comm.combine_send(ffn_out, h)
  │                                               │
  │ outputs.append(out)                           │
  │ merged = batcher.merge(outputs, mbs)          │
```

## 4. gRPC 协议草案

文件：`src/coordinator_arch/proto/coordinator.proto`（Phase 1 任务）

```proto
syntax = "proto3";
package coordinator;

message WorkerInfo {
  string role = 1;            // "attn" | "ffn"
  int32 rank = 2;             // global rank within role group
  string host = 3;            // hostname or IP
  int32 device_id = 4;        // local NPU index
  int32 world_size = 5;       // total ranks in this role
  repeated int32 local_experts = 6;   // FFN only; ignored for attn
  string deepep_endpoint = 7; // hostname:port, for control-side reference
}

message WorkerMetrics {
  string role = 1;
  int32 rank = 2;
  double queue_len_avg = 3;       // FFN only
  double dispatch_rate_tps = 4;   // tokens/sec recent
  double cache_miss_rate = 5;     // optional
  int64 timestamp_us = 6;
}

message RoutingTable {
  int64 version = 1;              // monotonic
  repeated int32 expert_to_rank = 2;  // len = num_experts
  string mode = 3;                // "normal" | "low_latency"
  int64 valid_from_us = 4;        // wall-clock; ranks SHOULD use only after this
}

message RegisterAck {
  int64 initial_table_version = 1;
  string assigned_group = 2;      // for future multi-group support
}

message Empty {}
message Ack { bool ok = 1; string msg = 2; }

service Coordinator {
  rpc RegisterWorker(WorkerInfo) returns (RegisterAck);
  rpc GetRoutingTable(Empty) returns (RoutingTable);
  rpc SubscribeRoutingTable(Empty) returns (stream RoutingTable);
  rpc UpdateMetrics(WorkerMetrics) returns (Ack);
}
```

**版本号语义**：
- `RoutingTable.version` 单调递增；ranks 收到新表后原子替换缓存
- attn worker 在 `dispatch` 中带上自己持有的 version；如果 FFN 端发现 mismatch，
  当前实现 **不拒收**（避免抖动），仅打 metric 告警，下个 step 由 attn 收到新表后自动对齐
- `valid_from_us` 给一个软的"生效时间"，避免 attn/ffn 同时换表造成跨表 dispatch 半丢

**容错**：
- coord 挂掉：worker 端 cache 旧表继续工作；订阅断开后定期重连
- worker 挂掉：coord sweep 心跳超时（5s）→ 重算 routing，version+1 推送

## 5. RoutingTable 数据结构

```python
{
  "version": 17,
  "expert_to_rank": [0, 0, 0, 0, 0, 0, 0, 0,
                     1, 1, 1, 1, 1, 1, 1, 1,
                     ...,
                     15,15,15,15,15,15,15,15],  # len=128, values in [0, ep_world)
  "mode": "low_latency",
  "valid_from_us": 1716172800123456
}
```

attn worker 每个 step：
```python
expert_to_rank = self.routing_table_tensor   # LongTensor[num_experts], on device
# topk_idx: [N, K]  experts each token is routed to
dest_rank = expert_to_rank[topk_idx]         # [N, K]  destination FFN ranks
```

## 6. 负载感知路由算法

Phase 1 任务 `p1-router-load-aware` 在 `src/coordinator_arch/router.py`。
算法 v1（贪心装箱）：

```python
def rebalance(metrics: List[WorkerMetrics],
              num_experts: int,
              ep_world: int,
              prev_table: List[int]) -> Optional[List[int]]:
    """Returns new expert_to_rank or None if no rebalance needed."""
    # 1. 估算每个 rank 的 cost = queue_len_avg + α * dispatch_rate_tps
    cost = [0.0] * ep_world
    for m in metrics:
        if m.role == 'ffn':
            cost[m.rank - ATTN_WORLD] = m.queue_len_avg + 0.01 * m.dispatch_rate_tps
    # 2. 检查不均衡度
    if (max(cost) - min(cost)) / max(min(cost), 1e-3) < 0.10:
        return None   # within 10% → 不动
    # 3. 估算每 expert 的"负载"= sum of dispatch_rate over recent windows (per-expert metrics)
    expert_load = self._estimate_per_expert_load()
    # 4. 贪心：按 expert_load 降序，放进当前 load 最小的 rank
    new_table = [-1] * num_experts
    bins = [0.0] * ep_world
    for e in sorted(range(num_experts), key=lambda i: -expert_load[i]):
        r = min(range(ep_world), key=lambda i: bins[i])
        new_table[e] = r
        bins[r] += expert_load[e]
    # 5. 平滑约束：每次 rebalance 最多迁移 K=10 个 expert，避免抖动
    return _project_with_max_moves(prev_table, new_table, max_moves=10)
```

**节流**：coord 主循环每 100ms 触发一次 rebalance 计算；变更才推送。

## 7. ContinuousBatcher / FFNQueue

详见 `src/coordinator_arch/batching/`。

**ATTN 端 (`continuous_batcher.py`)**：
- `MicroBatch(hidden, topk_indices, topk_weights, dest_ranks, token_indices)`
- `split(hidden, topk_idx, topk_w, expert_to_rank, max_tokens)` → `List[MicroBatch]`
  - 按 dest_rank 把 tokens 重排，再按 `max_tokens` 切片
- `merge(outputs, mbs, total_tokens)` → 原始 token 顺序的 `hidden_out`

**FFN 端 (`ffn_queue.py`)**：
- 线程安全 queue + 双触发：
  - `max_batch` (默认 64)：累计到上限立即 pop
  - `max_wait_ms` (默认 5)：超时强制 pop（避免低 QPS 饥饿）
- `pop_batch() -> FFNBatch | None`（非阻塞，返回 None 时 worker 主循环可做别的）

## 8. Pipeline 与计算-通信重叠

DeepEP-Ascend `buffer.dispatch()` 返回的张量在底层 stream 上异步可见。
我们的 `MoECommunicator.dispatch()` 返回的 handle dict 包含：
- `recv_hidden`：本 rank 收到的 tokens（异步）
- `expert_token_nums`：每 expert 收到的 token 数
- 一个 NPU event（Phase 2 实装），attn 端可在下个 MB attention 之前 `event.wait()`

**典型重叠 timeline**：
```
ATTN stream:  [attn_0]──[dispatch_0]──[attn_1]──[dispatch_1]──[combine_0]──[combine_1]
FFN stream:                   ▼                       ▼
                          [ffn_0]──────[ffn_1]
                                        ▲
                                    被 attn_1 隐藏
```

目标：FFN 计算期间 ATTN 等待时间 < 10%（性能目标见 §11）。

## 9. Mode Switching

DeepEP-Ascend `Buffer` 在 `low_latency_mode=True/False` 之间切换需要重新分配
buffer。`MoECommunicator.set_mode(mode)`：
1. `dist.barrier(ep_group)` — 等所有 rank 完成当前在飞 dispatch/combine
2. 析构旧 `Buffer`，分配新 `Buffer(group, num_nvl_bytes, low_latency_mode=new)`
3. `dist.barrier(ep_group)` — 同步完成
4. 更新 `self.mode`

**使用规则**：
- Prefill 阶段：`mode=normal`（大 batch 单次传输优先）
- Decode 阶段：`mode=low_latency`（小 batch 低延迟优先）
- 切换由 coord 通过 routing_table.mode 字段触发

**fallback**：若 DeepEP 不可用（import 失败），factory 自动用
`fallback_a2a.FallbackAllToAll`（torch.distributed `all_to_all_single`），性能差但
保证管线可跑通。

## 10. 失效与弹性

| 事件 | 检测 | 处理 |
|---|---|---|
| Worker 挂掉 | coord metrics 超时 (5s) | 从 registry 移除 → router 重算 → version+1 推送 |
| Worker 加入 | RegisterWorker RPC | router 重算 → version+1 推送 |
| Coord 挂掉 | worker 订阅断开 | 缓存上次 routing_table 继续工作；后台每 1s 重连 |
| 路由表 stale | dispatch/combine 时 version mismatch | 不拒收，metric 告警；下次 SubscribeRoutingTable push 后对齐 |
| HCCL/RoCE 抖动 | dispatch timeout | 本 step skip + log；连续 3 次触发 worker self-restart（out of scope Phase α）|

**out of scope**：Coordinator HA（单点）、跨数据中心、网络分区。

## 11. 性能目标

| 指标 | 目标 | 测量方式 |
|---|---|---|
| 端到端 P99 latency | < 200 ms | decode 1 batch 端到端墙钟 |
| 单 RTT 通信延迟（low_latency） | < 300 μs | bench_comm_transfer.py 双 rank |
| ATTN idle 比例 | < 10% | run_step 内部 timer：等 FFN combine 的时间 / 总时间 |
| FFN 线性扩展 | 接近线性 | 8/16/32 卡下 tokens/sec |

## 12. 与现有 prefill-DBO / decode-DBO 的关系

| 模块 | 旧 (`src/model/`) | 新 (`src/coordinator_arch/`) |
|---|---|---|
| 拓扑 | 静态 ATTN_SIZE/FFN_SIZE | 动态注册，coord 下发 |
| 通信 | 直接 HCCL all_to_all_single | DeepEP-Ascend (+fallback) |
| 路由 | 编译期固定 expert→rank | 运行时 routing_table |
| Pipeline | dual-batch micro stream | dispatch/combine event 重叠 |
| 测试 | results_npu/ 已有矩阵 | Phase 2-4 待补 |

旧路径保留作为：
- 性能对照（DBO 已优化过的 baseline）
- 单机 16 卡稳定运行的兜底（DeepEP 出问题时仍可跑）

## 13. Open Issues

- **DeepEP-Ascend over RoCE 未验证**：当前只在 Host1 单机 HCCS 跑过 kernel test；
  跨机 RoCE 在 Phase 3 (`p3-deepep-roce-validation`) 第一次实验
- **Qwen3 真权重接入延后**：Phase 1 worker 是 identity 占位；
  Phase 3 (`p3-end-to-end-decode`) 接入真 attention + grouped MoE
- **gRPC 鉴权/加密**：当前裸 TCP，假设内网；生产环境需要 TLS + mTLS
- **Coordinator HA**：单点失效后 worker 用旧表续命，但长期需 raft/etcd
- **Per-expert 负载估算**：算法 v1 只用 per-rank metrics，未来需要 FFN 上报
  per-expert 队列长度，更精细的 rebalance

## 14. 参考

- `src/coordinator_arch/comm/moe_communicator.py` — DeepEP 包装
- `src/coordinator_arch/comm/fallback_a2a.py` — torch.distributed 回退
- `src/coordinator_arch/comm/factory.py` — `CommunicatorProtocol` + `build_communicator`
- `src/coordinator_arch/batching/{continuous_batcher,ffn_queue}.py`
- `src/coordinator_arch/workers/{attention_worker,ffn_worker}.py`
- `src/coordinator_arch/coordinator_client.py` — gRPC client（Phase α 为 stub）
- `doc/deepep_ascend_install_report.md` — DeepEP-Ascend 安装与 API
- `results_npu/cross_host_bench/README.md` — 跨机 HCCL 验证
