# EP16 通信 msprof 与带宽分析

## 结论摘要

`all_to_all_single` 已正确接入真实 `EPFFNLayer` decode/overlap 路径，但当前实现不应默认启用。代码路径已经从 CLI、`EPFFNLayer`、`FallbackMoECommunicator` 到 `DecodeDBOScheduler` 串通；NPU 跨机 smoke 与 b16/b32/b256 代表 timing 均能完成。慢的主要原因不是链路带宽打满，而是 `all_to_all_single` 路径引入了更多 HCCL collective、变长 `hcom_alltoallv`、metadata/count exchange 以及同步等待。

当前有效带宽下界远低于 910C/HCCL 链路应有上限：b32/s256 下 `all_to_all_single` dispatch 逻辑 payload 约 0.469 MiB，但平均耗时 5.119 ms，只有约 0.096 GB/s；即使 b256/s1024 也只有约 0.979 GB/s。瓶颈更像固定延迟、collective 数量和变长 all-to-all 调度开销，而不是物理带宽。

## 接入正确性审计

### 真实接入口

- CLI 已允许 `--ffn-ep-backend all_to_all_single`，说明它是显式实验 backend，不会覆盖默认 broadcast/reduce。
- `DecodeDBOScheduler._use_ep_overlap()` 已把 `all_to_all_single` 纳入 EP overlap backend。
- `EPFFNLayer` 在 backend 为 `all_to_all_single` 时构造 `FallbackMoECommunicator`，使用当前静态 expert ownership 表，并设置 `metadata_src_rank=ctx.ffn_coordinator_rank`。
- `EPFFNLayer.dispatch_async()` / `finish_dispatch()` / `compute_local()` / `reduce_async()` / `finish_reduce()` 分别映射到 communicator dispatch、assignment-level local experts、combine。

### 关键语义

- 非 coordinator FFN ranks 使用 empty source tensors，但仍参与所有 collectives，满足 HCCL 所有 rank 同序进入 collective 的要求。
- `FallbackMoECommunicator` 先交换 per-rank counts，再用 `dist.all_to_all_single` 发送 hidden assignment，combine 阶段反向 `all_to_all_single` 收回 expert output。
- 真实 EPFFN 路径为了减少 payload，设置 `dispatch_weights=False` 和 `dispatch_experts=False`：routing weight 不随 dispatch 发送，而是在 combine 时统一使用原始 top-k weights；expert id 通过 coordinator broadcast 的 expert-count metadata 在接收端重建。
- `ShardedExperts.forward_dispatched()` 对收到的 assignment 执行未加权 local expert compute，并显式校验 expert id 必须属于本 rank；非法 ownership 会抛错，不会 silent ignore。

### 测试覆盖与缺口

已完成：

- Host1 容器直接运行 `tests/coordinator_arch/test_fallback_a2a.py` 通过。
- Host1 容器手动调用 `tests/test_ep_moe_reference.py` 相关 reference 函数通过。
- 跨机 1A16F/EP16 `all_to_all_single` smoke 与 b16/b32/b256 代表配置完成。

仍需补强：

- 需要增加一个真实 `EPFFNLayer` reference test：同一 hidden/top-k/routing 下，对比 broadcast/reduce dense partial 与 all-to-all assignment combine 的 coordinator 输出。
- 当前 all-to-all timing 中 `ep_dispatch_bytes` 仍沿用 source hidden/router/weight 口径，不能直接代表 all-to-all 物理发送量；报告中的带宽表使用公式计算逻辑 payload 下界。

## msprof 采集过程

### 首次全量采集失败

第一次尝试对 b32/s256 的 17 个 ranks 全量开启 msprof：

```bash
python scripts/run_crosshost_static_ep_matrix.py \
  --host1-workdir /workspace/afd_demo_all2all \
  --host2-workdir /workspace/afd_demo_all2all \
  --out-root crosshost_static_ep16_msprof_comm \
  --ep-sizes 16 \
  --backends broadcast_reduce_overlap,all_to_all_single \
  --modes decode-dbo \
  --configs 32:256 \
  --tokens 20 \
  --msprof --msprof-analyze
```

该 run 的模型执行已经进入/完成，但 Host2 `/workspace` 被 msprof raw export 写满，timing/profile 保存报 `OSError: [Errno 28] No space left on device`。因此后续改为支持选 rank 和 storage limit，并删除本任务产生的失败 profile 目录。

### 脚本新增能力

`scripts/run_crosshost_static_ep_smoke.sh` 与 `scripts/run_crosshost_static_ep_matrix.py` 增加：

- `--msprof-ranks all|0,1,16`：只 profile 指定 global ranks。
- `--msprof-storage-limit-mb N`：传给 `msprof --storage-limit=NMB`。
- 保持默认不启用 msprof；只有显式传 `--msprof` 才包装 rank 命令。

### 成功采集的 profile

由于 Host2 磁盘只剩约数百 MB 可写，最终成功采集的是 b32/s256/t3 的降级 profile：

```bash
python scripts/run_crosshost_static_ep_matrix.py \
  --host1-workdir /workspace/afd_demo_all2all \
  --host2-workdir /workspace/afd_demo_all2all \
  --out-root crosshost_static_ep16_msprof_comm_debug \
  --summary-csv crosshost_static_ep16_msprof_comm_debug/matrix_summary.csv \
  --ep-sizes 16 \
  --backends broadcast_reduce_overlap,all_to_all_single \
  --modes decode-dbo \
  --configs 32:256 \
  --tokens 3 \
  --debug-max-layers 2 \
  --base-master-port 38000 \
  --base-h1-hccl-port 45400 \
  --base-h2-hccl-port 46400 \
  --attn-kernel npu-official \
  --attn-precopy-layer-inputs \
  --attn-fused-rmsnorm \
  --attn-fused-rope \
  --no-resource-monitor \
  --msprof \
  --msprof-ranks 0,1,16 \
  --msprof-storage-limit-mb 200 \
  --no-resume \
  --timeout-sec 2400 \
  --poll-sec 30
```

结果目录：

- `crosshost_static_ep16_msprof_comm_debug/decode-dbo_ep16_broadcast_reduce_overlap_b32_s256_t3/`
- `crosshost_static_ep16_msprof_comm_debug/decode-dbo_ep16_all_to_all_single_b32_s256_t3/`
- 小型汇总 CSV：`crosshost_static_ep16_msprof_comm_debug/msprof_summaries/`

说明：虽然传了 `AFD_DEBUG_MAX_LAYERS=2`，当前 timing JSON 仍记录到 48 层事件，因此该 profile 可用于观察全层通信模式；但 tokens=3/profile overhead 使 TPOT 不可与 tokens=20 性能表直接比较。

## timing 对比

以下均为 layer 1..47、MB event 平均值，单位 ms。

| Backend | Shape | TPOT | dispatch | local experts | combine/reduce | dispatch wait | reduce wait |
|---|---:|---:|---:|---:|---:|---:|---:|
| broadcast_reduce_overlap | b16/s128/t20 | 315.294 | 1.264 | 0.904 | 0.721 | 0.005 | 0.010 |
| broadcast_reduce_overlap | b32/s256/t20 | 384.361 | 1.943 | 1.467 | 0.917 | 0.007 | 0.011 |
| broadcast_reduce_overlap | b256/s1024/t20 | 1137.204 | 1.916 | 1.440 | 1.017 | 0.005 | 0.010 |
| all_to_all_single | b16/s128/t20 | 596.726 | 4.964 | 0.935 | 1.145 | 0.195 | 0.146 |
| all_to_all_single | b32/s256/t20 | 664.915 | 5.119 | 1.075 | 1.197 | 0.199 | 0.148 |
| all_to_all_single_no_weight | b256/s1024/t20 | 1325.661 | 4.015 | 1.330 | 1.190 | 0.013 | 0.144 |
| msprof broadcast_reduce_overlap | b32/s256/t3 | 476.091 | 1.456 | 1.287 | 1.066 | 0.006 | 0.012 |
| msprof all_to_all_single | b32/s256/t3 | 733.459 | 6.481 | 0.898 | 1.174 | 0.276 | 0.153 |

关键观察：

1. `all_to_all_single` 的 local expert compute 不一定更慢，b32/s256 甚至低于 broadcast/reduce；端到端慢主要来自 dispatch。
2. b32/s256/t20 中，dispatch 从 `1.943ms` 增至 `5.119ms`，是 TPOT 从 `384.361ms` 退化到 `664.915ms` 的核心原因。
3. b256/s1024 中无 weight dispatch 的 all-to-all dispatch 仍为 `4.015ms`，明显高于 broadcast/reduce 的 `1.916ms`；说明仅减少 weight/expert-id payload 不足以解决问题。

## msprof HCCL 结果

`msprof` consolidated SQLite 中的 `COMMUNICATION_OP` 表显示：

| Config | Host/rank | 主要 HCCL op | 次数 | total_ms | avg_ms | 说明 |
|---|---:|---|---:|---:|---:|---|
| all_to_all_single b32/s256/t3 | Host2 rank1 | `hcom_alltoallv_` | 480 | 2797.623 | 5.828 | all-to-all dispatch/combine 的主要新增开销 |
| all_to_all_single b32/s256/t3 | Host2 rank1 | `hcom_broadcast_` | 241 | 993.829 | 4.124 | expert-count metadata broadcast；含一个明显慢的启动/异常值 |
| all_to_all_single b32/s256/t3 | Host2 rank1 | `hcom_alltoall_` | 240 | 49.715 | 0.207 | count exchange |
| broadcast_reduce_overlap b32/s256/t3 | Host2 rank1 | `hcom_reduce_` | 240 | 1836.210 | 7.651 | dense reduce/combine |
| broadcast_reduce_overlap b32/s256/t3 | Host2 rank1 | `hcom_broadcast_` | 241 | 50.656 | 0.210 | full hidden/router broadcast |
| broadcast_reduce_overlap b32/s256/t3 | Host2 rank16 | `hcom_reduce_` | 240 | 1910.122 | 7.959 | expert rank reduce 视角 |
| broadcast_reduce_overlap b32/s256/t3 | Host2 rank16 | `hcom_broadcast_` | 241 | 660.490 | 2.741 | 非 coordinator rank broadcast 视角，存在跨 rank 差异 |

解释：

- `all_to_all_single` 真实落到 HCCL 的 `hcom_alltoallv_`，不是普通 broadcast/reduce 的简单替换；它还增加了 `hcom_alltoall_` count exchange 和 metadata `hcom_broadcast_`。
- `hcom_alltoallv_` 在 b32/s256/t3 的 Host2 rank1 上总计约 `2.798s`，平均 `5.828ms`，与 timing JSON 中 `ep_dispatch` 明显偏高一致。
- msprof 中 Host1 rank0 的 `hcom_receive_`/`hcom_send_` 多来自 A2F/F2A P2P，不代表 EP dispatch 本身；EP 通信分析主要看 Host2 FFN ranks。
- msprof 表包含整个进程生命周期中的通信（包括建组、barrier、profile overhead、模型前后同步），因此单个 max 值不可直接等同某一 layer-MB 的 timing；更可靠的是 op 类型、次数和总趋势。

## 通信量与有效带宽下界

逻辑 payload 估算使用 Qwen3 当前形状：

- hidden size `H=2048`
- top-k `K=8`
- EP size `E=16`
- dtype bytes `2`
- MB2 下每个 event 的 token 数 `T=B/2`

公式：

- broadcast dispatch：`(T*H*2 + T*K*8 + T*K*2) * (E-1)`
- dense reduce：`T*H*2 * (E-1)`
- all-to-all dispatch/combine 下界：`T*K*H*2*(E-1)/E`

| Backend | Shape | Dispatch MiB | Dispatch ms | Dispatch GB/s | Combine MiB | Combine ms | Combine GB/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| broadcast_reduce_overlap | b16/s128 | 0.478 | 1.264 | 0.396 | 0.469 | 0.721 | 0.681 |
| broadcast_reduce_overlap | b32/s256 | 0.956 | 1.943 | 0.516 | 0.938 | 0.917 | 1.072 |
| broadcast_reduce_overlap | b256/s1024 | 7.646 | 1.916 | 4.186 | 7.500 | 1.017 | 7.731 |
| all_to_all_single | b16/s128 | 0.234 | 4.964 | 0.050 | 0.234 | 1.145 | 0.215 |
| all_to_all_single | b32/s256 | 0.469 | 5.119 | 0.096 | 0.469 | 1.197 | 0.411 |
| all_to_all_single_no_weight | b256/s1024 | 3.750 | 4.015 | 0.979 | 3.750 | 1.190 | 3.305 |

这些值均是逻辑 payload 下界，不包含 HCCL 协议头、notify、count/metadata、重复传输、算法内部中间流量和 profile overhead。因此它们不是链路物理带宽测量值；但它们足以说明当前离带宽上限很远。若链路带宽是主要瓶颈，batch/payload 增大时有效 GB/s 应快速接近平台上限；实际 all-to-all b32 仍只有 `0.096 GB/s`，b256 也只有 `0.979 GB/s`。

## 为什么 all-to-all 理论上更省流量但实际更慢

理论上，token-aware all-to-all 只发送 `(token, expert)` assignment，避免把完整 hidden 广播给所有 EP ranks，也避免 dense partial reduce。但当前实现有几个抵消项：

1. **collective 数量更多**：dispatch 阶段有 count exchange、metadata broadcast、hidden all-to-all；combine 阶段还有反向 all-to-all。
2. **变长 split 开销高**：HCCL profile 中主要 op 是 `hcom_alltoallv_`，变长 split 的调度成本在小/中 payload 下显著。
3. **metadata broadcast 有异常长尾**：b32 profile 中 `hcom_broadcast_` metadata 总时间接近 1s，含一个 959ms 的长尾。它可能包含首轮建链/profile/同步干扰，但说明 metadata 路径还不稳定。
4. **Python/tensor prep 与同步不可忽略**：`FallbackMoECommunicator` 需要 `dest_ranks`、`argsort`、`bincount`、`recv_counts.tolist()`、`repeat_interleave` 和 inverse permutation；这些在 timing 的 dispatch/combine 阶段内。
5. **当前 top-k=8 降低稀疏收益**：Qwen3 每 token 路由 8 个 experts，EP16 下 assignment 数是 token 数的 8 倍；all-to-all 不是“每 token 只发一次”，payload 下界并不总是小很多。

## 结论与下一步

1. `all_to_all_single` 接入正确，但当前 HCCL/PyTorch fallback 形态不适合作为默认 EPFFN backend。
2. 当前慢点不是物理带宽上限，而是 all-to-all fixed latency、变长 split、metadata/count exchange 和 tensor 重排。
3. 继续保留 `broadcast_reduce_overlap` 作为真实 decode EP 默认 baseline。
4. 下一步优化优先级：
   - 原型化 padded/equal-split all-to-all，减少 `alltoallv` 变长调度成本。
   - 合并/削减 metadata collective，避免每 layer/MB 额外 broadcast expert-count。
   - 给 `FallbackMoECommunicator` 增加更细 timing：count exchange、metadata broadcast、hidden all-to-all、expert local compute、combine all-to-all、restore/weight sum。
   - 在 Host2 空间充足或输出目录迁到更大磁盘后，补 b16/b256 的 rank1 msprof 原始采集。
   - 继续推进 CANN official MoE v2 / DeepEP，但只有真实 Qwen nonzero expert combine 稳定且端到端 TPOT 优于 broadcast/reduce 后才考虑切默认。

