# Cross-host 1A7F Coordinator 架构说明

本文说明当前已经验证的 coordinator-based cross-host 1A7F / EP7 架构。它描述的是 **真实 `src.main` decode-dbo 路径**，不是早期 skeleton worker，也不是 DeepEP 实验路径。

## 1. 当前状态

- 拓扑：Host1 attention rank0 + Host2 FFN ranks 1-7，world size 8。
- 模型：Qwen3-30B-A3B。
- 路由：`--routing-backend coordinator --routing-update-mode oneshot`。
- EP backend：`broadcast_reduce_overlap`。
- 已完成 full matrix：batch `2,4,8,16,32,64,128,256` × seq `128,256,512` × tokens `20`，24/24 OK。
- 主结果目录：`results_npu/coordinator_arch/crosshost_1a7f_decode_dbo/`。

核心结论：当前 cross-host 1A7F 已经不再卡在 Host2 HCCL `EJ0003` 或小配置 TBE/JIT；full matrix 的最终完成依赖稳定的实验 orchestrator、fresh ports、残留进程清理和 Host2 磁盘空间监控。

## 2. 进程与 rank 布局

| 主机 | 容器 | 进程 | rank | device | 角色 |
|---|---|---|---:|---:|---|
| Host1 `192.168.0.125` | `afd-npu-test` | `coordinator_server` | - | - | gRPC 控制面，提供 routing table |
| Host1 `192.168.0.125` | `afd-npu-test` | `python -m src.main` | 0 | NPU0 | attention rank |
| Host2 `192.168.0.192` | `afd-npu-test-h2` | `python -m src.main` | 1 | NPU0 | FFN coordinator / EP rank0 |
| Host2 `192.168.0.192` | `afd-npu-test-h2` | `python -m src.main` | 2-7 | NPU1-6 | FFN expert ranks |

当前验证的是 1A7F，因此只有一个 attention rank。FFN 侧使用 7 个 rank 承载 128 个 experts，expert ownership 由 coordinator 的 `expert_to_rank` 表显式下发。

## 3. HCCL group

所有 rank 先进入同一个 default process group：

```text
world ranks = [0,1,2,3,4,5,6,7]
rank0      = attention
rank1-7    = FFN EP ranks
```

当 `ffn_ep_size=7` 时，`src/distributed/__init__.py` 按固定顺序创建 3 个 FFN-only groups：

```text
ffn_ep_group          ranks=[1,2,3,4,5,6,7]
ffn_ep_dispatch_group ranks=[1,2,3,4,5,6,7]
ffn_ep_reduce_group   ranks=[1,2,3,4,5,6,7]
```

这些 groups 必须由所有 world ranks 按相同顺序创建。此前 Host2 `EJ0003` 的最小复现实验证明：清理残留 `src.main` 后，default group 和这三个 EP groups 都能通过，EP group 创建本身不是当前 blocker。

## 4. Coordinator 控制面

Coordinator 是一个 gRPC server，由 Host1 side launcher 启动：

```bash
python3 -m src.coordinator_arch.coordinator_server \
  --bind 0.0.0.0:<coord_port> \
  --num-experts 128 \
  --attn-world 1 \
  --ffn-world 7 \
  --mode low_latency
```

它的职责：

1. 生成并提供 `RoutingTable(version=1, expert_to_rank=[...])`。
2. 接收 attention / FFN worker registration。
3. 支持 `GetRoutingTable`，真实路径默认 one-shot 拉表。
4. 为未来 load-aware routing / poll 更新保留控制面入口。

当前真实 NPU decode path 不使用后台 streaming subscribe；`oneshot` 是默认稳定模式。`poll` 代码路径存在，但如果新表会改变 EP expert ownership，真实路径会拒绝在线迁移并保留旧表。

## 5. 启动流程

每个配置使用一组 fresh ports：

| 端口 | 用途 |
|---|---|
| `MASTER_PORT` | HCCL default process group rendezvous |
| Host1 `HCCL_IF_BASE_PORT` | Host1 HCCL port base |
| Host2 `HCCL_IF_BASE_PORT` | Host2 HCCL port base |
| `coord_port` | Coordinator gRPC bind / client addr |

单配置启动顺序：

1. Host1 创建输出目录。
2. Host1 启动 coordinator server。
3. Host1 启动 attention rank0，等待 distributed world。
4. Host2 创建输出目录。
5. Host2 启动 FFN ranks 1-7。
6. 所有 ranks 完成 HCCL init 与 EP group creation。
7. 所有 ranks 通过 coordinator one-shot 获取 routing table。
8. 权重加载、decode-dbo、timing JSON 写出。
9. side launcher 检查并清理本轮 coordinator/rank 进程。

`scripts/run_crosshost_coord_1a7f_smoke.sh` 是单侧 launcher；`scripts/run_crosshost_coord_1a7f_matrix.py` 是本地 orchestrator，负责把矩阵配置逐个变成两端 side launcher 调用。

## 6. Orchestrator 职责

`scripts/run_crosshost_coord_1a7f_matrix.py` 不参与模型计算，也不是模型里的 coordinator。它是实验控制脚本，职责如下：

- 展开 batch/seq/tokens 矩阵。
- 分配 fresh `MASTER_PORT` / HCCL base ports / coordinator port。
- 在 Host1/Host2 容器内写入临时 side script，并用 `docker exec -d` detached 启动，避免本地 SSH 会话绑定长时间模型执行。
- 每个配置启动后检查 rank log / side marker，避免“远端 rank 没启动但本地还在等”的假超时。
- 每个配置结束后读取 side rc marker、Host1 `decode_tpot`、Host2 free space。
- 遇到 OOM/HCCL/启动失败时，停止同 seq 更大 batch，避免盲跑污染环境。
- 每轮记录 summary CSV。

本轮修复过两个 orchestrator 相关问题：

1. Host2 输出目录必须先 `mkdir -p`，否则 `host2_side.out` 重定向失败会导致 ranks 根本不启动。
2. side logs 必须每轮截断，不能追加复用旧 `host*_rc=0` marker，否则会把旧 run 误判为当前 run 完成。

## 7. 结果与 metrics 流

`src.main` 当前仍把 decode timing JSON 写入远端 worktree 的 `results/prefill_dbo/`。矩阵完成后，本地收集并整理到：

```text
results_npu/coordinator_arch/crosshost_1a7f_decode_dbo/
  matrix_summary_final.csv
  decode-dbo/
    timing_attention_xhost_coord_matrix_b*_s*_t20.json
    timing_ffn_coordinator_xhost_coord_matrix_b*_s*_t20.json
    timing_ffn_expert_r*_xhost_coord_matrix_b*_s*_t20.json
  decode_mfu_summary.csv
  ep_bandwidth_summary.csv
  crosshost_vs_singlehost_summary.csv
  crosshost_vs_singlehost_comparison.md
```

指标来源：

| 指标 | 来源 |
|---|---|
| TPOT / throughput | Host1 attention timing JSON / `matrix_summary_final.csv` |
| MFU estimate | `scripts/report_decode_mfu.py` + Qwen3 config |
| EP dispatch/reduce bandwidth | `scripts/report_ep_bandwidth.py` 解析 Host2 FFN timing JSON |
| cross-host vs single-host | `crosshost_vs_singlehost_summary.csv` |
| static baseline | `results_npu/coordinator_arch/singlehost_ep7/coord_vs_static_summary.csv` |

## 8. 与 static EP7 的区别

| 维度 | static EP7 | coordinator cross-host 1A7F |
|---|---|---|
| expert ownership | 本地静态 policy，例如 round-robin | coordinator `RoutingTable.expert_to_rank` 显式下发 |
| 控制面 | 无独立 server | Host1 gRPC coordinator |
| 更新模式 | 启动时固定 | 当前真实路径使用 one-shot；poll 仅 safe-point 检查 |
| 部署 | 单机 EP7 或静态多机 | Host1 attention + Host2 FFN ranks |
| 当前验证 | 单机 full matrix 与 static baseline | cross-host full matrix 24/24 OK |

当前 coordinator one-shot 的性能结论见：

```text
results_npu/coordinator_arch/crosshost_1a7f_decode_dbo/crosshost_vs_singlehost_comparison.md
```

本轮 mean cross-host / single-host coordinator TPOT ratio 为 `1.018x`，说明跨机 1A7F 在该矩阵上整体接近单机 coordinator EP7，但这仍是 Qwen3-30B-A3B、1A7F、oneshot、fallback EP backend 下的结论。

## 9. 失败分层

| 层级 | 典型现象 | 判断方式 | 处理 |
|---|---|---|---|
| Orchestration failure | 没有 rank log / 没有 `src.main` 进程 | side out 无 rank PID | 修 launcher/orchestrator，不等模型 timeout |
| Coordinator bind failure | `Failed to bind to address 0.0.0.0:<port>` | coordinator log | 换 fresh coord port；launcher fail-fast |
| HCCL failure | `EJ0003` / `EI0006` / init group timeout | rank logs | 先查残留 `src.main`，必要时跑 `repro_hccl_ep7_ej0003.py` |
| TBE/JIT stall | 卡在 warmup / kernel_meta 异常增长 | rank logs + cache 目录 | 先确认 HCCL 已通过，再排查编译缓存 |
| Model timeout | rank 已启动且有模型日志，但超时无 timing | side rc / rank log | 可用小模型做 control，但性能结论仍回到 Qwen3 |
| Disk pressure | `/workspace` 可用空间持续下降 | orchestrator 每轮 `df` | 暂停矩阵，清理/迁移输出 |
| OOM | rank log 出现 memory/OOM | rank log | 停止同 seq 更大 batch |

## 10. 当前边界

- 当前只验证 coordinator `oneshot`；真实路径的动态 routing 迁移还未完成。
- Poll 模式不会在线迁移 EP expert ownership。
- DeepEP-Ascend 仍 deferred；当前数据面是 `broadcast_reduce_overlap` fallback EP backend。
- 结果目录中的 MFU 是模型配置估算值，不是硬件 counter。
- Host2 `/workspace` 仍约 97% 使用；full matrix 已完成，但后续更大实验仍应继续做空间监控。
