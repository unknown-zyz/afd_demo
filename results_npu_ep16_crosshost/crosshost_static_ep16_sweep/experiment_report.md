# 跨机 EP16 同拓扑实验报告

## 结论摘要

本轮实验完成了跨机 `1A16F / EP16` 的同拓扑矩阵：Host1 rank0 运行 Attention，Host2 rank1..16 运行 FFN EP ranks。结果目录为 `crosshost_static_ep16_sweep/`。

主要结论：

- 请求范围内 `96/96` 行全部完成且状态为 `OK`：`batch=2..256`、`seq=128..1024`，模式包括 `serial`、`decode-dbo`、`decode-dbo-crosslayer`。
- 每个 `(batch, seq)` 都补齐了同拓扑 serial baseline，后续 speedup 只使用同拓扑 serial TPOT 作为分母。
- 当前 DBO 没有超过 serial。最佳同拓扑 speedup 是 `decode-dbo b256/s1024 = 0.916x`。
- 大 batch 下 Attention 与 FFN local compute 已经接近对齐，但端到端 TPOT 仍被 EP 通信与等待气泡抵消。
- 已接入实验性 `all_to_all_single` sparse dispatch/combine 后，EP16 功能 smoke 和 `b32/s256/t20` 均可跑通；但当前 HCCL all-to-all dispatch 开销更高，代表配置 TPOT 仍慢于 broadcast/reduce。
- 请求范围内没有 OOM；最大已验证非 OOM 点为 `batch=256, seq=1024`，三种模式全部通过。

## 实验配置

| 项目 | 配置 |
|---|---|
| 结果目录 | `crosshost_static_ep16_sweep/` |
| 拓扑 | Host1 rank0 Attention + Host2 rank1..16 FFN EP |
| World size | `17` |
| EP backend | `broadcast_reduce_overlap` |
| 模式 | `serial`, `decode-dbo`, `decode-dbo-crosslayer` |
| Batch | `2,4,8,16,32,64,128,256` |
| Seq | `128,256,512,1024` |
| Tokens | `20` |
| Micro-batches | `2` |
| Attention 优化 | `npu-official`, precopy, fused RMSNorm, fused RoPE |
| Serial baseline | 同一 `1A16F/EP16` 拓扑，`--no-dbo --generate` |

关键产物：

- `matrix_summary.csv`：原始运行状态、TPOT、报告、pipeline 图、资源日志路径。
- `ep16_speedup_summary.csv/md`：同拓扑 serial join 与 speedup 汇总。
- `npu_utilization_summary.csv`：Host1/Host2 `npu-smi` 采样汇总。
- `*/pipeline_*.png`：64 张 DBO/crosslayer pipeline 图。
- `*/npu_smi_host1.log`, `*/npu_smi_host2.log`：每次运行的 NPU 资源采样。

## 为什么相比 serial 仍然是负收益

Speedup 口径为：

`same-topology serial decode_tpot_ms / DBO decode_tpot_ms`

因此数值大于 `1.0x` 才表示 DBO 快于 serial。本轮所有 DBO/crosslayer 都低于 `1.0x`。

从 pipeline 结构和 timing JSON 看，根因不是 Attention 与 FFN local compute 没有对齐，而是 EP 通信路径仍然太重。当前真实 EPFFN 路径是：

1. FFN coordinator rank 计算 layernorm/router。
2. coordinator 通过 `broadcast` 把完整 `hidden_2d + selected_experts + routing_weights` 发给所有 EP ranks。
3. 每个 EP rank 只计算自己拥有的专家，但输出的是 dense partial `[tokens, hidden]`。
4. 所有 EP ranks 通过 `reduce(SUM)` 把 dense partial 聚合回 coordinator。

这意味着当前实现没有利用 MoE routing 的稀疏性：dispatch 仍广播所有 token，combine/reduce 仍规约全量 dense hidden。EP16 虽然降低了每个 rank 的 local expert compute，但通信和等待开销仍然远大于 local compute。

代表配置：

| 配置 | Speedup | A avg/layer | F avg/layer | F/A | Dispatch | Reduce | A recv wait | 通信+等待 | 通信+等待 / F |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `decode-dbo b256/s1024` | 0.916x | 1.537ms | 1.560ms | 1.02 | 1.916ms | 1.017ms | 1.518ms | 4.451ms | 2.85x |
| `decode-dbo b256/s512` | 0.913x | 1.234ms | 1.818ms | 1.47 | 1.796ms | 1.136ms | 2.014ms | 4.946ms | 2.72x |
| `decode-dbo b32/s256` | 0.777x | 0.979ms | 1.616ms | 1.65 | 1.943ms | 0.917ms | 2.096ms | 4.956ms | 3.07x |
| `crosslayer b256/s256` | 0.913x | 1.080ms | 1.905ms | 1.76 | 1.896ms | 1.186ms | 2.269ms | 5.351ms | 2.81x |
| `crosslayer b16/s128` | 0.730x | 0.902ms | 1.362ms | 1.51 | 1.582ms | 0.925ms | 1.836ms | 4.342ms | 3.19x |

最好的 `decode-dbo b256/s1024` 已经达到 `F/A=1.02`，说明 EP16 能把 FFN local compute 压到与 Attention 接近；但每层仍额外支付约 `1.9ms` dispatch、`1.0ms` reduce 和 `1.5ms` Attention recv wait。通信/等待合计约 `4.45ms`，是 FFN local compute 的 `2.85x`，所以整体 TPOT 仍慢于 serial。

因此下一步优化重点不是继续盲目增大 EP，而是把 EP 通信从 full broadcast + dense reduce 改为 token-aware sparse dispatch/combine。当前实现计划新增 `all_to_all_single` backend，复用 coordinator 架构中的 `FallbackMoECommunicator`，只把 token-expert assignment 发到目标 expert rank，再按 assignment gather 回 coordinator。

## `all_to_all_single` 接入结果

已新增实验 backend：

```text
--ffn-ep-backend all_to_all_single
```

该 backend 复用 coordinator 架构中的 `FallbackMoECommunicator`，在真实 `EPFFNLayer` 路径中执行 token/expert assignment 级别的 `all_to_all_single` dispatch/combine。实现上保留了原有 `broadcast_reduce_sync` / `broadcast_reduce_overlap`，因此可以逐项 A/B 对比和快速回退。

验证结果：

| 配置 | 状态 | TPOT | Dispatch | Local experts | Combine/Reduce | 说明 |
|---|---|---:|---:|---:|---:|---|
| EP16 `b2/s32/t3`, `debug_max_layers=2` | OK | 483.923ms | N/A | N/A | N/A | metadata 优化后 HCCL smoke 通过。 |
| EP16 `b16/s128/t20`, `all_to_all_single` metadata 重构 experts | OK | 596.726ms | 4.972ms | 0.946ms | 1.154ms | 低 batch 代表点；慢于同拓扑 serial 243.727ms 和 broadcast/reduce 315.297ms。 |
| EP16 `b32/s256/t20`, `broadcast_reduce_overlap` | OK | 373.129ms | 1.404ms | 1.197ms | 0.902ms | 当前最快可用真实路径。 |
| EP16 `b32/s256/t20`, `all_to_all_single` 初版 | OK | 636.266ms | 4.370ms | 1.055ms | 1.052ms | sparse 语义跑通，但 dispatch 明显偏慢。 |
| EP16 `b32/s256/t20`, `all_to_all_single` 不发送 weights | OK | 611.773ms | 4.761ms | 1.432ms | 1.312ms | 减少 weight payload 没有带来稳定收益。 |
| EP16 `b32/s256/t20`, `all_to_all_single` metadata 重构 experts | OK | 664.915ms | 5.131ms | 1.090ms | 1.211ms | 避免 expert-id all-to-all 后仍更慢，metadata broadcast/变长 all-to-all 成本主导。 |
| EP16 `b256/s1024/t20`, `all_to_all_single` 不发送 weights | OK | 1325.661ms | 4.025ms | 1.345ms | 1.200ms | 高 batch 下 local experts 有下降，但 TPOT 仍慢于 broadcast/reduce 的 1137.203ms。 |

当前结论：

1. `all_to_all_single` 已证明可以接入真实 EPFFN decode path，不再只是 coordinator skeleton。
2. 它确实把 local expert compute 维持在较低水平，但 HCCL all-to-all dispatch 当前比 full broadcast 更贵；`b16/s128` 下 dispatch 为 `4.972ms`，`b32/s256` 下 dispatch 从 broadcast/reduce 的 `1.404ms` 增至 `5.131ms`。
3. metadata 优化修复了 earlier group/global rank bug：`dist.broadcast(group=...)` 的 `src` 必须传全局 rank，因此 FFN coordinator 源 rank 使用 `ctx.ffn_coordinator_rank`。
4. 现阶段不建议把 `all_to_all_single` 设为默认 backend。下一步应先用 `msprof` 确认变长 all-to-all/count exchange/metadata broadcast 的 HCCL 开销，再考虑 padded equal-split all-to-all、batch 合并、或回到 CANN official MoE dispatch/combine 路线。

## Speedup 结果

| Seq | Batch | DBO speedup | Crosslayer speedup |
|---:|---:|---:|---:|
| 128 | 2 | 0.767 | 0.780 |
| 128 | 4 | 0.769 | 0.774 |
| 128 | 8 | 0.772 | 0.752 |
| 128 | 16 | 0.773 | 0.730 |
| 128 | 32 | 0.838 | 0.832 |
| 128 | 64 | 0.853 | 0.841 |
| 128 | 128 | 0.871 | 0.859 |
| 128 | 256 | 0.912 | 0.909 |
| 256 | 2 | 0.787 | 0.760 |
| 256 | 4 | 0.773 | 0.798 |
| 256 | 8 | 0.798 | 0.809 |
| 256 | 16 | 0.796 | 0.777 |
| 256 | 32 | 0.777 | 0.798 |
| 256 | 64 | 0.831 | 0.824 |
| 256 | 128 | 0.887 | 0.878 |
| 256 | 256 | 0.901 | 0.913 |
| 512 | 2 | 0.812 | 0.787 |
| 512 | 4 | 0.781 | 0.788 |
| 512 | 8 | 0.787 | 0.770 |
| 512 | 16 | 0.791 | 0.788 |
| 512 | 32 | 0.796 | 0.795 |
| 512 | 64 | 0.847 | 0.826 |
| 512 | 128 | 0.859 | 0.867 |
| 512 | 256 | 0.913 | 0.913 |
| 1024 | 2 | 0.770 | 0.764 |
| 1024 | 4 | 0.808 | 0.775 |
| 1024 | 8 | 0.794 | 0.775 |
| 1024 | 16 | 0.766 | 0.812 |
| 1024 | 32 | 0.834 | 0.794 |
| 1024 | 64 | 0.847 | 0.842 |
| 1024 | 128 | 0.900 | 0.898 |
| 1024 | 256 | 0.916 | 0.907 |

最佳若干行：

| Mode | Batch | Seq | Serial TPOT | DBO TPOT | Speedup | A avg/layer | F avg/layer | F/A | Dispatch | Reduce |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| decode-dbo | 256 | 1024 | 1041.612ms | 1137.203ms | 0.916x | 1.537ms | 1.560ms | 1.02 | 1.916ms | 1.017ms |
| decode-dbo-crosslayer | 256 | 256 | 1012.814ms | 1109.202ms | 0.913x | 1.080ms | 1.905ms | 1.76 | 1.896ms | 1.186ms |
| decode-dbo | 256 | 512 | 1021.925ms | 1119.497ms | 0.913x | 1.234ms | 1.818ms | 1.47 | 1.796ms | 1.136ms |
| decode-dbo-crosslayer | 256 | 512 | 1021.925ms | 1119.637ms | 0.913x | 1.191ms | 2.041ms | 1.71 | 2.082ms | 1.232ms |
| decode-dbo | 256 | 128 | 1015.956ms | 1113.986ms | 0.912x | 1.042ms | 1.988ms | 1.91 | 1.896ms | 1.180ms |

## Pipeline 图

所有 `decode-dbo` 和 `decode-dbo-crosslayer` 行都生成了 pipeline 图。每张图的 speedup 标注都使用同一 `(batch, seq)` 下的同拓扑 serial TPOT。Serial 行本身没有 DBO microbatch pipeline event，因此不生成 DBO pipeline 图。

代表图：

- `crosshost_static_ep16_sweep/decode-dbo_ep16_broadcast_reduce_overlap_b256_s1024_t20/pipeline_xhost_static_decode-dbo_ep16_broadcast_reduce_overlap_b256_s1024_t20.png`
- `crosshost_static_ep16_sweep/decode-dbo-crosslayer_ep16_broadcast_reduce_overlap_b256_s1024_t20/pipeline_xhost_static_decode-dbo-crosslayer_ep16_broadcast_reduce_overlap_b256_s1024_t20.png`
- `crosshost_static_ep16_sweep/decode-dbo-crosslayer_ep16_broadcast_reduce_overlap_b256_s256_t20/pipeline_xhost_static_decode-dbo-crosslayer_ep16_broadcast_reduce_overlap_b256_s256_t20.png`

## NPU 利用率与 MFU

每次运行都采集了 Host1/Host2 的 `npu-smi info -t usages`。解析汇总位于 `npu_utilization_summary.csv`。

| 侧 | AICore 峰值 | 最高平均 AICore 配置 | HBM 峰值 | HBM 带宽峰值 | 说明 |
|---|---:|---|---:|---:|---|
| Host1 Attention | 100% | `decode-dbo-crosslayer b256/s512`, avg `48.651%` | 88% at `b256/s1024` | 65% at `b256/s1024` | Attention rank 在请求范围内更接近 HBM 上限。 |
| Host2 FFN EP | 52% | `decode-dbo b128/s1024`, avg `0.521%` | 72% at `b256/s1024` | 23% at `b256/s1024` | 1s 采样会漏掉短 FFN expert burst；单层耗时应以 timing JSON 为准。 |

`npu-smi` 提供利用率、HBM 使用率和带宽使用率，但不直接给 MFU。严格 MFU 需要估算 Qwen3 MoE 激活专家 FLOPs，再除以 TPOT 和 910C 理论峰值。当前报告只使用硬件 counter 与 timing JSON，不把估算 MFU 当作真实硬件指标。

后续建议只对少量代表配置跑 `msprof`：

- 高 batch 最优：`decode-dbo b256/s1024`
- crosslayer 最优附近：`decode-dbo-crosslayer b256/s256`
- 低 batch 慢配置：`decode-dbo-crosslayer b16/s128`

重点分析 Communication Time、AllReduce/AllToAll/Reduce、Overlap Analysis，并与 timing JSON 中的 `ep_dispatch`、`ep_local_experts`、`ep_reduce`、`attention_recv_wait` 对齐。

## OOM 边界

请求范围内没有 OOM。最大已验证非 OOM 点：

| Mode | Batch | Seq | TPOT | Host1 HBM peak | Host2 HBM peak |
|---|---:|---:|---:|---:|---:|
| serial | 256 | 1024 | 1041.612ms | 88% | 72% |
| decode-dbo | 256 | 1024 | 1137.203ms | 88% | 72% |
| decode-dbo-crosslayer | 256 | 1024 | 1148.943ms | 88% | 72% |

因为没有任何行 OOM，本轮不能证明超出请求范围后是 Attention 还是 FFN 先 OOM。在已测范围内，Host1 Attention HBM 峰值更高（88% vs Host2 72%），说明 Attention 更接近内存上限；但真正的 first-OOM 侧需要继续测更大的 batch/seq。

## 运行过程问题

实验期间遇到过 SSH reset 和 HCCL `EADDRINUSE`。处理方式是：

- 使用 fresh `MASTER_PORT` / `HCCL_IF_BASE_PORT` 继续 resume。
- 失败后检查 Host1/Host2 容器内 stale `src.main` / launcher 进程。
- 只按明确 PID 清理，未使用 `pkill` / `killall`。

## 后续优化方向

当前结果说明 EP16 已能把 FFN local compute 压到接近 Attention，但 TPOT 仍主要受 EP 通信与等待气泡限制。下一步优先级：

1. 对 `all_to_all_single` 的代表配置跑 `msprof`，确认 dispatch 慢在 count exchange、变长 payload all-to-all、metadata broadcast，还是 HCCL group 同步。
2. 尝试 padded/equal-split all-to-all 或分层批量化，降低变长 split sizes 和 per-layer collective 开销。
3. 继续评估 CANN official MoE dispatch/combine v2，但只有在真实 Qwen hidden=2048、真实 expert output combine 通过后才接入真实路径。
4. 在通信不再主导前，不建议继续盲目增大 EP 或默认启用 crosslayer/all-to-all。
