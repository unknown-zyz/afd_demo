# Single-host EP7 coordinator vs static decode-DBO

## 结论

本轮在 Host1 单机 1A7F / EP7 / Qwen3-30B-A3B 真实 decode-DBO 路径上完成了 24 个 coordinator one-shot 配置，网格为 batch `{2,4,8,16,32,64,128,256}` × seq `{128,256,512}` × tokens `20`。所有 coordinator 配置均跑通，routing table 固定为 `coordinator v1`，`routing_poll_count=0`，说明本轮测的是 **coordinator one-shot routing**，不是动态负载均衡收益。源数据见 `coordinator/experiment_matrix_summary.csv` 与 `coordinator/decode-dbo/timing_attention_*.json`。

static 全矩阵复用历史 `results_npu_ep7` baseline；当前代码下重跑了 3 个 static 校准点，TPOT 漂移均在 ±5% 内，因此复用历史 baseline 是可接受的。校准结果见 `static_reuse_validation.csv`，三点分别为 b8/s128 `-3.98%`、b32/s512 `-0.87%`、b128/s512 `-1.31%`。

## 数据来源

| 用途 | 路径 |
|---|---|
| coordinator summary | `results_npu/coordinator_arch/singlehost_ep7/coordinator/experiment_matrix_summary.csv` |
| coordinator timing | `results_npu/coordinator_arch/singlehost_ep7/coordinator/decode-dbo/timing_attention_*.json` |
| coordinator FFN timing | `results_npu/coordinator_arch/singlehost_ep7/coordinator/decode-dbo/timing_ffn_coordinator_*.json` |
| static historical baseline | `results_npu_ep7/experiment_matrix_summary.csv` |
| static timing | `results_npu_ep7/decode-dbo/timing_attention_*.json` |
| static validation | `results_npu/coordinator_arch/singlehost_ep7/static_reuse_validation.csv` |
| 聚合表 | `results_npu/coordinator_arch/singlehost_ep7/coord_vs_static_summary.csv` |
| 自动摘要 | `results_npu/coordinator_arch/singlehost_ep7/comparison_summary.md` |

## TPOT / throughput 对比

整体 24 个配置里，coordinator TPOT 优于 static 的配置为 16/24；平均 TPOT delta 为 `-2.40%`，median 为 `-2.52%`。按 seq 分组：

| seq | coordinator TPOT 更优 | 平均 TPOT delta | 平均 throughput delta |
|---:|---:|---:|---:|
| 128 | 2/8 | +0.86% | -0.77% |
| 256 | 7/8 | -3.85% | +4.11% |
| 512 | 7/8 | -4.22% | +4.57% |

最佳点是 b8/s512，coordinator TPOT 比 static 低 `12.77%`；最差点是 b16/s128，coordinator TPOT 高 `3.86%`。逐配置明细见 `coord_vs_static_summary.csv`，热力图见 `coord_vs_static_tpot_delta_heatmap.png` 与 `coord_vs_static_throughput_ratio_heatmap.png`。

seq=512 的 batch 趋势图已生成：

- `static_vs_coord_throughput_vs_batch_s512.png`
- `static_vs_coord_tpot_vs_batch_s512.png`

## pipeline / overlap 图

常规 pipeline 图已为 24 个 coordinator 配置全部生成在：

- `results_npu/coordinator_arch/singlehost_ep7/coordinator/decode-dbo/pipeline_decode-dbo_npu_ep7_broadcast_reduce_overlap_b*_s*_t20.png`

去掉 L0 的 no-L0 版本已生成在：

- `results_npu/coordinator_arch/singlehost_ep7/coordinator/pipeline_figs_no_l0/decode_dbo_npu_ep7_broadcast_reduce_overlap_b*_s*_t20_no_l0.png`

L0 分析见 `coordinator/decode_dbo_l0_warmup_analysis.csv` 和 `coordinator/decode_dbo_l0_warmup_analysis.md`。本轮 coordinator 结果中 `likely_l0_cold_start=False` 覆盖所有配置，最大 L0/tail ratio 约 `1.93`，没有看到之前那种明显由首层冷启动污染导致的 L0 极端慢点。

从 FFN timing 汇总看，seq=512 下 coordinator 的 `total_ep_overlap_hidden_ms` 平均比 static 低约 `3.54 ms`，与 b8/s512、b32/s512、b64/s512 等配置的 TPOT 改善方向一致；但 seq=128 平均反而高约 `3.97 ms`，因此当前 one-shot coordinator 的 overlap 改善不是全局稳定收益。该结论来自 `coord_vs_static_summary.csv` 中 `*_total_ep_dispatch_ms`、`*_total_ep_local_experts_ms`、`*_total_ep_reduce_ms`、`*_total_ep_overlap_hidden_ms` 列。

## 限制

本轮 coordinator 使用的是 one-shot 均匀 routing table，只在初始化时从 coordinator 获取 expert ownership；它与 static round-robin EP ownership 很接近。因此本轮结果只能说明真实 EP7 decode 路径已接通、指标链路可用、没有明显整体回退；不能证明动态 load-aware routing 已带来收益。真正的动态收益需要在 `routing-update-mode=poll` 和负载倾斜实验中验证。

## 后续保留事项

1. 在 1A7F / EP7 上验证 `routing-update-mode=poll` 的稳定性和 `routing_poll_ms` 开销。
2. 设计 load-aware / dynamic routing 实验，让 routing table 不再等价于 static round-robin。
3. 单机结果稳定后推进跨机 1A7F coordinator，对比单机 coordinator 与 historical static EP7，拆分 RoCE/HCCL 网络成本。
4. DeepEP 运行层仍暂缓，不作为当前 fallback/HCCL 主线的阻塞项。
