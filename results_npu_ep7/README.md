# results_npu_ep7 — 910C EP7 全矩阵 + crosslayer 实验汇总

> **目标**：在 Ascend 910C 上以 EP7（attention rank ×1 + FFN coordinator rank ×1 + FFN expert rank ×6 = 共 8 rank，experts 跨 7 个 ranks fan-out）拓扑下，覆盖 prefill / decode / decode-crosslayer 三种模式，从 batch 2 一直跑到 OOM 上界。本目录是面向用户的对外汇总；图全部使用 round-4 fourlane (compute/comm) 视角。

## 数据来源

| 模式 | 来源 | 状态 |
|---|---|---|
| `serial/` (基线) | `results_npu/full_matrix_v2/serial/cache/*.json` 直接复用 | 完整：batch 2..512 × seq 128..2048，已含 OOM 标记 |
| `decode-dbo/` 数值 | `results_npu/full_matrix_v2/decode-dbo/report_*.md`（统计） | 完整 |
| `decode-dbo/` pipeline 图 | `results_npu/ep7_matrix_v2/decode-dbo/timing_*.json` 全 24 配置 | ✅ 已批量生成到 `pipeline_figs/decode_dbo_b*_s*_t20.png` |
| `prefill-dbo/` 数值 | `results_npu/full_matrix_v2/prefill-dbo/report_*.md`（统计） | 完整 |
| `prefill-dbo/` pipeline 图 | NPU 重跑（b8/16/32/64 × s512/1024，t=20） | ✅ 8 配置 |
| `decode-dbo-crosslayer/` 数值 | NPU 重跑（全矩阵 batch 2..512 × seq 128..2048，t=20） | ✅ 34 ok / 11 OOM 或 fail |
| `decode-dbo-crosslayer/` pipeline 图 | 同上，跑回后批量画 | ✅ 34 配置 |

`summary_v2_baseline.csv` / `experiment_matrix_summary_v2_baseline.csv` 是从 `full_matrix_v2/` 直接 copy 的基础数据。`experiment_matrix_summary.csv` 现为合并后版本（baseline serial+decode-dbo + 新 prefill-dbo + 新 decode-dbo-crosslayer），`summary.csv` 是 aggregator 输出（含 tpot/speedup 字段）。

### OOM 边界（batch×seq）

- decode-dbo（baseline）：b256/s2048, b512/s1024+, b512/s2048
- decode-dbo-crosslayer（本轮新跑）：b512/s1024, **整行 s2048 全部 OOM**（pre-post irecv buffer 进一步拉高 HBM 占用）
- prefill-dbo（仅 b8..64）：未触发 OOM

## Pipeline 图说明（fourlane）

每个 PNG 顶部是 `Speedup`（vs cached serial baseline），下面 4 条泳道：

| 泳道 | 含义 |
|---|---|
| **Attention** | ATT 的 attn_compute |
| **A2F** | ATT.send.start → FFN.ep_local_experts.start（含 send + ATT-side recv_wait + router + dispatch） |
| **FFN** | ep_local_experts (FFN GEMM 主体) |
| **F2A** | FFN.ep_reduce.start → ATT.recv_wait.end（含 combine + send + ATT 串行 recv 排队） |

A2F / F2A 的 bar **保留"传输开始 → 接收方真正消费完成"语义**，bar 长度直接代表通信开销 + 接收侧排队，便于看出 pipeline 瓶颈（如 MB1 F2A 比 MB0 长是 ATT 端 irecv 串行造成的）。详见 `doc/QA.md` §3.4 / §3.5。

## 主图（聚合）

- `fig_decode_speedup_heatmap.png` — decode DBO speedup 热力图（vs serial）
- `fig_decode_speedup_curves.png` — decode DBO speedup 折线（按 batch）
- `fig_prefill_speedup_heatmap.png` — prefill DBO speedup 热力图
- `fig_decode_crosslayer_speedup_heatmap.png` — decode-crosslayer speedup 热力图
- `fig_decode_dbo_vs_crosslayer_curves.png` — DBO vs crosslayer 对比折线（同 seq 一组）

## Pipeline 图位置

`pipeline_figs/`，命名约定：`{mode}_b{batch}_s{seq}_t{tokens}.png`，其中 mode ∈ {`decode_dbo`, `decode_crosslayer`, `prefill_dbo`}。共 66 张：24 decode_dbo + 34 decode_crosslayer + 8 prefill_dbo。

## 复现命令

```bash
# Pipeline 图（fourlane 默认）
python3 scripts/visualize_dbo_pipeline.py \
  --attn-timing results_npu/ep7_matrix_v2/decode-dbo/timing_attention_decode-dbo_npu_ep7_broadcast_reduce_overlap_b16_s512_t20.json \
  --ffn-timing  results_npu/ep7_matrix_v2/decode-dbo/timing_ffn_coordinator_decode-dbo_npu_ep7_broadcast_reduce_overlap_b16_s512_t20.json \
  --output results_npu_ep7/pipeline_figs/decode_dbo_b16_s512_t20.png

# NPU 重跑（在 afd-npu-test 容器内）
bash scripts/run_experiment_matrix_npu.sh --modes decode-dbo-crosslayer
bash scripts/run_experiment_matrix_npu.sh --modes prefill-dbo \
  --batches 8,16,32,64 --seqs 512,1024 --tokens 8

# 重新汇总
python3 scripts/aggregate_full_matrix_v2.py --root results_npu_ep7
```
