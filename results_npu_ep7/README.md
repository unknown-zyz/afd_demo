# results_npu_ep7 — 910C EP7 全矩阵 + crosslayer 实验汇总

> **目标**：在 Ascend 910C 上以 EP7（attention rank ×1 + FFN coordinator rank ×1 + FFN expert rank ×6 = 共 8 rank，experts 跨 7 个 ranks fan-out）拓扑下，覆盖 prefill / decode / decode-crosslayer 三种模式，从 batch 2 一直跑到 OOM 上界。本目录是面向用户的对外汇总；图全部使用 round-4 fourlane (compute/comm) 视角。

## 数据来源 / 一致性（round-6）

- **单一来源**：本目录所有 decode-dbo / prefill-dbo 数值与图均来自 round-6 在 `exp/profile-viz-qa-fix` 分支上的 NPU 重跑（EP7 + `broadcast_reduce_overlap` 后端 + `round_robin` 调度，t=20）。`serial/` 为缓存基线（与 round-5 一致），`decode-dbo-crosslayer/` 沿用 round-5 的 34 个有效配置。
- **同源时序**：heatmap (`*_speedup_heatmap.png`) 和 `pipeline_figs/*.png` 现在均直接读取本目录下 report 配套的 timing JSON，不再出现 round-5 中"heatmap 用 full_matrix_v2、pipeline fig 用 ep7_matrix_v2"导致的数值漂移。
- **Spot-check**：decode-dbo `b16/s512/t20` → `speedup = 1.315×`（`experiment_matrix_summary.csv` 行与 `pipeline_figs/decode_dbo_b16_s512_t20.png` 顶部标题完全一致）。
- 聚合脚本：`scripts/aggregate_npu_ep7.py`（新增），`experiment_matrix_summary.csv` 为其唯一输出（156 行：45 serial + 42 decode-dbo + 35 prefill-dbo + 34 crosslayer）。
- 历史 round-4/5 文件 `summary_v2_baseline.csv` / `experiment_matrix_summary_v2_baseline.csv` 保留作历史快照。

## 配置覆盖

batch ∈ {2, 4, 8, 16, 32, 64, 128, 256, 512}（9 行）× seq ∈ {128, 256, 512, 1024, 2048}（5 列）= **45 配置**/模式。

| 模式 | 完成 / 计划 | 失败配置 |
|---|---|---|
| `serial/`（基线，cache） | 45 / 45 | — |
| `decode-dbo/` | **42 / 45** | OOM：`b512/s1024`、`b256/s2048`、`b512/s2048` |
| `prefill-dbo/` | **35 / 45** | OOM：`b256/s512`、`b256/s1024`、`b256/s2048`、`b128/s1024`、`b128/s2048`、`b64/s2048`、`b512/s512`、`b512/s1024`、`b512/s2048`、`b512/s256`（小 batch OOM 后 b256/b512 较大 seq 直接跳过） |
| `decode-dbo-crosslayer/` | 34 / 45（沿用 round-5） | `b512/s1024`、整行 `s2048` 全 OOM |

> "完成" = `experiment_matrix_summary.csv` 中该模式 status=ok 的行数。OOM/skipped 不写入 CSV，但在 `decode-dbo/` 与 `prefill-dbo/` 子目录下保留 stderr 日志。

### OOM 边界汇总

| 模式 | 第一处 OOM (按 seq 增大) | 备注 |
|---|---|---|
| decode-dbo | b256 在 s2048 OOM；b512 在 s1024 起 OOM | b≤128 全部跑通到 s2048 |
| decode-dbo-crosslayer | b512 在 s1024 OOM；**任意 batch 在 s2048 全 OOM** | pre-post irecv buffer 把 HBM 顶满 |
| prefill-dbo | b64 在 s2048 OOM；b128 在 s1024 起 OOM；b256 在 s512 起 OOM；b512 在 s256 起 OOM | prefill 激活 ×seq² 内存压力远大于 decode |
| serial（基线） | b512/s1024+, b512/s2048, b256/s2048 | 与 decode-dbo 边界一致 |

## Pipeline 图说明（fourlane）

每个 PNG 顶部是 `Speedup`（vs cached serial baseline），下面 4 条泳道：

| 泳道 | 含义 |
|---|---|
| **Attention** | ATT 的 attn_compute |
| **A2F** | ATT.send.start → FFN.ep_local_experts.start（含 send + ATT-side recv_wait + router + dispatch） |
| **FFN** | ep_local_experts (FFN GEMM 主体) |
| **F2A** | FFN.ep_reduce.start → ATT.recv_wait.end（含 combine + send + ATT 串行 recv 排队） |

A2F / F2A 的 bar **保留"传输开始 → 接收方真正消费完成"语义**，bar 长度直接代表通信开销 + 接收侧排队，便于看出 pipeline 瓶颈（如 MB1 F2A 比 MB0 长是 ATT 端 irecv 串行造成的）。详见 `doc/QA.md` §3.4 / §3.5。

## Q1: crosslayer 是什么？为什么效果不如不开？

**机制（`src/pipeline/decode_scheduler.py:480-525, 725-770`）**：常规 decode-dbo 在每层内部完成"ATT compute → A2F send → FFN compute → F2A send → 下一层 ATT"的串行 layer 边界。`use_crosslayer=True` 时调度器额外做两件事：

1. **预 post 下一层 A2F irecvs**：在当前层 F2A sends 还在排队时，下一层每个 micro-batch 的 A2F irecv 就 enqueue 到 NCCL stream（line 737-758），让下层 ATT 一发出 send，FFN 端立即 match 上 recv，跨层 micro-batch 流水
2. **预 post layer-0 F2A irecvs**：layer 0 进入时 ATT 端就把所有 layer F2A irecv 排好（line 492-502），消除首层 cold-start

**为什么 EP7 下不增益（甚至轻微负向）**：

1. **基础 DBO 已经把通信掩盖得很好** —— EP7 + `broadcast_reduce_overlap` 后端在 layer 内 MB0/MB1 重叠已经把 inter-layer bubble 压到很小，crosslayer 进一步抢的空间有限
2. **HBM 成本明显**：pre-posted irecv 的 buffer `(mb_size, 1, hidden)` × 层数 × MB 数被预分配占用 → s2048 整行 100% OOM 即直接证据；同 batch 下 baseline 还能跑、crosslayer OOM
3. **NCCL proxy queue 竞争**：跨层 send/recv 顺序乱了，proxy 调度时 next-layer recv 先排队，反而打乱了 current-layer send 的 fast path

**与历史结果对照**：旧 2-rank 配置下 b4 有 0.73×→0.94× 收益（仓库记忆 `cross-layer pipeline`），EP7 下因基础 DBO 已强而失去优势，是合理的。

## Q2: heatmap vs pipeline-fig speedup 不一致 — 已修复

round-5 中 heatmap 来源于 `full_matrix_v2/`，pipeline fig 来源于 `ep7_matrix_v2/`，两处对同一 (mode, batch, seq, t) 报出的 speedup 不同（典型偏差 ±5%）。round-6 起：

- `experiment_matrix_summary.csv` 由 `scripts/aggregate_npu_ep7.py` **直接扫描本目录** `decode-dbo/`、`prefill-dbo/`、`decode-dbo-crosslayer/`、`serial/` 下的 report.md / cache JSON 生成。
- `pipeline_figs/*.png` 顶部 speedup 取的 timing JSON 与同一 report 同源。
- 因此 heatmap 单元格、CSV 行、PNG 顶部三者一致；上文 spot-check 已验证。

## Q3: decode 全矩阵已扩展至 b512

round-5 decode-dbo 仅跑到 b256；本轮补跑 b512：

| seq | 128 | 256 | 512 | 1024 | 2048 |
|---|---|---|---|---|---|
| b512 | ok | ok | ok | **OOM** | **OOM** |

加上 b256/s2048 OOM，decode-dbo 完成度提升至 **42/45 (93%)**。具体单元格数值见 `decode_dbo_speedup_heatmap.png` 和 CSV。

## Q4: prefill 矩阵已从 8 配置扩到 35

round-5 prefill-dbo 仅 8 个配置（b8/16/32/64 × s512/1024）。本轮覆盖完整 9×5 矩阵（batch 2..512 × seq 128..2048），实跑成功 **35/45**，10 个因 prefill 激活 ~`O(b·s²·hidden)` 内存超限被 OOM/skip（见上方 OOM 表）。

- 小 batch（b≤32）下 prefill DBO speedup 显著（详见 `prefill_dbo_speedup_heatmap.png`）。
- batch 增大后 prefill compute 已经把通信吃满，DBO 边际收益缩窄。

## 主图（聚合）

顶层（round-6 新出，单一时序源）：

- `decode_dbo_speedup_heatmap.png` — decode DBO speedup 热力图（vs serial）
- `decode_dbo_tpot_heatmap.png` — decode DBO TPOT (ms)
- `prefill_dbo_speedup_heatmap.png` — prefill DBO speedup 热力图
- `decode_dbo_crosslayer_speedup_heatmap.png` — decode-crosslayer speedup 热力图
- `crosslayer_comparison.png` — DBO vs crosslayer 同 (b,s) 对比
- `serial_throughput_vs_batch_s512.png` — serial 在 seq=512 下的 throughput vs batch
- `serial_tpot_vs_batch_s512.png` — serial 在 seq=512 下的 TPOT vs batch

历史快照（round-4/5，保留参照，**勿用于结论**）：

- `fig_decode_speedup_heatmap.png`、`fig_decode_speedup_curves.png`
- `fig_prefill_speedup_heatmap.png`
- `fig_decode_crosslayer_speedup_heatmap.png`、`fig_decode_dbo_vs_crosslayer_curves.png`

## Pipeline 图位置

`pipeline_figs/`，命名约定：`{mode}_b{batch}_s{seq}_t{tokens}.png`，其中 mode ∈ {`decode_dbo`, `decode_crosslayer`, `prefill_dbo`}。

`pipeline_figs_no_l0/` 是 decode-dbo 的 L0-filtered 重绘目录：每张图从 L1 开始画（`--start-layer 1 --num-layers 3 --no-auto-skip-warmup`），用于排除首层/首 micro-batch 的冷启动视觉干扰。现有静态分析见 `decode_dbo_l0_warmup_analysis.{csv,md}`：当前 EP7 decode-dbo 数据中 L0/mb0 相对后续层的最大倍率约 2.06×，没有达到 5× cold-start 阈值，因此它不像“完全没做 warmup”导致的巨大异常；更像首层调度/接收排队与 decode 首步 lazy path 的轻量放大。

| 类别 | 张数 | 备注 |
|---|---|---|
| `decode_dbo_*` | 42 | 与 `decode-dbo/` 报告一一对应 |
| `prefill_dbo_*` | 35 | 与 `prefill-dbo/` 报告一一对应 |
| `decode_crosslayer_*` | 34 | 沿用 round-5 |
| **合计** | **111** | |

## Warmup 说明与 ablation

当前代码里有两类 warmup，预热对象不同：

- `--warmup-p2p --warmup-rounds N`：预热 HCCL/NCCL P2P 通信、communicator/proxy/lazy init 和 send/recv 路径，不跑模型 forward。
- `--prefill-warmup-rounds N`：正式计时前跑 untimed prefill forward，并关闭 scheduler timing，吸收 prefill shape 下的 NPU JIT / graph compile / kernel lazy init；它不保证覆盖 decode loop 首步的所有 lazy path。

NPU matrix runner 已支持四种组合，输出 suffix 会带 `wp2p*/pw*`，避免覆盖：

```bash
DRY_RUN=true BATCHES=4 SEQS=512 bash scripts/run_warmup_ablation_npu.sh

# 实跑默认小矩阵：decode-dbo, batches={4,16,64}, seq=512
bash scripts/run_warmup_ablation_npu.sh

# 聚合 warmup_ablation 目录
python3 scripts/aggregate_warmup_ablation_npu.py --root results_npu_ep7/warmup_ablation
```

四个变体：

| variant | P2P warmup | prefill warmup |
|---|---:|---:|
| `both_on` | on | 1 round |
| `p2p_only` | on | 0 round |
| `prefill_only` | off | 1 round |
| `both_off` | off | 0 round |

## 复现命令

```bash
# NPU 重跑（在 afd-npu-test 容器内）
bash scripts/run_experiment_matrix_npu.sh --modes decode-dbo \
  --backend broadcast_reduce_overlap --schedule round_robin --tokens 20
bash scripts/run_experiment_matrix_npu.sh --modes prefill-dbo \
  --backend broadcast_reduce_overlap --schedule round_robin --tokens 20

# 聚合（生成 experiment_matrix_summary.csv + heatmaps）
python3 scripts/aggregate_npu_ep7.py --root results_npu_ep7

# 单张 pipeline 图（fourlane）
python3 scripts/visualize_dbo_pipeline.py \
  --attn-timing results_npu_ep7/decode-dbo/timing_attention_decode-dbo_npu_ep7_broadcast_reduce_overlap_b16_s512_t20.json \
  --ffn-timing  results_npu_ep7/decode-dbo/timing_ffn_coordinator_decode-dbo_npu_ep7_broadcast_reduce_overlap_b16_s512_t20.json \
  --output results_npu_ep7/pipeline_figs/decode_dbo_b16_s512_t20.png
```
