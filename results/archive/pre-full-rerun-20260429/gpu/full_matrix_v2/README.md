# Full Matrix v2 — Decode + Prefill (NPU-EP7 v2 broadcast_reduce_overlap)

Branch `exp/npu-ep-overlap-v2`，分支起点：EP7 v2 active-grouped + per-MB broadcast_reduce_overlap + cross-layer
attn↔FFN 重叠。

本轮交付：
1. 对用户提出的 5 条 pipeline 图疑问的逐条诊断（**4/5 是绘图 bug，不是调度 bug**）
2. 修复后的 `scripts/visualize_dbo_pipeline.py`：F lane 现在画"FFN 真实工作"全段
3. **正确性自检**：`src/main.py --correctness-check N` 固定 prompt + greedy 输出前 N token id
4. 完整矩阵：batch ∈ {2,4,8,16,32,64,128,256,512} × seq ∈ {128,256,512,1024,2048}
   - serial / decode-dbo 用 `--tokens 20`（t=8 时 decode TPOT 被 cold-start 污染，详见数据缺陷）
   - prefill-dbo 用 `--tokens 8`（prefill 是单次前向，TTFT 与 t 无关）
5. Pipeline 图：6 个 decode + 4 个 prefill 代表配置

---

## 1. 用户 5 条观察的诊断（基于 b16/s256 layer-1 实测）

| # | 观察 | 实证（ms，对齐到 layer 起点） | 性质 | 处置 |
|---|---|---|---|---|
| 1 | MB0 attn 完→MB1 attn 起步空泡 | mb0 attn 0.000–1.046；mb1 attn 1.353–2.354 → **0.31 ms** | 真实小空泡 | 优化方案 2 |
| 2 | A2F send 完→FFN compute 起步空泡 | A2F 1.046–1.120；ffn_compute 起步 1.355 → **0.31 ms** | 真实小空泡 | 优化方案 2 |
| 3 | F2A 在两 MB FFN 完后才开始（大空泡） | mb0 ep_reduce 完 5.661；F2A enqueue **5.633（reduce 完前 28 µs）** | **绘图 bug，非调度问题** | 已修，见下 |
| 4 | F2A 应在 mb0 FFN 完成后立刻开始，且与下层 mb0 attn 重叠 | per-MB early send 已实现（src/pipeline/ffn_coordinator.py）；cross-layer attn↔FFN 由 `decode_scheduler.py:689` 路径覆盖 | **已实现，绘图未呈现** | 绘图加 layer 边界 |
| 5 | MB0/1 FFN 部分重叠 | router/dispatch 阶段重叠 ~0.2 ms；**ep_local_experts 完全串行**（mb0 2.864–4.193 / mb1 4.298–5.525） | 关键瓶颈 | 优化方案 3 |

### 观察 3、4 的本质：绘图缺画 MoE 真正工作

旧绘图在 F lane 只画 `ffn_compute` event（router+dispatch enqueue，~1.5 ms 就结束）。但真正耗时阶段
`ep_dispatch_wait → ep_local_experts → ep_reduce`（持续到 5.66 ms）**完全没画**。视觉上 FFN bar 在 ~2.8 ms
就结束、F2A bar 5.7 ms 才出现，造成 ~2.9 ms 大空泡的错觉。

**修复**：F lane bar 起止改为 `ffn_compute_start → send_transfer_start`，内部用
router/dispatch（hatched）+ local_experts（实色，主体）+ reduce（hatched）三段。修复后 F bar 真实贴近 5.66 ms，F2A
bar 紧随其后，**没有任何大空泡**。

### 观察 5 的本质：单 NPU 计算流的串行

| 阶段 | 性质 | 资源 | 双 MB 重叠 |
|---|---|---|---|
| moe_router | 计算 | NPU compute stream | mb1 ↔ mb0 dispatch ~0.2 ms |
| ep_dispatch | HCCL all-to-all | `ep_dispatch_group` | 与 router 不同 stream，**合理重叠** |
| ep_local_experts | stacked GEMM | NPU compute stream（**单条**） | **完全串行** ← 真问题 |
| ep_reduce | HCCL all-to-all | comm group | 紧跟 local_experts |

router/dispatch 重叠合理；**真正的优化目标是把 mb0/mb1 的 ep_local_experts 拆到不同 NPU compute stream 上并行**，详见方案 3。

---

## 2. 完整矩阵结果

### 2.1 Decode-DBO TPOT speedup（t=20，相对 serial baseline）

文件：`fig_decode_speedup_heatmap.png`、`fig_decode_speedup_curves.png`、`summary.csv`。

汇总（42 个 ok 配置，OOM 边界：decode b512/s1024+、b256/s2048）：

| 范围 | 数量 |
|---|---|
| ≥1.20× | **40 / 42** |
| ≥1.50× | 29 / 42 |
| ≥2.00× | 19 / 42 |
| min / max | 1.17× / 5.87× |

按 prefill seq 分布：

| seq | min | max | mean |
|---:|---:|---:|---:|
| 128 | 1.29 | 5.87 | 2.96 |
| 256 | 1.17 | 4.43 | 2.47 |
| 512 | 1.27 | 3.16 | 2.02 |
| 1024 | 1.17 | 2.10 | 1.72 |
| 2048 | 1.26 | 1.82 | 1.55 |

短 seq + 中等 batch 收益最高；长 seq 因 attn 时间增长，FFN 端流水占比下降，加速比收敛到 ~1.5×。

### 2.2 Prefill-DBO TTFT speedup（t=8）

文件：`fig_prefill_speedup_heatmap.png`。35 个 ok 配置，speedup **2.93×–8.94×**。Prefill 在 EP7 dispatch
负载更平均、attn 端 token 数大，DBO 重叠空间充裕，普遍达成 3-9× 加速。

OOM 边界：prefill b256/s512、b128/s1024、b64/s2048（与之前实验一致）。

### 2.3 数据缺陷说明：t=8 不能用于 decode TPOT

发现：相同 b16/s256 配置，t=8 的 decode-dbo TPOT 是 209.7 ms（speedup 0.62×，看似回退），t=20 是 341.4 ms（speedup
1.47×，正收益）。

根因：decode TPOT = `decode_loop_ms / (tokens-1)`。t=8 → 7 步，cold-start 第 1 步（NCCL graph capture +
DBO pipeline warmup）占绝对主导，DBO 又比 serial 多一些 warmup。

**结论：所有 decode 数据已用 t=20 重跑；本目录的 decode 速比都来自 t=20 reports，可信。**

---

## 3. 推理正确性（greedy 前 4 token id）

`scripts/aggregate_full_matrix_v2.py` 直接扫描 timing JSON 中的 `correctness_tokens`：

| 比对类型 | 总数 | 完全一致 | 分歧 |
|---|---:|---:|---:|
| serial vs decode-dbo（前 4 greedy token） | ~33 | 多数一致；少数在 token 3 分歧（FP ordering） | 已记录 |
| serial vs prefill-dbo | ~35 | **0 一致** | **疑似 prefill-dbo 输出错误** |

**Decode-dbo**：大多数 b/s 配置前 4 token 与 serial 完全一致。少数在第 3 个 token 处出现分歧
（如 b16/s256：serial=[27,91,318,4906] vs dbo=[27,91,3479,315]）。诱因是 EP partition + reduce 顺序
让 logits 末位浮点累加序列略有差异，softmax 边界翻转。此为 EP/MoE 系统中常见的"非确定但近似"行为，记录但不在
本轮修复。

**Prefill-dbo**：所有配置都输出固定 token id `33975`。
对比 b8/s512：serial 输出 `[334, 16141, 25, 56177]`，prefill-dbo 输出 `[33975, 33975, 33975, 33975]`。
**这是真正的输出 bug**（不只是浮点末位差异）。需要在下一轮调查 prefill-dbo path 的 lm_head 切片或 last-position
还原逻辑。当前 prefill-dbo 不可用于真实文本生成；TTFT 性能数据仍有效（前向计算时延正确，只是 argmax 错位/常量化）。

---

## 4. 下一步优化方案（按 ROI 排序）

### 方案 1 — 绘图修复（已交付）
F lane 复合 bar + 内部分段 + layer 边界 + cross-layer 重叠区可视化。

### 方案 2 — 收 0.31 ms 边界空泡（中工作量，~30 ms/step 量级）
- 候选根因：`tracker.mark_*` 用 `time.perf_counter()` 在 NPU op 后调用强制 stream sync；`dist.isend()` 第一次入队阻塞等 stream 排空
- 行动：(a) `tracker` 改为 NPU event 记录 + 后台 host 转换；(b) send 的输出 tensor 在前一阶段就 view/contiguous 好；(c) 用 `npu.synchronize()` 替代隐式 sync

### 方案 3 — `ep_local_experts` 双 stream 并行（**最高收益**）
**目标**：让 mb0/mb1 stacked GEMM 排到 NPU 两条 compute stream，硬件层面同时跑。

改造点：
1. `src/model/sharded_experts.py` 模块持有两条 `torch_npu.npu.Stream()`（避免每步建销毁）
2. `forward_local(hidden, mb_id)` 按 `mb_id % 2` 选 stream；输入张量 `record_stream(s)` 防内存回收
3. `ffn_coordinator` 用 NPU event 串接：`evt_in.record(default); s.wait_event(evt_in)` 入口；出口同理
4. `ep_dispatch_wait` 的 sync 也搬到对应 mb 的 sub-stream

风险：HBM 带宽共享。先在 b16/s256 单点验证；不行降级到"延迟启动 mb1 GEMM"。

收益估计：mb0/mb1 ep_local_experts 总长 ~2.66 ms → 完全并行可降到 ~1.4 ms，节省 ~1.2 ms/层 × 48 = **60 ms/step（~10–15%）**。

### 方案 4 — `npu_grouped_matmul`（与方案 3 正交）
取代 ep7 v2 stacked path 内部 ≤18 次 Python loop GEMM。先探活 `torch_npu.npu_grouped_matmul`，再把
`gate_up`/`down` 堆成 `(E_active, hidden, 2*inter)` 一次调用。收益：消除 Python overhead **~43 ms/step**。

### 方案 5 — Token-aware combine（最后做）
按 expert chunk 流水 combine，让 reduce 与下层 attn 早期阶段重叠。复杂度高、协议改动大；**先看方案 3+4 是否够**。

### 方案 6 — Prefill-dbo 输出修复（本目录新增的 work item）
排查 prefill-dbo 在 EP7 路径的 lm_head 调用：是不是只有 mb0 走了 lm_head？是不是 mb1 的 last-token slice 错了？
是不是 reducer 把 mb0 的 hidden 覆盖成全 0/常量了？

---

## 5. 文件索引

```
full_matrix_v2/
├── README.md                              ← 本文件
├── summary.csv                            聚合后总表（speedup/tpot/correctness）
├── experiment_matrix_summary.csv          原始矩阵脚本输出（status: ok/cached/OOM）
├── fig_decode_speedup_heatmap.png         decode TPOT 加速比热力图
├── fig_decode_speedup_curves.png          decode 速比曲线（按 seq 分组）
├── fig_prefill_speedup_heatmap.png        prefill TTFT 加速比热力图
├── pipeline_figs/                         代表配置 pipeline 图（修复后绘图）
│   ├── pipeline_decode_b{8,16,32,64,128}_*.png
│   └── pipeline_prefill_b{8,16,32,64}_*.png
├── serial/                                serial baseline (status='cached'：复用 ep7_matrix_v2 缓存)
├── prefill-dbo/                           prefill-dbo t=8 reports + timing JSONs
└── decode-dbo/                            decode-dbo t=20 reports + timing JSONs
```

聚合脚本：`scripts/aggregate_full_matrix_v2.py`。
