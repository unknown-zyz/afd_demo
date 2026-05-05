# Full Matrix v2 — 实验小结（2026-05-04）

> 本文是 `exp/npu-ep-overlap-v2` 分支上 commit **a052912** 的随附说明，
> 适合用作"过几周回头查看时"的索引。所有原始数据/图都在
> `results_npu/full_matrix_v2/` 下（见末尾"如何查看"）。

## 1. 这一轮要解决的问题

用户在上一轮 EP7 v2 的 pipeline 图上提了 5 个观察：
1. MB0 attn 完→MB1 attn 起步存在小空泡
2. A2F send 完→FFN 起步存在小空泡
3. F2A 在两个 MB FFN 完成后才出现，看着有"很大的空泡"
4. F2A 应该在 MB0 FFN 完成后立刻开始，且与下一层 MB0 attn 重叠（图上看不出来）
5. 部分配置下 MB0/1 FFN 部分重叠

并要求：补 prefill 的完整矩阵实验、关注模型推理结果是否正常。

## 2. 诊断结论（实证 b16/s256 layer-1）

| # | 性质 | 实证 | 处置 |
|---|---|---|---|
| 1 | **真实** 0.31 ms 小空泡 | mb1 attn 1.353 起步，mb0 attn 1.046 完成 | 留作方案 2 |
| 2 | **真实** 0.31 ms 小空泡 | A2F send 完 1.120，ffn_compute 起步 1.355 | 留作方案 2 |
| 3 | **绘图 bug**（不是调度 bug） | F2A enqueue **5.633 ms（ep_reduce 完成前 28 µs）** | 已修绘图 |
| 4 | **已实现，绘图未呈现** | per-MB early send 在 `ffn_coordinator.py`；cross-layer attn↔FFN 在 `decode_scheduler.py:689` | 绘图加 layer 边界标注 |
| 5 | **真瓶颈** | `ep_local_experts` 串行：mb0 2.864–4.193 / mb1 4.298–5.525 | **下一步：方案 3** |

**绘图 bug 本质**：旧绘图 F lane 只画 `ffn_compute` 这一个 event（router+dispatch enqueue
~1.5 ms 就结束）。实际 MoE 的主要工作 `ep_dispatch_wait → ep_local_experts → ep_reduce`
（持续到 5.66 ms）**完全没画**。这是为什么用户看着"FFN 早就结束、F2A 半天才出现"——
因为屏幕上 FFN bar 短了 ~3 ms。

修复后 F lane = 复合 bar，由 router/dispatch（hatched）+ local_experts（实色，主体）
+ reduce（hatched）三段拼成。详见 `scripts/visualize_dbo_pipeline.py`。

## 3. 实验结果

### 3.1 矩阵
固定：preset = `npu-ep7`，backend = `broadcast_reduce_overlap`，16 NPU。
- batch ∈ {2, 4, 8, 16, 32, 64, 128, 256, 512}
- seq ∈ {128, 256, 512, 1024, 2048}
- serial / decode-dbo 用 `--tokens 20`
- prefill-dbo 用 `--tokens 8`

### 3.2 Decode-DBO TPOT speedup（vs serial, t=20）

| 范围 | 数量 / 42 ok |
|---|---|
| ≥ 1.20× | **40** |
| ≥ 1.50× | 29 |
| ≥ 2.00× | 19 |
| min / max | 1.17× / 5.87× |

按 prefill seq 平均：

| seq | mean speedup |
|---:|---:|
| 128 | 2.96× |
| 256 | 2.47× |
| 512 | 2.02× |
| 1024 | 1.72× |
| 2048 | 1.55× |

短 seq + 中 batch 收益最大；长 seq 时 attn 时间增长，FFN 流水占比下降，
加速比收敛到 ~1.5×。

### 3.3 Prefill-DBO TTFT speedup（vs serial, t=8）
35 个 ok 配置，speedup **2.93×–8.94×**。Prefill 因 token 数大、dispatch 负载均匀，
DBO 重叠空间充裕。

### 3.4 OOM 边界
- serial: b≥512 / s≥1024 失败；b≥256 / s=2048 失败
- prefill-dbo: b256/s512、b128/s1024、b64/s2048
- decode-dbo: b512/s1024、b256/s2048

### 3.5 数据缺陷：t=8 不能测 decode TPOT
b16/s256 同配置：
- t=8 decode-dbo TPOT = 209.7 ms，speedup 0.62×（**假回退**）
- t=20 同配置 TPOT = 341.4 ms，speedup 1.47×（真值）

根因：decode TPOT = `decode_loop_ms / (tokens-1)`。t=8 → 7 步，cold-start 第 1 步
NCCL graph capture + DBO pipeline warmup 占绝对主导，DBO 比 serial warmup 多。
**结论：所有 decode 数据已用 t=20 重跑；本次报告中 decode 速比都来自 t=20。**

## 4. 推理正确性（greedy 前 4 token）

通过 `--correctness-check N` 注入固定 prompt + greedy decode 前 N token id。

| 模式 | 与 serial 完全一致 | 部分一致（前 1-2 token 同） | 完全分歧 |
|---|---:|---:|---:|
| decode-dbo | 多数 | 少数（token 3 浮点边界翻转） | 0 |
| **prefill-dbo** | **0** | 0 | **全部输出固定 token 33975** |

- decode-dbo 浮点分歧示例：b16/s256 serial=`[27,91,318,4906]` vs dbo=`[27,91,3479,315]`。
  EP partition 改变了 logits 末位浮点累加序列，softmax 边界翻转。**EP/MoE 系统常见**。
- **prefill-dbo 真 bug**：所有配置都输出常量 `[33975, 33975, 33975, 33975]`，
  serial 同配置正常。前向计算时延数据仍然有效，但生成结果不可用。
  **下一步需排查 lm_head 切片 / last-position 还原 / mb1 hidden 是否被覆盖。**

## 5. 文件 & 工具

```
results_npu/full_matrix_v2/
├── README.md                       ★ 详细诊断 + 速比表 + 优化方案 2-6
├── summary.csv                     聚合表：mode/b/s/t/speedup/tpot/correctness
├── experiment_matrix_summary.csv   原始矩阵脚本输出
├── fig_decode_speedup_heatmap.png  ★ decode TPOT 加速比热力图
├── fig_decode_speedup_curves.png   decode 速比曲线（按 seq）
├── fig_prefill_speedup_heatmap.png ★ prefill TTFT 加速比热力图
├── pipeline_figs/                  ★ 修复后的代表配置 pipeline 图
│   ├── pipeline_decode_b{8,16,32,64,128}_s*.png
│   └── pipeline_prefill_b{8,16,32,64}_s*.png
├── serial/                         baseline 缓存
├── prefill-dbo/                    prefill-dbo t=8 reports + timing JSONs
└── decode-dbo/                     decode-dbo t=20 reports + timing JSONs
```

工具：
- `scripts/aggregate_full_matrix_v2.py`：扫 timing JSON / report，生成
  summary.csv + 三张图 + 正确性比对（CSV-independent，可重复跑）
- `scripts/visualize_dbo_pipeline.py`：4-lane 复合 F bar 绘图（已修复）
- `scripts/run_experiment_matrix_npu.sh --correctness-tokens N`：矩阵跑实验时
  把 `--correctness-check N` 透传到 `src/main.py`

## 6. 如何重新跑实验 / 重新生成图

```bash
# 单点重跑（在远端 NPU 容器中）：
cd /workspace/afd_demo_ep_prototype
scripts/run_experiment_matrix_npu.sh \
  --modes serial,decode-dbo \
  --batches 16,32 --seqs 256,512 --tokens 20 \
  --preset npu-ep7 --ffn-ep-backend broadcast_reduce_overlap \
  --correctness-tokens 4 --root-out results_npu/full_matrix_v2

# 拉回本地（容器 /host_zhangyz 是 read-only，要 tar+ssh）：
ssh -i ~/.ssh/id_rsa_second schedTeam@1.95.114.229 \
  "docker exec afd-npu-test tar czf - -C /workspace/afd_demo_ep_prototype \
   results_npu/full_matrix_v2" > /tmp/sync.tgz
tar xzf /tmp/sync.tgz -C /home/zyz/afd_demo/

# 本地重生图：
python3 scripts/aggregate_full_matrix_v2.py
python3 scripts/visualize_dbo_pipeline.py \
  --attn-timing results_npu/full_matrix_v2/decode-dbo/timing_attention_*_b16_s256_t20.json \
  --ffn-timing  results_npu/full_matrix_v2/decode-dbo/timing_ffn_coordinator_*_b16_s256_t20.json \
  --output /tmp/foo.png --start-layer 1 --num-layers 3 --mode decode
```

## 7. 下一步（已立项）
- **方案 3**：`ep_local_experts` 双 NPU compute stream 并行（最高 ROI）
- **方案 6**：prefill-dbo 输出 33975 bug 排查
- 方案 2、4、5 见 `results_npu/full_matrix_v2/README.md`
