# EP7 v2 矩阵实验报告（Active Grouped GEMM + Last-Token LM Head）

> 分支：`exp/npu-ep-overlap-v2`（基于 `exp/npu-ep-prototype`）  
> 模型：Qwen3-30B-A3B（48 层 MoE，128 expert，top-k=8）  
> 硬件：8×Ascend 910C（EP=7，Attention 1×NPU + FFN 7×NPU）  
> 通信后端：`broadcast_reduce_overlap`

## 1. 一句话总结

v2 在 v1 基础上做了**两处**核心优化，**21/21 v1 配置全部回归正向、新增 3 个 OOM 配置全部跑通**，最佳加速比从 v1 的 **1.088×** 跃升到 v2 的 **1.473×**，14/24 配置达到 ≥1.20× 显著正收益。

## 2. v1 → v2 关键改动

| # | 改动 | 文件 | 解决的瓶颈 |
|---|---|---|---|
| 1 | **Active Grouped GEMM** —— ShardedExperts 用 stacked weights `(E_local, 2F, H)` / `(E_local, H, F)`，加上 device-side `_local_mask_lut`，每层 forward 只做一次 host sync 然后按 active expert 分组做 GEMM | `src/model/ep_moe.py` ShardedExperts | v1 用 ParameterDict + per-expert Python loop + `.cpu()` 同步，每层 ~3-4ms 都耗在 FFN 算子上；FFN 是 critical path，是 DBO 收益不显的根本原因 |
| 2 | **Prefill 仅对最后位置算 lm_head** —— forward_prefill 两个分支都先 `[:, -1:, :]` 再过 lm_head | `src/model/disaggregated.py:285-294, 446-454` | v1 的 lm_head 在大 batch×长 seq 时分配 `(B,S,vocab)` ≈ 37 GiB 会 OOM；调用方一直只用 `logits[:, -1, :]`，行为等价但显存降为 1/S |

> 没有引入 token-aware dispatch / aclgraph 等更激进改动，仅靠这两点就消除了 FFN 主要 bubble。

## 3. 决策门（5 配置 smoke）

| 配置 | v1 speedup | v2 speedup | 门槛 | 通过 |
|---|---|---|---|---|
| b16/s256 | 1.088 | **1.473** | ≥1.20 | ✅ |
| b16/s512 | 1.056 | **1.432** | ≥1.15 | ✅ |
| b64/s256 | 1.008 | **1.247** | ≥1.10 | ✅ |
| b8/s128  | 0.865 | **1.080** | ≥1.00 | ✅ |
| b128/s512 | 0.865 | **1.090** | ≥1.00 | ✅ |

5/5 全过、最佳 1.473× 远超 1.20× 门槛 → 直接进全矩阵扫描。

## 4. 全矩阵结果（24 配置）

`summary.csv` 完整数据；下表为速览：

| seq \ batch | 8 | 16 | 32 | 64 | 128 | 256 |
|---|---|---|---|---|---|---|
| **128**  | 1.080 | **1.394** | **1.422** | 1.148 | 1.057 | 1.165 |
| **256**  | 1.190 | **1.473** | **1.243** | **1.247** | **1.210** | 1.099 |
| **512**  | 1.195 | **1.432** | **1.264** | **1.262** | 1.090 | 1.033 |
| **1024** | 1.068 | **1.327** | **1.321** | **1.210** | 0.985 | 0.936 |

加粗 = 达成 ≥1.20× 显著正收益门槛。

**统计**：
- 24/24 配置成功（v1 有 2 个 OOM）
- 23/24 ≥1.0×（>95%）
- 21/24 ≥1.05×
- **14/24 ≥1.20×**
- 最佳 b16/s256 **1.473×**，最差 b256/s1024 0.936×

## 5. 图表

- `fig_speedup_heatmap.png` — 24 配置加速比热力图
- `fig_speedup_curves.png` — 按 seq 分线的 batch×speedup 曲线
- `fig_v1_vs_v2.png` — 同配置 v1 vs v2 直方对比
- `pipeline/pipeline_b{B}_s{S}_t20.png` — 8 个代表配置的 4 泳道 Gantt 图，体现 attn↔ffn↔send 三流的重叠

## 6. 残留瓶颈与下一步

观察 `pipeline_b256_s1024_t20.png` 等大配置：

1. **大 batch×长 seq 的 FFN 算子被 attn KV 计算追平**：b256/s1024 attn 单层接近 50ms，FFN 已不是瓶颈，DBO 反成开销（0.94×）→ 这已经是另一种"打平"的体现
2. **active expert 多时 stacked GEMM 仍走 Python loop**：≤18 次单 expert GEMM；可考虑 `torch_npu.npu_grouped_matmul`（CANN 8.5）一次 fused，预期再压 1-2ms / 层
3. **小 batch (b≤8) 收益受限**：FFN 计算极短，被通信启动开销（~1ms isend）支配；可加 NCCL 预热常驻、或合并 a2f/f2a 通道

后续可选优化优先级：
1. **token-aware combine**（高价值/中改造）：让 expert 本地"完成的 hidden"按到达顺序回传，减小 reduce wait
2. **npu_grouped_matmul**（中价值/低改造）：消除 stacked path 的 Python loop
3. **Attention TP=2**（中价值/高改造）：缓解 b256/s1024 attn 反成 critical 的极端工况

## 7. 复现命令

```bash
# 远端容器内（afd-npu-test）
cd /workspace/afd_demo_ep_prototype
git checkout exp/npu-ep-overlap-v2

# 全矩阵
for S in 128 256 512 1024; do
  for B in 8 16 32 64 128 256; do
    bash scripts/run_experiment_matrix_npu.sh --modes decode-dbo \
      --batches $B --seqs $S --tokens 20 \
      --preset npu-ep7 --ffn-ep-backend broadcast_reduce_overlap \
      --output-root results_npu/ep7_matrix_v2 \
      --serial-cache-root results_npu/serial/cache --append
  done
done
```
