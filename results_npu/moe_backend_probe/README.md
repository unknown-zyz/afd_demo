# NPU MoE Backend Probe

本目录用于下一阶段 NPU FFN/MoE backend 实验，独立于已有 `results_npu/{serial,prefill-dbo,decode-dbo,...}` baseline。

## 当前已完成：单层 MoE microbenchmark

实验分支：`exp/npu-moe-backend`

脚本：

- `scripts/bench_npu_moe_layer.py`
- `scripts/plot_npu_moe_microbench.py`

远端环境：

- 机器：910C
- 容器：`afd-npu-test`
- worktree：`/workspace/afd_demo_moe_backend`
- 模型：`/models/Qwen3-30B-A3B`
- device：`npu:0`
- dtype：`bfloat16`

测试对象：

- Qwen3 MoE layer 24
- repeat=3，warmup=1

## 关键结果

| Shape | Tokens | Total median (ms) | Router median (ms) | Experts median (ms) | Experts 占比 |
|---|---:|---:|---:|---:|---:|
| decode `32x1` | 32 | 4.747 | 0.178 | 4.278 | 90.1% |
| decode `64x1` | 64 | 4.925 | 0.173 | 4.522 | 91.8% |
| decode `128x1` | 128 | 5.109 | 0.164 | 4.725 | 92.5% |
| prefill `16x512` | 8192 | 19.519 | 0.233 | 18.879 | 96.7% |
| prefill `32x512` | 16384 | 35.588 | 0.292 | 34.570 | 97.1% |

结论：

- FFN/MoE 单层耗时几乎都在 `experts` 阶段，router 只占很小比例。
- decode 小 token 场景下 total 约 4.7-5.1ms，与完整 pipeline 中 FFN avg/layer 约 5.8-5.9ms 同量级。
- prefill 大 token 场景下 experts 占比进一步上升，说明下一步应优先优化 expert loop / grouped matmul，而不是 router 或 layer placement。

## 产物

位于 `microbench/`：

- `moe_layer_baseline.json`
- `moe_layer_baseline.csv`
- `moe_layer_baseline.png`
- `moe_stage_breakdown.png`
- `moe_shape_scaling.png`

## 下一步

基于这些真实 shape，继续验证：

1. `torch_npu.npu_moe_gating_top_k_softmax` 是否可替换 router。
2. `torch_npu.npu_grouped_matmul` 是否能替换 HF Qwen3 当前 Python expert loop。
3. 若 correctness 通过，再接入显式开启的 `AFD_NPU_MOE_BACKEND=torch_npu_fused` 原型。
