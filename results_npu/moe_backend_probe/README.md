# NPU MoE Backend Probe

本目录用于下一阶段 NPU FFN/MoE backend 实验，独立于已有 `results_npu/{serial,prefill-dbo,decode-dbo,...}` baseline。

## 当前已完成 1：单层 MoE microbenchmark

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

## 当前已完成 2：torch_npu MoE API smoke

脚本：

- `scripts/probe_npu_moe_api.py`
- `scripts/plot_npu_moe_api_probe.py`

输出目录：

```text
api_smoke/
```

测试 API：

- `torch_npu.npu_moe_gating_top_k_softmax`
- `torch_npu.npu_moe_token_permute`
- `torch_npu.npu_moe_token_unpermute`
- `torch_npu.npu_grouped_matmul`

关键 latency 结果：

| Shape | HF experts (ms) | Router topk (ms) | Token permute roundtrip (ms) | Grouped experts (ms) | Grouped / HF |
|---|---:|---:|---:|---:|---:|
| decode `32x1` | 4.616 | 0.198 | 0.255 | 50.933 | 11.0x |
| decode `64x1` | 4.828 | 0.100 | 0.199 | 53.791 | 11.1x |
| decode `128x1` | 5.167 | 0.100 | 0.201 | 57.540 | 11.1x |
| prefill `16x512` | 19.071 | 0.120 | 0.872 | 112.421 | 5.9x |
| prefill `32x512` | 34.745 | 0.151 | 1.528 | 170.130 | 4.9x |

正确性：

- Router topk 与 HF topk 完全一致，routing weight mean abs error 约 `2e-4`。
- Token permute/unpermute roundtrip 可用，decode 误差为 0，prefill mean abs error 在 `1e-7` 量级。
- Grouped experts 输出与 HF experts bf16 结果接近，mean abs error 约 `5e-4`。

结论：

- `npu_moe_gating_top_k_softmax` 和 token permute API 可用，但它们不是主瓶颈。
- `npu_grouped_matmul(group_type=-1)` 在当前 list-of-experts 调用方式下 correctness 通过，但 latency 明显劣于 HF experts，不应接入主 pipeline。
- 当前不实现 `AFD_NPU_MOE_BACKEND=torch_npu_fused` 主路径，避免引入确定的负优化。

位于 `api_smoke/`：

- `torch_npu_moe_api_probe.json`
- `correctness_summary.csv`
- `api_latency_comparison.png`
- `correctness_error.png`

## 下一步

## 当前已完成 3：decode NPUGraph / aclgraph 探测

脚本：

- `scripts/probe_npu_graph_ffn.py`
- `scripts/plot_npu_graph_ffn.py`

输出目录：

```text
graph_probe/
```

测试对象：

- `experts_only`：固定 `hidden_2d + selected_experts + routing_weights`，只捕获 HF experts。
- `full_ffn`：捕获 `post_attention_layernorm + gate + experts + residual add`。
- decode fixed shapes：`32x1`、`64x1`、`128x1`。
- `capture_error_mode=relaxed`。

结果：

| Shape | Probe | Eager median (ms) | Graph replay | 结论 |
|---|---|---:|---|---|
| decode `32x1` | experts_only | 4.433 | capture failed | 不可用 |
| decode `32x1` | full_ffn | 4.708 | capture failed | 不可用 |
| decode `64x1` | experts_only | 4.983 | capture failed | 不可用 |
| decode `64x1` | full_ffn | 5.239 | capture failed | 不可用 |
| decode `128x1` | experts_only | 5.117 | capture failed | 不可用 |
| decode `128x1` | full_ffn | 5.639 | capture failed | 不可用 |

失败原因：

- HF Qwen3 MoE experts 内部在 graph capture stream 上触发 host/device 同步拷贝。
- 即使使用 `capture_error_mode=relaxed`，仍报 `ACL stream synchronize failed, error code:107027` / captured-stream synchronize 不允许。
- 因此当前 HF experts 路线不能直接通过 NPUGraph/aclgraph 获得 replay 加速。

位于 `graph_probe/`：

- `decode_graph_probe.json`
- `decode_graph_probe.csv`
- `graph_latency_comparison.png`
- `graph_speedup.png`

## 下一步

1. 单独开 `exp/npu-ep-prototype` 做真实 expert-parallel 拓扑，而不是继续 list grouped matmul 或 HF graph capture 路线。
2. 最小原型建议先做 `1 attention rank + 2 ffn ranks`，FFN 按 expert 分片而不是按 layer 分片。
3. 若后续发现更合适的 single-tensor grouped matmul 或 fused MoE 调用方式，再重新进入 fused backend 原型。
