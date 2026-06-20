# AttentionWorker NPU 优化进展

## 总结

当前分支：`feat/attn-worker-npu-optimizations`。

本文结论全部来自 Ascend 910C NPU；GPU 结果不作为验收依据。

核心结论：**Attention 单层已拿到主要收益，端到端 decode-DBO 的主瓶颈已经转向 FFN、A/F 通信和 pipeline overlap。**

### 稳定判断

- Decode 侧 `official IFA + fused RMSNorm/RoPE` 是当前最优单层路径（9 个 shape 约 `0.632-0.659ms`，比 HF `0.716-1.051ms` 更稳定）。
- Prefill 侧 official PFA core 本身未转正，但 `official + RMSNorm/RoPE fusion` 在中/大 shape 上优于 HF（例如 b32/s1024 `14.733ms vs 17.755ms`）。
- 社区 `flash-attention-npu` 明显慢于官方 IFA，且不支持 RoPE 融合。
- precopy 是低风险优化，主要改善小 shape 的重复跨设备迁移。
- 单层 Attention 收益没有直接转化为 TPOT（b1 约 147ms，b8 约 326-363ms，b32 约 701-755ms），主瓶颈在 FFN/通信/pipeline。
- MoE distribute：当前容器 CANN 8.5.0 + torch_npu 2.6 的 v2 不暴露 `comm_alg`；隔离容器 CANN 8.5.1 + torch_npu 2.9 已暴露且 `dispatch_v2/combine_v2` 可跑通。

## 已实现内容

| 项目 | 状态 |
|---|---|
| 优化配置与 timing 元数据 | 已实现 |
| Layer 输入预拷贝 | 已实现 |
| 官方 NPU attention 适配 | 已实现 |
| RMSNorm / RoPE 融合 | 已实现 |
| msprof 工作流 | 已实现 |
| 预留 NPU 资源池 | 已实现 |
| NPU stream overlap | 已实现（静态 A/F 路径） |
| Attention TP | 未实现（blocked，需先完成拓扑设计） |

## 关键 Benchmark 数据

### Decode core：HF/SDPA vs official IFA（ms/layer，48 层中位数）

| Batch | Cache | HF/SDPA | official IFA | speedup |
|---:|---:|---:|---:|---:|
| 1 | 128 | 0.049 | 0.046 | 1.07x |
| 8 | 512 | 0.088 | 0.047 | 1.87x |
| 32 | 1024 | 0.431 | 0.078 | 5.53x |

### Decode full-layer：HF vs official+fusion（ms/layer，48 层中位数）

| Batch | Cache | HF | official+RMSNorm/RoPE | speedup |
|---:|---:|---:|---:|---:|
| 1 | 128 | 0.737 | 0.646 | 1.14x |
| 8 | 512 | 0.727 | 0.644 | 1.13x |
| 32 | 1024 | 1.051 | 0.649 | 1.62x |

### 真实 decode-DBO TPOT（`--attn-kernel npu-official --attn-fused-rmsnorm --attn-fused-rope`，3 次中位数）

| Batch | Seq | Decode TPOT | Prefill |
|---:|---:|---:|---:|
| 1 | 128 | 147.9 ms | 1505.8 ms |
| 8 | 512 | 325.7 ms | 2130.3 ms |
| 32 | 512 | 755.0 ms | 3480.6 ms |

## 剩余 blocked 待办

| 待办 | 为什么 blocked | 优先级 |
|---|---|---|
| Attention TP | 需先完成拓扑设计、Q/K/V/O 切分与 KV cache ownership 设计 | 中 |
| MoE dispatch real routing benchmark | 当前容器 torch_npu 2.6 的 v2 不暴露 `comm_alg` | 中 |
| 单机端到端 TPOT 转正 | FFN/通信/pipeline 瓶颈未解决，跨机只会放大问题 | 高 |

详细 benchmark 原始数据见 `results_npu/attention_scope_aligned/`。
