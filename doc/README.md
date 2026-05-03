# 文档目录

本目录保存当前维护文档。文件名保持英文，主题文档统一使用两位数字编号。
命令、参数、类名、字段名保留英文原文，说明文字统一使用中文。

## 推荐阅读顺序

| 顺序 | 文档 | 内容 |
|---:|---|---|
| 1 | [01-architecture.md](01-architecture.md) | AFD/DBO 架构、A/F/EP 拆分、NPU EP overlap、token-aware 设计、KV cache、CUDA/NPU backend。 |
| 2 | [02-usage.md](02-usage.md) | Serial、prefill DBO、decode DBO、crosslayer 和矩阵实验命令。 |
| 3 | [03-api-reference.md](03-api-reference.md) | 当前公开代码接口和脚本接口。 |
| 4 | [04-deployment.md](04-deployment.md) | GPU local、GPU multinode、Ascend 910C 容器部署。 |
| 5 | [05-code-review-guide.md](05-code-review-guide.md) | 审查 timing、distributed、scheduler 和结果可信度。 |
| 6 | [06-npu-910c-adaptation.md](06-npu-910c-adaptation.md) | 910C / HCCL 适配、验证拓扑、已知限制。 |
| 7 | [07-npu-vs-gpu-experiment-analysis.md](07-npu-vs-gpu-experiment-analysis.md) | TTFT/TPOT 口径、旧 NPU 5x 误判原因、baseline audit。 |
| 8 | [08-gpu-npu-experiment-summary.md](08-gpu-npu-experiment-summary.md) | 最新 GPU/NPU 矩阵覆盖率、speedup、OOM 边界和结论。 |

根目录 [`README.md`](../README.md) 是项目入口；[`scripts/README.md`](../scripts/README.md)
是脚本索引。

## 快速定位

| 需求 | 阅读 |
|---|---|
| 理解 A/F 分离与 DBO 的执行方式 | [01-architecture.md](01-architecture.md) |
| 跑一个 serial baseline | [02-usage.md](02-usage.md) |
| 跑 prefill DBO | [02-usage.md](02-usage.md) |
| 跑 decode DBO 或 crosslayer | [02-usage.md](02-usage.md) |
| 跑 GPU 全矩阵 | [02-usage.md](02-usage.md) |
| 跑 NPU 全矩阵 | [02-usage.md](02-usage.md) |
| 部署 GPU 多机 | [04-deployment.md](04-deployment.md) |
| 使用 910C 容器 | [04-deployment.md](04-deployment.md) |
| 判断 speedup 是否可信 | [07-npu-vs-gpu-experiment-analysis.md](07-npu-vs-gpu-experiment-analysis.md) |
| 查看最新实验结论 | [08-gpu-npu-experiment-summary.md](08-gpu-npu-experiment-summary.md) |
| 理解 NPU EP4/EP7 的探索过程 | [01-architecture.md](01-architecture.md) |

## 当前实验结论摘要

- Speedup 统一为 `serial / DBO`，大于 `1.0x` 才表示 DBO 更快。
- Prefill DBO 使用模型侧 TTFT-path：`serial_prefill_ms / dbo_total_time_ms`。
- Decode DBO 和 crosslayer 使用准确 TPOT：`serial_decode_tpot_ms / dbo_decode_tpot_ms`。
- Decode DBO 的 pipeline 明细来自 0-based decode step 1，只用于观察 overlap，不用于最终加速比。
- 旧的 “NPU decode DBO 约 5x 加速” 是口径误用导致的历史结论，不能继续引用。
- 当前 fresh rerun 中，最稳定的正收益来自 NPU prefill DBO；GPU DBO 和 NPU decode DBO 的中位数都低于 `1.0x`。
- NPU EP overlap 已找到一个 decode 正收益点：EP7 b16/s512/t20，详见 `results_npu/ep_overlap/README.md`；EP4 sync 负结果保留在 `results_npu/ep4_broadcast_reduce_sync/`。

## 维护原则

1. 文档中的命令必须能对应当前脚本参数。
2. 结论必须注明使用的指标口径。
3. 发布 speedup 前必须确认 `baseline_audit.csv` 为 `ok`。
4. OOM 是容量边界，不是缺失数据。
5. 修改 `doc/` 文件名时必须同步更新根 README、本文档和所有内部链接。
