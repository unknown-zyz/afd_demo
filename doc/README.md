# 文档目录

本目录保存当前维护文档。文件名保持英文，主题文档统一使用两位数字编号。
命令、参数、类名、字段名保留英文原文，说明文字统一使用中文。

## 推荐阅读顺序

| 顺序 | 文档 | 内容 |
|---:|---|---|
| 1 | [25-ea-disaggregated-moe-system-design.pdf](25-ea-disaggregated-moe-system-design.pdf) / [Typst 源文件](25-ea-disaggregated-moe-system-design.typ) | 中文系统设计文档：E/A 分离、去中心化协同调度、通信时延掩盖、技术先进性与可行性。 |
| 2 | [01-architecture.md](01-architecture.md) | AFD/DBO 架构、A/F/EP 拆分、NPU EP overlap、token-aware 设计、KV cache、CUDA/NPU backend。 |
| 3 | [02-usage.md](02-usage.md) | Serial、prefill DBO、decode DBO、crosslayer 和矩阵实验命令。 |
| 4 | [03-api-reference.md](03-api-reference.md) | 当前公开代码接口和脚本接口。 |
| 5 | [04-deployment.md](04-deployment.md) | GPU local、GPU multinode、Ascend 910C 容器部署。 |
| 6 | [05-code-review-guide.md](05-code-review-guide.md) | 审查 timing、distributed、scheduler 和结果可信度。 |
| 7 | [06-npu-910c-adaptation.md](06-npu-910c-adaptation.md) | 910C / HCCL 适配、验证拓扑、已知限制。 |
| 8 | [07-npu-vs-gpu-experiment-analysis.md](07-npu-vs-gpu-experiment-analysis.md) | TTFT/TPOT 口径、旧 NPU 5x 误判原因、baseline audit。 |
| 9 | [08-gpu-npu-experiment-summary.md](08-gpu-npu-experiment-summary.md) | 最新 GPU/NPU 矩阵覆盖率、speedup、OOM 边界和结论。 |
| 10 | [23-communication-architecture-comparison.md](23-communication-architecture-comparison.md) | 项目内通信架构对比：broadcast/reduce、all-to-all、sparse P2P、DeepEP、official MoE v2。 |
| 11 | [24-msprof-ep-communication-analysis.md](24-msprof-ep-communication-analysis.md) | EP16 通信 msprof、HCCL op、带宽下界与瓶颈分析。 |
| 12 | [branch_consolidation.md](branch_consolidation.md) | 分支合入/删除建议与主线合并依据。 |
| 13 | [experiment_archive.md](experiment_archive.md) | MB4、controller、dual-stream、MoE backend 等实验结论归档。 |
| 14 | [10-npu-910c-container-deployment.md](10-npu-910c-container-deployment.md) | 910C 远程容器创建、环境部署、代码同步与冒烟流程。 |
| 15 | [13-deepep-install-test-error-guide.md](13-deepep-install-test-error-guide.md) | DeepEP 安装、测试、报错与 Fallback 路径说明。 |
| 16 | [14-communication-modes.md](14-communication-modes.md) | Coordinator 通信方式、默认策略与切换方法。 |
| 17 | [16-routing-load-balancing-plan.md](16-routing-load-balancing-plan.md) | Coordinator routing / 负载均衡现状、限制、实验顺序与后续计划。 |
| 18 | [18-crosshost-coordinator-1a7f-architecture.md](18-crosshost-coordinator-1a7f-architecture.md) | 当前 cross-host 1A7F coordinator 架构、orchestrator、metrics flow 与失败分层。 |
| 19 | [19-code-review-system-architecture.md](19-code-review-system-architecture.md) | 面向代码审查的系统架构、类交互、调用链、数据流和 Review 关注点。 |

根目录 [`README.md`](../README.md) 是项目入口；[`scripts/README.md`](../scripts/README.md)
是脚本索引。

## 快速定位

| 需求 | 阅读 |
|---|---|
| 了解项目总体设计思想、技术先进性和可行性 | [25-ea-disaggregated-moe-system-design.pdf](25-ea-disaggregated-moe-system-design.pdf) |
| 理解 A/F 分离与 DBO 的执行方式 | [01-architecture.md](01-architecture.md) |
| 跑一个 serial baseline | [02-usage.md](02-usage.md) |
| 跑 prefill DBO | [02-usage.md](02-usage.md) |
| 跑 decode DBO 或 crosslayer | [02-usage.md](02-usage.md) |
| 跑 GPU 全矩阵 | [02-usage.md](02-usage.md) |
| 跑 NPU 全矩阵 | [02-usage.md](02-usage.md) |
| 部署 GPU 多机 | [04-deployment.md](04-deployment.md) |
| 使用 910C 容器 | [10-npu-910c-container-deployment.md](10-npu-910c-container-deployment.md) |
| 排查 DeepEP / fallback 数据面 | [13-deepep-install-test-error-guide.md](13-deepep-install-test-error-guide.md) |
| 切换 Coordinator 通信方式 | [14-communication-modes.md](14-communication-modes.md) |
| 理解 routing / 负载均衡当前到底做到哪一步 | [16-routing-load-balancing-plan.md](16-routing-load-balancing-plan.md) |
| 快速 Review 代码结构、类交互和调用链 | [19-code-review-system-architecture.md](19-code-review-system-architecture.md) |
| 对比当前可选 EP/MoE 通信架构 | [23-communication-architecture-comparison.md](23-communication-architecture-comparison.md) |
| 查看 EP16 通信 msprof 与带宽瓶颈 | [24-msprof-ep-communication-analysis.md](24-msprof-ep-communication-analysis.md) |
| 判断 speedup 是否可信 | [07-npu-vs-gpu-experiment-analysis.md](07-npu-vs-gpu-experiment-analysis.md) |
| 查看最新实验结论 | [08-gpu-npu-experiment-summary.md](08-gpu-npu-experiment-summary.md) |
| 理解 NPU EP4/EP7 的探索过程 | [01-architecture.md](01-architecture.md) |
| 判断哪些实验分支可以删除 | [branch_consolidation.md](branch_consolidation.md) |
| 查负结果实验结论 | [experiment_archive.md](experiment_archive.md) |
| 理解当前跨机 1A7F coordinator 架构和 full matrix 结果 | [18-crosshost-coordinator-1a7f-architecture.md](18-crosshost-coordinator-1a7f-architecture.md) |

## 当前实验结论摘要

- Speedup 统一为 `serial / DBO`，大于 `1.0x` 才表示 DBO 更快。
- Prefill DBO 使用模型侧 TTFT-path：`serial_prefill_ms / dbo_total_time_ms`。
- Decode DBO 和 crosslayer 使用准确 TPOT：`serial_decode_tpot_ms / dbo_decode_tpot_ms`。
- Decode DBO 的 pipeline 明细来自 0-based decode step 1，只用于观察 overlap，不用于最终加速比。
- 旧的 “NPU decode DBO 约 5x 加速” 是口径误用导致的历史结论，不能继续引用。
- Attention 单层 NPU official IFA、RMSNorm/RoPE fusion、layer input precopy、msprof workflow 和 reserved NPU 能力已落地，端到端 decode TPOT 是否转正仍受 FFN/通信/pipeline 影响。
- 当前真实 EPFFN decode 路径建议继续以 `broadcast_reduce_overlap` 作为默认对照；`all_to_all_single` 功能可用但当前 HCCL all-to-allv 固定开销较高，不应默认启用。
- `sparse_p2p_overlap`、DeepEP、official MoE v2 都是后续演进候选，必须先在 NPU 真实路径通过正确性、稳定性和 TPOT 门槛。

## 维护原则

1. 文档中的命令必须能对应当前脚本参数。
2. 结论必须注明使用的指标口径。
3. 发布 speedup 前必须确认 `baseline_audit.csv` 为 `ok`。
4. OOM 是容量边界，不是缺失数据。
5. 修改 `doc/` 文件名时必须同步更新根 README、本文档和所有内部链接。
