# full-matrix-v2 QA / Review 说明（精简版）

本文保留 `results_npu/full_matrix_v2/` 实验中未在其他文档覆盖的关键发现和优化方案。
详细的 pipeline 图解读、t=8 vs t=20 口径、crosslayer 说明和 profile 代码走查见
`01-architecture.md`、`02-usage.md`、`scripts/README.md` 和 `experiment_archive.md`。

---

## 1. prefill-dbo 推理 bug

**状态**：待排查。`prefill-dbo` 有真实输出 bug，不能用于文本生成；TTFT 性能数据仍可用于时延趋势分析。

现象：多个 prefill-dbo 配置都输出固定 token id `33975`，serial 同配置正常。

可能根因：

1. last-token slice 错位
2. micro-batch 合并错误
3. lm_head 输入错误
4. combine/reduce 覆盖
5. attention_mask / position_ids 不匹配

---

## 2. 核心结论

1. `decode-dbo` 主结果使用 t=20（t=8 只有 7 个 decode step，cold-start 占比过高，不适合 TPOT 结论）。
2. 旧 pipeline 图中 F2A 大空泡主要是绘图漏画 MoE 主体；修复后 F lane 看 composite bar。
3. `ep_local_experts` GEMM 主体在单 NPU compute stream 上基本串行；MB0/MB1 router/dispatch 轻微重叠是合理的。
4. 双 stream 方案已验证无收益（910C MoE GEMM 受 HBM 带宽限制，两条 stream 竞争同一内存子系统）。

---

## 3. 方案 4：`npu_grouped_matmul` / fused-MoE（下一步最高 ROI）

目标：减少 Python loop、kernel launch 和不必要的 HBM 往返。

当前瓶颈：`forward_local` 按 active expert 循环执行 stacked GEMM，每层每 MB 有多次小 GEMM。

方案：

1. 保留现有 routing / sort / counts 逻辑
2. 构造 `group_list` 表示每个 active expert 的 token 数
3. 用 `npu_grouped_matmul` 一次完成所有 active expert 的 gate/up GEMM
4. 执行 SiLU * up
5. 再用 `npu_grouped_matmul` 一次完成 down GEMM
6. 保持原有 `index_add_` combine

预期收益：中高。如果 `ep_local_experts` 从 1-3 ms/层降低 15%-30%，decode TPOT 可能有可见收益。

风险：weight layout、group_list 语义、dtype/shape 限制、correctness 验证。

---

## 4. 方案 5：Token-aware combine / reduce-F2A overlap（方案 4 之后）

目标：不等整个 MB 所有 expert 完成后再整体 combine，而是按 token chunk 推进，让 F2A 更早发送。

预期收益：中等，依赖方案 4 后 FFN 主体是否仍是瓶颈。

建议执行顺序：先做方案 4 → 复测 F2A recv-wait → 再考虑方案 5。

---

## 5. 绘图修复备忘

- 修复后 F lane 默认使用 staged 视图（`F/router`、`F/dispatch`、`F/local_experts`、`F/combine`）
- A2F bar 跨度改为 `[ATT.send.start, FFN.recv_wait.end]`（接收方拿到数据时刻）
- `ep_overlap_hidden` 不再单独画，combine 整段在 F2A 通信泳道
- `--ffn-view legacy` 保留旧 composite F lane 用于前后对比
