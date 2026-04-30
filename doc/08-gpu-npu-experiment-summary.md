# 08. GPU / NPU DBO 实验总结

本文总结分支清理和准确 TTFT/TPOT 口径修正后的最新全量重跑。Speedup 统一为
`serial / DBO`，大于 `1.0x` 表示 DBO 更快。

## 1. 分支与产物

| 平台 | 分支 / 来源 | 结果目录 | 说明 |
|---|---|---|---|
| GPU | `main` @ `49bb17c` | `results/` | fresh default、high-batch、long-seq matrix；图表和 audit 已重新生成。 |
| Ascend 910C NPU | `npu` after `5b424a6` | `results_npu/` | 在长期容器 `afd-npu-test` 中完成 fresh matrix，结果拷回本地生成图。 |

关键文件：

| 文件 | 用途 |
|---|---|
| `results/experiment_matrix_summary.csv` | GPU 合并矩阵状态。 |
| `results/baseline_audit.csv` | GPU mode-matched baseline audit。 |
| `results_npu/experiment_matrix_summary.csv` | NPU 合并矩阵状态。 |
| `results_npu/baseline_audit.csv` | NPU mode-matched baseline audit。 |
| `results_npu/*/pipelines_index.md` | NPU pipeline 图索引。 |

## 2. 指标定义

| 模式 | 服务指标 | DBO timing 字段 | Serial baseline 字段 |
|---|---|---|---|
| `prefill-dbo` | 模型侧 TTFT / TTFT-path | `total_time_ms` | `prefill_ms` |
| `decode-dbo` | 准确 TPOT | `decode_tpot_ms` | `decode_tpot_ms` |
| `decode-dbo-crosslayer` | 准确 TPOT | `decode_tpot_ms` | `decode_tpot_ms` |

Pipeline Gantt 图仍然展示 representative layer / step 事件，但所有报告和 audit
speedup 都使用上表中的准确 TTFT/TPOT 字段。

## 3. 矩阵覆盖率

### 3.1 GPU

| 模式 | OK / cached | OOM | 说明 |
|---|---:|---:|---|
| `serial` | 45（27 ok + 18 cached） | 5 | `b256` 在 `seq=128/256/512/1024/2048` 均 OOM。 |
| `prefill-dbo` | 20 | 7 | OOM 边界随 sequence length 变紧。 |
| `decode-dbo` | 45 | 5 | OK 到 `b192`；`b256` 在所有测试 seq 上 OOM。 |
| `decode-dbo-crosslayer` | 45 | 5 | 容量边界与 decode DBO 相同。 |

GPU baseline audit：**110 / 110 OK**，没有 fallback 或缺失 baseline。

### 3.2 NPU 910C

| 模式 | OK | OOM | 说明 |
|---|---:|---:|---|
| `serial` | 45 | 5 | 所有测试 seq 到 `b512` OK；`b1024` 全部 OOM。 |
| `prefill-dbo` | 25 | 8 | OOM 边界：`b256/s128`、`b128/s256`、`b64/s512`、`b32/s1024`、`b16/s2048`；额外 `b512` probes 在 `s128/256/512` OOM。 |
| `decode-dbo` | 45 | 1 | 所有测试 seq 到 `b512` OK；`b1024/s128` OOM。 |
| `decode-dbo-crosslayer` | 45 | 1 | 所有测试 seq 到 `b512` OK；`b1024/s128` OOM。 |

NPU baseline audit：**115 / 115 OK**，没有 fallback 或缺失 baseline。
Pipeline 图：**115 / 115 已生成**。图在本地生成，因为 NPU 容器没有安装
`matplotlib`。

## 4. 加速比总结

| 平台 | 模式 | 样本数 | Median | Min | Max | 解读 |
|---|---|---:|---:|---:|---:|---|
| GPU | `prefill-dbo` TTFT-path | 20 | 0.50x | 0.31x | 0.72x | fresh GPU prefill DBO 慢于 serial。 |
| GPU | `decode-dbo` TPOT | 45 | 0.75x | 0.63x | 1.07x | 大多慢于 serial，少数接近或略高于 `1.0x`。 |
| GPU | `decode-dbo-crosslayer` TPOT | 45 | 0.75x | 0.64x | 1.09x | crosslayer 没有改变 GPU 中位结论。 |
| NPU | `prefill-dbo` TTFT-path | 25 | 3.32x | 1.14x | 5.18x | fitting configs 中 NPU prefill DBO 稳定快于 serial。 |
| NPU | `decode-dbo` TPOT | 45 | 0.85x | 0.70x | 1.16x | 中位数慢于 serial，少数配置略有正收益。 |
| NPU | `decode-dbo-crosslayer` TPOT | 45 | 0.85x | 0.70x | 1.18x | 与 decode DBO 类似，没有广泛中位收益。 |

## 5. 容量边界

| 平台 | 模式 | 最大完全 OK batch | 首个 OOM probes |
|---|---|---|---|
| GPU | `serial` | `b192` for `s1024/2048`；default/high-batch grids 在 `b256` OOM | `b256` for all tested seqs。 |
| GPU | `prefill-dbo` | `b64/s128`、`b64/s256`、`b32/s512`、`b8/s1024`、`b4/s2048` | 下一档 batch OOM。 |
| GPU | `decode-dbo` | high/long seq grids 到 `b192` | `b256` for all tested seqs。 |
| GPU | `decode-dbo-crosslayer` | high/long seq grids 到 `b192` | `b256` for all tested seqs。 |
| NPU | `serial` | 所有测试 seq 到 `b512` | `b1024` for all tested seqs。 |
| NPU | `prefill-dbo` | `b128/s128`、`b64/s256`、`b32/s512`、`b16/s1024`、`b8/s2048` | 每个 seq 的下一档 batch；`b512` 在 `s128/256/512` 也 OOM。 |
| NPU | `decode-dbo` | 所有测试 seq 到 `b512` | `b1024/s128`。 |
| NPU | `decode-dbo-crosslayer` | 所有测试 seq 到 `b512` | `b1024/s128`。 |

## 6. 结论

1. 旧的 “NPU decode DBO 有很大加速 / 约 5x” 结论不能继续使用。按准确
   `decode_tpot_ms` 重跑后，NPU decode DBO 中位 speedup 低于 `1.0x`，仅少数
   cell 略高于 `1.0x`。
2. 当前最强的正向结果是 NPU prefill DBO：所有 fitting NPU prefill 样本在模型侧
   TTFT-path 指标下都快于 serial。
3. GPU DBO 在本次 full rerun 中没有中位正收益；GPU decode / crosslayer 大多低于
   `1.0x`，GPU prefill 也低于 `1.0x`。
4. 两个结果目录的 baseline 完整性已清理干净：当前报告没有依赖 legacy
   representative ITL 或 `total_time_ms/tokens` fallback。
5. OOM 是容量边界，不是缺失数据。矩阵 summary 保留 OOM 行，避免图表和报告把它们
   错当作可用 speedup cell。
