# NPU EP7 Decode 矩阵实验报告

本目录记录 `npu-ep7 + broadcast_reduce_overlap` 的 decode DBO 矩阵实验。拓扑为 **1 个 Attention rank + 7 个 FFN EP ranks**，使用 NPU 0-7，模型为 `/models/Qwen3-30B-A3B`，所有配置均为 `tokens=20`。serial baseline 来自 `results_npu/serial/cache/b<B>_s<S>_t20.json`，speedup 统一按 `serial decode_tpot_ms / EP7 decode_tpot_ms` 计算。

运行命令：

```bash
./scripts/run_experiment_matrix_npu.sh \
  --preset npu-ep7 \
  --ffn-ep-backend broadcast_reduce_overlap \
  --output-root results_npu/ep7_matrix \
  --modes decode-dbo \
  --batches 8,16,32,64,128,256 \
  --seqs 128,256,512,1024 \
  --tokens 20 \
  --visible-devs 0,1,2,3,4,5,6,7
```

## 结论

- **EP7 不是稳定正收益**：21 个成功配置中只有 3 个 TPOT speedup > 1.0。最佳配置是 `b16/s256`，speedup=1.088x；最差成功配置是 `b128/s512`，speedup=0.865x。
- **正收益区域很窄**：`b16/s256` 为 1.088x，`b16/s512` 为 1.056x，`b64/s256` 为 1.008x；其余成功配置均为持平或负收益。
- **大 batch/长 seq 受 Attention 侧 OOM 限制**：`b256/s512` 与 `b128/s1024` 都在 rank0 的 prefill `lm_head` 分配 logits 时 OOM，报错分配约 37.10 GiB；FFN EP ranks 随后停在 collective 等待，需要清理残留进程。
- **pipeline 图已显示 speedup**：`pipeline/` 下每个成功配置都有 pipeline PNG，标题中包含 Serial TPOT、EP7 TPOT 和 TPOT Speedup。注意当前图中的 A2F/F2A send bars 是 `comm_timing_mode=enqueue` 下的 `isend()` 入队/返回开销，不是真实链路传输完成时间。
- **当前 backend 仍不是 token-aware**：`broadcast_reduce_overlap` 仍做 full hidden broadcast + dense reduce，只是在 EP 内部尝试把 reduce 和下一 micro-batch compute 重叠；下一步应实现 token-aware dispatch/combine，减少无效通信和 dense reduce 负担。

## 汇总图

- [`fig_ep7_tpot_speedup.png`](fig_ep7_tpot_speedup.png)：不同 seq 下 TPOT speedup 随 batch 的变化。
- [`fig_ep7_oom_boundary.png`](fig_ep7_oom_boundary.png)：batch/seq heatmap 和 OOM 边界。
- [`fig_ep7_balance_wait.png`](fig_ep7_balance_wait.png)：代表配置的 Attention/FFN compute、recv wait、EP reduce/overlap hidden 对比。

## 全矩阵结果

| batch | seq | status | serial TPOT (ms) | EP7 TPOT (ms) | speedup | note |
|---:|---:|---|---:|---:|---:|---|
| 8 | 128 | ok | 331.7 | 383.3 | 0.865x | 负收益/持平 |
| 16 | 128 | ok | 482.9 | 484.3 | 0.997x | 负收益/持平 |
| 32 | 128 | ok | 608.1 | 616.3 | 0.987x | 负收益/持平 |
| 64 | 128 | ok | 727.7 | 806.0 | 0.903x | 负收益/持平 |
| 128 | 128 | ok | 943.2 | 1089.0 | 0.866x | 负收益/持平 |
| 256 | 128 | ok | 1551.1 | 1655.2 | 0.937x | 负收益/持平 |
| 8 | 256 | ok | 356.9 | 366.6 | 0.974x | 负收益/持平 |
| 16 | 256 | ok | 503.0 | 462.3 | 1.088x | 正收益 |
| 32 | 256 | ok | 562.8 | 567.1 | 0.992x | 负收益/持平 |
| 64 | 256 | ok | 778.5 | 772.4 | 1.008x | 正收益 |
| 128 | 256 | ok | 1019.6 | 1175.7 | 0.867x | 负收益/持平 |
| 256 | 256 | ok | 1501.1 | 1606.8 | 0.934x | 负收益/持平 |
| 8 | 512 | ok | 351.5 | 362.4 | 0.970x | 负收益/持平 |
| 16 | 512 | ok | 502.9 | 476.4 | 1.056x | 正收益 |
| 32 | 512 | ok | 567.0 | 588.7 | 0.963x | 负收益/持平 |
| 64 | 512 | ok | 787.9 | 890.7 | 0.885x | 负收益/持平 |
| 128 | 512 | ok | 993.8 | 1149.2 | 0.865x | 负收益/持平 |
| 256 | 512 | OOM | 1498.0 | - | - | rank0 prefill lm_head OOM |
| 8 | 1024 | ok | 332.9 | 349.4 | 0.953x | 负收益/持平 |
| 16 | 1024 | ok | 476.0 | 478.1 | 0.996x | 负收益/持平 |
| 32 | 1024 | ok | 614.1 | 614.4 | 0.999x | 负收益/持平 |
| 64 | 1024 | ok | 825.4 | 867.1 | 0.952x | 负收益/持平 |
| 128 | 1024 | OOM | 982.8 | - | - | rank0 prefill lm_head OOM |

## 代表 pipeline 图

- `b16/s256/t20`: [`pipeline/pipeline_decode-dbo_npu_ep7_broadcast_reduce_overlap_b16_s256_t20.png`](pipeline/pipeline_decode-dbo_npu_ep7_broadcast_reduce_overlap_b16_s256_t20.png)，speedup=1.088x。
- `b16/s512/t20`: [`pipeline/pipeline_decode-dbo_npu_ep7_broadcast_reduce_overlap_b16_s512_t20.png`](pipeline/pipeline_decode-dbo_npu_ep7_broadcast_reduce_overlap_b16_s512_t20.png)，speedup=1.056x。
- `b64/s1024/t20`: [`pipeline/pipeline_decode-dbo_npu_ep7_broadcast_reduce_overlap_b64_s1024_t20.png`](pipeline/pipeline_decode-dbo_npu_ep7_broadcast_reduce_overlap_b64_s1024_t20.png)，speedup=0.952x。
- `b128/s512/t20`: [`pipeline/pipeline_decode-dbo_npu_ep7_broadcast_reduce_overlap_b128_s512_t20.png`](pipeline/pipeline_decode-dbo_npu_ep7_broadcast_reduce_overlap_b128_s512_t20.png)，speedup=0.865x。

完整 pipeline 图位于 [`pipeline/`](pipeline/)；每个成功配置均有一张，文件名格式：

```text
pipeline_decode-dbo_npu_ep7_broadcast_reduce_overlap_b<B>_s<S>_t20.png
```

## 指标文件

- [`experiment_matrix_summary.csv`](experiment_matrix_summary.csv)：矩阵脚本原始状态汇总，包含 ok/OOM。
- [`summary.csv`](summary.csv)：后处理后的数值汇总，包含 serial/EP7 TPOT、speedup、Attention/FFN compute、EP dispatch/reduce/overlap 等字段。
- [`decode-dbo/`](decode-dbo/)：每个成功配置的 attention/coordinator/expert timing JSON 与单次 Markdown report。

## 现象解释

EP7 相比 EP4 sync 的改进在于 coordinator 不再同步等待每个 micro-batch 的 reduce 完成，而是让 reduce 与下一 micro-batch 的 local experts 尽量重叠。从 `summary.csv` 可见，部分配置存在 `ep_overlap_hidden_ms_per_event`，说明 overlap 机制生效；但端到端 TPOT 仍常常不优于 serial，主要原因是：

1. Attention rank 的 F2A recv wait 仍然较高，A/F 外层流水线空泡没有完全消失。
2. EP7 的 fan-out、broadcast、dense reduce 固定开销在小 batch 和部分大 batch 下抵消了专家并行收益。
3. 当前 decode payload 是按 micro-batch token 的 hidden state 发送，未按专家实际命中 token 做 compact dispatch，因此 7 个 EP ranks 的通信和 reduce 仍有大量无效工作。
4. 长 seq/大 batch 在进入 decode 前还要经历 prefill，rank0 `lm_head` 当前会形成很大的 logits 分配，导致 OOM 边界先出现在 Attention 侧。

## 下一步

1. 实现 token-aware dispatch/combine：coordinator 根据 `selected_experts` 构造 per-rank compact token buffer，expert rank 只计算自己拥有专家命中的 token，combine 用 `index_add_` 聚合。
2. 对 decode 路径避免 prefill 对全 prompt position 做不必要的 `lm_head`，只保留最后 token logits，降低 `b256/s512` 和 `b128/s1024` 的 OOM 风险。
3. 用 `comm_timing_mode=completion` 对代表配置补跑通信完成跨度，和现有 enqueue 图区分“入队开销”与“有效完成等待”。
