# NPU EP overlap 修复实验报告

本报告对应 `broadcast_reduce_overlap` 修复路径。同步版 EP4 负结果保留在
`results_npu/ep4_broadcast_reduce_sync/`，用于解释为什么需要 overlap 和后续
token-aware dispatch/combine。

## 1. 结论

本轮修复后，已经找到一个 Decode DBO 正收益配置：

```text
1 Attention rank + 7 FFN EP ranks
batch=16, seq=512, tokens=20
```

该配置在 8 张 910C NPU 上运行，结果：

| 配置 | Serial TPOT | 旧 2-rank Decode DBO TPOT | 修复后 EP overlap TPOT | vs Serial | vs 旧 DBO |
|---|---:|---:|---:|---:|---:|
| EP7 b16/s512/t20 | 502.899 ms | 546.922 ms | 463.440 ms | 1.085x | 1.180x |

因此，DBO 负优化不是不可逆，核心修复点是：

1. EP 内部 reduce 必须与下一 micro-batch local expert compute 重叠。
2. EP hot path 不能为了细粒度 timing 插入多次 NPU device synchronize。
3. decode 小 batch 下不能遍历所有本地 experts，应只计算当前 micro-batch 实际命中的 experts。
4. 在当前 8 卡环境中，EP7 比 EP4 更接近 Attention/FFN 平衡点。

## 2. 为什么原 EP4 变慢

原始 EP4 sync 路径为：

```text
router -> broadcast -> local experts -> reduce -> F2A
```

每个 micro-batch 都严格串行，且实现中为了记录 EP stage timing 多次调用 `torch.npu.synchronize()`。这导致：

- FFN wall compute 是 Attention compute 的 `7x~9x`。
- Attention rank 每层大部分时间都在等 F2A。
- reduce 没有被下一 micro-batch 的计算隐藏。
- local experts 使用 Python loop 固定遍历本 rank 32 个 experts，decode 小 batch 下大量 experts 未命中但仍要发 `where` kernel。

原 EP4 sync 结果：

| 配置 | Serial TPOT | 旧 2-rank DBO | EP4 sync | EP4 sync vs Serial |
|---|---:|---:|---:|---:|
| b4/s128/t20 | 252.722 ms | 273.469 ms | 783.681 ms | 0.322x |
| b8/s512/t20 | 351.484 ms | 332.727 ms | 931.040 ms | 0.378x |

原始 EP4 sync 详细数据见 `results_npu/ep4_broadcast_reduce_sync/`。

## 3. 本轮代码修复

### 3.1 `broadcast_reduce_overlap`

新增 overlapped EP decode backend：

```text
dispatch MB0
dispatch MB1
compute MB0
reduce MB0 async || compute MB1
wait MB0 reduce
reduce MB1 async
wait MB1 reduce
```

修改文件：

- `src/model/ep_moe.py`
  - 新增 `EPWorkItem`
  - 新增 `create_work_item`
  - 新增 `dispatch_async`
  - 新增 `finish_dispatch`
  - 新增 `compute_local`
  - 新增 `reduce_async`
  - 新增 `finish_reduce`
  - 新增 `finish_output`
- `src/pipeline/decode_scheduler.py`
  - 当 `--ffn-ep-backend broadcast_reduce_overlap` 时，FFN coordinator/expert-only ranks 走 overlap decode path。
  - Attention path 保持原 DBO，不直接参与 EP collectives。
- `src/utils/timing.py`
  - 新增 `ep_dispatch_wait`
  - 新增 `ep_reduce_wait`
  - 新增 `ep_overlap_hidden`
- `tests/test_timing_comm.py`
  - 增加 EP overlap timing 聚合测试。

### 3.2 去掉侵入式 EP stage sync

`sync_if_needed()` 默认不再同步设备，只有设置：

```bash
AFD_EP_SYNC_TIMING=1
```

才启用 stage 内部 device synchronize。这样避免为了细粒度 timing 破坏真实 overlap。

### 3.3 active-only local experts

`ShardedExperts.forward_local()` 不再固定遍历本 rank 全部 local experts，而是先找当前 micro-batch 中实际出现的 routed experts，只对命中的本地 experts 执行 `where + GEMM + index_add_`。

该修改对 b8/s512/t20 影响很大：

| 阶段 | TPOT |
|---|---:|
| EP4 sync | 931.040 ms |
| EP4 overlap + low-sync | 730.971 ms |
| EP4 overlap + active-only experts | 464.009 ms |
| EP7 overlap + active-only experts | 380.658 ms |

## 4. 最终实验结果

| EP | 配置 | Serial TPOT | 旧 DBO TPOT | EP overlap TPOT | vs Serial | vs 旧 DBO | FFN/Attention compute ratio |
|---:|---|---:|---:|---:|---:|---:|---:|
| 4 | b4/s128/t20 | 252.722 | 273.469 | 753.925 | 0.335x | 0.363x | 5.10x |
| 4 | b8/s512/t20 | 351.484 | 332.727 | 464.009 | 0.757x | 0.717x | 2.24x |
| 7 | b8/s512/t20 | 351.484 | 332.727 | 380.658 | 0.923x | 0.874x | 1.39x |
| 7 | b16/s512/t20 | 502.899 | 546.922 | 463.440 | 1.085x | 1.180x | 2.07x |

观察：

- EP4 经过修复后仍不足以稳定正收益，但已从 931ms 降到 464ms。
- EP7 在 b8/s512/t20 上接近 serial，但仍未超过。
- EP7 在 b16/s512/t20 上超过 serial 和旧 DBO，说明更大 batch 能摊薄 EP fan-out，并让 FFN compute 更有效覆盖通信。
- reduce wait 已经很低，b16/s512/t20 的 FFN coordinator reduce wait 约 `0.010 ms/layer/MB`，主要瓶颈转移到 local experts 与 Attention 等待。

## 5. 图表

- `fig_overlap_tpot_comparison.png`：serial、旧 DBO、EP overlap TPOT 对比。
- `fig_overlap_balance_wait.png`：Attention compute、FFN compute、Attention recv-wait 平衡情况。
- `fig_overlap_ep_breakdown.png`：local experts、reduce wait、hidden reduce window 分解。

## 6. 运行命令

EP4：

```bash
./scripts/run_npu.sh --preset npu-ep4 \
  --ffn-ep-backend broadcast_reduce_overlap \
  --batch 8 --seq 512 --tokens 20 \
  --model-name /models/Qwen3-30B-A3B
```

EP7（8 卡：1 Attention + 7 FFN EP）：

```bash
./scripts/run_npu.sh --preset npu-ep7 \
  --ffn-ep-backend broadcast_reduce_overlap \
  --batch 16 --seq 512 --tokens 20 \
  --model-name /models/Qwen3-30B-A3B
```

EP7 矩阵入口：

```bash
./scripts/run_experiment_matrix_npu.sh \
  --preset npu-ep7 \
  --ffn-ep-backend broadcast_reduce_overlap \
  --output-root results_npu/ep7_matrix \
  --modes decode-dbo \
  --batches 8,16,32,64,128,256 \
  --seqs 128,256,512,1024 \
  --tokens 20
```

## 7. 下一步

当前已经证明 EP overlap 可以让 DBO 出现正收益，但收益只在较大 batch 上出现。下一步优先级：

1. 在 EP7 上扩展 batch/seq 扫描，找稳定正收益区域。
2. 做 token-aware dispatch/combine，减少 broadcast full hidden 与 dense reduce 的浪费。
3. 优化 local experts 的 Python loop，尝试按 active experts 聚合小 GEMM。
4. 若容器能暴露 16 张 NPU，再测 EP15；否则当前 8 卡环境下 EP7 是主候选。

注意：当前 `broadcast_reduce_overlap` 仍然是 full hidden broadcast + dense reduce，
不是 token-aware 稀疏通信。token-aware dispatch/combine 是下一阶段计划。
