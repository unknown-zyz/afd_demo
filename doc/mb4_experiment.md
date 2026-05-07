# MB4 实验报告（Round-8 Track B）

> 分支 `exp/mb4-experiment` · 拓扑 npu-ep7（1 attn + 7 ffn EP）· 模型 Qwen3-30B-A3B · backend `broadcast_reduce_overlap`

## 1. 改动总览

| 文件 | 变更 |
|---|---|
| `scripts/run_npu.sh` | 新增 `--num-micro-batches N`，向 `python -m src.main` 传 `--num-micro-batches`；`SUFFIX` 在 `N!=2` 时附加 `_mb${N}`，使 timing JSON / report 不会与 MB2 同名覆盖 |
| `scripts/run_experiment_matrix_npu.sh` | 同样接 `--num-micro-batches` 透传；`raw_suffix` 与 SUMMARY report 名都带 `_mb${N}` |
| `scripts/aggregate_mb4_vs_mb2.py` | 新脚本：读 MB2/MB4 timing + serial baseline，输出对比 CSV / TPOT 折线 / 吞吐折线 / speedup 折线 |

代码本身（`src/main.py`、`MicroBatchManager`、`DecodeDBOScheduler`、`AsyncPipelineScheduler`）已经按 `num_mb` 泛化，不需要新逻辑。

## 2. 实验配置

NPU-910C，`results_npu_ep7_mb4/` 输出：

- decode-dbo：batch ∈ {8, 16, 32, 64, 128, 256}，seq=512，t=20，MB=4
- prefill-dbo：batch ∈ {8, 16}，seq=512，t=20，MB=4
- 对比基线：MB=2（来自 `results_npu_ep7/`，同分支同 backend）+ serial（同源）

## 3. 结果

### Decode TPOT (ms) at seq=512 — 越小越好

| batch | serial | DBO MB=2 | DBO MB=4 | MB=2 speedup | MB=4 speedup |
|---:|---:|---:|---:|---:|---:|
| 8   | 351.5  | 325.7  | 481.5  | 1.08× | 0.73× |
| 16  | 502.9  | 382.4  | 501.8  | 1.32× | 1.00× |
| 32  | 567.0  | 456.3  | 680.0  | 1.24× | 0.83× |
| 64  | 787.9  | 656.7  | 861.8  | 1.20× | 0.91× |
| 128 | 993.8  | 900.3  | 1178.4 | 1.10× | 0.84× |
| 256 | 1498.0 | 1452.5 | 1658.4 | 1.03× | 0.90× |

### Prefill total time (ms) at seq=512 — 越小越好

| batch | serial prefill | DBO MB=2 total | DBO MB=4 total | MB=2 speedup | MB=4 speedup |
|---:|---:|---:|---:|---:|---:|
| 8  | 1925.6 | 378.0 | 685.5 | 5.09× | 2.81× |
| 16 | 1827.0 | 449.7 | 752.3 | 4.06× | 2.43× |

### 图

- TPOT 对比：`results_npu_ep7_mb4/mb2_vs_mb4_decode_tpot.png`
- 吞吐对比：`results_npu_ep7_mb4/mb2_vs_mb4_decode_throughput.png`
- 加速比对比：`results_npu_ep7_mb4/mb2_vs_mb4_decode_speedup.png`
- MB4 4-lane pipeline：`results_npu_ep7_mb4/pipeline_figs/decode_dbo_mb4_b16_s512_fourlane.png`

## 4. 结论

**在当前 Qwen3-30B-A3B + npu-ep7 + broadcast_reduce_overlap 配置下，MB=4 全面劣于 MB=2，没有出现 OOM。**

- decode：MB=4 在所有 batch 上都比 MB=2 慢 9%~48%；b≤16 时甚至跌破 serial（speedup<1）。
- prefill：MB=4 仍比 serial 快 2.4×~2.8×，但比 MB=2 慢约 1.7×~1.8×。
- 测试到 b=256 仍未触发 OOM，说明 MB4 增加的 in-flight buffer 不是当前瓶颈。

### 退化原因（与 plan.md 风险一致）

1. **MoE GEMM 更碎**。MB=4 把 routed-expert 算子分 4 段调用 `npu_grouped_matmul`，单次 GEMM batch tokens 砍半，AICore 利用率下降；910C MoE expert 本身已经被认证为通信/launch overhead 主导（仓库 NPU MoE 记忆显示 `npu_grouped_matmul` 比 HF expert 慢 4.9×~11.1×），多分段进一步放大此惩罚。
2. **HCCL collective 次数翻倍**。每层 A2F / F2A / EP dispatch / EP reduce 都从 2 次变 4 次，HCCL proxy queue 与 stream 同步开销线性增加。
3. **router/dispatch/combine 启动开销 ×N**。这些小 kernel 的 host launch / 编译 caching 不会随 MB 缩放，固定开销变 2 倍。
4. **更小 MB 让 attention 也变慢**。b=8 时单 MB tokens=2，attn kernel 调度开销远大于真正算力，导致 attn 也比 MB=2 慢。

### 何时可能有收益

- 计算时间远长于通信时间的场景（FFN/Attn ratio ≫ 1，且单 MB 仍能填满 AICore）。当前最甜区在 b∈[8, 64]，MB=2 已经做到接近 FFN/Attn ≈ 1.6~1.9（见 `doc/compute_time_vs_batch_s512.md`），通信掩盖空间已被吃掉。
- 更大 batch（b≥1024）+ 长 seq + 弱通信链路；910C-EP7 当前不属于这种情形。
- 计算 backend 改进（fused MoE / NPUGraph / 更高效 grouped_matmul）后，单 MB 计算更快，再分 MB 才有意义。

## 5. 风险与遗留

- 当前 MB4 实验 batch 上限只到 256；如需更高 batch（≥512）的 OOM 边界，要单独跑（MB2 的全矩阵在 b=512/seq=2048 OOM，MB4 预计更早）。
- prefill 只测了 b8/b16 验证可用；更全 prefill 矩阵未跑（不在本轮目标范围内）。
- 没有跑 `decode-dbo-crosslayer + MB4` 组合；推断收益不会更好（crosslayer 在 EP7 已不增益）。

## 6. 复现命令

```bash
# 同步 MB4 分支到 NPU 容器后
ASCEND_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  bash scripts/run_experiment_matrix_npu.sh \
    --modes decode-dbo \
    --batches 8,16,32,64,128,256 \
    --seqs 512 --tokens 20 \
    --preset npu-ep7 \
    --ffn-ep-backend broadcast_reduce_overlap \
    --num-micro-batches 4 \
    --output-root results_npu_ep7_mb4 \
    --serial-cache-root results_npu_ep7_mb4/serial/cache

# 拉回结果后
python3 scripts/aggregate_mb4_vs_mb2.py
```

---

## 7. Round-9: Fused dispatch 优化（v2）

### 7.1 瓶颈定位

per-layer per-MB）拆解 decode b=16/s=512  ms/layer）：

| metric | MB2 | MB4 | 倍数 |
|---|---:|---:|---:|
| compute (attn) | 4.17 | 6.10 | 1.46× |
| router | 0.26 | 0.50 | 1.92× |
| **ep_dispatch** | **3.78** | **16.09** | **4.26×** ⚠️ |
| ep_local_experts | 3.90 | 5.60 | 1.43× |
| ep_reduce | 2.40 | 5.32 | 2.22× |
| dispatch_wait | ≈0 | ≈0 | — |

`ep_dispatch` 4.26× 超线性放大是 MB4 退化主因。`dispatch_wait≈0` 说明 stream 不是被传输等待阻塞，而 enqueue/HCCL 串行化阻塞——4 个 MB × 3 broadcasts/MB = 12 broadcasts/layer 排队抢同一个 `ffn_ep_dispatch_group`。

### 7.2 优化：3 broadcast → 1 fused broadcast

`src/model/ep_moe.py` 改造（commit `79b42ab`）：

- 把 `hidden_2d` (bf16) / `selected_experts` (int64) / `routing_weights` (bf16) 三张 tensor pack 进单个 `uint8` buffer
- coordinator 端 1 次 `dist.broadcast`，experts 端用 view 切片回原 dtype（zero-copy）
同时 -         `dispatch_async`（async/decode 路径）和 `_broadcast_inputs`（sync/prefill 路径）

### 7.3 v2 实测（4-way 对比，seq=512，t=20）

| batch | serial | MB2-orig | **MB2-fused** | MB4-fused | MB2-orig× | **MB2-fused×** | MB4-fused× | fuse 收益 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8   | 351.5  | 325.7  | **300.4**  | 450.8 | 1.08× | **1.17×** | 0.78× | +7.8% |
| 16  | 502.9  | 382.4  | **360.0**  | 495.4 | 1.32× | **1.40×** | 1.02× | +5.9% |
| 32  | 567.0  | 456.3  | **433.3**  | 602.3 | 1.24× | **1.31×** | 0.94× | +5.0% |
| 64  | 787.9  | 656.7  | 642.8      | 848.3 | 1.20× | 1.23×     | 0.93× | +2.1% |
| 128 | 993.8  | 900.3  | 894.7      | 1164.9| 1.10× | 1.11×     | 0.85× | +0.6% |
| 256 | 1498.0 | 1452.5 | 1420.0     | 1682.5| 1.03× | 1.05×     | 0.89× | +2.2% |
| 512 | 2452.4 | 2394.4 | 2374.9     | 2594.8| 1.02× | 1.03×     | 0.95× | +0.8% |

Prefill (s=512) total time (ms)：

| batch | serial | MB2-orig | MB2-fused | MB4-fused |
|---:|---:|---:|---:|---:|
| 8  | 1925.6 | 378.0 | 402.7 | 684.0 |
| 16 | 1827.0 | 449.7 | 460.5 | 792.6 |
| 32 | 1811.8 | 654.7 | 662.3 | 902.4 |

### 7.4 结论

- ✅ **MB2-fused 在小/中 batch（8–32）上稳定带来 +5–8% TPOT 收益**，最高加速比从 1.32× → 1.40×（b=16）。这是可发布的优化。
- ❌ **MB4 即使融合后仍跑不过 MB2**：原因是 HCCL stream 串行化是真正瓶颈，3→1 broadcast 只把每 MB 的启动数减少 2/3，并不改变跨 MB 的队列竞争。证据：experts 端 mb0 dispatch ≈ 600µs（快），mb1+ ≈ 5500µs（慢）。
- ⚠️ **Prefill 上 fusion 略微负向**（+1.3% ~ +6.5% 总时间）：prefill 单 broadcast payload 已经较大，3→1 packing 的额外 device copy 成本反而占主导。后续可以考虑只在 decode 路径走 fused，prefill 保留 3-broadcast——目前未做这个分叉。

.git .github .gitignore .pytest_cache README.md config doc requirements.txt results results_npu results_npu_ep7 results_npu_ep7_mb4 results_npu_ep7_mb4_v2 scripts src tests venv 

- **跨 MB coalescing**：把同 layer 所有 MB 的 hidden 打成一个大 broadcast。代价：丢失 MB 间流水重叠机会。
- **多 HCCL group**：把 N 个 MB round-robin 拆到 2 个独立 dispatch_group，让 broadcast 并行下发。910C 上 HCCL 是否真的会在多 group 上并行需实测。
 2D-grid alltoall，启动次数可降到 O(1)/layer，但调度逻辑要重写。

### 7.6 复现命令

```bash
git checkout exp/mb4-experiment   # commit 79b42ab+

bash scripts/run_experiment_matrix_npu.sh \
  --preset npu-ep7 --ffn-ep-backend broadcast_reduce_overlap \
  --modes decode-dbo --num-micro-batches 2 \
  --batches 8,16,32,64,128,256,512 --seqs 512 --tokens 20 \
  --output-root results_npu_ep7_mb4_v2 \
  --serial-cache-root results_npu_ep7_mb4_v2/serial/cache --no-cache

# 同上跑 --num-micro-batches 4 + prefill-dbo
python3 scripts/aggregate_mb4_v2.py
```

ls`results_npu_ep7_mb4_v2/mb2_vs_mb4_v2_{summary.csv, decode_tpot.png, decode_throughput.png, decode_speedup.png}`
