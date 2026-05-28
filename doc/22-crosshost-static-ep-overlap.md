# 跨机 static EP overlap 实验说明

本实验用于验证在 AttentionWorker 优化之后，扩大 FFN EP 规模是否能把单层 FFN 时间压到与单层 Attention 时间接近，从而改善 decode-DBO pipeline overlap。

## 拓扑

- Host1：只运行 rank0，角色为 Attention。
- Host2：运行 rank1..EP，角色为 FFN EP ranks。
- `WORLD_SIZE = 1 + EP`，`--ffn-size = --ffn-ep-size = EP`。
- 当前优先测试 EP8、EP12、EP16，但脚本不写死这些值，可以手动指定任意 Host2 可承载的 EP 数量。

## 单次实验

Host2 侧先启动，Host1 侧后启动。手动运行时示例：

```bash
# Host2 容器内
bash scripts/run_crosshost_static_ep_smoke.sh \
  --side host2 \
  --master-addr 192.168.0.125 \
  --master-port 35601 \
  --hccl-if-base-port 43120 \
  --hccl-if-ip 192.168.0.192 \
  --ffn-ep-size 16 \
  --host2-ffn-devices 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
  --ffn-ep-backend broadcast_reduce_overlap \
  --mode decode-dbo \
  --batch 32 --seq 256 --tokens 20 --num-micro-batches 2 \
  --attn-kernel npu-official \
  --attn-fused-rmsnorm --attn-fused-rope --attn-precopy-layer-inputs \
  --model-name /models/Qwen3-30B-A3B

# Host1 容器内
bash scripts/run_crosshost_static_ep_smoke.sh \
  --side host1 \
  --master-addr 192.168.0.125 \
  --master-port 35601 \
  --hccl-if-base-port 42120 \
  --hccl-if-ip 192.168.0.125 \
  --ffn-ep-size 16 \
  --ffn-ep-backend broadcast_reduce_overlap \
  --mode decode-dbo \
  --batch 32 --seq 256 --tokens 20 --num-micro-batches 2 \
  --attn-kernel npu-official \
  --attn-fused-rmsnorm --attn-fused-rope --attn-precopy-layer-inputs \
  --model-name /models/Qwen3-30B-A3B
```

每轮必须使用新的 `MASTER_PORT` 和两侧新的 `HCCL_IF_BASE_PORT`。失败后只能检查 PID 并用 `kill <PID>` 清理，不使用 `pkill` 或 `killall`，也不重启/删除长期容器。

## 矩阵实验

在本地仓库运行编排脚本，它会通过 Host1 跳板控制两个容器，并把结果 JSON 拉回本地 `results_npu/crosshost_static_ep/`：

```bash
python scripts/run_crosshost_static_ep_matrix.py \
  --host1-workdir /workspace/afd_demo_crosshost_ep \
  --host2-workdir /workspace/afd_demo_repo_crosshost_ep \
  --ep-sizes 16,12,8 \
  --host2-ffn-devices 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
  --backends broadcast_reduce_overlap,broadcast_reduce_sync \
  --modes decode-dbo,decode-dbo-crosslayer \
  --num-micro-batches 2,4 \
  --configs 32:256,16:256,16:128,8:256,4:128,2:64 \
  --tokens 20 \
  --attn-kernel npu-official \
  --attn-fused-rmsnorm --attn-fused-rope --attn-precopy-layer-inputs
```

大 batch/seq 会优先运行。推荐先 dry-run 确认命令：

```bash
python scripts/run_crosshost_static_ep_matrix.py \
  --dry-run \
  --host1-workdir /workspace/afd_demo_crosshost_ep \
  --host2-workdir /workspace/afd_demo_repo_crosshost_ep \
  --ep-sizes 16 \
  --backends broadcast_reduce_overlap \
  --modes decode-dbo \
  --num-micro-batches 2 \
  --configs 32:256 \
  --tokens 20
```

smoke 阶段可加 `--debug-max-layers 2` 和 `--tokens 4`，先确认 HCCL group、A/F P2P 与 FFN EP group 都能跑通。

如果需要生成 pipeline 图，`--tokens` 至少设为 3。decode timing 默认跳过 0-based decode step 0 以避开 warmup，`--tokens 2` 只有 1 个 decode step，只能得到 TPOT，不能得到 per-layer pipeline events。

## 结果汇总

矩阵脚本成功后会为每个配置生成：

- `timing_attention_*.json`
- `timing_ffn_coordinator_*.json`
- `report_*.md`
- `pipeline_*.png`

再运行：

```bash
python scripts/summarize_crosshost_ep_timing.py \
  --root results_npu/crosshost_static_ep
```

汇总重点看：

- `attention_avg_layer_ms_excl_l0`
- `ffn_avg_layer_ms_excl_l0`
- `ep_dispatch_avg_layer_ms_excl_l0`
- `ep_local_experts_avg_layer_ms_excl_l0`
- `ep_reduce_avg_layer_ms_excl_l0`
- `attention_recv_wait_avg_layer_ms_excl_l0`
- `decode_tpot_ms`

判断标准不是 EP 越大越好，而是 FFN 单层时间与 Attention 单层时间是否接近、recv wait 是否下降、真实 `decode_tpot_ms` 是否改善。如果 EP 增大后 dispatch/reduce 超过 local expert compute 降幅，应停止继续扩大 EP，转向优化通信或 expert packing。

## 当前实验结果

本轮使用分支 `exp/crosshost-large-ep-overlap`，两端独立 worktree：

- Host1：`/workspace/afd_demo_crosshost_ep`
- Host2：`/workspace/afd_demo_repo_crosshost_ep`

执行中修复了三个脚本问题：

1. Host1 SSH 偶发 `rc=255` 时，矩阵脚本现在会重试；否则 stdout 中的 SSH 错误可能被误判为 stale process。
2. `scripts/run_crosshost_static_ep_smoke.sh` 不再 source 会直接 `exit` 的 vendor OPP `set_env.bash`，只保留 Ascend toolkit 基础环境；否则 Host1 rank0 不会启动。
3. 跨机脚本现在显式透传 Attention 优化开关：`--attn-kernel npu-official --attn-fused-rmsnorm --attn-fused-rope --attn-precopy-layer-inputs`。

### Smoke

`b2/s64` 小配置用于确认 HCCL group、A/F P2P 和 EP ranks 都能启动：

| EP | Backend | Tokens | 状态 | 备注 |
|---:|---|---:|---|---|
| 8 | `broadcast_reduce_sync` | 2 | OK | TPOT 273.8ms；tokens=2 无 per-layer events。 |
| 8 | `broadcast_reduce_overlap` | 2 | OK | TPOT 233.5ms；tokens=2 无 per-layer events。 |
| 12 | `broadcast_reduce_sync` | 2 | OK | TPOT 243.8ms；tokens=2 无 per-layer events。 |
| 12 | `broadcast_reduce_overlap` | 2 | OK | TPOT 254.6ms；tokens=2 无 per-layer events。 |
| 16 | `broadcast_reduce_sync` | 2 | OK | TPOT 264.0ms；tokens=2 无 per-layer events。 |
| 16 | `broadcast_reduce_overlap` | 3 | OK | fresh port 重试后通过并生成 pipeline 图。第一次 tokens=2 run 在 `ctx.barrier()` 触发 Host1 HCCL `ERR02005`，Host2 遗留 FFN ranks 已按明确 PID 清理。 |

### 大配置 decode-DBO

主结果使用 `b32/s256/t20`、MB2、`npu-official + RMSNorm/RoPE fusion + precopy`：

| EP | Mode | Backend | TPOT ms | A avg/layer ms | F avg/layer ms | F/A | recv-wait ms | dispatch ms | local experts ms | reduce ms | Pipeline |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 16 | decode-dbo | `broadcast_reduce_overlap` | 373.13 | 0.880 | 1.301 | 1.48 | 1.687 | 1.393 | 1.179 | 0.890 | `results_npu/crosshost_static_ep/decode-dbo_ep16_broadcast_reduce_overlap_b32_s256_t20/pipeline_xhost_static_decode-dbo_ep16_broadcast_reduce_overlap_b32_s256_t20.png` |
| 12 | decode-dbo | `broadcast_reduce_overlap` | 381.81 | 0.982 | 1.453 | 1.48 | 1.771 | 1.548 | 1.327 | 0.984 | `results_npu/crosshost_static_ep/decode-dbo_ep12_broadcast_reduce_overlap_b32_s256_t20/pipeline_xhost_static_decode-dbo_ep12_broadcast_reduce_overlap_b32_s256_t20.png` |
| 8 | decode-dbo | `broadcast_reduce_overlap` | 415.43 | 0.879 | 1.848 | 2.10 | 2.241 | 1.787 | 1.729 | 1.134 | `results_npu/crosshost_static_ep/decode-dbo_ep8_broadcast_reduce_overlap_b32_s256_t20/pipeline_xhost_static_decode-dbo_ep8_broadcast_reduce_overlap_b32_s256_t20.png` |
| 16 | decode-dbo-crosslayer | `broadcast_reduce_overlap` | 383.88 | 0.963 | 1.397 | 1.45 | 1.809 | 1.639 | 1.260 | 0.897 | `results_npu/crosshost_static_ep/decode-dbo-crosslayer_ep16_broadcast_reduce_overlap_b32_s256_t20/pipeline_xhost_static_decode-dbo-crosslayer_ep16_broadcast_reduce_overlap_b32_s256_t20.png` |

阶段性结论：

1. 在 `b32/s256/t20` 下，EP16 是当前最优：TPOT 373.13ms，优于 EP12 的 381.81ms 和 EP8 的 415.43ms。
2. EP 从 8 扩到 16 后，FFN local experts 从 1.729ms 降到 1.179ms，单层 FFN 从 1.848ms 降到 1.301ms，F/A 从 2.10 降到 1.48，说明扩大 EP 确实在把 FFN 向 Attention 对齐。
3. EP16 仍未完全对齐 Attention：FFN 单层仍比 Attention 高约 48%，且 dispatch/reduce 合计约 2.28ms，通信已经和 local expert compute 同量级。下一步继续扩大 EP 的收益可能被通信抵消，应优先优化 EP 通信/packed expert compute。
4. EP16 crosslayer 在该配置下 TPOT 383.88ms，慢于普通 decode-DBO 的 373.13ms；当前不作为最佳路径。
