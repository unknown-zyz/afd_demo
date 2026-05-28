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
  --model-name /models/Qwen3-30B-A3B
```

每轮必须使用新的 `MASTER_PORT` 和两侧新的 `HCCL_IF_BASE_PORT`。失败后只能检查 PID 并用 `kill <PID>` 清理，不使用 `pkill` 或 `killall`，也不重启/删除长期容器。

## 矩阵实验

在本地仓库运行编排脚本，它会通过 Host1 跳板控制两个容器，并把结果 JSON 拉回本地 `results_npu/crosshost_static_ep/`：

```bash
python scripts/run_crosshost_static_ep_matrix.py \
  --ep-sizes 16,12,8 \
  --host2-ffn-devices 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
  --backends broadcast_reduce_overlap,broadcast_reduce_sync \
  --modes decode-dbo,decode-dbo-crosslayer \
  --num-micro-batches 2,4 \
  --configs 32:256,16:256,16:128,8:256,4:128,2:64 \
  --tokens 20
```

大 batch/seq 会优先运行。推荐先 dry-run 确认命令：

```bash
python scripts/run_crosshost_static_ep_matrix.py \
  --dry-run \
  --ep-sizes 16 \
  --backends broadcast_reduce_overlap \
  --modes decode-dbo \
  --num-micro-batches 2 \
  --configs 32:256 \
  --tokens 20
```

smoke 阶段可加 `--debug-max-layers 2` 和 `--tokens 4`，先确认 HCCL group、A/F P2P 与 FFN EP group 都能跑通。

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
