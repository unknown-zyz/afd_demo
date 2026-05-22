# 02. 使用与实验命令

本文是当前实验命令手册，覆盖 GPU / NPU 单机、矩阵、warmup ablation、
跨机 smoke / fallback / DeepEP / 当前 P3 真实 decode 调试，以及常用后处理命令。

## 1. 环境准备

```bash
cd /path/to/afd_demo
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pytest tests/ -q
```

GPU 模型路径：

```bash
export MODEL_PATH=/data/Qwen/Qwen3-30B-A3B/
```

NPU 模型路径：

```bash
export MODEL_NAME=/models/Qwen3-30B-A3B
```

当前最常用脚本：

| 脚本 | 场景 |
|---|---|
| `scripts/run_single.sh` | GPU 单次 serial / prefill DBO / decode DBO / crosslayer。 |
| `scripts/run_experiment_matrix.sh` | GPU 矩阵实验。 |
| `scripts/run_npu.sh` | 910C 单次实验；支持 1A1F、EP4、EP7、EP8、EP15。 |
| `scripts/run_experiment_matrix_npu.sh` | 910C 矩阵实验。 |
| `scripts/run_warmup_ablation_npu.sh` | P2P warmup / prefill warmup 的 2×2 消融。 |
| `scripts/run_tbe_cache_warmup_npu.sh` | 跨机前的 TBE / `kernel_meta` 预编译，带日志和 cache 轮询。 |
| `scripts/run_npu_coordinator_single_host.sh` | Host1 单机 coordinator smoke；自动先起 coordinator，再走真实 `src.main` 1A1F decode/prefill 路径。当前仅支持 1A1F、无 FFN EP。 |
| `scripts/cross_host_*.py` | 双机 HCCL / fallback / DeepEP 通信冒烟与 RT bench。 |
| `scripts/plot_all_pipelines.py` / `scripts/gen_experiment_report.py` | 图表与单次报告生成。 |

## 2. GPU 单次入口

```bash
./scripts/run_single.sh <local|multinode> <batch> <seq> [options]
```

常用参数：

| 参数 | 含义 |
|---|---|
| `--tokens N` | 生成 token 数 / `max_new_tokens`；单次默认 `5`。 |
| `--no-dbo` | 关闭 DBO，运行 serial AF baseline。 |
| `--generate` | 启用 prefill + autoregressive decode；不加时只跑 prefill。 |
| `--crosslayer` | 启用 decode cross-layer pipeline；只在 DBO + generation 下有意义。 |
| `--visualize` | 单次运行结束后生成 pipeline PNG。 |
| `--warmup-p2p` | timing 前执行未计时 NCCL P2P warmup。 |
| `--warmup-rounds N` | P2P warmup 轮数；单次默认 `3`，matrix 使用 `5`。 |
| `--comm-timing-mode enqueue\|completion` | send event 计时模式；`enqueue` 记录 `isend()` 返回开销，`completion` 记录有效 Work 完成跨度。 |
| `--no-timing` | 关闭详细 timing / report 输出，用于评估 timing 开销。 |
| `--verbose` | 打印更详细日志。 |

输入构造：

- `batch` 目前表示把同一个 prompt 复制成 `batch` 份，不是自动构造不同文本样本。
- `seq` 会传入 `--prefill-seq-len`。在指定该参数时，prefill-only 和 generation
  都会使用 `padding="max_length"` / `truncation=True`，把输入固定到 `[batch, seq]`。
- Timing JSON 会写入 `prefill_seq_len` 和 `actual_prompt_len`。两者一致时，说明
  文件名中的 `s<seq>` 与实际 prompt/KV cache 长度一致。

## 3. 串行基线（Serial baseline）

Serial 表示关闭 DBO，但仍使用 A/F 分离。

```bash
# 串行生成基线：用于准确 decode TPOT。
./scripts/run_single.sh local 4 128 --tokens 20 --no-dbo --generate

# 串行预填充基线：用于补采 prefill_ms。
./scripts/run_single.sh local 4 128 --tokens 20 --no-dbo
```

矩阵：

```bash
./scripts/run_experiment_matrix.sh \
  --modes serial \
  --batches 2,4,8,16,32,64 \
  --seqs 128,256,512 \
  --tokens 20 \
  --no-cache
```

输出：

| 路径 | 含义 |
|---|---|
| `results/serial/timing_attention_serial_b<B>_s<S>_t<T>.json` | Attention 侧 serial timing。 |
| `results/serial/timing_ffn_serial_b<B>_s<S>_t<T>.json` | FFN 侧 serial timing。 |
| `results/serial/cache/b<B>_s<S>_t<T>.json` | 报告和图表使用的 baseline cache；prefill 对比读取其中的 `prefill_ms`，decode 对比读取其中的 `decode_tpot_ms`。 |

## 4. 预填充 DBO（Prefill DBO）

Prefill DBO 表示 DBO 开启、generation 关闭。`run_single.sh` 默认就是
prefill-only，除非显式加 `--generate`。

```bash
./scripts/run_single.sh local 4 128 --tokens 20
```

矩阵：

```bash
./scripts/run_experiment_matrix.sh \
  --modes prefill-dbo \
  --batches 2,4,8,16,32,64 \
  --seqs 128,256,512 \
  --tokens 20
```

Prefill speedup 使用模型侧 TTFT-path：

```text
serial_prefill_ms / dbo_total_time_ms
```

其中 `serial_prefill_ms` 来自 `results/serial/cache/b<B>_s<S>_t<T>.json` 的
`prefill_ms`，不是生成 prefill 报告时临时重跑 serial；`dbo_total_time_ms` 来自
prefill DBO timing JSON。

## 5. 解码 DBO / 跨层流水（Decode DBO / Crosslayer）

Decode DBO 必须加 `--generate`。

```bash
# 解码 DBO
./scripts/run_single.sh local 4 128 --tokens 20 --generate

# 解码跨层流水
./scripts/run_single.sh local 4 128 --tokens 20 --generate --crosslayer
```

矩阵：

```bash
./scripts/run_experiment_matrix.sh \
  --modes decode-dbo,decode-dbo-crosslayer \
  --batches 2,4,8,16,32,64 \
  --seqs 128,256,512 \
  --tokens 20
```

Decode speedup 使用准确 TPOT：

```text
serial_decode_tpot_ms / dbo_decode_tpot_ms
```

这里的 `decode_tpot_ms` 是 batch-level per-step latency：

```text
decode_tpot_ms = decode_loop_ms / (max_new_tokens - 1)
```

一次 decode step 会同时为 batch 内每条序列各生成 1 个 token。因此它表示整个
batch 每推进一步的平均等待时间，不是 `decode_loop_ms / (batch * steps)` 的
per-output-token latency。吞吐口径可换算为：

```text
decode_tokens_per_sec = 1000 * batch / decode_tpot_ms
```

Decode DBO 的 pipeline 明细固定记录 0-based decode step 1，也就是第 2 个
decode-loop iteration；step 0 被跳过以避开 warmup / 冷启动。这个 step 1 timing
只用于观察 overlap，不用于最终 speedup。

A2F/F2A send event 有两种口径：

- `enqueue`：默认，记录非阻塞 `isend()` 返回/排队开销，通常约 0.1ms；它不是
  链路传输完成时间。
- `completion`：记录从 `isend()` 发起到 distributed Work 完成的有效通信完成跨度，
  包含排队、接收端 readiness、传输和完成通知；它不是纯硬件链路时延，但更适合在
  pipeline 图上观察通信是否被计算掩盖。

换句话说，completion 图展示的是“从框架视角这次 send 何时完成”的上层可观测跨度。
真实数据搬运发生在这个跨度内，但该跨度还可能包含 NCCL/HCCL 排队、对端尚未 post
`irecv`、通信流等待和 callback / `wait()` 观察开销。因此它通常大于或等于纯链路传输
时间，不能直接当作硬件带宽测试；它适合分析 pipeline overlap。若要校准纯通信量级，
使用 `scripts/bench_comm_transfer.py` 的独立 P2P microbenchmark。

评估 profiling 开销时，对同一配置分别运行：

```bash
./scripts/run_single.sh local 2 128 --tokens 20 --generate --no-timing
./scripts/run_single.sh local 2 128 --tokens 20 --generate --comm-timing-mode enqueue
./scripts/run_single.sh local 2 128 --tokens 20 --generate --comm-timing-mode completion --visualize
```

历史注意：旧版本 generation 路径没有按 `--prefill-seq-len` 固定 padding，部分
decode / serial generation 长序列结果中的 `s512/s1024/s2048` 可能只是文件名标签，
不一定对应真实长 KV cache。修复后的结果应检查 timing JSON 中
`prefill_seq_len == actual_prompt_len`。

## 6. GPU 矩阵参数

```bash
./scripts/run_experiment_matrix.sh [options]
```

| 参数 | 默认值 | 含义 |
|---|---|---|
| `--modes list` | 四种模式全部运行 | `serial,prefill-dbo,decode-dbo,decode-dbo-crosslayer` 的子集。 |
| `--batches list` | `2,4,8,16,32,64` | batch size 列表。 |
| `--seqs list` | `128,256,512` | prefill sequence length 列表。 |
| `--tokens N` | `20` | decode token 数。 |
| `--deployment` | `local` | `local` 或 `multinode`。 |
| `--no-cache` | false | 强制重跑 serial，不复用 cache。 |
| `--dry-run` | false | 只打印命令，不执行。 |

参考 GPU rerun phases：

```bash
# 默认网格
./scripts/run_experiment_matrix.sh \
  --modes serial,prefill-dbo,decode-dbo,decode-dbo-crosslayer \
  --batches 2,4,8,16,32,64 \
  --seqs 128,256,512 \
  --tokens 20

# 高 batch 扩展
./scripts/run_experiment_matrix.sh \
  --modes serial,decode-dbo,decode-dbo-crosslayer \
  --batches 96,128,192,256 \
  --seqs 128,256,512 \
  --tokens 20

# 长序列扩展
./scripts/run_experiment_matrix.sh \
  --modes serial,prefill-dbo,decode-dbo,decode-dbo-crosslayer \
  --batches 2,4,8,16,32,64,96,128,192,256 \
  --seqs 1024,2048 \
  --tokens 20
```

脚本启动时会重写 `results/experiment_matrix_summary.csv`。如果分阶段运行，需要
手动保存或合并阶段 summary。

## 7. NPU 矩阵参数

NPU matrix 需要在 910C 容器 / worktree 内运行。

```bash
./scripts/run_experiment_matrix_npu.sh [options]
```

| 参数 | 默认值 | 含义 |
|---|---|---|
| `--modes list` | 四种模式全部运行 | 与 GPU matrix 相同。 |
| `--batches list` | `2,4,8,16,32,64,128,256` | batch size 列表。 |
| `--seqs list` | `128,256,512` | prefill sequence length 列表。 |
| `--tokens N` | `20` | decode token 数。 |
| `--visible-devs list` | `0..15` | `ASCEND_VISIBLE_DEVICES` 设备池。 |
| `--attn-devs list` | 空 | Attention rank 的设备池覆盖。 |
| `--ffn-devs list` | 空 | FFN rank 的设备池覆盖。 |
| `--no-cache` | false | 强制重跑 serial。 |
| `--append` | false | 追加到已有 summary，而不是重写。 |
| `--dry-run` | false | 只打印命令，不执行。 |

参考 NPU rerun phases：

```bash
# 默认网格
./scripts/run_experiment_matrix_npu.sh \
  --modes serial,prefill-dbo,decode-dbo,decode-dbo-crosslayer \
  --batches 2,4,8,16,32,64,128,256 \
  --seqs 128,256,512 \
  --tokens 20 \
  --no-cache

# b512 高 batch 探测
./scripts/run_experiment_matrix_npu.sh \
  --append \
  --modes serial,prefill-dbo,decode-dbo,decode-dbo-crosslayer \
  --batches 512 \
  --seqs 128,256,512 \
  --tokens 20 \
  --no-cache

# 长序列网格
./scripts/run_experiment_matrix_npu.sh \
  --append \
  --modes serial,prefill-dbo,decode-dbo,decode-dbo-crosslayer \
  --batches 2,4,8,16,32,64,128,256,512 \
  --seqs 1024,2048 \
  --tokens 20 \
  --no-cache

# b1024 边界探测
./scripts/run_experiment_matrix_npu.sh \
  --append \
  --modes serial,decode-dbo,decode-dbo-crosslayer \
  --batches 1024 \
  --seqs 128,256,512,1024,2048 \
  --tokens 20 \
  --no-cache
```

NPU 输出使用 `results_npu/`，目录布局与 GPU 相同，并额外记录 visible chip pool
和 active world size。

### 7.1 NPU EP 单次实验

EP4 同步版负结果对应 `broadcast_reduce_sync`，保留在
`results_npu/ep4_broadcast_reduce_sync/`：

```bash
./scripts/run_npu.sh --preset npu-ep4 \
  --ffn-ep-backend broadcast_reduce_sync \
  --batch 8 --seq 512 --tokens 20 \
  --model-name /models/Qwen3-30B-A3B
```

EP overlap 使用 `broadcast_reduce_overlap`。当前 8 卡环境中的代表性正收益配置是
`1 Attention + 7 FFN EP ranks`：

```bash
./scripts/run_npu.sh --preset npu-ep7 \
  --ffn-ep-backend broadcast_reduce_overlap \
  --batch 16 --seq 512 --tokens 20 \
  --model-name /models/Qwen3-30B-A3B
```

当前 `broadcast_reduce_overlap` 仍是 full hidden broadcast + dense reduce；它不是
token-aware dispatch/combine。EP 架构细节见 [01-architecture.md](01-architecture.md)。

EP7 矩阵示例：

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

该矩阵会在每个 seq 遇到 OOM 后停止更大 batch，并复用
`results_npu/serial/cache/` 中的 serial baseline 计算 TPOT speedup。

### 7.2 NPU warmup 消融实验

`scripts/run_warmup_ablation_npu.sh` 固定跑四种组合：

1. `both_on`：P2P warmup + prefill warmup 都开
2. `p2p_only`：只开 P2P warmup
3. `prefill_only`：只开 prefill warmup
4. `both_off`：都不开

默认聚焦 decode L0 问题（`batches=4,16,64`、`seq=512`、`preset=npu-ep7`）：

```bash
bash scripts/run_warmup_ablation_npu.sh
```

自定义网格：

```bash
BATCHES=2,4,8,16,32,64 \
SEQS=512 \
TOKENS=20 \
PRESET=npu-ep7 \
FFN_EP_BACKEND=broadcast_reduce_overlap \
WARMUP_ROUNDS=5 \
bash scripts/run_warmup_ablation_npu.sh
```

结果输出到 `results_npu_ep7/warmup_ablation/`，汇总由
`scripts/aggregate_warmup_ablation_npu.py` 自动生成。

## 8. 跨机调试与当前 P3 真实 decode

### 8.1 跨机脚本分工

| 脚本 | 用途 |
|---|---|
| `scripts/cross_host_hccl_smoke.py` | 验证双机 HCCL 基础可用。 |
| `scripts/cross_host_fallback_comm_smoke.py` | 验证 `FallbackMoECommunicator` 真实 dispatch/combine 路径。 |
| `scripts/cross_host_fallback_rt_bench.py` | 测 fallback round-trip latency。 |
| `scripts/cross_host_deepep_smoke.py` | 验证 DeepEP buffer 能否建起来。 |
| `scripts/cross_host_deepep_rt_bench.py` | DeepEP normal mode RT。 |
| `scripts/cross_host_deepep_lowlatency_rt_bench.py` | DeepEP low_latency RT。 |

当前推荐排查顺序：

1. 先跑 `cross_host_hccl_smoke.py`
2. 再跑 `cross_host_fallback_rt_bench.py`
3. 再看 `cross_host_deepep_*.py`
4. 最后再尝试 P3 真实 1A7F decode

### 8.2 双机 HCCL / fallback / DeepEP 命令参考

以下命令都假设在各自主机的容器内执行，且 Host1 控制 IP 为 `192.168.0.125`、Host2 控制 IP 为 `192.168.0.192`。

HCCL smoke：

```bash
# Host2
cd /workspace/afd_demo_repo
source venv/bin/activate
source /usr/local/Ascend/ascend-toolkit/set_env.sh
RANK=1 WORLD_SIZE=2 LOCAL_RANK=0 ASCEND_VISIBLE_DEVICES=0 \
MASTER_ADDR=192.168.0.125 MASTER_PORT=29701 \
HCCL_IF_BASE_PORT=29711 HCCL_IF_IP=192.168.0.192 \
HCCL_CONNECT_TIMEOUT=600 HCCL_EXEC_TIMEOUT=600 \
python3 scripts/cross_host_hccl_smoke.py

# Host1
cd /workspace/afd_demo
source venv/bin/activate
source /usr/local/Ascend/ascend-toolkit/set_env.sh
RANK=0 WORLD_SIZE=2 LOCAL_RANK=0 ASCEND_VISIBLE_DEVICES=0 \
MASTER_ADDR=192.168.0.125 MASTER_PORT=29701 \
HCCL_IF_BASE_PORT=29711 HCCL_IF_IP=192.168.0.125 \
HCCL_CONNECT_TIMEOUT=600 HCCL_EXEC_TIMEOUT=600 \
python3 scripts/cross_host_hccl_smoke.py
```

Fallback RT：

```bash
# Host2
cd /workspace/afd_demo_repo
source venv/bin/activate
source /usr/local/Ascend/ascend-toolkit/set_env.sh
RANK=1 WORLD_SIZE=2 LOCAL_RANK=0 ASCEND_VISIBLE_DEVICES=0 \
MASTER_ADDR=192.168.0.125 MASTER_PORT=29721 \
HCCL_IF_BASE_PORT=29731 HCCL_IF_IP=192.168.0.192 \
HCCL_CONNECT_TIMEOUT=600 HCCL_EXEC_TIMEOUT=600 \
python3 scripts/cross_host_fallback_rt_bench.py --iters 50 --warmup 10 --num-tokens 64 --hidden 4096

# Host1
cd /workspace/afd_demo
source venv/bin/activate
source /usr/local/Ascend/ascend-toolkit/set_env.sh
RANK=0 WORLD_SIZE=2 LOCAL_RANK=0 ASCEND_VISIBLE_DEVICES=0 \
MASTER_ADDR=192.168.0.125 MASTER_PORT=29721 \
HCCL_IF_BASE_PORT=29731 HCCL_IF_IP=192.168.0.125 \
HCCL_CONNECT_TIMEOUT=600 HCCL_EXEC_TIMEOUT=600 \
python3 scripts/cross_host_fallback_rt_bench.py --iters 50 --warmup 10 --num-tokens 64 --hidden 4096
```

DeepEP normal / low_latency 的命令模板与 `doc/15-cross-host-communication-diagnosis.md`
和 `doc/13-deepep-install-test-error-guide.md` 保持一致；如果只是验证当前生产可行路径，优先跑 fallback。

### 8.2.1 Host1 单机 coordinator smoke（真实 decode/prefill 路径）

当前 `coordinator_arch` 已经通过 `src.main` 打通 **Host1 单机 1A1F smoke**：
会先起 gRPC coordinator，再走真实 Qwen3 `decode-dbo` / `prefill` 路径。当前限制：

- 只支持 **1 attention + 1 ffn**
- 不支持 FFN EP
- 目前是 **注册 + 一次性拉表** 的 control-plane smoke，不包含真实动态路由更新

```bash
cd /workspace/afd_demo
source venv/bin/activate
source /usr/local/Ascend/ascend-toolkit/set_env.sh
ASCEND_VISIBLE_DEVICES=0,1 \
bash scripts/run_npu_coordinator_single_host.sh \
  --batch 2 --seq 128 --tokens 5 \
  --model-name /models/Qwen3-30B-A3B \
  --timing --timing-suffix coordinator_b2_s128_t5
```

### 8.3 Host1 / Host2 单机预编译（TBE warmup）

P3 当前真正的阻塞点是首次冷编译，而不是 HCCL bootstrap。推荐先在两台机器本地把代表性 prefill shape 编出来，再跑跨机。

注意：`scripts/run_npu.sh` 会把每个 rank 的输出写到 `results/logs/npu_..._r*.log`，
所以 Host2 EP7 冷编译时直接执行 `run_npu.sh` 会看起来“前台没反应”。使用下面的
wrapper 可以持续看到 rank 数、`kernel_meta/` 大小和日志尾部；超时会返回 `124`
并清理本次 suffix 对应的残留 `python -m src.main` rank。

Host1（本地 1A1F 预编译 attention 路径为主）：

```bash
cd /workspace/afd_demo
source venv/bin/activate
source /usr/local/Ascend/ascend-toolkit/set_env.sh
bash scripts/run_tbe_cache_warmup_npu.sh \
  --profile host1-attn \
  --batch 2 --seq 128 --tokens 5 \
  --model-name /models/Qwen3-30B-A3B \
  --timeout-sec 7200 --poll-sec 60
```

Host2（本地 1A7F / EP7 预编译 FFN + EP 路径）：

```bash
cd /workspace/afd_demo_repo
source venv/bin/activate
source /usr/local/Ascend/ascend-toolkit/set_env.sh
bash scripts/run_tbe_cache_warmup_npu.sh \
  --profile host2-ep7 \
  --batch 2 --seq 128 --tokens 5 \
  --model-name /models/Qwen3-30B-A3B \
  --timeout-sec 7200 --poll-sec 60
```

观察点：

- `kernel_meta/` 目录明显增大
- 日志不再长期停在 `Running 1 prefill warmup round(s) to absorb JIT compile cost`
- 如果返回 `exit=124`，表示仍在冷编译阶段超时；确认 `active_src_main` 清零后再复跑或增大 `--timeout-sec`

### 8.4 当前 P3 真实 decode 复现命令（1A7F，fallback）

下面命令用于复现当前“已过 HCCL bootstrap，但卡在首次 prefill warmup / TBE JIT”的现象。两边都在容器内执行。

Host2 先起 7 个 FFN ranks：

```bash
cd /workspace/afd_demo_repo
source venv/bin/activate
source /usr/local/Ascend/ascend-toolkit/set_env.sh
mkdir -p results_npu/coordinator_arch/p3_real_xhost

export MASTER_ADDR=192.168.0.125
export MASTER_PORT=29795
export HCCL_IF_BASE_PORT=30400
export HCCL_IF_IP=192.168.0.192
export HCCL_CONNECT_TIMEOUT=3600
export HCCL_EXEC_TIMEOUT=3600
export HCCL_BUFFSIZE=200
export AFD_DIST_TIMEOUT_SEC=7200

for R in 1 2 3 4 5 6 7; do
  LRANK=$((R - 1))
  ASCEND_VISIBLE_DEVICES=$LRANK ASCEND_RT_VISIBLE_DEVICES=$LRANK \
  nohup timeout 3600 python3 -u -m src.main \
    --backend npu --role ffn --world-size 8 --rank $R --local-rank 0 \
    --attn-node-rank 0 --ffn-node-rank 1 \
    --attn-size 1 --ffn-size 7 --ffn-tp-size 1 --ffn-ep-size 7 \
    --ffn-ep-backend broadcast_reduce_overlap --ffn-coordinator-rank 1 \
    --ep-expert-policy round_robin \
    --batch-size 2 --prefill-seq-len 128 --max-new-tokens 5 \
    --num-micro-batches 2 --timing \
    --timing-suffix xhost_real_b2_s128_t5_v6 \
    --master-addr $MASTER_ADDR --master-port $MASTER_PORT \
    --model-name /models/Qwen3-30B-A3B \
    --no-dbo --no-generate \
    > results_npu/coordinator_arch/p3_real_xhost/h2_rank${R}_xhost_real_b2_s128_t5_v6.log 2>&1 &
done
```

Host1 再起 attention rank：

```bash
cd /workspace/afd_demo
source venv/bin/activate
source /usr/local/Ascend/ascend-toolkit/set_env.sh
mkdir -p results_npu/coordinator_arch/p3_real_xhost

export MASTER_ADDR=192.168.0.125
export MASTER_PORT=29795
export HCCL_IF_BASE_PORT=30300
export HCCL_IF_IP=192.168.0.125
export HCCL_CONNECT_TIMEOUT=3600
export HCCL_EXEC_TIMEOUT=3600
export HCCL_BUFFSIZE=200
export AFD_DIST_TIMEOUT_SEC=7200
export ASCEND_VISIBLE_DEVICES=0
export ASCEND_RT_VISIBLE_DEVICES=0

nohup timeout 3600 python3 -u -m src.main \
  --backend npu --role attention --world-size 8 --rank 0 --local-rank 0 \
  --attn-node-rank 0 --ffn-node-rank 1 \
  --attn-size 1 --ffn-size 7 --ffn-tp-size 1 --ffn-ep-size 7 \
  --ffn-ep-backend broadcast_reduce_overlap --ffn-coordinator-rank 1 \
  --ep-expert-policy round_robin \
  --batch-size 2 --prefill-seq-len 128 --max-new-tokens 5 \
  --num-micro-batches 2 --timing \
  --timing-suffix xhost_real_b2_s128_t5_v6 \
  --master-addr $MASTER_ADDR --master-port $MASTER_PORT \
  --model-name /models/Qwen3-30B-A3B \
  --no-dbo --no-generate \
  > results_npu/coordinator_arch/p3_real_xhost/h1_rank0_xhost_real_b2_s128_t5_v6.log 2>&1 &
```

当前预期现象：

- 所有 rank 能越过 `Distributed initialized`
- 日志会进入 `Running 1 prefill warmup round(s) to absorb JIT compile cost`
- 冷编译场景下可能 60 分钟内仍无法完成第一轮 warmup

若已提前完成 §8.3 的单机 warmup，再重复本节命令，通常更有机会快速越过首轮 prefill。

## 9. 后处理

```bash
# GPU
python scripts/plot_all_pipelines.py --root results
python scripts/audit_experiment_baselines.py --root results --output-csv results/baseline_audit.csv

# NPU
python scripts/plot_all_pipelines.py --root results_npu
python scripts/audit_experiment_baselines.py --root results_npu --output-csv results_npu/baseline_audit.csv
```

`baseline_audit.csv` 的常见状态：

| 状态 | 含义 |
|---|---|
| `ok` | 存在 mode-matched baseline，speedup 可信。 |
| `serial-cache-missing` | 缺少相同 `(batch, seq, tokens)` 的 serial cache。 |
| `baseline-missing` | cache 存在，但缺少 `prefill_ms` 或 `decode_tpot_ms`。 |
| `serial-cache-invalid` | cache 无法解析。 |

## 10. 通信 microbenchmark

`scripts/bench_comm_transfer.py` 用两个 rank 独立测相同 payload 下的 P2P 通信：

- sender `isend()` enqueue 时间；
- sender `isend -> Work.wait()` completion 时间；
- receiver `irecv -> Work.wait()` 等待时间；
- 可选 blocking `send/recv`；
- completion 口径的等效带宽。

GPU 示例：

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 \
  scripts/bench_comm_transfer.py \
  --backend cuda \
  --sizes-mib 0.004,0.031,1,16,32 \
  --warmup 5 --iters 50 --blocking \
  --output results/comm_bench/gpu_comm.json
```

NPU 示例（在 `afd-npu-test` 容器和 `npu` 分支内）：

```bash
ASCEND_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 \
  scripts/bench_comm_transfer.py \
  --backend npu \
  --sizes-mib 0.004,0.031,1,16,32 \
  --warmup 5 --iters 50 --blocking \
  --output results_npu/comm_bench/npu_comm.json
```

对比方法：

- decode DBO payload 约为 `batch * 1 * hidden * dtype_bytes`，随 batch 增长，基本不随 prefill seq 增长；
- prefill DBO payload 约为 `batch * seq * hidden * dtype_bytes`，随 `batch * seq` 增长；
- 如果 DBO completion 明显大于 microbench completion，多出来的部分通常来自 pipeline 调度、对端 readiness 或 backend queueing，而不是纯数据搬运。

## 11. 常见问题

| 问题 | 处理方式 |
|---|---|
| Speedup 是 `N/A` | 检查 `baseline_audit.csv`，补跑或修复对应 serial cache。 |
| 启动到输出耗时远大于 timing | 多数时间在模型加载、进程启动、warmup，不属于 scheduler timing。 |
| 只有一个 token 输出 | 确认是否加了 `--generate`，以及 `--tokens` 是否大于 1。 |
| NPU rank 挂住 | 先看双 rank 日志；确认一侧 OOM 后，只 kill 对应 stuck peer PID。 |
