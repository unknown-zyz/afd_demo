# Code Review 指南（05）

> 本文档面向第一次接触本仓库、需要做完整 code review 的同事。  
> 目标是用最短的篇幅把"系统设计 → 关键模块 → 调度器 → Profile → 画图 → Warmup → 实验启动方式 → OOM 矩阵"串成一条线，  
> 所有论断都给出 `path/file.py:line` 引用，方便对照原文。

---

## 1. 项目定位

本项目是一个 **AFD（Attention-FFN Disaggregated）解耦推理 Demo**，模型固定为
**Qwen3-30B-A3B**（48 层 MoE，hidden=2048，128 个 experts，每 token top-8）。
单机本地模式下用 4 张 V100-32GB 模拟"双机"：

| 角色 | GPU | 层数 | 备注 |
|---|---|---|---|
| Attention 节点 | GPU 0, 1 | 21 | 含 embedding、LM head、KV cache |
| FFN 节点 | GPU 2, 3 | 27 | 含 MoE experts |

层数切分 **21/27 非对称** 是经验数据：GPU 0 上 NCCL 上下文额外吃约 6 GiB
显存（见 `src/model/ffn_worker.py:165`、`src/model/attention_worker.py:210`）。

### 1.1 数据流（每层）

```
Attention 节点                 FFN 节点
   ┌────────┐    A→F send         ┌────────┐
   │ Attn   │ ──────────────────▶ │ recv   │
   │ Compute│                     │ FFN    │
   │  ...   │ ◀────────────────── │ Compute│
   └────────┘    F→A send         │ send   │
                                  └────────┘
```

每一层产生四类时间事件，是后续 Profile / 画图的基本单位：

| 事件名（`EventType`） | 发生在 | 含义 |
|---|---|---|
| `ATTN_COMPUTE` | Attn 节点 | Attention 层正向 |
| `SEND_TRANSFER` | Attn / FFN 节点 | NCCL `isend`/`send` |
| `FFN_COMPUTE` | FFN 节点 | MoE FFN 层正向 |
| `RECV_WAIT` | Attn / FFN 节点 | NCCL `recv`/`irecv.wait` |

定义见 `src/utils/timing.py:30`。

---

## 2. 模块结构

```
src/
├── main.py                      # CLI 入口；scheduler 派发；timing JSON 落盘
├── distributed/
│   ├── __init__.py              # init_dist；建 a2f_group / f2a_group
│   ├── communicator.py          # send_sync / recv_sync / isend / irecv 封装
│   └── warmup.py                # NCCL P2P warmup（解决 40-60 ms 冷启动）
├── model/
│   ├── disaggregated.py         # 顶层 model；forward_prefill；generate
│   ├── attention_worker.py      # Attn 节点持有的层
│   └── ffn_worker.py            # FFN 节点持有的层
├── pipeline/
│   ├── micro_batch.py           # MicroBatchManager.split_batch
│   ├── scheduler.py             # SimplePipelineScheduler（同步串行）
│   ├── async_scheduler.py       # AsyncPipelineScheduler（prefill DBO）
│   └── decode_scheduler.py      # DecodeDBOScheduler（decode DBO + crosslayer）
└── utils/
    └── timing.py                # TimingTracker（CUDA Events 模式）
```

---

## 3. 调度器（Scheduler）家族

仓库里有 **三个**真正在跑的调度器：serial baseline、prefill DBO、decode DBO。
历史 `PipelineScheduler` 抽象基类已删除，避免 API 与当前主路径混淆。

### 3.1 `SimplePipelineScheduler`（同步串行参考实现）

- 位置：`src/pipeline/scheduler.py:293`
- 启动方式：`--no-dbo --no-generate`
- 行为：把 batch 切成 2 个 micro-batch（默认 `num_micro_batches=2`，
  `src/main.py:119`），逐层、逐 MB **同步** 发收。
  - Attn 节点：`compute → send_sync → recv_sync`，line 320-370。
  - FFN 节点：`recv_sync → compute → send_sync`，line 378-403。
- 内存特征：任何时刻 FFN 节点上只持有 **1 个 recv buffer + 1 个 output buffer**。

### 3.2 `AsyncPipelineScheduler`（Prefill DBO）

- 位置：`src/pipeline/async_scheduler.py:80`
- 启动方式：默认（不加 `--no-dbo`）+ `--no-generate`
- 行为：每层把 2 个 MB 的 send/recv 全部 `isend`/`irecv`，配合
  `compute_stream` 与 `comm_stream` 两条 CUDA stream 重叠
  通信与计算。
  - Attn 节点核心：`_run_attention_node_async`，line 631-726。
  - FFN 节点核心：`_run_ffn_node_async`，line 728-813。
- 关键内存代价（对 review 很重要）：
  - **FFN 节点每层持有 4 块 in-flight buffer**：2 个 `recv_tensor`（line
    750-762）+ 2 个 `output_tensor`（line 766，必须保留引用直到
    `handle.wait()`，line 793-794）。
  - 这是后面 OOM 矩阵能解释清楚的根本原因。

### 3.3 `DecodeDBOScheduler`（Decode DBO + Crosslayer）

- 位置：`src/pipeline/decode_scheduler.py:65`
- 启动方式：`model.generate(..., use_decode_dbo=True,
  decode_use_crosslayer=...)`，对应 CLI 默认开（不加 `--no-dbo`），
  `--crosslayer` 控制是否跨层流水。
- KV cache 切分：HuggingFace `DynamicCache` 按 `[batch, heads, seq, dim]`
  组织，按 batch 维切片重新装回，line 211-213。
- Crosslayer 区别（line 417-443、551-580）：
  - **关闭**（默认）：当前层所有 A→F send 全部 `wait()` 后，再统一
    post 下一层的 F→A `irecv`。层间存在 ~60 ms 气泡。
  - **开启**：每个 MB 计算完成后**立即** post 自己那条 F→A `irecv`，
    用 `f2a_group` 单独的 NCCL group 发送，避免和 A→F FIFO 串行。
- 注意：**早 post irecv 必须用独立的 NCCL group**，否则 NCCL 内部 stream FIFO
  会让 irecv 阻塞后续 isend，导致死锁（仓库 memory `NCCL P2P limitations`
  记录的踩坑历史）。

### 3.4 三者对比一表

| 特性 | Simple | Async (prefill DBO) | DecodeDBOScheduler |
|---|---|---|---|
| 通信原语 | send/recv 同步 | isend/irecv | isend/irecv |
| 多 CUDA stream | 否 | 是（compute+comm） | 是 |
| MB 数 | 2 | 2 | 2（≥2 才走 DBO） |
| 层间 Pre-post recv | 否 | 否（每层屏障） | 可选（`--crosslayer`） |
| FFN 节点单层 in-flight buffer | 1+1 | 2+2 | 2+2 |
| 适合场景 | 参考 baseline | Prefill | Decode |

---

## 4. Profile / Timing

### 4.1 TimingTracker

- 位置：`src/utils/timing.py:159`
- 两种模式（`--timing-mode`）：
  - `cuda_events`（**默认，零开销**）：用 `torch.cuda.Event` 记录
    GPU 时间戳，事后 `synchronize` 一次性算 elapsed。
  - `sync`（**遗留，+16.4% 开销**）：每次 `mark_*` 都做
    `cuda.synchronize()` + `time.perf_counter()`。
- 接口：`mark_start(event_type, layer_idx, mb_idx)` /
  `mark_end(...)`，line 210-251。
- `finish()` 写入 `total_time_ms`：整个 run 的端到端时间。

### 4.2 输出 JSON 结构

每个角色（attention / ffn）各落一份：

```json
{
  "node": "attention",
  "num_layers": 48,
  "num_micro_batches": 2,
  "total_time_ms": 3637.2,
  "layers": [
    {
      "layer_idx": 0,
      "events": [
        {"event_type": "attn_compute", "mb_idx": 0,
         "start_ms": 12.34, "end_ms": 14.56},
        ...
      ]
    },
    ...
  ]
}
```

落盘路径见 `src/main.py:318-337`（serial）和 `src/main.py:475-494`
（generate 路径）。

---

## 5. 画图链路

### 5.1 单图：`scripts/visualize_dbo_pipeline.py`

- 4 通道甘特图：A、A→F、F、F→A，攻 attn 与 ffn 两份 JSON。
- **clock_offset** 通过锚事件对齐两节点时钟（line 165 附近）。
- 模式选择（`--mode {prefill,decode,auto}`）：
  - `prefill` → baseline 用 `prefill_ms`（serial cache 中字段）。
  - `decode` → baseline 用 `decode_step_ms`（同上）。
  - `auto`（默认） → 从 timing JSON 路径推断。
- Speedup 公式：`speedup = baseline / dbo_total_time_ms`。
  baseline 字段缺失时显示 "N/A"，不再用错误的 amortized 数字。

### 5.2 批量：`scripts/plot_all_pipelines.py`

驱动 63 张 PNG（21 × {prefill-dbo, decode-dbo, decode-dbo-crosslayer}），
输出到对应子目录的 `pipelines_index.md`。

### 5.3 Serial baseline cache

- 路径：`results/serial/cache/b<B>_s<S>_t<T>.json`
- 字段：`total_time_ms`、`max_new_tokens`、`prefill_ms`、`decode_step_ms`
  （`decode_step_ms = (total_ms − prefill_ms) / N`）。
- `scripts/capture_serial_prefill.sh` 用 `--no-generate` 单跑一遍 prefill
  把 `prefill_ms` 合并进 cache。当前 24 个配置中 15 个有 prefill_ms，
  剩下 9 个 prefill-only 路径 OOM（详见 §7）。

---

## 6. NCCL Warmup

- 位置：`src/distributed/warmup.py:12`（`warmup_p2p`）。
- 现象：第一次 NCCL P2P 有 **40-60 ms 冷启动**，会污染 timing。
- 用法：CLI 加 `--warmup-p2p --warmup-rounds 5`，跑 5 轮 dummy
  `isend/irecv` 把 NCCL channel 预热完。
- 历史 `P2PKeepalive` 路径已删除。原因：双向心跳容易与 scheduler NCCL
  操作争用并死锁；只保留显式 warmup。

---

## 7. 实验启动方式（命令速查）

统一脚本 `scripts/run_single.sh`，用法：

```bash
./scripts/run_single.sh local <BATCH> <SEQ> --tokens N [flags...]
```

常用 flag：

| Flag | 含义 |
|---|---|
| `--no-dbo` | 关 DBO，走 SimplePipelineScheduler |
| `--no-generate` | 只跑 prefill，不进 decode |
| `--crosslayer` | 开 decode 跨层流水（仅对 decode 生效） |
| `--warmup-p2p --warmup-rounds 5` | 必加，消除 NCCL 冷启动 |
| `--timing --timing-suffix <s>` | 写 timing JSON |
| `--timing-mode {cuda_events,sync}` | 默认 `cuda_events` |

按场景：

| 场景 | 命令片段 |
|---|---|
| 串行 + generate（baseline） | `--no-dbo` |
| 串行 prefill | `--no-dbo --no-generate` |
| Prefill DBO | `--no-generate` |
| Decode DBO | （默认） |
| Decode DBO + Crosslayer | `--crosslayer` |

环境变量（`scripts/run_single.sh:45` 附近）：
- `NCCL_BUFFSIZE=33554432`（≥ 单条 send 张量大小，否则 isend 卡住）
- `NCCL_NCHANNELS_PER_NET_PEER=1`
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

---

## 8. OOM 矩阵 ——「为什么串行会 OOM 而 DBO 能跑」

### 8.1 真实数据

| 模式 | 命令 | OOM 出现位置 |
|---|---|---|
| 串行 + generate | `--no-dbo` | **24 个配置全过**（含 b128_s512） |
| 串行 prefill-only | `--no-dbo --no-generate` | b32_s512、b64_s≥256、b96_*、b128_* |
| Prefill DBO | `--no-generate` | **b96_s128、b64_s256、b32_s512**（仅 3 个，CSV 标 OOM） |
| Decode DBO（± crosslayer） | （默认）/ `--crosslayer` | **从未 OOM** |

数据来源：`results/experiment_matrix_summary.csv`。

### 8.2 关键纠正

> 用户的直观印象「串行 OOM、DBO 能跑」**只在 decode 路径成立**。

- **Decode**：每一步 seq=1，配合 KV cache 增量推理。激活内存与
  模式无关，所以 decode 在任何配置下都不 OOM。
- **Prefill**：`AsyncPipelineScheduler` 在 FFN 节点单层会同时持有
  **2 个 recv + 2 个 output buffer**（`async_scheduler.py:750-794`），
  比同步 Simple 的 1+1 多一倍 in-flight 显存。所以 prefill DBO **更容易
  OOM**，而不是更省。这与 CSV 中 prefill DBO 在 `b96_s128` 等 3 个配置
  OOM、而串行+generate 全过的事实一致。

### 8.3 为什么 OOM 发生在 FFN MoE 的 `gate*up`

报错栈：

```
File "transformers/integrations/moe.py", line 472, in _default_apply_gate
    return self.act_fn(gate) * up
torch.OutOfMemoryError: Tried to allocate 96.00 MiB.
GPU 1 has a total capacity of 31.73 GiB of which 96.75 MiB is free.
```

具体原因 5 条：

1. **`gate * up` 单次分配大小** ≈ `tokens × intermediate × dtype_bytes`。
   B=128, S=512, intermediate=768 → `128 × 512 × 768 × 2B ≈ 96 MiB`，
   恰好就是上面报错要分配的大小。
2. **MoE 激活峰值**：每个 expert 的 `gate*up` 都是临时大张量，repeated
   `act_fn(gate) * up` 在 allocator 层造成显著 transient peak。
3. **In-flight buffer 多寡**：Async DBO 的 4 块 buffer vs Sync Simple 的
   2 块；前者已多消耗 ~96-200 MiB（取决于配置）。
4. **KV cache 预留**：`setup_kv_cache(B, max_seq_len)` 给 21 层
   attention 全量预分配 `B × max_seq_len × heads × dim`，这是固定开销，
   FFN 节点只剩 `~32 GiB − KV − weights ≈ 22 GiB` 给激活和通信。
5. **CUDA allocator 碎片化**：`expandable_segments:True` 缓解但不
   消除；MoE 的"小张量短生命"模式天然容易碎片化。Async 因为同时
   持有更多临时张量，碎片化进一步放大。

### 8.4 可验证 follow-up（未做）

如要把上面定量化，建议：

```python
# 在 ffn_worker.forward_ffn_layer 入口和出口
torch.cuda.reset_peak_memory_stats()
...
peak = torch.cuda.max_memory_allocated()
```

对相同 `(B, S)` 跑 serial 与 DBO，比较峰值。

---

## 9. Code Review 重点 / 已知技术债

| 文件:行 | 问题 | 严重度 |
|---|---|---|
| `src/pipeline/scheduler.py:400` | `send_sync` 阻塞到对端 recv，timing 中 SEND_TRANSFER 被夸大 | 中 |
| `src/pipeline/async_scheduler.py:670, 783` | `compute_stream.synchronize()` 让"双 stream"退化为串行 | 中 |
| `src/pipeline/decode_scheduler.py` | step 1 单次采样，方差大；建议 ≥5 次重复 | 中 |
| `src/distributed/warmup.py` | 只保留显式 warmup；不要恢复历史 keepalive 心跳 | 低 |
| `src/main.py:434` | `model.generate` 在 attn 节点跑、FFN 节点走另一分支，控制流分离不显眼 | 低 |
| `src/model/disaggregated.py:425` | `forward_prefill` 走全 batch（不切 MB），与 SimplePipelineScheduler 不一致 | 中 |

---

## 10. 与现有文档的关系

- `doc/01-architecture.md`：模块层级 / 类图。
- `doc/02-usage.md`：用户视角的 CLI 教程。
- `doc/03-api-reference.md`：函数级 API。
- `doc/04-deployment.md`：双机部署。
- **本文档（05）**：综合 review 入口，把以上几篇 + 调度器 / Profile /
  画图 / OOM 串成一条线，并给出每个论断的源代码位置。

---

附：阅读顺序建议  
**§1 → §3 → §4 → §8 → §9 →（选读）§2/§5/§6/§7**
