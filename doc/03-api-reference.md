# API 参考

本文只记录当前主路径仍在使用的接口；已删除的历史 `PipelineScheduler` 抽象、`LayerCommunicator`、手写 `KVCacheManager` 和 baseline validation helper 不再作为公开 API。

## 1. 分布式上下文

**文件**: `src/distributed/__init__.py`

```python
from src.distributed import init_distributed, get_distributed_context, DistributedConfig
```

`DistributedContext` 负责：

| 属性/方法 | 说明 |
|---|---|
| `rank`, `world_size`, `local_rank` | 分布式 rank 信息 |
| `role` | `"attention"` 或 `"ffn"` |
| `device` | 当前 rank 使用的 CUDA device |
| `peer_rank` | 对端 rank |
| `a2f_group`, `f2a_group` | Decode cross-layer 使用的方向 NCCL group |
| `initialize(config)` | 初始化 process group |
| `warmup(num_rounds=3)` | 预热 P2P 和已创建的方向 group |
| `barrier()` / `cleanup()` | 同步与清理 |

## 2. 通信

**文件**: `src/distributed/communicator.py`

`AFDCommunicator` 是模型同步路径使用的通信封装，提供：

| 方法 | 说明 |
|---|---|
| `send_sync(tensor, tag)` | 同步发送 |
| `recv_sync(shape, tag)` | 同步接收 |
| `send_async(tensor, tag)` | 使用内部 buffer 异步发送 |
| `recv_async(shape, tag)` / `wait_recv(idx)` | 异步接收并等待 |
| `wait_all_sends()` / `wait_send()` | 等待 pending send 完成 |

Prefill / decode DBO 中的高性能路径直接使用 `torch.distributed.isend/irecv`，以便控制 NCCL group、tensor lifetime 和 timing。

## 3. 模型入口

**文件**: `src/model/disaggregated.py`

```python
model = DisaggregatedQwenModel.from_pretrained(
    model_name,
    device=device,
    dtype=torch.bfloat16,
    max_seq_len=seq,
    max_batch_size=batch,
)
```

关键方法：

| 方法 | 说明 |
|---|---|
| `forward_prefill(input_ids, attention_mask=None)` | 不经 scheduler 的同步 prefill 路径 |
| `forward_layer_sync(...)` | 单层同步 AFD 前向，被 `SimplePipelineScheduler` 调用 |
| `generate(...)` | prefill 后自回归 decode，可启用 decode DBO |

KV cache 当前使用 HuggingFace `DynamicCache`，保存在 Attention 节点，不再维护仓库内手写 cache manager。

## 4. Workers

### `AttentionWorker`

**文件**: `src/model/attention_worker.py`

| 方法 | 说明 |
|---|---|
| `embed(input_ids)` | token embedding |
| `get_position_embeddings(hidden_states, position_ids)` | RoPE position embeddings |
| `forward_attention_layer(...)` | 单层 attention + residual |
| `forward_lm_head(hidden_states)` | logits |

### `FFNWorker`

**文件**: `src/model/ffn_worker.py`

| 方法 | 说明 |
|---|---|
| `forward_ffn_layer(layer_idx, hidden_states)` | 单层 FFN/MoE |
| `supports_moe_timing` | 是否可记录 MoE router/expert 分段 |

## 5. 调度器

### `SimplePipelineScheduler`

**文件**: `src/pipeline/scheduler.py`

同步 serial baseline。逐层逐 micro-batch 执行：

```text
Attention compute -> A2F send -> FFN compute -> F2A send
```

构造：

```python
SimplePipelineScheduler(model, num_micro_batches=2, enable_timing=False)
```

### `AsyncPipelineScheduler`

**文件**: `src/pipeline/async_scheduler.py`

Prefill DBO。用 micro-batch + `isend/irecv` 尝试重叠通信和计算。

```python
AsyncPipelineScheduler(
    model,
    num_micro_batches=2,
    use_cuda_streams=True,
    enable_timing=True,
    timing_mode="cuda_events",
)
```

### `DecodeDBOScheduler`

**文件**: `src/pipeline/decode_scheduler.py`

Decode DBO。默认记录 step 1 作为 representative step；`use_crosslayer=True` 时使用 `a2f_group` / `f2a_group` 拆分方向通信。

```python
DecodeDBOScheduler(
    model,
    num_micro_batches=2,
    enable_timing=True,
    timing_mode="cuda_events",
    use_crosslayer=False,
)
```

## 6. Timing

**文件**: `src/utils/timing.py`

| 类型 | 说明 |
|---|---|
| `EventType` | `ATTN_COMPUTE` / `FFN_COMPUTE` / `SEND_TRANSFER` / `RECV_WAIT` 等 |
| `TimingTracker` | 记录 per-layer/per-micro-batch 事件 |
| `PipelineTiming.save(path)` | 保存 JSON |

`timing_mode="cuda_events"` 是默认路径；`sync` 只用于调试，会改变 pipeline 行为和开销。

## 7. 工具函数

| 文件 | 接口 |
|---|---|
| `src/utils/sampling.py` | `sample_next_token`、top-k/top-p filtering |
| `src/utils/profiler.py` | `Timer`、`CUDATimer`、`profile_function` |
| `src/utils/validation.py` | `compare_tensors`、`validate_output` |

## 8. 脚本 API

| 脚本 | 输入 | 输出 |
|---|---|---|
| `scripts/run_single.sh` | deployment/batch/seq/options | `results/prefill_dbo/timing_*.json` + report |
| `scripts/run_experiment_matrix.sh` | mode/batch/seq matrix | `results/{mode}/` + summary CSV |
| `scripts/gen_experiment_report.py` | attention/ffn timing JSON | markdown report |
| `scripts/visualize_dbo_pipeline.py` | attention/ffn timing JSON | PNG |
| `scripts/plot_all_pipelines.py` | results root | batch PNG + `pipelines_index.md` |
