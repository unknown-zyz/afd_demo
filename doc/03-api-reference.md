# API reference

This document lists only the current supported code and script surfaces.

## 1. Backend/device helpers

File: `src/utils/device.py`

| API | Purpose |
|---|---|
| `init_backend(name)` | Resolve and initialize `cuda`, `npu`, `cpu`, or `auto`. |
| `DEVICE_TYPE` | Current logical device type. |
| `DIST_BACKEND` | Distributed backend (`nccl`, `hccl`, or `gloo`). |
| `device_module()` | Return backend module such as `torch.cuda` / `torch.npu`. |
| `set_device`, `synchronize`, `Stream`, `Event` | Backend-neutral accelerator helpers. |
| `apply_backend_envs()` | Apply backend-specific environment defaults. |

NPU support relies on `torch_npu.contrib.transfer_to_npu` so much of the CUDA
call surface maps to NPU at runtime.

## 2. Distributed context

File: `src/distributed/__init__.py`

```python
from src.distributed import DistributedConfig, init_distributed, get_distributed_context
```

| Attribute/method | Meaning |
|---|---|
| `rank`, `world_size`, `local_rank` | Distributed rank metadata. |
| `role` | `attention` or `ffn`. |
| `device` | Current backend device. |
| `peer_rank` | Peer rank for the 2-rank A/F path. |
| `a2f_group`, `f2a_group` | Directional groups used by decode crosslayer. |
| `initialize(config)` | Initialize process group and device. |
| `warmup(num_rounds)` | Warm up P2P communication. |
| `barrier()`, `cleanup()` | Synchronization and teardown helpers. |

## 3. Model

File: `src/model/disaggregated.py`

```python
model = DisaggregatedQwenModel.from_pretrained(
    model_name,
    device=device,
    dtype=torch.bfloat16,
    max_seq_len=seq,
    max_batch_size=batch,
)
```

| Method | Meaning |
|---|---|
| `forward_prefill(input_ids, attention_mask=None)` | Full-batch prefill helper without scheduler overlap. |
| `forward_layer_sync(...)` | One layer of synchronous AF execution for serial scheduling. |
| `forward_decode(next_token)` | One decode token path without decode DBO. |
| `generate(...)` | Prefill + autoregressive generation, optionally with decode DBO/crosslayer. |

KV cache is HuggingFace `DynamicCache`, owned by the attention role.

## 4. Workers

### `AttentionWorker`

File: `src/model/attention_worker.py`

| Method | Meaning |
|---|---|
| `embed(input_ids)` | Token embedding. |
| `get_position_embeddings(hidden_states, position_ids)` | RoPE position embedding preparation. |
| `forward_attention_layer(...)` | One attention layer plus residual packaging. |
| `forward_lm_head(hidden_states)` | Final logits. |

### `FFNWorker`

File: `src/model/ffn_worker.py`

| Method/property | Meaning |
|---|---|
| `forward_ffn_layer(layer_idx, hidden_states, ...)` | One FFN/MoE layer. |
| `supports_moe_timing` | Whether router/expert timing can be emitted. |

## 5. Schedulers

### `SimplePipelineScheduler`

File: `src/pipeline/scheduler.py`

Serial AF baseline:

```text
Attention compute -> A2F transfer -> FFN compute -> F2A transfer
```

Constructed by `src/main.py` when `--no-dbo` is set.

### `AsyncPipelineScheduler`

File: `src/pipeline/async_scheduler.py`

Prefill DBO scheduler. Constructed when DBO is enabled and generation is
disabled (`--no-generate`). Key options:

| Argument | Meaning |
|---|---|
| `num_micro_batches` | Number of micro-batches, normally `2`. |
| `use_cuda_streams` | Use accelerator streams for compute/comm separation. |
| `enable_timing` | Emit timing events. |
| `timing_mode` | `cuda_events` or `sync`. |

### `DecodeDBOScheduler`

File: `src/pipeline/decode_scheduler.py`

Decode DBO scheduler. Constructed by `DisaggregatedQwenModel.generate` when DBO
is enabled for generation.

| Argument | Meaning |
|---|---|
| `num_micro_batches` | Decode micro-batches. |
| `use_crosslayer` | Enable cross-layer directional pipeline. |
| `enable_timing` | Emit representative decode pipeline events. |
| `timing_mode` | `cuda_events` or `sync`. |

## 6. Timing JSON

File: `src/utils/timing.py`

| Concept | Meaning |
|---|---|
| `TimingTracker` | Records per-layer/per-micro-batch events. |
| `PipelineTiming.save(path)` | Writes JSON timing data. |
| `ATTN_COMPUTE` / `FFN_COMPUTE` | Compute events. |
| `SEND_TRANSFER` / `RECV_WAIT` | Communication events. |

Important JSON fields:

| Field | Meaning |
|---|---|
| `total_time_ms` | Timed path total. |
| `prefill_ms` | Serial prefill baseline. |
| `decode_loop_ms`, `decode_steps`, `decode_tpot_ms` | Exact decode TPOT components. |
| `events` | Representative Gantt events. |

## 7. Script APIs

| Script | Purpose |
|---|---|
| `scripts/run_single.sh` | One GPU local/multinode config. |
| `scripts/run_experiment_matrix.sh` | GPU matrix runner for serial/prefill/decode/crosslayer. |
| `scripts/run_npu.sh` | One local NPU/HCCL config. |
| `scripts/run_experiment_matrix_npu.sh` | NPU matrix runner under `results_npu/`. |
| `scripts/gen_experiment_report.py` | Timing JSON pair -> markdown report. |
| `scripts/visualize_dbo_pipeline.py` | Timing JSON pair -> pipeline PNG. |
| `scripts/plot_all_pipelines.py` | Batch plot all successful DBO timings under a result root. |
| `scripts/audit_experiment_baselines.py` | Check mode-matched serial baseline availability. |
| `scripts/experiment_baselines.py` | Shared TTFT/TPOT baseline resolution helpers. |
| `scripts/capture_serial_split.py` | Utility for enriching serial cache fields when needed. |

Removed historical helper surfaces such as hand-written KV cache managers,
generic pipeline base classes, and legacy fallback speedup logic are not public
APIs.
