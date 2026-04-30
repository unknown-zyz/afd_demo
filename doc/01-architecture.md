# Architecture

## 1. Goal

AFD Demo validates **Attention/FFN disaggregated inference** and **Dual Batch
Overlap (DBO)** for Qwen3-30B-A3B. The model is split by function rather than by
consecutive layers:

```text
Attention role                         FFN role
------------------------------         ------------------------------
Embedding                              Post-attention layernorm
Self-attention                         MLP / MoE router / experts
KV cache                               FFN output
LM head / sampling
```

The two roles exchange hidden states every transformer layer. The attention
role owns KV cache because only attention needs keys/values during decode.

## 2. Runtime layout

```text
input_ids
   |
   v
Attention role: embedding + attention
   |
   | A2F hidden/residual transfer
   v
FFN role: layernorm + MLP/MoE
   |
   | F2A hidden transfer
   v
Attention role: next layer or lm_head
```

On GPU local mode, `scripts/run_single.sh local` uses:

| Role | Default visible GPUs |
|---|---|
| Attention | `0,1` |
| FFN | `2,3` |

On NPU, `scripts/run_npu.sh` starts local ranks with HCCL. The validated matrix
uses active topology `attn_size=1`, `ffn_size=1`, `ffn_tp_size=1`; the visible
chip pool can be larger than the active world size.

## 3. Core modules

| Component | File | Responsibility |
|---|---|---|
| Backend helpers | `src/utils/device.py` | Select CUDA/NPU/CPU backend and wrap device APIs. |
| Distributed context | `src/distributed/__init__.py` | Rank, role, backend, peer ranks, directional groups. |
| Communicator | `src/distributed/communicator.py` | Basic synchronous/asynchronous tensor transfer helper. |
| Attention worker | `src/model/attention_worker.py` | Embedding, attention layers, RoPE, LM head. |
| FFN worker | `src/model/ffn_worker.py` | LayerNorm, MLP/MoE router and experts. |
| Top-level model | `src/model/disaggregated.py` | Prefill/generation orchestration and KV cache. |
| Serial scheduler | `src/pipeline/scheduler.py` | Synchronous AF baseline. |
| Prefill DBO scheduler | `src/pipeline/async_scheduler.py` | Micro-batch overlap for prefill. |
| Decode DBO scheduler | `src/pipeline/decode_scheduler.py` | Decode micro-batch and optional cross-layer pipeline. |
| Timing | `src/utils/timing.py` | Per-layer/per-micro-batch event recording. |

## 4. Schedulers

### 4.1 Serial baseline: `SimplePipelineScheduler`

Serial mode disables DBO:

```bash
./scripts/run_single.sh local 4 128 --tokens 20 --no-dbo --generate
```

For each layer and micro-batch:

```text
Attention compute -> A2F send -> FFN compute -> F2A send -> next layer
```

This path is used as the mode-matched baseline. Serial timing cache files live
under `results/serial/cache/` or `results_npu/serial/cache/`.

### 4.2 Prefill DBO: `AsyncPipelineScheduler`

Prefill DBO is the default single-run path:

```bash
./scripts/run_single.sh local 4 128 --tokens 20
```

It splits the batch into micro-batches, uses asynchronous P2P sends/receives,
and overlaps A/FFN work where backend scheduling permits. Prefill runs over the
full prompt length, so activation memory scales with `batch * seq`.

### 4.3 Decode DBO: `DecodeDBOScheduler`

Decode DBO is enabled for generation when DBO is not disabled:

```bash
./scripts/run_single.sh local 4 128 --tokens 20 --generate
```

The prefill phase builds the KV cache. Then each decode step processes one new
token per request while reusing the cache. Timing uses exact decode-loop TPOT:

```text
decode_tpot_ms = decode_loop_ms / decode_steps
decode_steps = max_new_tokens - 1
```

The first sampled token after prefill belongs to the TTFT-path, not the decode
loop.

### 4.4 Decode cross-layer

Cross-layer decode uses separate directional process groups and posts selected
receive/send operations earlier across layers:

```bash
./scripts/run_single.sh local 4 128 --tokens 20 --generate --crosslayer
```

It is experimental and must be judged by exact TPOT, not by one representative
Gantt sample.

## 5. KV cache

KV cache uses HuggingFace `DynamicCache` and is owned by the attention role.
Decode DBO slices cache by batch/micro-batch and keeps the cache local to the
attention side; keys/values are not transferred to the FFN role.

## 6. Timing and metrics

Each timing run writes one JSON per role:

```text
timing_attention_<tag>.json
timing_ffn_<tag>.json
```

Important fields:

| Field | Meaning |
|---|---|
| `total_time_ms` | Mode-dependent timed path. For prefill it is the prefill TTFT-path. |
| `prefill_ms` | Serial prefill-only baseline field. |
| `decode_loop_ms` | Full decode loop time excluding the prefill first-token path. |
| `decode_steps` | Number of decode loop steps, normally `max_new_tokens - 1`. |
| `decode_tpot_ms` | Exact TPOT. |
| `events` | Representative per-layer/per-micro-batch events used for Gantt plots. |

Speedup semantics:

| Mode | Metric | Formula |
|---|---|---|
| Prefill DBO | model-side TTFT-path | `serial_prefill_ms / dbo_total_time_ms` |
| Decode DBO | exact TPOT | `serial_decode_tpot_ms / dbo_decode_tpot_ms` |
| Cross-layer decode | exact TPOT | `serial_decode_tpot_ms / dbo_decode_tpot_ms` |

`events` can show overlap/bubbles, but they are not the denominator for speedup.

## 7. Backend abstraction

`src/utils/device.py` resolves:

| Backend | Device API | Distributed backend |
|---|---|---|
| CUDA | `torch.cuda` | NCCL |
| NPU | `torch.npu` via `torch_npu.contrib.transfer_to_npu` | HCCL |
| CPU | no accelerator module | Gloo |

The same high-level scheduler code runs on CUDA and NPU. NPU launch scripts set
HCCL-specific environment variables and device pools.

## 8. Result interpretation

Do not use old Qwen2-era performance notes as current guidance. The current
Qwen3 GPU/NPU conclusions are maintained in:

- `doc/gpu_npu_experiment_summary.md`
- `results/baseline_audit.csv`
- `results_npu/baseline_audit.csv`

OOM rows in matrix summaries are capacity boundaries, not missing data.
