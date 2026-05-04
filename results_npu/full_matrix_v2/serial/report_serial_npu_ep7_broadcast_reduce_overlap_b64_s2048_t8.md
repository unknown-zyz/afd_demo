# Experiment report — serial b64 s2048 t8

## Configuration

- **Mode**: `serial`
- **Batch size**: 64
- **Prefill seq**: 2048
- **Decode tokens**: 8
- **Layers**: ?
- **Micro-batches**: ?
- **Comm timing mode**: `enqueue`
- **Requested prefill seq**: 2048
- **Actual prompt len**: 2048

## Serial timing (model-side total + decode fields)

| Metric | Attention rank view | FFN rank view |
|---|---:|---:|
| Model-side generation total | 15049.650 ms | 15049.712 ms |
| Prefill / TTFT-path | 12394.502 ms | 12393.198 ms |
| Decode loop total | 2650.518 ms | 2650.533 ms |
| Decode steps | 7 | 7 |
| Decode TPOT | 378.645 ms | 378.648 ms |
| Legacy decode step (not exact TPOT) | - | - |
| Compute | - | - |
| Recv wait | - | - |
| MoE router | - | - |
| MoE experts | - | - |
| MoE shared/dense | - | - |
| Compute ratio | - | - |
| Tokens/sec | 0.53 | - |

- `Model-side generation total` is `total_time_ms` for the full generation call.
- The Attention/FFN columns are rank-level wall-clock views of the same serial run; they are not per-role compute decomposition.
- `Decode TPOT` is the serial decode baseline used for decode speedup.

## Layer averages summary

_No per-layer events recorded._

## Per-layer breakdown

_No per-layer events recorded._

---
_Generated from `timing_attention_serial_npu_ep7_broadcast_reduce_overlap_b64_s2048_t8.json` + `timing_ffn_coordinator_serial_npu_ep7_broadcast_reduce_overlap_b64_s2048_t8.json`._
