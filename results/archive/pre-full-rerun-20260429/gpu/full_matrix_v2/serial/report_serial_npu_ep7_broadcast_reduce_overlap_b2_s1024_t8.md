# Experiment report — serial b2 s1024 t8

## Configuration

- **Mode**: `serial`
- **Batch size**: 2
- **Prefill seq**: 1024
- **Decode tokens**: 8
- **Layers**: ?
- **Micro-batches**: ?
- **Comm timing mode**: `enqueue`
- **Requested prefill seq**: 1024
- **Actual prompt len**: 1024

## Serial timing (model-side total + decode fields)

| Metric | Attention rank view | FFN rank view |
|---|---:|---:|
| Model-side generation total | 3287.798 ms | 3287.759 ms |
| Prefill / TTFT-path | 2536.653 ms | 2535.981 ms |
| Decode loop total | 747.297 ms | 747.308 ms |
| Decode steps | 7 | 7 |
| Decode TPOT | 106.757 ms | 106.758 ms |
| Legacy decode step (not exact TPOT) | - | - |
| Compute | - | - |
| Recv wait | - | - |
| MoE router | - | - |
| MoE experts | - | - |
| MoE shared/dense | - | - |
| Compute ratio | - | - |
| Tokens/sec | 2.43 | - |

- `Model-side generation total` is `total_time_ms` for the full generation call.
- The Attention/FFN columns are rank-level wall-clock views of the same serial run; they are not per-role compute decomposition.
- `Decode TPOT` is the serial decode baseline used for decode speedup.

## Layer averages summary

_No per-layer events recorded._

## Per-layer breakdown

_No per-layer events recorded._

---
_Generated from `timing_attention_serial_npu_ep7_broadcast_reduce_overlap_b2_s1024_t8.json` + `timing_ffn_coordinator_serial_npu_ep7_broadcast_reduce_overlap_b2_s1024_t8.json`._
