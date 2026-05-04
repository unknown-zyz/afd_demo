# Experiment report — serial b128 s2048 t8

## Configuration

- **Mode**: `serial`
- **Batch size**: 128
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
| Model-side generation total | 28681.566 ms | 28681.446 ms |
| Prefill / TTFT-path | 24455.536 ms | 24453.138 ms |
| Decode loop total | 4220.640 ms | 4220.635 ms |
| Decode steps | 7 | 7 |
| Decode TPOT | 602.949 ms | 602.948 ms |
| Legacy decode step (not exact TPOT) | - | - |
| Compute | - | - |
| Recv wait | - | - |
| MoE router | - | - |
| MoE experts | - | - |
| MoE shared/dense | - | - |
| Compute ratio | - | - |
| Tokens/sec | 0.28 | - |

- `Model-side generation total` is `total_time_ms` for the full generation call.
- The Attention/FFN columns are rank-level wall-clock views of the same serial run; they are not per-role compute decomposition.
- `Decode TPOT` is the serial decode baseline used for decode speedup.

## Layer averages summary

_No per-layer events recorded._

## Per-layer breakdown

_No per-layer events recorded._

---
_Generated from `timing_attention_serial_npu_ep7_broadcast_reduce_overlap_b128_s2048_t8.json` + `timing_ffn_coordinator_serial_npu_ep7_broadcast_reduce_overlap_b128_s2048_t8.json`._
