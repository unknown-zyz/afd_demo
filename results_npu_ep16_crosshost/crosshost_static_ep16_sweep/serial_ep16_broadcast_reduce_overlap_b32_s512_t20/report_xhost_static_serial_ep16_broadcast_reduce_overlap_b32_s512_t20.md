# Experiment report — serial b32 s512 t20

## Configuration

- **Mode**: `serial`
- **Batch size**: 32
- **Prefill seq**: 512
- **Decode tokens**: 20
- **Layers**: ?
- **Micro-batches**: ?
- **Comm timing mode**: `completion`
- **Requested prefill seq**: 512
- **Actual prompt len**: 512

## Serial timing (model-side total + decode fields)

| Metric | Attention rank view | FFN rank view |
|---|---:|---:|
| Model-side generation total | 13266.056 ms | 13266.116 ms |
| Prefill / TTFT-path | 7423.544 ms | 7422.873 ms |
| Decode loop total | 5722.135 ms | 5722.148 ms |
| Decode steps | 19 | 19 |
| Decode TPOT | 301.165 ms | 301.166 ms |
| Legacy decode step (not exact TPOT) | - | - |
| Compute | - | - |
| Recv wait | - | - |
| MoE router | - | - |
| MoE experts | - | - |
| MoE shared/dense | - | - |
| Compute ratio | - | - |
| Tokens/sec | 1.51 | - |

- `Model-side generation total` is `total_time_ms` for the full generation call.
- The Attention/FFN columns are rank-level wall-clock views of the same serial run; they are not per-role compute decomposition.
- `Decode TPOT` is the serial decode baseline used for decode speedup.

## Layer averages summary

_No per-layer events recorded._

## Per-layer breakdown

_No per-layer events recorded._

---
_Generated from `timing_attention_xhost_static_serial_ep16_broadcast_reduce_overlap_b32_s512_t20.json` + `timing_ffn_coordinator_xhost_static_serial_ep16_broadcast_reduce_overlap_b32_s512_t20.json`._
