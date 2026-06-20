# Experiment report — serial b16 s1024 t20

## Configuration

- **Mode**: `serial`
- **Batch size**: 16
- **Prefill seq**: 1024
- **Decode tokens**: 20
- **Layers**: ?
- **Micro-batches**: ?
- **Comm timing mode**: `completion`
- **Requested prefill seq**: 1024
- **Actual prompt len**: 1024

## Serial timing (model-side total + decode fields)

| Metric | Attention rank view | FFN rank view |
|---|---:|---:|
| Model-side generation total | 12441.293 ms | 12441.617 ms |
| Prefill / TTFT-path | 7591.693 ms | 7590.997 ms |
| Decode loop total | 4760.951 ms | 4760.978 ms |
| Decode steps | 19 | 19 |
| Decode TPOT | 250.576 ms | 250.578 ms |
| Legacy decode step (not exact TPOT) | - | - |
| Compute | - | - |
| Recv wait | - | - |
| MoE router | - | - |
| MoE experts | - | - |
| MoE shared/dense | - | - |
| Compute ratio | - | - |
| Tokens/sec | 1.61 | - |

- `Model-side generation total` is `total_time_ms` for the full generation call.
- The Attention/FFN columns are rank-level wall-clock views of the same serial run; they are not per-role compute decomposition.
- `Decode TPOT` is the serial decode baseline used for decode speedup.

## Layer averages summary

_No per-layer events recorded._

## Per-layer breakdown

_No per-layer events recorded._

---
_Generated from `timing_attention_xhost_static_serial_ep16_broadcast_reduce_overlap_b16_s1024_t20.json` + `timing_ffn_coordinator_xhost_static_serial_ep16_broadcast_reduce_overlap_b16_s1024_t20.json`._
