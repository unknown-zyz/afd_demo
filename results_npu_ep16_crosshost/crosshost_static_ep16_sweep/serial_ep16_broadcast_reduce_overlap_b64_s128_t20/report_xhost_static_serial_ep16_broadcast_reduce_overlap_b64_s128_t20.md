# Experiment report — serial b64 s128 t20

## Configuration

- **Mode**: `serial`
- **Batch size**: 64
- **Prefill seq**: 128
- **Decode tokens**: 20
- **Layers**: ?
- **Micro-batches**: ?
- **Comm timing mode**: `completion`
- **Requested prefill seq**: 128
- **Actual prompt len**: 128

## Serial timing (model-side total + decode fields)

| Metric | Attention rank view | FFN rank view |
|---|---:|---:|
| Model-side generation total | 15153.080 ms | 15152.680 ms |
| Prefill / TTFT-path | 6668.883 ms | 6667.670 ms |
| Decode loop total | 8258.509 ms | 8258.471 ms |
| Decode steps | 19 | 19 |
| Decode TPOT | 434.658 ms | 434.656 ms |
| Legacy decode step (not exact TPOT) | - | - |
| Compute | - | - |
| Recv wait | - | - |
| MoE router | - | - |
| MoE experts | - | - |
| MoE shared/dense | - | - |
| Compute ratio | - | - |
| Tokens/sec | 1.32 | - |

- `Model-side generation total` is `total_time_ms` for the full generation call.
- The Attention/FFN columns are rank-level wall-clock views of the same serial run; they are not per-role compute decomposition.
- `Decode TPOT` is the serial decode baseline used for decode speedup.

## Layer averages summary

_No per-layer events recorded._

## Per-layer breakdown

_No per-layer events recorded._

---
_Generated from `timing_attention_xhost_static_serial_ep16_broadcast_reduce_overlap_b64_s128_t20.json` + `timing_ffn_coordinator_xhost_static_serial_ep16_broadcast_reduce_overlap_b64_s128_t20.json`._
