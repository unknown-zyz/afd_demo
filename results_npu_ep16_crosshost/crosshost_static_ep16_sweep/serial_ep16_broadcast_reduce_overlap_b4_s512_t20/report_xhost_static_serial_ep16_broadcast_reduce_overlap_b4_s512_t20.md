# Experiment report — serial b4 s512 t20

## Configuration

- **Mode**: `serial`
- **Batch size**: 4
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
| Model-side generation total | 8199.329 ms | 8199.345 ms |
| Prefill / TTFT-path | 4432.757 ms | 4400.509 ms |
| Decode loop total | 3701.097 ms | 3701.092 ms |
| Decode steps | 19 | 19 |
| Decode TPOT | 194.795 ms | 194.794 ms |
| Legacy decode step (not exact TPOT) | - | - |
| Compute | - | - |
| Recv wait | - | - |
| MoE router | - | - |
| MoE experts | - | - |
| MoE shared/dense | - | - |
| Compute ratio | - | - |
| Tokens/sec | 2.44 | - |

- `Model-side generation total` is `total_time_ms` for the full generation call.
- The Attention/FFN columns are rank-level wall-clock views of the same serial run; they are not per-role compute decomposition.
- `Decode TPOT` is the serial decode baseline used for decode speedup.

## Layer averages summary

_No per-layer events recorded._

## Per-layer breakdown

_No per-layer events recorded._

---
_Generated from `timing_attention_xhost_static_serial_ep16_broadcast_reduce_overlap_b4_s512_t20.json` + `timing_ffn_coordinator_xhost_static_serial_ep16_broadcast_reduce_overlap_b4_s512_t20.json`._
