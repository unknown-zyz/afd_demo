# Experiment report — serial b256 s512 t20

## Configuration

- **Mode**: `serial`
- **Batch size**: 256
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
| Model-side generation total | 37668.930 ms | 37669.533 ms |
| Prefill / TTFT-path | 17488.510 ms | 17486.654 ms |
| Decode loop total | 19416.573 ms | 19416.744 ms |
| Decode steps | 19 | 19 |
| Decode TPOT | 1021.925 ms | 1021.934 ms |
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
_Generated from `timing_attention_xhost_static_serial_ep16_broadcast_reduce_overlap_b256_s512_t20.json` + `timing_ffn_coordinator_xhost_static_serial_ep16_broadcast_reduce_overlap_b256_s512_t20.json`._
