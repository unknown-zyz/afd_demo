# Experiment report — serial b16 s256 t20

## Configuration

- **Mode**: `serial`
- **Batch size**: 16
- **Prefill seq**: 256
- **Decode tokens**: 20
- **Layers**: ?
- **Micro-batches**: ?

## Serial timing (model-side total + decode fields)

| Metric | Attention rank view | FFN rank view |
|---|---:|---:|
| Model-side generation total | 14359.732 ms | 14359.646 ms |
| Prefill / TTFT-path | 2162.702 ms | 2144.721 ms |
| Decode loop total | 11969.672 ms | 11969.637 ms |
| Decode steps | 19 | 19 |
| Decode TPOT | 629.983 ms | 629.981 ms |
| Legacy decode step (not exact TPOT) | - | - |
| Compute | - | - |
| Recv wait | - | - |
| MoE router | - | - |
| MoE experts | - | - |
| MoE shared/dense | - | - |
| Compute ratio | - | - |
| Tokens/sec | 1.39 | - |

- `Model-side generation total` is `total_time_ms` for the full generation call.
- The Attention/FFN columns are rank-level wall-clock views of the same serial run; they are not per-role compute decomposition.
- `Decode TPOT` is the serial decode baseline used for decode speedup.

## Layer averages summary

_No per-layer events recorded._

## Per-layer breakdown

_No per-layer events recorded._

---
_Generated from `timing_attention_serial_b16_s256_t20.json` + `timing_ffn_serial_b16_s256_t20.json`._
