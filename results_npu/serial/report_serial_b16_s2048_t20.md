# Experiment report — serial b16 s2048 t20

## Configuration

- **Mode**: `serial`
- **Batch size**: 16
- **Prefill seq**: 2048
- **Decode tokens**: 20
- **Layers**: ?
- **Micro-batches**: ?

## Serial timing (model-side total + decode fields)

| Metric | Attention rank view | FFN rank view |
|---|---:|---:|
| Model-side generation total | 11012.704 ms | 11012.636 ms |
| Prefill / TTFT-path | 1808.416 ms | 1807.805 ms |
| Decode loop total | 9083.618 ms | 9083.624 ms |
| Decode steps | 19 | 19 |
| Decode TPOT | 478.085 ms | 478.085 ms |
| Legacy decode step (not exact TPOT) | - | - |
| Compute | - | - |
| Recv wait | - | - |
| MoE router | - | - |
| MoE experts | - | - |
| MoE shared/dense | - | - |
| Compute ratio | - | - |
| Tokens/sec | 1.82 | - |

- `Model-side generation total` is `total_time_ms` for the full generation call.
- The Attention/FFN columns are rank-level wall-clock views of the same serial run; they are not per-role compute decomposition.
- `Decode TPOT` is the serial decode baseline used for decode speedup.

## Layer averages summary

_No per-layer events recorded._

## Per-layer breakdown

_No per-layer events recorded._

---
_Generated from `timing_attention_serial_b16_s2048_t20.json` + `timing_ffn_serial_b16_s2048_t20.json`._
