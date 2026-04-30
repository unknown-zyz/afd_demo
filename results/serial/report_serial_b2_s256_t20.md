# Experiment report — serial b2 s256 t20

## Configuration

- **Mode**: `serial`
- **Batch size**: 2
- **Prefill seq**: 256
- **Decode tokens**: 20
- **Layers**: ?
- **Micro-batches**: ?

## Serial timing (model-side total + decode fields)

| Metric | Attention rank view | FFN rank view |
|---|---:|---:|
| Model-side generation total | 6497.464 ms | 6496.868 ms |
| Prefill / TTFT-path | 2096.785 ms | 2092.118 ms |
| Decode loop total | 4174.770 ms | 4174.776 ms |
| Decode steps | 19 | 19 |
| Decode TPOT | 219.725 ms | 219.725 ms |
| Legacy decode step (not exact TPOT) | - | - |
| Compute | - | - |
| Recv wait | - | - |
| MoE router | - | - |
| MoE experts | - | - |
| MoE shared/dense | - | - |
| Compute ratio | - | - |
| Tokens/sec | 3.08 | - |

- `Model-side generation total` is `total_time_ms` for the full generation call.
- The Attention/FFN columns are rank-level wall-clock views of the same serial run; they are not per-role compute decomposition.
- `Decode TPOT` is the serial decode baseline used for decode speedup.

## Layer averages summary

_No per-layer events recorded._

## Per-layer breakdown

_No per-layer events recorded._

---
_Generated from `timing_attention_serial_b2_s256_t20.json` + `timing_ffn_serial_b2_s256_t20.json`._
