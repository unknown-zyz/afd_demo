# Experiment report — serial b96 s256 t20

## Configuration

- **Mode**: `serial`
- **Batch size**: 96
- **Prefill seq**: 256
- **Decode tokens**: 20
- **Layers**: ?
- **Micro-batches**: ?

## Serial timing (model-side total + decode fields)

| Metric | Attention rank view | FFN rank view |
|---|---:|---:|
| Model-side generation total | 23550.425 ms | 23549.785 ms |
| Prefill / TTFT-path | 2827.276 ms | 2733.441 ms |
| Decode loop total | 20491.740 ms | 20491.771 ms |
| Decode steps | 19 | 19 |
| Decode TPOT | 1078.513 ms | 1078.514 ms |
| Legacy decode step (not exact TPOT) | - | - |
| Compute | - | - |
| Recv wait | - | - |
| MoE router | - | - |
| MoE experts | - | - |
| MoE shared/dense | - | - |
| Compute ratio | - | - |
| Tokens/sec | 0.85 | - |

- `Model-side generation total` is `total_time_ms` for the full generation call.
- The Attention/FFN columns are rank-level wall-clock views of the same serial run; they are not per-role compute decomposition.
- `Decode TPOT` is the serial decode baseline used for decode speedup.

## Layer averages summary

_No per-layer events recorded._

## Per-layer breakdown

_No per-layer events recorded._

---
_Generated from `timing_attention_serial_b96_s256_t20.json` + `timing_ffn_serial_b96_s256_t20.json`._
