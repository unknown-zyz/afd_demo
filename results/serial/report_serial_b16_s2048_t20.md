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
| Model-side generation total | 12995.143 ms | 12995.076 ms |
| Prefill / TTFT-path | 2167.426 ms | 2149.777 ms |
| Decode loop total | 10597.072 ms | 10597.087 ms |
| Decode steps | 19 | 19 |
| Decode TPOT | 557.741 ms | 557.741 ms |
| Legacy decode step (not exact TPOT) | - | - |
| Compute | - | - |
| Recv wait | - | - |
| MoE router | - | - |
| MoE experts | - | - |
| MoE shared/dense | - | - |
| Compute ratio | - | - |
| Tokens/sec | 1.54 | - |

- `Model-side generation total` is `total_time_ms` for the full generation call.
- The Attention/FFN columns are rank-level wall-clock views of the same serial run; they are not per-role compute decomposition.
- `Decode TPOT` is the serial decode baseline used for decode speedup.

## Layer averages summary

_No per-layer events recorded._

## Per-layer breakdown

_No per-layer events recorded._

---
_Generated from `timing_attention_serial_b16_s2048_t20.json` + `timing_ffn_serial_b16_s2048_t20.json`._
