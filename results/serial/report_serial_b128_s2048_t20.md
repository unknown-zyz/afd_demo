# Experiment report — serial b128 s2048 t20

## Configuration

- **Mode**: `serial`
- **Batch size**: 128
- **Prefill seq**: 2048
- **Decode tokens**: 20
- **Layers**: ?
- **Micro-batches**: ?

## Serial timing (model-side total + decode fields)

| Metric | Attention rank view | FFN rank view |
|---|---:|---:|
| Model-side generation total | 25891.677 ms | 25891.029 ms |
| Prefill / TTFT-path | 3121.868 ms | 2998.953 ms |
| Decode loop total | 22588.903 ms | 22588.891 ms |
| Decode steps | 19 | 19 |
| Decode TPOT | 1188.890 ms | 1188.889 ms |
| Legacy decode step (not exact TPOT) | - | - |
| Compute | - | - |
| Recv wait | - | - |
| MoE router | - | - |
| MoE experts | - | - |
| MoE shared/dense | - | - |
| Compute ratio | - | - |
| Tokens/sec | 0.77 | - |

- `Model-side generation total` is `total_time_ms` for the full generation call.
- The Attention/FFN columns are rank-level wall-clock views of the same serial run; they are not per-role compute decomposition.
- `Decode TPOT` is the serial decode baseline used for decode speedup.

## Layer averages summary

_No per-layer events recorded._

## Per-layer breakdown

_No per-layer events recorded._

---
_Generated from `timing_attention_serial_b128_s2048_t20.json` + `timing_ffn_serial_b128_s2048_t20.json`._
