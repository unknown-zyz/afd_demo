# Experiment report — serial b128 s256 t20

## Configuration

- **Mode**: `serial`
- **Batch size**: 128
- **Prefill seq**: 256
- **Decode tokens**: 20
- **Layers**: ?
- **Micro-batches**: ?

## Serial timing (model-side total + decode fields)

| Metric | Attention rank view | FFN rank view |
|---|---:|---:|
| Model-side generation total | 25350.047 ms | 25350.000 ms |
| Prefill / TTFT-path | 3101.923 ms | 2980.160 ms |
| Decode loop total | 22064.613 ms | 22064.517 ms |
| Decode steps | 19 | 19 |
| Decode TPOT | 1161.295 ms | 1161.290 ms |
| Legacy decode step (not exact TPOT) | - | - |
| Compute | - | - |
| Recv wait | - | - |
| MoE router | - | - |
| MoE experts | - | - |
| MoE shared/dense | - | - |
| Compute ratio | - | - |
| Tokens/sec | 0.79 | - |

- `Model-side generation total` is `total_time_ms` for the full generation call.
- The Attention/FFN columns are rank-level wall-clock views of the same serial run; they are not per-role compute decomposition.
- `Decode TPOT` is the serial decode baseline used for decode speedup.

## Layer averages summary

_No per-layer events recorded._

## Per-layer breakdown

_No per-layer events recorded._

---
_Generated from `timing_attention_serial_b128_s256_t20.json` + `timing_ffn_serial_b128_s256_t20.json`._
