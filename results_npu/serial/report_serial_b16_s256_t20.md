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
| Model-side generation total | 11506.989 ms | 11506.725 ms |
| Prefill / TTFT-path | 1828.426 ms | 1827.473 ms |
| Decode loop total | 9556.811 ms | 9556.819 ms |
| Decode steps | 19 | 19 |
| Decode TPOT | 502.990 ms | 502.990 ms |
| Legacy decode step (not exact TPOT) | - | - |
| Compute | - | - |
| Recv wait | - | - |
| MoE router | - | - |
| MoE experts | - | - |
| MoE shared/dense | - | - |
| Compute ratio | - | - |
| Tokens/sec | 1.74 | - |

- `Model-side generation total` is `total_time_ms` for the full generation call.
- The Attention/FFN columns are rank-level wall-clock views of the same serial run; they are not per-role compute decomposition.
- `Decode TPOT` is the serial decode baseline used for decode speedup.

## Layer averages summary

_No per-layer events recorded._

## Per-layer breakdown

_No per-layer events recorded._

---
_Generated from `timing_attention_serial_b16_s256_t20.json` + `timing_ffn_serial_b16_s256_t20.json`._
