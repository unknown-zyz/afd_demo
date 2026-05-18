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
| Model-side generation total | 22055.438 ms | 22055.287 ms |
| Prefill / TTFT-path | 1946.123 ms | 1943.868 ms |
| Decode loop total | 19685.664 ms | 19685.656 ms |
| Decode steps | 19 | 19 |
| Decode TPOT | 1036.088 ms | 1036.087 ms |
| Legacy decode step (not exact TPOT) | - | - |
| Compute | - | - |
| Recv wait | - | - |
| MoE router | - | - |
| MoE experts | - | - |
| MoE shared/dense | - | - |
| Compute ratio | - | - |
| Tokens/sec | 0.91 | - |

- `Model-side generation total` is `total_time_ms` for the full generation call.
- The Attention/FFN columns are rank-level wall-clock views of the same serial run; they are not per-role compute decomposition.
- `Decode TPOT` is the serial decode baseline used for decode speedup.

## Layer averages summary

_No per-layer events recorded._

## Per-layer breakdown

_No per-layer events recorded._

---
_Generated from `timing_attention_serial_b128_s2048_t20.json` + `timing_ffn_serial_b128_s2048_t20.json`._
