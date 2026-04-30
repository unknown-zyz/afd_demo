# Experiment report — serial b192 s128 t20

## Configuration

- **Mode**: `serial`
- **Batch size**: 192
- **Prefill seq**: 128
- **Decode tokens**: 20
- **Layers**: ?
- **Micro-batches**: ?

## Serial timing (model-side total + decode fields)

| Metric | Attention rank view | FFN rank view |
|---|---:|---:|
| Model-side generation total | 28126.038 ms | 28125.210 ms |
| Prefill / TTFT-path | 3791.846 ms | 3607.819 ms |
| Decode loop total | 24150.453 ms | 24150.425 ms |
| Decode steps | 19 | 19 |
| Decode TPOT | 1271.076 ms | 1271.075 ms |
| Legacy decode step (not exact TPOT) | - | - |
| Compute | - | - |
| Recv wait | - | - |
| MoE router | - | - |
| MoE experts | - | - |
| MoE shared/dense | - | - |
| Compute ratio | - | - |
| Tokens/sec | 0.71 | - |

- `Model-side generation total` is `total_time_ms` for the full generation call.
- The Attention/FFN columns are rank-level wall-clock views of the same serial run; they are not per-role compute decomposition.
- `Decode TPOT` is the serial decode baseline used for decode speedup.

## Layer averages summary

_No per-layer events recorded._

## Per-layer breakdown

_No per-layer events recorded._

---
_Generated from `timing_attention_serial_b192_s128_t20.json` + `timing_ffn_serial_b192_s128_t20.json`._
