# Experiment report — serial b128 s512 t20

## Configuration

- **Mode**: `serial`
- **Batch size**: 128
- **Prefill seq**: 512
- **Decode tokens**: 20
- **Layers**: ?
- **Micro-batches**: ?

## Serial timing (model-side total + decode fields)

| Metric | Attention rank view | FFN rank view |
|---|---:|---:|
| Model-side generation total | 21188.406 ms | 21188.320 ms |
| Prefill / TTFT-path | 1874.325 ms | 1872.226 ms |
| Decode loop total | 18882.219 ms | 18882.219 ms |
| Decode steps | 19 | 19 |
| Decode TPOT | 993.801 ms | 993.801 ms |
| Legacy decode step (not exact TPOT) | - | - |
| Compute | - | - |
| Recv wait | - | - |
| MoE router | - | - |
| MoE experts | - | - |
| MoE shared/dense | - | - |
| Compute ratio | - | - |
| Tokens/sec | 0.94 | - |

- `Model-side generation total` is `total_time_ms` for the full generation call.
- The Attention/FFN columns are rank-level wall-clock views of the same serial run; they are not per-role compute decomposition.
- `Decode TPOT` is the serial decode baseline used for decode speedup.

## Layer averages summary

_No per-layer events recorded._

## Per-layer breakdown

_No per-layer events recorded._

---
_Generated from `timing_attention_serial_b128_s512_t20.json` + `timing_ffn_serial_b128_s512_t20.json`._
