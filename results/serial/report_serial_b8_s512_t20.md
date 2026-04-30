# Experiment report — serial b8 s512 t20

## Configuration

- **Mode**: `serial`
- **Batch size**: 8
- **Prefill seq**: 512
- **Decode tokens**: 20
- **Layers**: ?
- **Micro-batches**: ?

## Serial timing (model-side total + decode fields)

| Metric | Attention rank view | FFN rank view |
|---|---:|---:|
| Model-side generation total | 11454.103 ms | 11453.986 ms |
| Prefill / TTFT-path | 2098.744 ms | 2087.812 ms |
| Decode loop total | 9125.308 ms | 9125.305 ms |
| Decode steps | 19 | 19 |
| Decode TPOT | 480.279 ms | 480.279 ms |
| Legacy decode step (not exact TPOT) | - | - |
| Compute | - | - |
| Recv wait | - | - |
| MoE router | - | - |
| MoE experts | - | - |
| MoE shared/dense | - | - |
| Compute ratio | - | - |
| Tokens/sec | 1.75 | - |

- `Model-side generation total` is `total_time_ms` for the full generation call.
- The Attention/FFN columns are rank-level wall-clock views of the same serial run; they are not per-role compute decomposition.
- `Decode TPOT` is the serial decode baseline used for decode speedup.

## Layer averages summary

_No per-layer events recorded._

## Per-layer breakdown

_No per-layer events recorded._

---
_Generated from `timing_attention_serial_b8_s512_t20.json` + `timing_ffn_serial_b8_s512_t20.json`._
