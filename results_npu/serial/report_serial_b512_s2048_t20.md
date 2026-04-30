# Experiment report — serial b512 s2048 t20

## Configuration

- **Mode**: `serial`
- **Batch size**: 512
- **Prefill seq**: 2048
- **Decode tokens**: 20
- **Layers**: ?
- **Micro-batches**: ?

## Serial timing (model-side total + decode fields)

| Metric | Attention rank view | FFN rank view |
|---|---:|---:|
| Model-side generation total | 49879.136 ms | 49879.059 ms |
| Prefill / TTFT-path | 2462.120 ms | 2454.714 ms |
| Decode loop total | 45902.843 ms | 45902.856 ms |
| Decode steps | 19 | 19 |
| Decode TPOT | 2415.939 ms | 2415.940 ms |
| Legacy decode step (not exact TPOT) | - | - |
| Compute | - | - |
| Recv wait | - | - |
| MoE router | - | - |
| MoE experts | - | - |
| MoE shared/dense | - | - |
| Compute ratio | - | - |
| Tokens/sec | 0.40 | - |

- `Model-side generation total` is `total_time_ms` for the full generation call.
- The Attention/FFN columns are rank-level wall-clock views of the same serial run; they are not per-role compute decomposition.
- `Decode TPOT` is the serial decode baseline used for decode speedup.

## Layer averages summary

_No per-layer events recorded._

## Per-layer breakdown

_No per-layer events recorded._

---
_Generated from `timing_attention_serial_b512_s2048_t20.json` + `timing_ffn_serial_b512_s2048_t20.json`._
