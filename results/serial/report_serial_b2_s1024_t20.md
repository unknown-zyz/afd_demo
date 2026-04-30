# Experiment report — serial b2 s1024 t20

## Configuration

- **Mode**: `serial`
- **Batch size**: 2
- **Prefill seq**: 1024
- **Decode tokens**: 20
- **Layers**: ?
- **Micro-batches**: ?

## Serial timing (model-side total + decode fields)

| Metric | Attention rank view | FFN rank view |
|---|---:|---:|
| Model-side generation total | 6348.249 ms | 6347.908 ms |
| Prefill / TTFT-path | 2084.717 ms | 2080.490 ms |
| Decode loop total | 4039.406 ms | 4039.447 ms |
| Decode steps | 19 | 19 |
| Decode TPOT | 212.600 ms | 212.602 ms |
| Legacy decode step (not exact TPOT) | - | - |
| Compute | - | - |
| Recv wait | - | - |
| MoE router | - | - |
| MoE experts | - | - |
| MoE shared/dense | - | - |
| Compute ratio | - | - |
| Tokens/sec | 3.15 | - |

- `Model-side generation total` is `total_time_ms` for the full generation call.
- The Attention/FFN columns are rank-level wall-clock views of the same serial run; they are not per-role compute decomposition.
- `Decode TPOT` is the serial decode baseline used for decode speedup.

## Layer averages summary

_No per-layer events recorded._

## Per-layer breakdown

_No per-layer events recorded._

---
_Generated from `timing_attention_serial_b2_s1024_t20.json` + `timing_ffn_serial_b2_s1024_t20.json`._
