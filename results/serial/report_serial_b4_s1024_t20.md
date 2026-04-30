# Experiment report — serial b4 s1024 t20

## Configuration

- **Mode**: `serial`
- **Batch size**: 4
- **Prefill seq**: 1024
- **Decode tokens**: 20
- **Layers**: ?
- **Micro-batches**: ?

## Serial timing (model-side total + decode fields)

| Metric | Attention rank view | FFN rank view |
|---|---:|---:|
| Model-side generation total | 8811.634 ms | 8811.534 ms |
| Prefill / TTFT-path | 2070.915 ms | 2063.862 ms |
| Decode loop total | 6515.140 ms | 6515.143 ms |
| Decode steps | 19 | 19 |
| Decode TPOT | 342.902 ms | 342.902 ms |
| Legacy decode step (not exact TPOT) | - | - |
| Compute | - | - |
| Recv wait | - | - |
| MoE router | - | - |
| MoE experts | - | - |
| MoE shared/dense | - | - |
| Compute ratio | - | - |
| Tokens/sec | 2.27 | - |

- `Model-side generation total` is `total_time_ms` for the full generation call.
- The Attention/FFN columns are rank-level wall-clock views of the same serial run; they are not per-role compute decomposition.
- `Decode TPOT` is the serial decode baseline used for decode speedup.

## Layer averages summary

_No per-layer events recorded._

## Per-layer breakdown

_No per-layer events recorded._

---
_Generated from `timing_attention_serial_b4_s1024_t20.json` + `timing_ffn_serial_b4_s1024_t20.json`._
