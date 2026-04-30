# Experiment report — serial b8 s1024 t20

## Configuration

- **Mode**: `serial`
- **Batch size**: 8
- **Prefill seq**: 1024
- **Decode tokens**: 20
- **Layers**: ?
- **Micro-batches**: ?

## Serial timing (model-side total + decode fields)

| Metric | Attention rank view | FFN rank view |
|---|---:|---:|
| Model-side generation total | 8215.829 ms | 8215.734 ms |
| Prefill / TTFT-path | 1787.238 ms | 1786.655 ms |
| Decode loop total | 6325.139 ms | 6325.127 ms |
| Decode steps | 19 | 19 |
| Decode TPOT | 332.902 ms | 332.901 ms |
| Legacy decode step (not exact TPOT) | - | - |
| Compute | - | - |
| Recv wait | - | - |
| MoE router | - | - |
| MoE experts | - | - |
| MoE shared/dense | - | - |
| Compute ratio | - | - |
| Tokens/sec | 2.43 | - |

- `Model-side generation total` is `total_time_ms` for the full generation call.
- The Attention/FFN columns are rank-level wall-clock views of the same serial run; they are not per-role compute decomposition.
- `Decode TPOT` is the serial decode baseline used for decode speedup.

## Layer averages summary

_No per-layer events recorded._

## Per-layer breakdown

_No per-layer events recorded._

---
_Generated from `timing_attention_serial_b8_s1024_t20.json` + `timing_ffn_serial_b8_s1024_t20.json`._
