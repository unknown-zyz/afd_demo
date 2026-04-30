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
| Model-side generation total | 10772.417 ms | 10771.890 ms |
| Prefill / TTFT-path | 2106.727 ms | 2094.927 ms |
| Decode loop total | 8434.459 ms | 8434.473 ms |
| Decode steps | 19 | 19 |
| Decode TPOT | 443.919 ms | 443.920 ms |
| Legacy decode step (not exact TPOT) | - | - |
| Compute | - | - |
| Recv wait | - | - |
| MoE router | - | - |
| MoE experts | - | - |
| MoE shared/dense | - | - |
| Compute ratio | - | - |
| Tokens/sec | 1.86 | - |

- `Model-side generation total` is `total_time_ms` for the full generation call.
- The Attention/FFN columns are rank-level wall-clock views of the same serial run; they are not per-role compute decomposition.
- `Decode TPOT` is the serial decode baseline used for decode speedup.

## Layer averages summary

_No per-layer events recorded._

## Per-layer breakdown

_No per-layer events recorded._

---
_Generated from `timing_attention_serial_b8_s1024_t20.json` + `timing_ffn_serial_b8_s1024_t20.json`._
