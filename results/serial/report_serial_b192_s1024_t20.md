# Experiment report — serial b192 s1024 t20

## Configuration

- **Mode**: `serial`
- **Batch size**: 192
- **Prefill seq**: 1024
- **Decode tokens**: 20
- **Layers**: ?
- **Micro-batches**: ?

## Serial timing (model-side total + decode fields)

| Metric | Attention rank view | FFN rank view |
|---|---:|---:|
| Model-side generation total | 28808.886 ms | 28808.806 ms |
| Prefill / TTFT-path | 3810.169 ms | 3627.500 ms |
| Decode loop total | 24809.739 ms | 24809.759 ms |
| Decode steps | 19 | 19 |
| Decode TPOT | 1305.776 ms | 1305.777 ms |
| Legacy decode step (not exact TPOT) | - | - |
| Compute | - | - |
| Recv wait | - | - |
| MoE router | - | - |
| MoE experts | - | - |
| MoE shared/dense | - | - |
| Compute ratio | - | - |
| Tokens/sec | 0.69 | - |

- `Model-side generation total` is `total_time_ms` for the full generation call.
- The Attention/FFN columns are rank-level wall-clock views of the same serial run; they are not per-role compute decomposition.
- `Decode TPOT` is the serial decode baseline used for decode speedup.

## Layer averages summary

_No per-layer events recorded._

## Per-layer breakdown

_No per-layer events recorded._

---
_Generated from `timing_attention_serial_b192_s1024_t20.json` + `timing_ffn_serial_b192_s1024_t20.json`._
