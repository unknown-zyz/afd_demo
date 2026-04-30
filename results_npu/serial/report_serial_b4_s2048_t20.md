# Experiment report — serial b4 s2048 t20

## Configuration

- **Mode**: `serial`
- **Batch size**: 4
- **Prefill seq**: 2048
- **Decode tokens**: 20
- **Layers**: ?
- **Micro-batches**: ?

## Serial timing (model-side total + decode fields)

| Metric | Attention rank view | FFN rank view |
|---|---:|---:|
| Model-side generation total | 6864.834 ms | 6864.763 ms |
| Prefill / TTFT-path | 1810.688 ms | 1810.242 ms |
| Decode loop total | 4960.072 ms | 4960.067 ms |
| Decode steps | 19 | 19 |
| Decode TPOT | 261.056 ms | 261.056 ms |
| Legacy decode step (not exact TPOT) | - | - |
| Compute | - | - |
| Recv wait | - | - |
| MoE router | - | - |
| MoE experts | - | - |
| MoE shared/dense | - | - |
| Compute ratio | - | - |
| Tokens/sec | 2.91 | - |

- `Model-side generation total` is `total_time_ms` for the full generation call.
- The Attention/FFN columns are rank-level wall-clock views of the same serial run; they are not per-role compute decomposition.
- `Decode TPOT` is the serial decode baseline used for decode speedup.

## Layer averages summary

_No per-layer events recorded._

## Per-layer breakdown

_No per-layer events recorded._

---
_Generated from `timing_attention_serial_b4_s2048_t20.json` + `timing_ffn_serial_b4_s2048_t20.json`._
