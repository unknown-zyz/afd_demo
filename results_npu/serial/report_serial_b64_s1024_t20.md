# Experiment report — serial b64 s1024 t20

## Configuration

- **Mode**: `serial`
- **Batch size**: 64
- **Prefill seq**: 1024
- **Decode tokens**: 20
- **Layers**: ?
- **Micro-batches**: ?

## Serial timing (model-side total + decode fields)

| Metric | Attention rank view | FFN rank view |
|---|---:|---:|
| Model-side generation total | 17782.732 ms | 17782.653 ms |
| Prefill / TTFT-path | 1840.623 ms | 1839.478 ms |
| Decode loop total | 15682.607 ms | 15682.599 ms |
| Decode steps | 19 | 19 |
| Decode TPOT | 825.400 ms | 825.400 ms |
| Legacy decode step (not exact TPOT) | - | - |
| Compute | - | - |
| Recv wait | - | - |
| MoE router | - | - |
| MoE experts | - | - |
| MoE shared/dense | - | - |
| Compute ratio | - | - |
| Tokens/sec | 1.12 | - |

- `Model-side generation total` is `total_time_ms` for the full generation call.
- The Attention/FFN columns are rank-level wall-clock views of the same serial run; they are not per-role compute decomposition.
- `Decode TPOT` is the serial decode baseline used for decode speedup.

## Layer averages summary

_No per-layer events recorded._

## Per-layer breakdown

_No per-layer events recorded._

---
_Generated from `timing_attention_serial_b64_s1024_t20.json` + `timing_ffn_serial_b64_s1024_t20.json`._
