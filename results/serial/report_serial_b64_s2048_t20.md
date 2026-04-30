# Experiment report — serial b64 s2048 t20

## Configuration

- **Mode**: `serial`
- **Batch size**: 64
- **Prefill seq**: 2048
- **Decode tokens**: 20
- **Layers**: ?
- **Micro-batches**: ?

## Serial timing (model-side total + decode fields)

| Metric | Attention rank view | FFN rank view |
|---|---:|---:|
| Model-side generation total | 20696.468 ms | 20696.496 ms |
| Prefill / TTFT-path | 2585.714 ms | 2525.051 ms |
| Decode loop total | 17868.616 ms | 17868.611 ms |
| Decode steps | 19 | 19 |
| Decode TPOT | 940.453 ms | 940.453 ms |
| Legacy decode step (not exact TPOT) | - | - |
| Compute | - | - |
| Recv wait | - | - |
| MoE router | - | - |
| MoE experts | - | - |
| MoE shared/dense | - | - |
| Compute ratio | - | - |
| Tokens/sec | 0.97 | - |

- `Model-side generation total` is `total_time_ms` for the full generation call.
- The Attention/FFN columns are rank-level wall-clock views of the same serial run; they are not per-role compute decomposition.
- `Decode TPOT` is the serial decode baseline used for decode speedup.

## Layer averages summary

_No per-layer events recorded._

## Per-layer breakdown

_No per-layer events recorded._

---
_Generated from `timing_attention_serial_b64_s2048_t20.json` + `timing_ffn_serial_b64_s2048_t20.json`._
