# Experiment report — serial b8 s256 t20

## Configuration

- **Mode**: `serial`
- **Batch size**: 8
- **Prefill seq**: 256
- **Decode tokens**: 20
- **Layers**: ?
- **Micro-batches**: ?

## Serial timing (model-side total + decode fields)

| Metric | Attention rank view | FFN rank view |
|---|---:|---:|
| Model-side generation total | 10655.027 ms | 10649.818 ms |
| Prefill / TTFT-path | 4797.951 ms | - |
| Decode loop total | - | - |
| Decode steps | - | - |
| Decode TPOT | - | - |
| Legacy decode step (not exact TPOT) | 292.854 ms | - |
| Compute | - | - |
| Recv wait | - | - |
| MoE router | - | - |
| MoE experts | - | - |
| MoE shared/dense | - | - |
| Compute ratio | - | - |
| Tokens/sec | 1.88 | - |

- `Model-side generation total` is `total_time_ms` for the full generation call.
- The Attention/FFN columns are rank-level wall-clock views of the same serial run; they are not per-role compute decomposition.
- `Decode TPOT` is the serial decode baseline used for decode speedup.
- Missing serial display fields were filled from the matching serial cache JSON.
- `Legacy decode step` is shown for audit only; it is not exact TPOT and is not used for speedup.

## Layer averages summary

_No per-layer events recorded._

## Per-layer breakdown

_No per-layer events recorded._

---
_Generated from `timing_attention_serial_b8_s256_t20.json` + `timing_ffn_serial_b8_s256_t20.json`._
