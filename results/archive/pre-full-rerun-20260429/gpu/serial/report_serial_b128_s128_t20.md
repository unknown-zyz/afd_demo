# Experiment report — serial b128 s128 t20

## Configuration

- **Mode**: `serial`
- **Batch size**: 128
- **Prefill seq**: 128
- **Decode tokens**: 20
- **Layers**: ?
- **Micro-batches**: ?

## Serial timing (model-side total + decode fields)

| Metric | Attention rank view | FFN rank view |
|---|---:|---:|
| Model-side generation total | 20250.437 ms | 20250.339 ms |
| Prefill / TTFT-path | 1899.866 ms | 1897.662 ms |
| Decode loop total | 17920.735 ms | 17920.742 ms |
| Decode steps | 19 | 19 |
| Decode TPOT | 943.197 ms | 943.197 ms |
| Legacy decode step (not exact TPOT) | - | - |
| Compute | - | - |
| Recv wait | - | - |
| MoE router | - | - |
| MoE experts | - | - |
| MoE shared/dense | - | - |
| Compute ratio | - | - |
| Tokens/sec | 0.99 | - |

- `Model-side generation total` is `total_time_ms` for the full generation call.
- The Attention/FFN columns are rank-level wall-clock views of the same serial run; they are not per-role compute decomposition.
- `Decode TPOT` is the serial decode baseline used for decode speedup.

## Layer averages summary

_No per-layer events recorded._

## Per-layer breakdown

_No per-layer events recorded._

---
_Generated from `timing_attention_serial_b128_s128_t20.json` + `timing_ffn_serial_b128_s128_t20.json`._
