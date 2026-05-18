# Experiment report — serial b512 s128 t20

## Configuration

- **Mode**: `serial`
- **Batch size**: 512
- **Prefill seq**: 128
- **Decode tokens**: 20
- **Layers**: ?
- **Micro-batches**: ?

## Serial timing (model-side total + decode fields)

| Metric | Attention rank view | FFN rank view |
|---|---:|---:|
| Model-side generation total | 50119.344 ms | 50119.238 ms |
| Prefill / TTFT-path | 2473.322 ms | 2465.917 ms |
| Decode loop total | 46110.652 ms | 46110.634 ms |
| Decode steps | 19 | 19 |
| Decode TPOT | 2426.876 ms | 2426.875 ms |
| Legacy decode step (not exact TPOT) | - | - |
| Compute | - | - |
| Recv wait | - | - |
| MoE router | - | - |
| MoE experts | - | - |
| MoE shared/dense | - | - |
| Compute ratio | - | - |
| Tokens/sec | 0.40 | - |

- `Model-side generation total` is `total_time_ms` for the full generation call.
- The Attention/FFN columns are rank-level wall-clock views of the same serial run; they are not per-role compute decomposition.
- `Decode TPOT` is the serial decode baseline used for decode speedup.

## Layer averages summary

_No per-layer events recorded._

## Per-layer breakdown

_No per-layer events recorded._

---
_Generated from `timing_attention_serial_b512_s128_t20.json` + `timing_ffn_serial_b512_s128_t20.json`._
