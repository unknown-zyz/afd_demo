# Experiment report — serial b512 s1024 t20

## Configuration

- **Mode**: `serial`
- **Batch size**: 512
- **Prefill seq**: 1024
- **Decode tokens**: 20
- **Layers**: ?
- **Micro-batches**: ?

## Serial timing (model-side total + decode fields)

| Metric | Attention rank view | FFN rank view |
|---|---:|---:|
| Model-side generation total | 49970.597 ms | 49970.536 ms |
| Prefill / TTFT-path | 2399.121 ms | 2391.708 ms |
| Decode loop total | 46039.742 ms | 46039.738 ms |
| Decode steps | 19 | 19 |
| Decode TPOT | 2423.144 ms | 2423.144 ms |
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
_Generated from `timing_attention_serial_b512_s1024_t20.json` + `timing_ffn_serial_b512_s1024_t20.json`._
