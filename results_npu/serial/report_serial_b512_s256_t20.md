# Experiment report — serial b512 s256 t20

## Configuration

- **Mode**: `serial`
- **Batch size**: 512
- **Prefill seq**: 256
- **Decode tokens**: 20
- **Layers**: ?
- **Micro-batches**: ?

## Serial timing (model-side total + decode fields)

| Metric | Attention rank view | FFN rank view |
|---|---:|---:|
| Model-side generation total | 49114.353 ms | 49114.260 ms |
| Prefill / TTFT-path | 2432.173 ms | 2424.751 ms |
| Decode loop total | 45159.701 ms | 45159.703 ms |
| Decode steps | 19 | 19 |
| Decode TPOT | 2376.826 ms | 2376.826 ms |
| Legacy decode step (not exact TPOT) | - | - |
| Compute | - | - |
| Recv wait | - | - |
| MoE router | - | - |
| MoE experts | - | - |
| MoE shared/dense | - | - |
| Compute ratio | - | - |
| Tokens/sec | 0.41 | - |

- `Model-side generation total` is `total_time_ms` for the full generation call.
- The Attention/FFN columns are rank-level wall-clock views of the same serial run; they are not per-role compute decomposition.
- `Decode TPOT` is the serial decode baseline used for decode speedup.

## Layer averages summary

_No per-layer events recorded._

## Per-layer breakdown

_No per-layer events recorded._

---
_Generated from `timing_attention_serial_b512_s256_t20.json` + `timing_ffn_serial_b512_s256_t20.json`._
