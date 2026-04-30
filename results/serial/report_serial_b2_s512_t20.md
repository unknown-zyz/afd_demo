# Experiment report — serial b2 s512 t20

## Configuration

- **Mode**: `serial`
- **Batch size**: 2
- **Prefill seq**: 512
- **Decode tokens**: 20
- **Layers**: ?
- **Micro-batches**: ?

## Serial timing (model-side total + decode fields)

| Metric | Attention rank view | FFN rank view |
|---|---:|---:|
| Model-side generation total | 6646.154 ms | 6645.984 ms |
| Prefill / TTFT-path | 2098.686 ms | 2094.797 ms |
| Decode loop total | 4319.534 ms | 4319.549 ms |
| Decode steps | 19 | 19 |
| Decode TPOT | 227.344 ms | 227.345 ms |
| Legacy decode step (not exact TPOT) | - | - |
| Compute | - | - |
| Recv wait | - | - |
| MoE router | - | - |
| MoE experts | - | - |
| MoE shared/dense | - | - |
| Compute ratio | - | - |
| Tokens/sec | 3.01 | - |

- `Model-side generation total` is `total_time_ms` for the full generation call.
- The Attention/FFN columns are rank-level wall-clock views of the same serial run; they are not per-role compute decomposition.
- `Decode TPOT` is the serial decode baseline used for decode speedup.

## Layer averages summary

_No per-layer events recorded._

## Per-layer breakdown

_No per-layer events recorded._

---
_Generated from `timing_attention_serial_b2_s512_t20.json` + `timing_ffn_serial_b2_s512_t20.json`._
