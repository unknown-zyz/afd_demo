# Experiment report — serial b64 s256 t20

## Configuration

- **Mode**: `serial`
- **Batch size**: 64
- **Prefill seq**: 256
- **Decode tokens**: 20
- **Layers**: ?
- **Micro-batches**: ?

## Serial timing (model-side total + decode fields)

| Metric | Attention rank view | FFN rank view |
|---|---:|---:|
| Model-side generation total | 21253.562 ms | 21253.507 ms |
| Prefill / TTFT-path | 2510.809 ms | 2450.141 ms |
| Decode loop total | 18516.854 ms | 18516.733 ms |
| Decode steps | 19 | 19 |
| Decode TPOT | 974.571 ms | 974.565 ms |
| Legacy decode step (not exact TPOT) | - | - |
| Compute | - | - |
| Recv wait | - | - |
| MoE router | - | - |
| MoE experts | - | - |
| MoE shared/dense | - | - |
| Compute ratio | - | - |
| Tokens/sec | 0.94 | - |

- `Model-side generation total` is `total_time_ms` for the full generation call.
- The Attention/FFN columns are rank-level wall-clock views of the same serial run; they are not per-role compute decomposition.
- `Decode TPOT` is the serial decode baseline used for decode speedup.

## Layer averages summary

_No per-layer events recorded._

## Per-layer breakdown

_No per-layer events recorded._

---
_Generated from `timing_attention_serial_b64_s256_t20.json` + `timing_ffn_serial_b64_s256_t20.json`._
