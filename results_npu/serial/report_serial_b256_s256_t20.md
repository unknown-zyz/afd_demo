# Experiment report — serial b256 s256 t20

## Configuration

- **Mode**: `serial`
- **Batch size**: 256
- **Prefill seq**: 256
- **Decode tokens**: 20
- **Layers**: ?
- **Micro-batches**: ?

## Serial timing (model-side total + decode fields)

| Metric | Attention rank view | FFN rank view |
|---|---:|---:|
| Model-side generation total | 31412.074 ms | 31411.932 ms |
| Prefill / TTFT-path | 2071.600 ms | 2067.847 ms |
| Decode loop total | 28520.183 ms | 28520.145 ms |
| Decode steps | 19 | 19 |
| Decode TPOT | 1501.062 ms | 1501.060 ms |
| Legacy decode step (not exact TPOT) | - | - |
| Compute | - | - |
| Recv wait | - | - |
| MoE router | - | - |
| MoE experts | - | - |
| MoE shared/dense | - | - |
| Compute ratio | - | - |
| Tokens/sec | 0.64 | - |

- `Model-side generation total` is `total_time_ms` for the full generation call.
- The Attention/FFN columns are rank-level wall-clock views of the same serial run; they are not per-role compute decomposition.
- `Decode TPOT` is the serial decode baseline used for decode speedup.

## Layer averages summary

_No per-layer events recorded._

## Per-layer breakdown

_No per-layer events recorded._

---
_Generated from `timing_attention_serial_b256_s256_t20.json` + `timing_ffn_serial_b256_s256_t20.json`._
