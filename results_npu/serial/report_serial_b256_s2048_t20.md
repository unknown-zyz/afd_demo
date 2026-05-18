# Experiment report — serial b256 s2048 t20

## Configuration

- **Mode**: `serial`
- **Batch size**: 256
- **Prefill seq**: 2048
- **Decode tokens**: 20
- **Layers**: ?
- **Micro-batches**: ?

## Serial timing (model-side total + decode fields)

| Metric | Attention rank view | FFN rank view |
|---|---:|---:|
| Model-side generation total | 32764.426 ms | 32764.385 ms |
| Prefill / TTFT-path | 2117.754 ms | 2113.991 ms |
| Decode loop total | 29844.177 ms | 29844.179 ms |
| Decode steps | 19 | 19 |
| Decode TPOT | 1570.746 ms | 1570.746 ms |
| Legacy decode step (not exact TPOT) | - | - |
| Compute | - | - |
| Recv wait | - | - |
| MoE router | - | - |
| MoE experts | - | - |
| MoE shared/dense | - | - |
| Compute ratio | - | - |
| Tokens/sec | 0.61 | - |

- `Model-side generation total` is `total_time_ms` for the full generation call.
- The Attention/FFN columns are rank-level wall-clock views of the same serial run; they are not per-role compute decomposition.
- `Decode TPOT` is the serial decode baseline used for decode speedup.

## Layer averages summary

_No per-layer events recorded._

## Per-layer breakdown

_No per-layer events recorded._

---
_Generated from `timing_attention_serial_b256_s2048_t20.json` + `timing_ffn_serial_b256_s2048_t20.json`._
