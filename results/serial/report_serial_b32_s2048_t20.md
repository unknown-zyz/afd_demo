# Experiment report — serial b32 s2048 t20

## Configuration

- **Mode**: `serial`
- **Batch size**: 32
- **Prefill seq**: 2048
- **Decode tokens**: 20
- **Layers**: ?
- **Micro-batches**: ?

## Serial timing (model-side total + decode fields)

| Metric | Attention rank view | FFN rank view |
|---|---:|---:|
| Model-side generation total | 17052.479 ms | 17052.324 ms |
| Prefill / TTFT-path | 2303.307 ms | 2271.165 ms |
| Decode loop total | 14516.476 ms | 14516.505 ms |
| Decode steps | 19 | 19 |
| Decode TPOT | 764.025 ms | 764.027 ms |
| Legacy decode step (not exact TPOT) | - | - |
| Compute | - | - |
| Recv wait | - | - |
| MoE router | - | - |
| MoE experts | - | - |
| MoE shared/dense | - | - |
| Compute ratio | - | - |
| Tokens/sec | 1.17 | - |

- `Model-side generation total` is `total_time_ms` for the full generation call.
- The Attention/FFN columns are rank-level wall-clock views of the same serial run; they are not per-role compute decomposition.
- `Decode TPOT` is the serial decode baseline used for decode speedup.

## Layer averages summary

_No per-layer events recorded._

## Per-layer breakdown

_No per-layer events recorded._

---
_Generated from `timing_attention_serial_b32_s2048_t20.json` + `timing_ffn_serial_b32_s2048_t20.json`._
