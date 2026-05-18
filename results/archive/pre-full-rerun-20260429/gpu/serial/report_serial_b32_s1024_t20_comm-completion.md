# Experiment report — serial b32 s1024 t20

## Configuration

- **Mode**: `serial`
- **Batch size**: 32
- **Prefill seq**: 1024
- **Decode tokens**: 20
- **Layers**: ?
- **Micro-batches**: ?
- **Comm timing mode**: `completion`
- **Requested prefill seq**: 1024
- **Actual prompt len**: 1024

## Serial timing (model-side total + decode fields)

| Metric | Attention rank view | FFN rank view |
|---|---:|---:|
| Model-side generation total | 21420.465 ms | 21420.128 ms |
| Prefill / TTFT-path | 6081.692 ms | 6005.398 ms |
| Decode loop total | 15182.467 ms | 15182.389 ms |
| Decode steps | 19 | 19 |
| Decode TPOT | 799.077 ms | 799.073 ms |
| Legacy decode step (not exact TPOT) | - | - |
| Compute | - | - |
| Recv wait | - | - |
| MoE router | - | - |
| MoE experts | - | - |
| MoE shared/dense | - | - |
| Compute ratio | - | - |
| Tokens/sec | 0.93 | - |

- `Model-side generation total` is `total_time_ms` for the full generation call.
- The Attention/FFN columns are rank-level wall-clock views of the same serial run; they are not per-role compute decomposition.
- `Decode TPOT` is the serial decode baseline used for decode speedup.

## Layer averages summary

_No per-layer events recorded._

## Per-layer breakdown

_No per-layer events recorded._

---
_Generated from `timing_attention_serial_b32_s1024_t20_comm-completion.json` + `timing_ffn_serial_b32_s1024_t20_comm-completion.json`._
