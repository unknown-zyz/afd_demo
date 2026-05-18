# Experiment report — serial b32 s512 t20

## Configuration

- **Mode**: `serial`
- **Batch size**: 32
- **Prefill seq**: 512
- **Decode tokens**: 20
- **Layers**: ?
- **Micro-batches**: ?
- **Comm timing mode**: `completion`
- **Requested prefill seq**: 512
- **Actual prompt len**: 512

## Serial timing (model-side total + decode fields)

| Metric | Attention rank view | FFN rank view |
|---|---:|---:|
| Model-side generation total | 20187.538 ms | 20187.306 ms |
| Prefill / TTFT-path | 3873.666 ms | 3838.331 ms |
| Decode loop total | 16162.172 ms | 16162.118 ms |
| Decode steps | 19 | 19 |
| Decode TPOT | 850.641 ms | 850.638 ms |
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
_Generated from `timing_attention_serial_b32_s512_t20_comm-completion.json` + `timing_ffn_serial_b32_s512_t20_comm-completion.json`._
