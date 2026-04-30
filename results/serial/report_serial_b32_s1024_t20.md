# Experiment report — serial b32 s1024 t20

## Configuration

- **Mode**: `serial`
- **Batch size**: 32
- **Prefill seq**: 1024
- **Decode tokens**: 20
- **Layers**: ?
- **Micro-batches**: ?

## Serial timing (model-side total + decode fields)

| Metric | Attention rank view | FFN rank view |
|---|---:|---:|
| Model-side generation total | 16293.907 ms | 16293.817 ms |
| Prefill / TTFT-path | 2307.817 ms | 2275.776 ms |
| Decode loop total | 13756.979 ms | 13757.000 ms |
| Decode steps | 19 | 19 |
| Decode TPOT | 724.052 ms | 724.053 ms |
| Legacy decode step (not exact TPOT) | - | - |
| Compute | - | - |
| Recv wait | - | - |
| MoE router | - | - |
| MoE experts | - | - |
| MoE shared/dense | - | - |
| Compute ratio | - | - |
| Tokens/sec | 1.23 | - |

- `Model-side generation total` is `total_time_ms` for the full generation call.
- The Attention/FFN columns are rank-level wall-clock views of the same serial run; they are not per-role compute decomposition.
- `Decode TPOT` is the serial decode baseline used for decode speedup.

## Layer averages summary

_No per-layer events recorded._

## Per-layer breakdown

_No per-layer events recorded._

---
_Generated from `timing_attention_serial_b32_s1024_t20.json` + `timing_ffn_serial_b32_s1024_t20.json`._
