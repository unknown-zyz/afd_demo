# Experiment report — serial b16 s1024 t20

## Configuration

- **Mode**: `serial`
- **Batch size**: 16
- **Prefill seq**: 1024
- **Decode tokens**: 20
- **Layers**: ?
- **Micro-batches**: ?

## Decode timing (exact TPOT + representative ITL detail)

| Metric | Attention | FFN |
|---|---:|---:|
| Representative ITL sample total | 11006.304 ms | 11006.496 ms |
| Decode loop total | 9044.471 ms | 9044.491 ms |
| Decode steps | 19 | 19 |
| Decode TPOT | 476.025 ms | 476.026 ms |
| Compute | - | - |
| Recv wait | - | - |
| MoE router | - | - |
| MoE experts | - | - |
| MoE shared/dense | - | - |
| Compute ratio | - | - |
| Tokens/sec | 1.82 | - |

## Layer averages summary

_No per-layer events recorded._

## Per-layer breakdown

_No per-layer events recorded._

---
_Generated from `timing_attention_serial_b16_s1024_t20.json` + `timing_ffn_serial_b16_s1024_t20.json`._
