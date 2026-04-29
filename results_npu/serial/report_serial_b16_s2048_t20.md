# Experiment report — serial b16 s2048 t20

## Configuration

- **Mode**: `serial`
- **Batch size**: 16
- **Prefill seq**: 2048
- **Decode tokens**: 20
- **Layers**: ?
- **Micro-batches**: ?

## Decode timing (exact TPOT + representative ITL detail)

| Metric | Attention | FFN |
|---|---:|---:|
| Representative ITL sample total | 11012.704 ms | 11012.636 ms |
| Decode loop total | 9083.618 ms | 9083.624 ms |
| Decode steps | 19 | 19 |
| Decode TPOT | 478.085 ms | 478.085 ms |
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
_Generated from `timing_attention_serial_b16_s2048_t20.json` + `timing_ffn_serial_b16_s2048_t20.json`._
