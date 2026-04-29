# Experiment report — serial b8 s2048 t20

## Configuration

- **Mode**: `serial`
- **Batch size**: 8
- **Prefill seq**: 2048
- **Decode tokens**: 20
- **Layers**: ?
- **Micro-batches**: ?

## Decode timing (exact TPOT + representative ITL detail)

| Metric | Attention | FFN |
|---|---:|---:|
| Representative ITL sample total | 11310.330 ms | 11310.045 ms |
| Decode loop total | 8993.668 ms | 8993.671 ms |
| Decode steps | 19 | 19 |
| Decode TPOT | 473.351 ms | 473.351 ms |
| Compute | - | - |
| Recv wait | - | - |
| MoE router | - | - |
| MoE experts | - | - |
| MoE shared/dense | - | - |
| Compute ratio | - | - |
| Tokens/sec | 1.77 | - |

## Layer averages summary

_No per-layer events recorded._

## Per-layer breakdown

_No per-layer events recorded._

---
_Generated from `timing_attention_serial_b8_s2048_t20.json` + `timing_ffn_serial_b8_s2048_t20.json`._
