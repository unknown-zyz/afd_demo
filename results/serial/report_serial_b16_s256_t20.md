# Experiment report — serial b16 s256 t20

## Configuration

- **Mode**: `serial`
- **Batch size**: 16
- **Prefill seq**: 256
- **Decode tokens**: 20
- **Layers**: ?
- **Micro-batches**: ?

## Decode timing (exact TPOT + representative ITL detail)

| Metric | Attention | FFN |
|---|---:|---:|
| Representative ITL sample total | 14359.732 ms | 14359.646 ms |
| Decode loop total | 11969.672 ms | 11969.637 ms |
| Decode steps | 19 | 19 |
| Decode TPOT | 629.983 ms | 629.981 ms |
| Compute | - | - |
| Recv wait | - | - |
| MoE router | - | - |
| MoE experts | - | - |
| MoE shared/dense | - | - |
| Compute ratio | - | - |
| Tokens/sec | 1.39 | - |

## Layer averages summary

_No per-layer events recorded._

## Per-layer breakdown

_No per-layer events recorded._

---
_Generated from `timing_attention_serial_b16_s256_t20.json` + `timing_ffn_serial_b16_s256_t20.json`._
