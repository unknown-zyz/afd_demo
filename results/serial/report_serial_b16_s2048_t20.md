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
| Representative ITL sample total | 12995.143 ms | 12995.076 ms |
| Decode loop total | 10597.072 ms | 10597.087 ms |
| Decode steps | 19 | 19 |
| Decode TPOT | 557.741 ms | 557.741 ms |
| Compute | - | - |
| Recv wait | - | - |
| MoE router | - | - |
| MoE experts | - | - |
| MoE shared/dense | - | - |
| Compute ratio | - | - |
| Tokens/sec | 1.54 | - |

## Layer averages summary

_No per-layer events recorded._

## Per-layer breakdown

_No per-layer events recorded._

---
_Generated from `timing_attention_serial_b16_s2048_t20.json` + `timing_ffn_serial_b16_s2048_t20.json`._
