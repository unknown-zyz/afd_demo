# Experiment report — serial b16 s512 t20

## Configuration

- **Mode**: `serial`
- **Batch size**: 16
- **Prefill seq**: 512
- **Decode tokens**: 20
- **Layers**: ?
- **Micro-batches**: ?

## Decode timing (exact TPOT + representative ITL detail)

| Metric | Attention | FFN |
|---|---:|---:|
| Representative ITL sample total | 11503.560 ms | 11503.265 ms |
| Decode loop total | 9555.074 ms | 9555.079 ms |
| Decode steps | 19 | 19 |
| Decode TPOT | 502.899 ms | 502.899 ms |
| Compute | - | - |
| Recv wait | - | - |
| MoE router | - | - |
| MoE experts | - | - |
| MoE shared/dense | - | - |
| Compute ratio | - | - |
| Tokens/sec | 1.74 | - |

## Layer averages summary

_No per-layer events recorded._

## Per-layer breakdown

_No per-layer events recorded._

---
_Generated from `timing_attention_serial_b16_s512_t20.json` + `timing_ffn_serial_b16_s512_t20.json`._
