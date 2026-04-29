# Experiment report — serial b32 s2048 t20

## Configuration

- **Mode**: `serial`
- **Batch size**: 32
- **Prefill seq**: 2048
- **Decode tokens**: 20
- **Layers**: ?
- **Micro-batches**: ?

## Decode timing (exact TPOT + representative ITL detail)

| Metric | Attention | FFN |
|---|---:|---:|
| Representative ITL sample total | 13029.311 ms | 13029.226 ms |
| Decode loop total | 10919.871 ms | 10919.871 ms |
| Decode steps | 19 | 19 |
| Decode TPOT | 574.730 ms | 574.730 ms |
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
_Generated from `timing_attention_serial_b32_s2048_t20.json` + `timing_ffn_serial_b32_s2048_t20.json`._
