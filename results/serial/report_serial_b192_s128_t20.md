# Experiment report — serial b192 s128 t20

## Configuration

- **Mode**: `serial`
- **Batch size**: 192
- **Prefill seq**: 128
- **Decode tokens**: 20
- **Layers**: ?
- **Micro-batches**: ?

## Decode timing (exact TPOT + representative ITL detail)

| Metric | Attention | FFN |
|---|---:|---:|
| Representative ITL sample total | 28126.038 ms | 28125.210 ms |
| Decode loop total | 24150.453 ms | 24150.425 ms |
| Decode steps | 19 | 19 |
| Decode TPOT | 1271.076 ms | 1271.075 ms |
| Compute | - | - |
| Recv wait | - | - |
| MoE router | - | - |
| MoE experts | - | - |
| MoE shared/dense | - | - |
| Compute ratio | - | - |
| Tokens/sec | 0.71 | - |

## Layer averages summary

_No per-layer events recorded._

## Per-layer breakdown

_No per-layer events recorded._

---
_Generated from `timing_attention_serial_b192_s128_t20.json` + `timing_ffn_serial_b192_s128_t20.json`._
