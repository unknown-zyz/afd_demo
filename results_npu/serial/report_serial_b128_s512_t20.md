# Experiment report — serial b128 s512 t20

## Configuration

- **Mode**: `serial`
- **Batch size**: 128
- **Prefill seq**: 512
- **Decode tokens**: 20
- **Layers**: ?
- **Micro-batches**: ?

## Decode timing (exact TPOT + representative ITL detail)

| Metric | Attention | FFN |
|---|---:|---:|
| Representative ITL sample total | 21188.406 ms | 21188.320 ms |
| Decode loop total | 18882.219 ms | 18882.219 ms |
| Decode steps | 19 | 19 |
| Decode TPOT | 993.801 ms | 993.801 ms |
| Compute | - | - |
| Recv wait | - | - |
| MoE router | - | - |
| MoE experts | - | - |
| MoE shared/dense | - | - |
| Compute ratio | - | - |
| Tokens/sec | 0.94 | - |

## Layer averages summary

_No per-layer events recorded._

## Per-layer breakdown

_No per-layer events recorded._

---
_Generated from `timing_attention_serial_b128_s512_t20.json` + `timing_ffn_serial_b128_s512_t20.json`._
