# Experiment report — serial b2 s128 t20

## Configuration

- **Mode**: `serial`
- **Batch size**: 2
- **Prefill seq**: 128
- **Decode tokens**: 20
- **Layers**: ?
- **Micro-batches**: ?

## Decode timing (exact TPOT + representative ITL detail)

| Metric | Attention | FFN |
|---|---:|---:|
| Representative ITL sample total | 6336.377 ms | 6336.304 ms |
| Decode loop total | 4426.987 ms | 4426.979 ms |
| Decode steps | 19 | 19 |
| Decode TPOT | 232.999 ms | 232.999 ms |
| Compute | - | - |
| Recv wait | - | - |
| MoE router | - | - |
| MoE experts | - | - |
| MoE shared/dense | - | - |
| Compute ratio | - | - |
| Tokens/sec | 3.16 | - |

## Layer averages summary

_No per-layer events recorded._

## Per-layer breakdown

_No per-layer events recorded._

---
_Generated from `timing_attention_serial_b2_s128_t20.json` + `timing_ffn_serial_b2_s128_t20.json`._
