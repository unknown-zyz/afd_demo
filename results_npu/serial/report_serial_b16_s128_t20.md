# Experiment report — serial b16 s128 t20

## Configuration

- **Mode**: `serial`
- **Batch size**: 16
- **Prefill seq**: 128
- **Decode tokens**: 20
- **Layers**: ?
- **Micro-batches**: ?

## Decode timing (exact TPOT + representative ITL detail)

| Metric | Attention | FFN |
|---|---:|---:|
| Representative ITL sample total | 11159.263 ms | 11159.184 ms |
| Decode loop total | 9175.650 ms | 9175.639 ms |
| Decode steps | 19 | 19 |
| Decode TPOT | 482.929 ms | 482.928 ms |
| Compute | - | - |
| Recv wait | - | - |
| MoE router | - | - |
| MoE experts | - | - |
| MoE shared/dense | - | - |
| Compute ratio | - | - |
| Tokens/sec | 1.79 | - |

## Layer averages summary

_No per-layer events recorded._

## Per-layer breakdown

_No per-layer events recorded._

---
_Generated from `timing_attention_serial_b16_s128_t20.json` + `timing_ffn_serial_b16_s128_t20.json`._
