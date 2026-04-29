# Experiment report — serial b128 s128 t20

## Configuration

- **Mode**: `serial`
- **Batch size**: 128
- **Prefill seq**: 128
- **Decode tokens**: 20
- **Layers**: ?
- **Micro-batches**: ?

## Decode timing (exact TPOT + representative ITL detail)

| Metric | Attention | FFN |
|---|---:|---:|
| Representative ITL sample total | 25061.687 ms | 25061.106 ms |
| Decode loop total | 21777.759 ms | 21777.758 ms |
| Decode steps | 19 | 19 |
| Decode TPOT | 1146.198 ms | 1146.198 ms |
| Compute | - | - |
| Recv wait | - | - |
| MoE router | - | - |
| MoE experts | - | - |
| MoE shared/dense | - | - |
| Compute ratio | - | - |
| Tokens/sec | 0.80 | - |

## Layer averages summary

_No per-layer events recorded._

## Per-layer breakdown

_No per-layer events recorded._

---
_Generated from `timing_attention_serial_b128_s128_t20.json` + `timing_ffn_serial_b128_s128_t20.json`._
