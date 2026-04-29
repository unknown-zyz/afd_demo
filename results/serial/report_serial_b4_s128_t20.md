# Experiment report — serial b4 s128 t20

## Configuration

- **Mode**: `serial`
- **Batch size**: 4
- **Prefill seq**: 128
- **Decode tokens**: 20
- **Layers**: ?
- **Micro-batches**: ?

## Decode timing (exact TPOT + representative ITL detail)

| Metric | Attention | FFN |
|---|---:|---:|
| Representative ITL sample total | 8302.549 ms | 8302.453 ms |
| Decode loop total | 5980.125 ms | 5980.171 ms |
| Decode steps | 19 | 19 |
| Decode TPOT | 314.743 ms | 314.746 ms |
| Compute | - | - |
| Recv wait | - | - |
| MoE router | - | - |
| MoE experts | - | - |
| MoE shared/dense | - | - |
| Compute ratio | - | - |
| Tokens/sec | 2.41 | - |

## Layer averages summary

_No per-layer events recorded._

## Per-layer breakdown

_No per-layer events recorded._

---
_Generated from `timing_attention_serial_b4_s128_t20.json` + `timing_ffn_serial_b4_s128_t20.json`._
