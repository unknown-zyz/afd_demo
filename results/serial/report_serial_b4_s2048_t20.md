# Experiment report — serial b4 s2048 t20

## Configuration

- **Mode**: `serial`
- **Batch size**: 4
- **Prefill seq**: 2048
- **Decode tokens**: 20
- **Layers**: ?
- **Micro-batches**: ?

## Decode timing (exact TPOT + representative ITL detail)

| Metric | Attention | FFN |
|---|---:|---:|
| Representative ITL sample total | 7922.347 ms | 7921.815 ms |
| Decode loop total | 5634.763 ms | 5634.776 ms |
| Decode steps | 19 | 19 |
| Decode TPOT | 296.566 ms | 296.567 ms |
| Compute | - | - |
| Recv wait | - | - |
| MoE router | - | - |
| MoE experts | - | - |
| MoE shared/dense | - | - |
| Compute ratio | - | - |
| Tokens/sec | 2.52 | - |

## Layer averages summary

_No per-layer events recorded._

## Per-layer breakdown

_No per-layer events recorded._

---
_Generated from `timing_attention_serial_b4_s2048_t20.json` + `timing_ffn_serial_b4_s2048_t20.json`._
