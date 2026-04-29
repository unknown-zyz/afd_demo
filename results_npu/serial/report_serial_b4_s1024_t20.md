# Experiment report — serial b4 s1024 t20

## Configuration

- **Mode**: `serial`
- **Batch size**: 4
- **Prefill seq**: 1024
- **Decode tokens**: 20
- **Layers**: ?
- **Micro-batches**: ?

## Decode timing (exact TPOT + representative ITL detail)

| Metric | Attention | FFN |
|---|---:|---:|
| Representative ITL sample total | 6660.439 ms | 6660.354 ms |
| Decode loop total | 4761.499 ms | 4761.495 ms |
| Decode steps | 19 | 19 |
| Decode TPOT | 250.605 ms | 250.605 ms |
| Compute | - | - |
| Recv wait | - | - |
| MoE router | - | - |
| MoE experts | - | - |
| MoE shared/dense | - | - |
| Compute ratio | - | - |
| Tokens/sec | 3.00 | - |

## Layer averages summary

_No per-layer events recorded._

## Per-layer breakdown

_No per-layer events recorded._

---
_Generated from `timing_attention_serial_b4_s1024_t20.json` + `timing_ffn_serial_b4_s1024_t20.json`._
