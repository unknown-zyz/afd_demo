# Experiment report — serial b2 s2048 t20

## Configuration

- **Mode**: `serial`
- **Batch size**: 2
- **Prefill seq**: 2048
- **Decode tokens**: 20
- **Layers**: ?
- **Micro-batches**: ?

## Decode timing (exact TPOT + representative ITL detail)

| Metric | Attention | FFN |
|---|---:|---:|
| Representative ITL sample total | 6352.161 ms | 6351.596 ms |
| Decode loop total | 4045.366 ms | 4045.356 ms |
| Decode steps | 19 | 19 |
| Decode TPOT | 212.914 ms | 212.913 ms |
| Compute | - | - |
| Recv wait | - | - |
| MoE router | - | - |
| MoE experts | - | - |
| MoE shared/dense | - | - |
| Compute ratio | - | - |
| Tokens/sec | 3.15 | - |

## Layer averages summary

_No per-layer events recorded._

## Per-layer breakdown

_No per-layer events recorded._

---
_Generated from `timing_attention_serial_b2_s2048_t20.json` + `timing_ffn_serial_b2_s2048_t20.json`._
