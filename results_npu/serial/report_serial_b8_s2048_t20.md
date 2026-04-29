# Experiment report — serial b8 s2048 t20

## Configuration

- **Mode**: `serial`
- **Batch size**: 8
- **Prefill seq**: 2048
- **Decode tokens**: 20
- **Layers**: ?
- **Micro-batches**: ?

## Decode timing (exact TPOT + representative ITL detail)

| Metric | Attention | FFN |
|---|---:|---:|
| Representative ITL sample total | 9016.790 ms | 9016.676 ms |
| Decode loop total | 7071.651 ms | 7071.660 ms |
| Decode steps | 19 | 19 |
| Decode TPOT | 372.192 ms | 372.193 ms |
| Compute | - | - |
| Recv wait | - | - |
| MoE router | - | - |
| MoE experts | - | - |
| MoE shared/dense | - | - |
| Compute ratio | - | - |
| Tokens/sec | 2.22 | - |

## Layer averages summary

_No per-layer events recorded._

## Per-layer breakdown

_No per-layer events recorded._

---
_Generated from `timing_attention_serial_b8_s2048_t20.json` + `timing_ffn_serial_b8_s2048_t20.json`._
