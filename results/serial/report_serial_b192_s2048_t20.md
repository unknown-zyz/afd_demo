# Experiment report — serial b192 s2048 t20

## Configuration

- **Mode**: `serial`
- **Batch size**: 192
- **Prefill seq**: 2048
- **Decode tokens**: 20
- **Layers**: ?
- **Micro-batches**: ?

## Decode timing (exact TPOT + representative ITL detail)

| Metric | Attention | FFN |
|---|---:|---:|
| Representative ITL sample total | 28685.754 ms | 28685.687 ms |
| Decode loop total | 24727.420 ms | 24727.401 ms |
| Decode steps | 19 | 19 |
| Decode TPOT | 1301.443 ms | 1301.442 ms |
| Compute | - | - |
| Recv wait | - | - |
| MoE router | - | - |
| MoE experts | - | - |
| MoE shared/dense | - | - |
| Compute ratio | - | - |
| Tokens/sec | 0.70 | - |

## Layer averages summary

_No per-layer events recorded._

## Per-layer breakdown

_No per-layer events recorded._

---
_Generated from `timing_attention_serial_b192_s2048_t20.json` + `timing_ffn_serial_b192_s2048_t20.json`._
