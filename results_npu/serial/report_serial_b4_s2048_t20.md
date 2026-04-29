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
| Representative ITL sample total | 6864.834 ms | 6864.763 ms |
| Decode loop total | 4960.072 ms | 4960.067 ms |
| Decode steps | 19 | 19 |
| Decode TPOT | 261.056 ms | 261.056 ms |
| Compute | - | - |
| Recv wait | - | - |
| MoE router | - | - |
| MoE experts | - | - |
| MoE shared/dense | - | - |
| Compute ratio | - | - |
| Tokens/sec | 2.91 | - |

## Layer averages summary

_No per-layer events recorded._

## Per-layer breakdown

_No per-layer events recorded._

---
_Generated from `timing_attention_serial_b4_s2048_t20.json` + `timing_ffn_serial_b4_s2048_t20.json`._
