# Experiment report — serial b8 s1024 t20

## Configuration

- **Mode**: `serial`
- **Batch size**: 8
- **Prefill seq**: 1024
- **Decode tokens**: 20
- **Layers**: ?
- **Micro-batches**: ?

## Decode timing (exact TPOT + representative ITL detail)

| Metric | Attention | FFN |
|---|---:|---:|
| Representative ITL sample total | 10772.417 ms | 10771.890 ms |
| Decode loop total | 8434.459 ms | 8434.473 ms |
| Decode steps | 19 | 19 |
| Decode TPOT | 443.919 ms | 443.920 ms |
| Compute | - | - |
| Recv wait | - | - |
| MoE router | - | - |
| MoE experts | - | - |
| MoE shared/dense | - | - |
| Compute ratio | - | - |
| Tokens/sec | 1.86 | - |

## Layer averages summary

_No per-layer events recorded._

## Per-layer breakdown

_No per-layer events recorded._

---
_Generated from `timing_attention_serial_b8_s1024_t20.json` + `timing_ffn_serial_b8_s1024_t20.json`._
