# Experiment report — serial b2 s1024 t20

## Configuration

- **Mode**: `serial`
- **Batch size**: 2
- **Prefill seq**: 1024
- **Decode tokens**: 20
- **Layers**: ?
- **Micro-batches**: ?

## Decode timing (exact TPOT + representative ITL detail)

| Metric | Attention | FFN |
|---|---:|---:|
| Representative ITL sample total | 6348.249 ms | 6347.908 ms |
| Decode loop total | 4039.406 ms | 4039.447 ms |
| Decode steps | 19 | 19 |
| Decode TPOT | 212.600 ms | 212.602 ms |
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
_Generated from `timing_attention_serial_b2_s1024_t20.json` + `timing_ffn_serial_b2_s1024_t20.json`._
