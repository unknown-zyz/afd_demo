# Experiment report — serial b32 s1024 t20

## Configuration

- **Mode**: `serial`
- **Batch size**: 32
- **Prefill seq**: 1024
- **Decode tokens**: 20
- **Layers**: ?
- **Micro-batches**: ?

## Decode timing (exact TPOT + representative ITL detail)

| Metric | Attention | FFN |
|---|---:|---:|
| Representative ITL sample total | 13668.351 ms | 13668.030 ms |
| Decode loop total | 11667.277 ms | 11667.208 ms |
| Decode steps | 19 | 19 |
| Decode TPOT | 614.067 ms | 614.064 ms |
| Compute | - | - |
| Recv wait | - | - |
| MoE router | - | - |
| MoE experts | - | - |
| MoE shared/dense | - | - |
| Compute ratio | - | - |
| Tokens/sec | 1.46 | - |

## Layer averages summary

_No per-layer events recorded._

## Per-layer breakdown

_No per-layer events recorded._

---
_Generated from `timing_attention_serial_b32_s1024_t20.json` + `timing_ffn_serial_b32_s1024_t20.json`._
