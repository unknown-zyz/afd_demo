# Experiment report — serial b256 s1024 t20

## Configuration

- **Mode**: `serial`
- **Batch size**: 256
- **Prefill seq**: 1024
- **Decode tokens**: 20
- **Layers**: ?
- **Micro-batches**: ?

## Decode timing (exact TPOT + representative ITL detail)

| Metric | Attention | FFN |
|---|---:|---:|
| Representative ITL sample total | 32299.242 ms | 32299.161 ms |
| Decode loop total | 29328.422 ms | 29328.449 ms |
| Decode steps | 19 | 19 |
| Decode TPOT | 1543.601 ms | 1543.603 ms |
| Compute | - | - |
| Recv wait | - | - |
| MoE router | - | - |
| MoE experts | - | - |
| MoE shared/dense | - | - |
| Compute ratio | - | - |
| Tokens/sec | 0.62 | - |

## Layer averages summary

_No per-layer events recorded._

## Per-layer breakdown

_No per-layer events recorded._

---
_Generated from `timing_attention_serial_b256_s1024_t20.json` + `timing_ffn_serial_b256_s1024_t20.json`._
