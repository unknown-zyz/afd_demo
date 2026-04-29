# Experiment report — serial b64 s1024 t20

## Configuration

- **Mode**: `serial`
- **Batch size**: 64
- **Prefill seq**: 1024
- **Decode tokens**: 20
- **Layers**: ?
- **Micro-batches**: ?

## Decode timing (exact TPOT + representative ITL detail)

| Metric | Attention | FFN |
|---|---:|---:|
| Representative ITL sample total | 21593.427 ms | 21592.920 ms |
| Decode loop total | 18807.396 ms | 18807.445 ms |
| Decode steps | 19 | 19 |
| Decode TPOT | 989.863 ms | 989.866 ms |
| Compute | - | - |
| Recv wait | - | - |
| MoE router | - | - |
| MoE experts | - | - |
| MoE shared/dense | - | - |
| Compute ratio | - | - |
| Tokens/sec | 0.93 | - |

## Layer averages summary

_No per-layer events recorded._

## Per-layer breakdown

_No per-layer events recorded._

---
_Generated from `timing_attention_serial_b64_s1024_t20.json` + `timing_ffn_serial_b64_s1024_t20.json`._
