# Experiment report — serial b192 s256 t20

## Configuration

- **Mode**: `serial`
- **Batch size**: 192
- **Prefill seq**: 256
- **Decode tokens**: 20
- **Layers**: ?
- **Micro-batches**: ?

## Decode timing (exact TPOT + representative ITL detail)

| Metric | Attention | FFN |
|---|---:|---:|
| Representative ITL sample total | 28190.657 ms | 28190.347 ms |
| Decode loop total | 24208.365 ms | 24208.385 ms |
| Decode steps | 19 | 19 |
| Decode TPOT | 1274.124 ms | 1274.126 ms |
| Compute | - | - |
| Recv wait | - | - |
| MoE router | - | - |
| MoE experts | - | - |
| MoE shared/dense | - | - |
| Compute ratio | - | - |
| Tokens/sec | 0.71 | - |

## Layer averages summary

_No per-layer events recorded._

## Per-layer breakdown

_No per-layer events recorded._

---
_Generated from `timing_attention_serial_b192_s256_t20.json` + `timing_ffn_serial_b192_s256_t20.json`._
