# Experiment report — serial b128 s256 t20

## Configuration

- **Mode**: `serial`
- **Batch size**: 128
- **Prefill seq**: 256
- **Decode tokens**: 20
- **Layers**: ?
- **Micro-batches**: ?

## Decode timing (exact TPOT + representative ITL detail)

| Metric | Attention | FFN |
|---|---:|---:|
| Representative ITL sample total | 25350.047 ms | 25350.000 ms |
| Decode loop total | 22064.613 ms | 22064.517 ms |
| Decode steps | 19 | 19 |
| Decode TPOT | 1161.295 ms | 1161.290 ms |
| Compute | - | - |
| Recv wait | - | - |
| MoE router | - | - |
| MoE experts | - | - |
| MoE shared/dense | - | - |
| Compute ratio | - | - |
| Tokens/sec | 0.79 | - |

## Layer averages summary

_No per-layer events recorded._

## Per-layer breakdown

_No per-layer events recorded._

---
_Generated from `timing_attention_serial_b128_s256_t20.json` + `timing_ffn_serial_b128_s256_t20.json`._
