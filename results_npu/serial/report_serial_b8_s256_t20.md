# Experiment report — serial b8 s256 t20

## Configuration

- **Mode**: `serial`
- **Batch size**: 8
- **Prefill seq**: 256
- **Decode tokens**: 20
- **Layers**: ?
- **Micro-batches**: ?

## Decode timing (exact TPOT + representative ITL detail)

| Metric | Attention | FFN |
|---|---:|---:|
| Representative ITL sample total | 8772.774 ms | 8772.552 ms |
| Decode loop total | 6781.807 ms | 6781.757 ms |
| Decode steps | 19 | 19 |
| Decode TPOT | 356.937 ms | 356.935 ms |
| Compute | - | - |
| Recv wait | - | - |
| MoE router | - | - |
| MoE experts | - | - |
| MoE shared/dense | - | - |
| Compute ratio | - | - |
| Tokens/sec | 2.28 | - |

## Layer averages summary

_No per-layer events recorded._

## Per-layer breakdown

_No per-layer events recorded._

---
_Generated from `timing_attention_serial_b8_s256_t20.json` + `timing_ffn_serial_b8_s256_t20.json`._
