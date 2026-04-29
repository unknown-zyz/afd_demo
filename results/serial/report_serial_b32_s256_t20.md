# Experiment report — serial b32 s256 t20

## Configuration

- **Mode**: `serial`
- **Batch size**: 32
- **Prefill seq**: 256
- **Decode tokens**: 20
- **Layers**: ?
- **Micro-batches**: ?

## Decode timing (exact TPOT + representative ITL detail)

| Metric | Attention | FFN |
|---|---:|---:|
| Representative ITL sample total | 18631.681 ms | 18631.089 ms |
| Decode loop total | 16128.796 ms | 16128.852 ms |
| Decode steps | 19 | 19 |
| Decode TPOT | 848.884 ms | 848.887 ms |
| Compute | - | - |
| Recv wait | - | - |
| MoE router | - | - |
| MoE experts | - | - |
| MoE shared/dense | - | - |
| Compute ratio | - | - |
| Tokens/sec | 1.07 | - |

## Layer averages summary

_No per-layer events recorded._

## Per-layer breakdown

_No per-layer events recorded._

---
_Generated from `timing_attention_serial_b32_s256_t20.json` + `timing_ffn_serial_b32_s256_t20.json`._
