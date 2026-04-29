# Experiment report — serial b96 s256 t20

## Configuration

- **Mode**: `serial`
- **Batch size**: 96
- **Prefill seq**: 256
- **Decode tokens**: 20
- **Layers**: ?
- **Micro-batches**: ?

## Decode timing (exact TPOT + representative ITL detail)

| Metric | Attention | FFN |
|---|---:|---:|
| Representative ITL sample total | 23550.425 ms | 23549.785 ms |
| Decode loop total | 20491.740 ms | 20491.771 ms |
| Decode steps | 19 | 19 |
| Decode TPOT | 1078.513 ms | 1078.514 ms |
| Compute | - | - |
| Recv wait | - | - |
| MoE router | - | - |
| MoE experts | - | - |
| MoE shared/dense | - | - |
| Compute ratio | - | - |
| Tokens/sec | 0.85 | - |

## Layer averages summary

_No per-layer events recorded._

## Per-layer breakdown

_No per-layer events recorded._

---
_Generated from `timing_attention_serial_b96_s256_t20.json` + `timing_ffn_serial_b96_s256_t20.json`._
