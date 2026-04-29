# Experiment report — serial b96 s128 t20

## Configuration

- **Mode**: `serial`
- **Batch size**: 96
- **Prefill seq**: 128
- **Decode tokens**: 20
- **Layers**: ?
- **Micro-batches**: ?

## Decode timing (exact TPOT + representative ITL detail)

| Metric | Attention | FFN |
|---|---:|---:|
| Representative ITL sample total | 24598.900 ms | 24598.841 ms |
| Decode loop total | 21507.747 ms | 21507.750 ms |
| Decode steps | 19 | 19 |
| Decode TPOT | 1131.987 ms | 1131.987 ms |
| Compute | - | - |
| Recv wait | - | - |
| MoE router | - | - |
| MoE experts | - | - |
| MoE shared/dense | - | - |
| Compute ratio | - | - |
| Tokens/sec | 0.81 | - |

## Layer averages summary

_No per-layer events recorded._

## Per-layer breakdown

_No per-layer events recorded._

---
_Generated from `timing_attention_serial_b96_s128_t20.json` + `timing_ffn_serial_b96_s128_t20.json`._
