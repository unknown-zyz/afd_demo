# Experiment report — serial b16 s512 t20

## Configuration

- **Mode**: `serial`
- **Batch size**: 16
- **Prefill seq**: 512
- **Decode tokens**: 20
- **Layers**: ?
- **Micro-batches**: ?

## Decode timing (exact TPOT + representative ITL detail)

| Metric | Attention | FFN |
|---|---:|---:|
| Representative ITL sample total | 13124.870 ms | 13124.328 ms |
| Decode loop total | 10743.969 ms | 10743.970 ms |
| Decode steps | 19 | 19 |
| Decode TPOT | 565.472 ms | 565.472 ms |
| Compute | - | - |
| Recv wait | - | - |
| MoE router | - | - |
| MoE experts | - | - |
| MoE shared/dense | - | - |
| Compute ratio | - | - |
| Tokens/sec | 1.52 | - |

## Layer averages summary

_No per-layer events recorded._

## Per-layer breakdown

_No per-layer events recorded._

---
_Generated from `timing_attention_serial_b16_s512_t20.json` + `timing_ffn_serial_b16_s512_t20.json`._
