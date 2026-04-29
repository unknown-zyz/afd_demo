# Experiment report — serial b96 s1024 t20

## Configuration

- **Mode**: `serial`
- **Batch size**: 96
- **Prefill seq**: 1024
- **Decode tokens**: 20
- **Layers**: ?
- **Micro-batches**: ?

## Decode timing (exact TPOT + representative ITL detail)

| Metric | Attention | FFN |
|---|---:|---:|
| Representative ITL sample total | 23770.401 ms | 23770.362 ms |
| Decode loop total | 20651.326 ms | 20651.317 ms |
| Decode steps | 19 | 19 |
| Decode TPOT | 1086.912 ms | 1086.911 ms |
| Compute | - | - |
| Recv wait | - | - |
| MoE router | - | - |
| MoE experts | - | - |
| MoE shared/dense | - | - |
| Compute ratio | - | - |
| Tokens/sec | 0.84 | - |

## Layer averages summary

_No per-layer events recorded._

## Per-layer breakdown

_No per-layer events recorded._

---
_Generated from `timing_attention_serial_b96_s1024_t20.json` + `timing_ffn_serial_b96_s1024_t20.json`._
