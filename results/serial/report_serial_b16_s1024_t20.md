# Experiment report — serial b16 s1024 t20

## Configuration

- **Mode**: `serial`
- **Batch size**: 16
- **Prefill seq**: 1024
- **Decode tokens**: 20
- **Layers**: ?
- **Micro-batches**: ?

## Decode timing (exact TPOT + representative ITL detail)

| Metric | Attention | FFN |
|---|---:|---:|
| Representative ITL sample total | 14687.277 ms | 14686.736 ms |
| Decode loop total | 12324.883 ms | 12324.867 ms |
| Decode steps | 19 | 19 |
| Decode TPOT | 648.678 ms | 648.677 ms |
| Compute | - | - |
| Recv wait | - | - |
| MoE router | - | - |
| MoE experts | - | - |
| MoE shared/dense | - | - |
| Compute ratio | - | - |
| Tokens/sec | 1.36 | - |

## Layer averages summary

_No per-layer events recorded._

## Per-layer breakdown

_No per-layer events recorded._

---
_Generated from `timing_attention_serial_b16_s1024_t20.json` + `timing_ffn_serial_b16_s1024_t20.json`._
