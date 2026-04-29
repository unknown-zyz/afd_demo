# Experiment report — serial b32 s512 t20

## Configuration

- **Mode**: `serial`
- **Batch size**: 32
- **Prefill seq**: 512
- **Decode tokens**: 20
- **Layers**: ?
- **Micro-batches**: ?

## Decode timing (exact TPOT + representative ITL detail)

| Metric | Attention | FFN |
|---|---:|---:|
| Representative ITL sample total | 18642.060 ms | 18642.070 ms |
| Decode loop total | 16092.983 ms | 16093.010 ms |
| Decode steps | 19 | 19 |
| Decode TPOT | 846.999 ms | 847.001 ms |
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
_Generated from `timing_attention_serial_b32_s512_t20.json` + `timing_ffn_serial_b32_s512_t20.json`._
