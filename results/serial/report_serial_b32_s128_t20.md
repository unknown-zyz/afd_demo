# Experiment report — serial b32 s128 t20

## Configuration

- **Mode**: `serial`
- **Batch size**: 32
- **Prefill seq**: 128
- **Decode tokens**: 20
- **Layers**: ?
- **Micro-batches**: ?

## Decode timing (exact TPOT + representative ITL detail)

| Metric | Attention | FFN |
|---|---:|---:|
| Representative ITL sample total | 17493.335 ms | 17493.256 ms |
| Decode loop total | 14968.221 ms | 14968.225 ms |
| Decode steps | 19 | 19 |
| Decode TPOT | 787.801 ms | 787.801 ms |
| Compute | - | - |
| Recv wait | - | - |
| MoE router | - | - |
| MoE experts | - | - |
| MoE shared/dense | - | - |
| Compute ratio | - | - |
| Tokens/sec | 1.14 | - |

## Layer averages summary

_No per-layer events recorded._

## Per-layer breakdown

_No per-layer events recorded._

---
_Generated from `timing_attention_serial_b32_s128_t20.json` + `timing_ffn_serial_b32_s128_t20.json`._
