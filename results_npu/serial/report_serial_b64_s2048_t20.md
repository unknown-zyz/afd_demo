# Experiment report — serial b64 s2048 t20

## Configuration

- **Mode**: `serial`
- **Batch size**: 64
- **Prefill seq**: 2048
- **Decode tokens**: 20
- **Layers**: ?
- **Micro-batches**: ?

## Decode timing (exact TPOT + representative ITL detail)

| Metric | Attention | FFN |
|---|---:|---:|
| Representative ITL sample total | 15664.231 ms | 15664.134 ms |
| Decode loop total | 13610.751 ms | 13610.756 ms |
| Decode steps | 19 | 19 |
| Decode TPOT | 716.355 ms | 716.356 ms |
| Compute | - | - |
| Recv wait | - | - |
| MoE router | - | - |
| MoE experts | - | - |
| MoE shared/dense | - | - |
| Compute ratio | - | - |
| Tokens/sec | 1.28 | - |

## Layer averages summary

_No per-layer events recorded._

## Per-layer breakdown

_No per-layer events recorded._

---
_Generated from `timing_attention_serial_b64_s2048_t20.json` + `timing_ffn_serial_b64_s2048_t20.json`._
