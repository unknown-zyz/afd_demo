# Experiment report — serial b64 s128 t20

## Configuration

- **Mode**: `serial`
- **Batch size**: 64
- **Prefill seq**: 128
- **Decode tokens**: 20
- **Layers**: ?
- **Micro-batches**: ?

## Decode timing (exact TPOT + representative ITL detail)

| Metric | Attention | FFN |
|---|---:|---:|
| Representative ITL sample total | 21742.406 ms | 21742.316 ms |
| Decode loop total | 18961.226 ms | 18961.209 ms |
| Decode steps | 19 | 19 |
| Decode TPOT | 997.959 ms | 997.958 ms |
| Compute | - | - |
| Recv wait | - | - |
| MoE router | - | - |
| MoE experts | - | - |
| MoE shared/dense | - | - |
| Compute ratio | - | - |
| Tokens/sec | 0.92 | - |

## Layer averages summary

_No per-layer events recorded._

## Per-layer breakdown

_No per-layer events recorded._

---
_Generated from `timing_attention_serial_b64_s128_t20.json` + `timing_ffn_serial_b64_s128_t20.json`._
