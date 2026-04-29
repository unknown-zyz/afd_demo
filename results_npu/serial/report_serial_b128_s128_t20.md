# Experiment report — serial b128 s128 t20

## Configuration

- **Mode**: `serial`
- **Batch size**: 128
- **Prefill seq**: 128
- **Decode tokens**: 20
- **Layers**: ?
- **Micro-batches**: ?

## Decode timing (exact TPOT + representative ITL detail)

| Metric | Attention | FFN |
|---|---:|---:|
| Representative ITL sample total | 20250.437 ms | 20250.339 ms |
| Decode loop total | 17920.735 ms | 17920.742 ms |
| Decode steps | 19 | 19 |
| Decode TPOT | 943.197 ms | 943.197 ms |
| Compute | - | - |
| Recv wait | - | - |
| MoE router | - | - |
| MoE experts | - | - |
| MoE shared/dense | - | - |
| Compute ratio | - | - |
| Tokens/sec | 0.99 | - |

## Layer averages summary

_No per-layer events recorded._

## Per-layer breakdown

_No per-layer events recorded._

---
_Generated from `timing_attention_serial_b128_s128_t20.json` + `timing_ffn_serial_b128_s128_t20.json`._
