# Experiment report — serial b128 s1024 t20

## Configuration

- **Mode**: `serial`
- **Batch size**: 128
- **Prefill seq**: 1024
- **Decode tokens**: 20
- **Layers**: ?
- **Micro-batches**: ?

## Decode timing (exact TPOT + representative ITL detail)

| Metric | Attention | FFN |
|---|---:|---:|
| Representative ITL sample total | 21022.902 ms | 21022.792 ms |
| Decode loop total | 18672.370 ms | 18672.362 ms |
| Decode steps | 19 | 19 |
| Decode TPOT | 982.756 ms | 982.756 ms |
| Compute | - | - |
| Recv wait | - | - |
| MoE router | - | - |
| MoE experts | - | - |
| MoE shared/dense | - | - |
| Compute ratio | - | - |
| Tokens/sec | 0.95 | - |

## Layer averages summary

_No per-layer events recorded._

## Per-layer breakdown

_No per-layer events recorded._

---
_Generated from `timing_attention_serial_b128_s1024_t20.json` + `timing_ffn_serial_b128_s1024_t20.json`._
