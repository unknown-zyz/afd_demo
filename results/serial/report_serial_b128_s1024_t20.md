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
| Representative ITL sample total | 25759.843 ms | 25759.792 ms |
| Decode loop total | 22465.591 ms | 22465.597 ms |
| Decode steps | 19 | 19 |
| Decode TPOT | 1182.400 ms | 1182.400 ms |
| Compute | - | - |
| Recv wait | - | - |
| MoE router | - | - |
| MoE experts | - | - |
| MoE shared/dense | - | - |
| Compute ratio | - | - |
| Tokens/sec | 0.78 | - |

## Layer averages summary

_No per-layer events recorded._

## Per-layer breakdown

_No per-layer events recorded._

---
_Generated from `timing_attention_serial_b128_s1024_t20.json` + `timing_ffn_serial_b128_s1024_t20.json`._
