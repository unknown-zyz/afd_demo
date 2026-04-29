# Experiment report — serial b256 s128 t20

## Configuration

- **Mode**: `serial`
- **Batch size**: 256
- **Prefill seq**: 128
- **Decode tokens**: 20
- **Layers**: ?
- **Micro-batches**: ?

## Decode timing (exact TPOT + representative ITL detail)

| Metric | Attention | FFN |
|---|---:|---:|
| Representative ITL sample total | 32431.164 ms | 32431.108 ms |
| Decode loop total | 29471.559 ms | 29471.559 ms |
| Decode steps | 19 | 19 |
| Decode TPOT | 1551.135 ms | 1551.135 ms |
| Compute | - | - |
| Recv wait | - | - |
| MoE router | - | - |
| MoE experts | - | - |
| MoE shared/dense | - | - |
| Compute ratio | - | - |
| Tokens/sec | 0.62 | - |

## Layer averages summary

_No per-layer events recorded._

## Per-layer breakdown

_No per-layer events recorded._

---
_Generated from `timing_attention_serial_b256_s128_t20.json` + `timing_ffn_serial_b256_s128_t20.json`._
