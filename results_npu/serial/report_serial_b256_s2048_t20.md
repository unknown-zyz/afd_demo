# Experiment report — serial b256 s2048 t20

## Configuration

- **Mode**: `serial`
- **Batch size**: 256
- **Prefill seq**: 2048
- **Decode tokens**: 20
- **Layers**: ?
- **Micro-batches**: ?

## Decode timing (exact TPOT + representative ITL detail)

| Metric | Attention | FFN |
|---|---:|---:|
| Representative ITL sample total | 32764.426 ms | 32764.385 ms |
| Decode loop total | 29844.177 ms | 29844.179 ms |
| Decode steps | 19 | 19 |
| Decode TPOT | 1570.746 ms | 1570.746 ms |
| Compute | - | - |
| Recv wait | - | - |
| MoE router | - | - |
| MoE experts | - | - |
| MoE shared/dense | - | - |
| Compute ratio | - | - |
| Tokens/sec | 0.61 | - |

## Layer averages summary

_No per-layer events recorded._

## Per-layer breakdown

_No per-layer events recorded._

---
_Generated from `timing_attention_serial_b256_s2048_t20.json` + `timing_ffn_serial_b256_s2048_t20.json`._
