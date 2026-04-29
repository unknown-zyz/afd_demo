# Experiment report — serial b8 s128 t20

## Configuration

- **Mode**: `serial`
- **Batch size**: 8
- **Prefill seq**: 128
- **Decode tokens**: 20
- **Layers**: ?
- **Micro-batches**: ?

## Decode timing (exact TPOT + representative ITL detail)

| Metric | Attention | FFN |
|---|---:|---:|
| Representative ITL sample total | 10294.818 ms | 10294.300 ms |
| Decode loop total | 7990.129 ms | 7990.117 ms |
| Decode steps | 19 | 19 |
| Decode TPOT | 420.533 ms | 420.532 ms |
| Compute | - | - |
| Recv wait | - | - |
| MoE router | - | - |
| MoE experts | - | - |
| MoE shared/dense | - | - |
| Compute ratio | - | - |
| Tokens/sec | 1.94 | - |

## Layer averages summary

_No per-layer events recorded._

## Per-layer breakdown

_No per-layer events recorded._

---
_Generated from `timing_attention_serial_b8_s128_t20.json` + `timing_ffn_serial_b8_s128_t20.json`._
