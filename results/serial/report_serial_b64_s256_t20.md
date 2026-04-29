# Experiment report — serial b64 s256 t20

## Configuration

- **Mode**: `serial`
- **Batch size**: 64
- **Prefill seq**: 256
- **Decode tokens**: 20
- **Layers**: ?
- **Micro-batches**: ?

## Decode timing (exact TPOT + representative ITL detail)

| Metric | Attention | FFN |
|---|---:|---:|
| Representative ITL sample total | 21253.562 ms | 21253.507 ms |
| Decode loop total | 18516.854 ms | 18516.733 ms |
| Decode steps | 19 | 19 |
| Decode TPOT | 974.571 ms | 974.565 ms |
| Compute | - | - |
| Recv wait | - | - |
| MoE router | - | - |
| MoE experts | - | - |
| MoE shared/dense | - | - |
| Compute ratio | - | - |
| Tokens/sec | 0.94 | - |

## Layer averages summary

_No per-layer events recorded._

## Per-layer breakdown

_No per-layer events recorded._

---
_Generated from `timing_attention_serial_b64_s256_t20.json` + `timing_ffn_serial_b64_s256_t20.json`._
