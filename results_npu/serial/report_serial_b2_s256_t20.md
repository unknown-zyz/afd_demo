# Experiment report — serial b2 s256 t20

## Configuration

- **Mode**: `serial`
- **Batch size**: 2
- **Prefill seq**: 256
- **Decode tokens**: 20
- **Layers**: ?
- **Micro-batches**: ?

## Decode timing (exact TPOT + representative ITL detail)

| Metric | Attention | FFN |
|---|---:|---:|
| Representative ITL sample total | 6252.151 ms | 6251.849 ms |
| Decode loop total | 4347.867 ms | 4347.874 ms |
| Decode steps | 19 | 19 |
| Decode TPOT | 228.835 ms | 228.835 ms |
| Compute | - | - |
| Recv wait | - | - |
| MoE router | - | - |
| MoE experts | - | - |
| MoE shared/dense | - | - |
| Compute ratio | - | - |
| Tokens/sec | 3.20 | - |

## Layer averages summary

_No per-layer events recorded._

## Per-layer breakdown

_No per-layer events recorded._

---
_Generated from `timing_attention_serial_b2_s256_t20.json` + `timing_ffn_serial_b2_s256_t20.json`._
