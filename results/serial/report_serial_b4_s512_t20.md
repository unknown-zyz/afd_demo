# Experiment report — serial b4 s512 t20

## Configuration

- **Mode**: `serial`
- **Batch size**: 4
- **Prefill seq**: 512
- **Decode tokens**: 20
- **Layers**: ?
- **Micro-batches**: ?

## Decode timing (exact TPOT + representative ITL detail)

| Metric | Attention | FFN |
|---|---:|---:|
| Representative ITL sample total | 7889.284 ms | 7889.277 ms |
| Decode loop total | 5538.312 ms | 5538.346 ms |
| Decode steps | 19 | 19 |
| Decode TPOT | 291.490 ms | 291.492 ms |
| Compute | - | - |
| Recv wait | - | - |
| MoE router | - | - |
| MoE experts | - | - |
| MoE shared/dense | - | - |
| Compute ratio | - | - |
| Tokens/sec | 2.54 | - |

## Layer averages summary

_No per-layer events recorded._

## Per-layer breakdown

_No per-layer events recorded._

---
_Generated from `timing_attention_serial_b4_s512_t20.json` + `timing_ffn_serial_b4_s512_t20.json`._
