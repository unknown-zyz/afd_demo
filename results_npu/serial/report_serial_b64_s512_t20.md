# Experiment report — serial b64 s512 t20

## Configuration

- **Mode**: `serial`
- **Batch size**: 64
- **Prefill seq**: 512
- **Decode tokens**: 20
- **Layers**: ?
- **Micro-batches**: ?

## Decode timing (exact TPOT + representative ITL detail)

| Metric | Attention | FFN |
|---|---:|---:|
| Representative ITL sample total | 17133.841 ms | 17133.753 ms |
| Decode loop total | 14969.662 ms | 14969.664 ms |
| Decode steps | 19 | 19 |
| Decode TPOT | 787.877 ms | 787.877 ms |
| Compute | - | - |
| Recv wait | - | - |
| MoE router | - | - |
| MoE experts | - | - |
| MoE shared/dense | - | - |
| Compute ratio | - | - |
| Tokens/sec | 1.17 | - |

## Layer averages summary

_No per-layer events recorded._

## Per-layer breakdown

_No per-layer events recorded._

---
_Generated from `timing_attention_serial_b64_s512_t20.json` + `timing_ffn_serial_b64_s512_t20.json`._
