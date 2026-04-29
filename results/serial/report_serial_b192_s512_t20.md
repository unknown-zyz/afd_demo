# Experiment report — serial b192 s512 t20

## Configuration

- **Mode**: `serial`
- **Batch size**: 192
- **Prefill seq**: 512
- **Decode tokens**: 20
- **Layers**: ?
- **Micro-batches**: ?

## Decode timing (exact TPOT + representative ITL detail)

| Metric | Attention | FFN |
|---|---:|---:|
| Representative ITL sample total | 28589.808 ms | 28589.212 ms |
| Decode loop total | 24672.814 ms | 24672.810 ms |
| Decode steps | 19 | 19 |
| Decode TPOT | 1298.569 ms | 1298.569 ms |
| Compute | - | - |
| Recv wait | - | - |
| MoE router | - | - |
| MoE experts | - | - |
| MoE shared/dense | - | - |
| Compute ratio | - | - |
| Tokens/sec | 0.70 | - |

## Layer averages summary

_No per-layer events recorded._

## Per-layer breakdown

_No per-layer events recorded._

---
_Generated from `timing_attention_serial_b192_s512_t20.json` + `timing_ffn_serial_b192_s512_t20.json`._
