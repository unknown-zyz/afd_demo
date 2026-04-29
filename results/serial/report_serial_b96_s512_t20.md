# Experiment report — serial b96 s512 t20

## Configuration

- **Mode**: `serial`
- **Batch size**: 96
- **Prefill seq**: 512
- **Decode tokens**: 20
- **Layers**: ?
- **Micro-batches**: ?

## Decode timing (exact TPOT + representative ITL detail)

| Metric | Attention | FFN |
|---|---:|---:|
| Representative ITL sample total | 23899.679 ms | 23899.599 ms |
| Decode loop total | 20794.684 ms | 20794.668 ms |
| Decode steps | 19 | 19 |
| Decode TPOT | 1094.457 ms | 1094.456 ms |
| Compute | - | - |
| Recv wait | - | - |
| MoE router | - | - |
| MoE experts | - | - |
| MoE shared/dense | - | - |
| Compute ratio | - | - |
| Tokens/sec | 0.84 | - |

## Layer averages summary

_No per-layer events recorded._

## Per-layer breakdown

_No per-layer events recorded._

---
_Generated from `timing_attention_serial_b96_s512_t20.json` + `timing_ffn_serial_b96_s512_t20.json`._
