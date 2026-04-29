# Experiment report — serial b8 s512 t20

## Configuration

- **Mode**: `serial`
- **Batch size**: 8
- **Prefill seq**: 512
- **Decode tokens**: 20
- **Layers**: ?
- **Micro-batches**: ?

## Decode timing (exact TPOT + representative ITL detail)

| Metric | Attention | FFN |
|---|---:|---:|
| Representative ITL sample total | 8707.136 ms | 8707.080 ms |
| Decode loop total | 6678.202 ms | 6678.214 ms |
| Decode steps | 19 | 19 |
| Decode TPOT | 351.484 ms | 351.485 ms |
| Compute | - | - |
| Recv wait | - | - |
| MoE router | - | - |
| MoE experts | - | - |
| MoE shared/dense | - | - |
| Compute ratio | - | - |
| Tokens/sec | 2.30 | - |

## Layer averages summary

_No per-layer events recorded._

## Per-layer breakdown

_No per-layer events recorded._

---
_Generated from `timing_attention_serial_b8_s512_t20.json` + `timing_ffn_serial_b8_s512_t20.json`._
