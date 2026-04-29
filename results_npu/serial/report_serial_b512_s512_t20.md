# Experiment report — serial b512 s512 t20

## Configuration

- **Mode**: `serial`
- **Batch size**: 512
- **Prefill seq**: 512
- **Decode tokens**: 20
- **Layers**: ?
- **Micro-batches**: ?

## Decode timing (exact TPOT + representative ITL detail)

| Metric | Attention | FFN |
|---|---:|---:|
| Representative ITL sample total | 50540.170 ms | 50540.164 ms |
| Decode loop total | 46594.828 ms | 46594.880 ms |
| Decode steps | 19 | 19 |
| Decode TPOT | 2452.359 ms | 2452.362 ms |
| Compute | - | - |
| Recv wait | - | - |
| MoE router | - | - |
| MoE experts | - | - |
| MoE shared/dense | - | - |
| Compute ratio | - | - |
| Tokens/sec | 0.40 | - |

## Layer averages summary

_No per-layer events recorded._

## Per-layer breakdown

_No per-layer events recorded._

---
_Generated from `timing_attention_serial_b512_s512_t20.json` + `timing_ffn_serial_b512_s512_t20.json`._
