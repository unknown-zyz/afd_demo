# Experiment report — serial b512 s256 t20

## Configuration

- **Mode**: `serial`
- **Batch size**: 512
- **Prefill seq**: 256
- **Decode tokens**: 20
- **Layers**: ?
- **Micro-batches**: ?

## Decode timing (exact TPOT + representative ITL detail)

| Metric | Attention | FFN |
|---|---:|---:|
| Representative ITL sample total | 49114.353 ms | 49114.260 ms |
| Decode loop total | 45159.701 ms | 45159.703 ms |
| Decode steps | 19 | 19 |
| Decode TPOT | 2376.826 ms | 2376.826 ms |
| Compute | - | - |
| Recv wait | - | - |
| MoE router | - | - |
| MoE experts | - | - |
| MoE shared/dense | - | - |
| Compute ratio | - | - |
| Tokens/sec | 0.41 | - |

## Layer averages summary

_No per-layer events recorded._

## Per-layer breakdown

_No per-layer events recorded._

---
_Generated from `timing_attention_serial_b512_s256_t20.json` + `timing_ffn_serial_b512_s256_t20.json`._
