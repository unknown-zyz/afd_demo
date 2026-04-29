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
| Representative ITL sample total | 21965.785 ms | 21965.675 ms |
| Decode loop total | 19206.914 ms | 19206.915 ms |
| Decode steps | 19 | 19 |
| Decode TPOT | 1010.890 ms | 1010.890 ms |
| Compute | - | - |
| Recv wait | - | - |
| MoE router | - | - |
| MoE experts | - | - |
| MoE shared/dense | - | - |
| Compute ratio | - | - |
| Tokens/sec | 0.91 | - |

## Layer averages summary

_No per-layer events recorded._

## Per-layer breakdown

_No per-layer events recorded._

---
_Generated from `timing_attention_serial_b64_s512_t20.json` + `timing_ffn_serial_b64_s512_t20.json`._
