# Experiment report — serial b2 s128 t5

## Configuration

- **Mode**: `serial`
- **Batch size**: 2
- **Prefill seq**: 128
- **Decode tokens**: 5
- **Layers**: ?
- **Micro-batches**: ?

## Decode timing (exact TPOT + representative ITL detail)

| Metric | Attention | FFN |
|---|---:|---:|
| Representative ITL sample total | 3202.660 ms | 3202.251 ms |
| Decode loop total | 912.954 ms | 912.998 ms |
| Decode steps | 4 | 4 |
| Decode TPOT | 228.238 ms | 228.250 ms |
| Compute | - | - |
| Recv wait | - | - |
| MoE router | - | - |
| MoE experts | - | - |
| MoE shared/dense | - | - |
| Compute ratio | - | - |
| Tokens/sec | 1.56 | - |

## Per-layer breakdown

_No per-layer events recorded._

---
_Generated from `timing_attention_serial_b2_s128_t5.json` + `timing_ffn_serial_b2_s128_t5.json`._
