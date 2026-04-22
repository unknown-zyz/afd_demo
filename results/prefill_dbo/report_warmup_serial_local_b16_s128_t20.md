# Experiment report — serial b16 s128 t20

## Configuration

- **Mode**: `serial`
- **Batch size**: 16
- **Prefill seq**: 128
- **Decode tokens**: 20
- **Layers**: ?
- **Micro-batches**: ?

## End-to-end decode timing (representative step)

| Metric | Attention | FFN |
|---|---:|---:|
| Representative-step total | 4677.260 ms | 4560.078 ms |
| Compute | - | - |
| Recv wait | - | - |
| MoE router | - | - |
| MoE experts | - | - |
| MoE shared/dense | - | - |
| Compute ratio | - | - |

## Per-layer breakdown

_No per-layer events recorded._

---
_Generated from `timing_attention_warmup_serial_local_b16_s128_t20.json` + `timing_ffn_warmup_serial_local_b16_s128_t20.json`._
