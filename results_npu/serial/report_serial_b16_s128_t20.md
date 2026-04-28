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
| Representative-step total | 10444.359 ms | 10366.071 ms |
| Compute | - | - |
| Recv wait | - | - |
| MoE router | - | - |
| MoE experts | - | - |
| MoE shared/dense | - | - |
| Compute ratio | - | - |
| Tokens/sec | 1.91 | - |

## Per-layer breakdown

_No per-layer events recorded._

---
_Generated from `timing_attention_serial_b16_s128_t20.json` + `timing_ffn_serial_b16_s128_t20.json`._
