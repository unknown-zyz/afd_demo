# Experiment report — serial b4 s256 t20

## Configuration

- **Mode**: `serial`
- **Batch size**: 4
- **Prefill seq**: 256
- **Decode tokens**: 20
- **Layers**: ?
- **Micro-batches**: ?

## End-to-end decode timing (representative step)

| Metric | Attention | FFN |
|---|---:|---:|
| Representative-step total | 6670.060 ms | 6617.834 ms |
| Compute | - | - |
| Recv wait | - | - |
| MoE router | - | - |
| MoE experts | - | - |
| MoE shared/dense | - | - |
| Compute ratio | - | - |
| Tokens/sec | 3.00 | - |

## Per-layer breakdown

_No per-layer events recorded._

---
_Generated from `timing_attention_serial_b4_s256_t20.json` + `timing_ffn_serial_b4_s256_t20.json`._
