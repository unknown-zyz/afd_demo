# Experiment report — serial b8 s1024 t20

## Configuration

- **Mode**: `serial`
- **Batch size**: 8
- **Prefill seq**: 1024
- **Decode tokens**: 20
- **Layers**: ?
- **Micro-batches**: ?

## End-to-end decode timing (representative step)

| Metric | Attention | FFN |
|---|---:|---:|
| Representative-step total | 8219.749 ms | 8160.302 ms |
| Compute | - | - |
| Recv wait | - | - |
| MoE router | - | - |
| MoE experts | - | - |
| MoE shared/dense | - | - |
| Compute ratio | - | - |
| Tokens/sec | 2.43 | - |

## Per-layer breakdown

_No per-layer events recorded._

---
_Generated from `timing_attention_serial_b8_s1024_t20.json` + `timing_ffn_serial_b8_s1024_t20.json`._
