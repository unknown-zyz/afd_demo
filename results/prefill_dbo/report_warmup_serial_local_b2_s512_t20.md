# Experiment report — serial b2 s512 t20

## Configuration

- **Mode**: `serial`
- **Batch size**: 2
- **Prefill seq**: 512
- **Decode tokens**: 20
- **Layers**: ?
- **Micro-batches**: ?

## End-to-end decode timing (representative step)

| Metric | Attention | FFN |
|---|---:|---:|
| Representative-step total | 3454.769 ms | 3396.025 ms |
| Compute | - | - |
| Recv wait | - | - |
| MoE router | - | - |
| MoE experts | - | - |
| MoE shared/dense | - | - |
| Compute ratio | - | - |

## Per-layer breakdown

_No per-layer events recorded._

---
_Generated from `timing_attention_warmup_serial_local_b2_s512_t20.json` + `timing_ffn_warmup_serial_local_b2_s512_t20.json`._
