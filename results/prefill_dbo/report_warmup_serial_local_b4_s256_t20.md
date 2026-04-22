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
| Representative-step total | 4009.610 ms | 3951.142 ms |
| Compute | - | - |
| Recv wait | - | - |
| MoE router | - | - |
| MoE experts | - | - |
| MoE shared/dense | - | - |
| Compute ratio | - | - |

## Per-layer breakdown

_No per-layer events recorded._

---
_Generated from `timing_attention_warmup_serial_local_b4_s256_t20.json` + `timing_ffn_warmup_serial_local_b4_s256_t20.json`._
