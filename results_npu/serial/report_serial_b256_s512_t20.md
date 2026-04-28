# Experiment report — serial b256 s512 t20

## Configuration

- **Mode**: `serial`
- **Batch size**: 256
- **Prefill seq**: 512
- **Decode tokens**: 20
- **Layers**: ?
- **Micro-batches**: ?

## End-to-end decode timing (representative step)

| Metric | Attention | FFN |
|---|---:|---:|
| Representative-step total | 31881.786 ms | 31114.235 ms |
| Compute | - | - |
| Recv wait | - | - |
| MoE router | - | - |
| MoE experts | - | - |
| MoE shared/dense | - | - |
| Compute ratio | - | - |
| Tokens/sec | 0.63 | - |

## Per-layer breakdown

_No per-layer events recorded._

---
_Generated from `timing_attention_serial_b256_s512_t20.json` + `timing_ffn_serial_b256_s512_t20.json`._
