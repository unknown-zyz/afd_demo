# Experiment report — serial b512 s256 t20

## Configuration

- **Mode**: `serial`
- **Batch size**: 512
- **Prefill seq**: 256
- **Decode tokens**: 20
- **Layers**: ?
- **Micro-batches**: ?

## End-to-end decode timing (representative step)

| Metric | Attention | FFN |
|---|---:|---:|
| Representative-step total | 47540.810 ms | 46053.214 ms |
| Compute | - | - |
| Recv wait | - | - |
| MoE router | - | - |
| MoE experts | - | - |
| MoE shared/dense | - | - |
| Compute ratio | - | - |
| Tokens/sec | 0.42 | - |

## Per-layer breakdown

_No per-layer events recorded._

---
_Generated from `timing_attention_serial_b512_s256_t20.json` + `timing_ffn_serial_b512_s256_t20.json`._
