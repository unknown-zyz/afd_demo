# Experiment report — serial b64 s512 t20

## Configuration

- **Mode**: `serial`
- **Batch size**: 64
- **Prefill seq**: 512
- **Decode tokens**: 20
- **Layers**: ?
- **Micro-batches**: ?
- **Comm timing mode**: `completion`
- **Requested prefill seq**: 512
- **Actual prompt len**: 512

## Serial timing (model-side total + decode fields)

| Metric | Attention rank view | FFN rank view |
|---|---:|---:|
| Model-side generation total | 29486.864 ms | 29486.753 ms |
| Prefill / TTFT-path | 5969.327 ms | 5893.261 ms |
| Decode loop total | 23261.343 ms | 23261.353 ms |
| Decode steps | 19 | 19 |
| Decode TPOT | 1224.281 ms | 1224.282 ms |
| Legacy decode step (not exact TPOT) | - | - |
| Compute | - | - |
| Recv wait | - | - |
| MoE router | - | - |
| MoE experts | - | - |
| MoE shared/dense | - | - |
| Compute ratio | - | - |
| Tokens/sec | 0.68 | - |

- `Model-side generation total` is `total_time_ms` for the full generation call.
- The Attention/FFN columns are rank-level wall-clock views of the same serial run; they are not per-role compute decomposition.
- `Decode TPOT` is the serial decode baseline used for decode speedup.

## Layer averages summary

_No per-layer events recorded._

## Per-layer breakdown

_No per-layer events recorded._

---
_Generated from `timing_attention_serial_b64_s512_t20_comm-completion.json` + `timing_ffn_serial_b64_s512_t20_comm-completion.json`._
