# Experiment report — serial b64 s512 t20

## Configuration

- **Mode**: `serial`
- **Batch size**: 64
- **Prefill seq**: 512
- **Decode tokens**: 20
- **Layers**: ?
- **Micro-batches**: ?

## Serial timing (model-side total + decode fields)

| Metric | Attention rank view | FFN rank view |
|---|---:|---:|
| Model-side generation total | 17133.841 ms | 17133.753 ms |
| Prefill / TTFT-path | 1905.664 ms | 1904.568 ms |
| Decode loop total | 14969.662 ms | 14969.664 ms |
| Decode steps | 19 | 19 |
| Decode TPOT | 787.877 ms | 787.877 ms |
| Legacy decode step (not exact TPOT) | - | - |
| Compute | - | - |
| Recv wait | - | - |
| MoE router | - | - |
| MoE experts | - | - |
| MoE shared/dense | - | - |
| Compute ratio | - | - |
| Tokens/sec | 1.17 | - |

- `Model-side generation total` is `total_time_ms` for the full generation call.
- The Attention/FFN columns are rank-level wall-clock views of the same serial run; they are not per-role compute decomposition.
- `Decode TPOT` is the serial decode baseline used for decode speedup.

## Layer averages summary

_No per-layer events recorded._

## Per-layer breakdown

_No per-layer events recorded._

---
_Generated from `timing_attention_serial_b64_s512_t20.json` + `timing_ffn_serial_b64_s512_t20.json`._
