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
| Model-side generation total | 21965.785 ms | 21965.675 ms |
| Prefill / TTFT-path | 2527.793 ms | 2467.166 ms |
| Decode loop total | 19206.914 ms | 19206.915 ms |
| Decode steps | 19 | 19 |
| Decode TPOT | 1010.890 ms | 1010.890 ms |
| Legacy decode step (not exact TPOT) | - | - |
| Compute | - | - |
| Recv wait | - | - |
| MoE router | - | - |
| MoE experts | - | - |
| MoE shared/dense | - | - |
| Compute ratio | - | - |
| Tokens/sec | 0.91 | - |

- `Model-side generation total` is `total_time_ms` for the full generation call.
- The Attention/FFN columns are rank-level wall-clock views of the same serial run; they are not per-role compute decomposition.
- `Decode TPOT` is the serial decode baseline used for decode speedup.

## Layer averages summary

_No per-layer events recorded._

## Per-layer breakdown

_No per-layer events recorded._

---
_Generated from `timing_attention_serial_b64_s512_t20.json` + `timing_ffn_serial_b64_s512_t20.json`._
