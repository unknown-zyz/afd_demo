# Experiment report — serial b2 s128 t20

## Configuration

- **Mode**: `serial`
- **Batch size**: 2
- **Prefill seq**: 128
- **Decode tokens**: 20
- **Layers**: ?
- **Micro-batches**: ?
- **Comm timing mode**: `enqueue`
- **Requested prefill seq**: 128
- **Actual prompt len**: 128

## Serial timing (model-side total + decode fields)

| Metric | Attention rank view | FFN rank view |
|---|---:|---:|
| Model-side generation total | 6554.260 ms | 6553.584 ms |
| Prefill / TTFT-path | 2303.293 ms | 2288.032 ms |
| Decode loop total | 4025.828 ms | 4025.747 ms |
| Decode steps | 19 | 19 |
| Decode TPOT | 211.886 ms | 211.881 ms |
| Legacy decode step (not exact TPOT) | - | - |
| Compute | - | - |
| Recv wait | - | - |
| MoE router | - | - |
| MoE experts | - | - |
| MoE shared/dense | - | - |
| Compute ratio | - | - |
| Tokens/sec | 3.05 | - |

- `Model-side generation total` is `total_time_ms` for the full generation call.
- The Attention/FFN columns are rank-level wall-clock views of the same serial run; they are not per-role compute decomposition.
- `Decode TPOT` is the serial decode baseline used for decode speedup.

## Layer averages summary

_No per-layer events recorded._

## Per-layer breakdown

_No per-layer events recorded._

---
_Generated from `timing_attention_serial_b2_s128_t20.json` + `timing_ffn_serial_b2_s128_t20.json`._
