# Experiment report — serial b4 s1024 t20

## Configuration

- **Mode**: `serial`
- **Batch size**: 4
- **Prefill seq**: 1024
- **Decode tokens**: 20
- **Layers**: ?
- **Micro-batches**: ?
- **Comm timing mode**: `enqueue`
- **Requested prefill seq**: 1024
- **Actual prompt len**: 1024

## Serial timing (model-side total + decode fields)

| Metric | Attention rank view | FFN rank view |
|---|---:|---:|
| Model-side generation total | 12212.759 ms | 12212.271 ms |
| Prefill / TTFT-path | 5422.596 ms | 5191.871 ms |
| Decode loop total | 6563.821 ms | 6563.871 ms |
| Decode steps | 19 | 19 |
| Decode TPOT | 345.464 ms | 345.467 ms |
| Legacy decode step (not exact TPOT) | - | - |
| Compute | - | - |
| Recv wait | - | - |
| MoE router | - | - |
| MoE experts | - | - |
| MoE shared/dense | - | - |
| Compute ratio | - | - |
| Tokens/sec | 1.64 | - |

- `Model-side generation total` is `total_time_ms` for the full generation call.
- The Attention/FFN columns are rank-level wall-clock views of the same serial run; they are not per-role compute decomposition.
- `Decode TPOT` is the serial decode baseline used for decode speedup.

## Layer averages summary

_No per-layer events recorded._

## Per-layer breakdown

_No per-layer events recorded._

---
_Generated from `timing_attention_serial_b4_s1024_t20_comm-completion.json` + `timing_ffn_serial_b4_s1024_t20_comm-completion.json`._
