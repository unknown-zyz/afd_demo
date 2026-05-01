# Experiment report — serial b8 s512 t20

## Configuration

- **Mode**: `serial`
- **Batch size**: 8
- **Prefill seq**: 512
- **Decode tokens**: 20
- **Layers**: ?
- **Micro-batches**: ?
- **Comm timing mode**: `enqueue`
- **Requested prefill seq**: 512
- **Actual prompt len**: 512

## Serial timing (model-side total + decode fields)

| Metric | Attention rank view | FFN rank view |
|---|---:|---:|
| Model-side generation total | 14708.050 ms | 14707.559 ms |
| Prefill / TTFT-path | 5080.869 ms | 4851.078 ms |
| Decode loop total | 9402.202 ms | 9402.254 ms |
| Decode steps | 19 | 19 |
| Decode TPOT | 494.853 ms | 494.855 ms |
| Legacy decode step (not exact TPOT) | - | - |
| Compute | - | - |
| Recv wait | - | - |
| MoE router | - | - |
| MoE experts | - | - |
| MoE shared/dense | - | - |
| Compute ratio | - | - |
| Tokens/sec | 1.36 | - |

- `Model-side generation total` is `total_time_ms` for the full generation call.
- The Attention/FFN columns are rank-level wall-clock views of the same serial run; they are not per-role compute decomposition.
- `Decode TPOT` is the serial decode baseline used for decode speedup.

## Layer averages summary

_No per-layer events recorded._

## Per-layer breakdown

_No per-layer events recorded._

---
_Generated from `timing_attention_serial_b8_s512_t20_comm-completion.json` + `timing_ffn_serial_b8_s512_t20_comm-completion.json`._
