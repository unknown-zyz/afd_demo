# Experiment report — serial b64 s1024 t20

## Configuration

- **Mode**: `serial`
- **Batch size**: 64
- **Prefill seq**: 1024
- **Decode tokens**: 20
- **Layers**: ?
- **Micro-batches**: ?
- **Comm timing mode**: `completion`
- **Requested prefill seq**: 1024
- **Actual prompt len**: 1024

## Serial timing (model-side total + decode fields)

| Metric | Attention rank view | FFN rank view |
|---|---:|---:|
| Model-side generation total | 35768.588 ms | 35768.517 ms |
| Prefill / TTFT-path | 11976.681 ms | 11822.163 ms |
| Decode loop total | 23530.651 ms | 23530.648 ms |
| Decode steps | 19 | 19 |
| Decode TPOT | 1238.455 ms | 1238.455 ms |
| Legacy decode step (not exact TPOT) | - | - |
| Compute | - | - |
| Recv wait | - | - |
| MoE router | - | - |
| MoE experts | - | - |
| MoE shared/dense | - | - |
| Compute ratio | - | - |
| Tokens/sec | 0.56 | - |

- `Model-side generation total` is `total_time_ms` for the full generation call.
- The Attention/FFN columns are rank-level wall-clock views of the same serial run; they are not per-role compute decomposition.
- `Decode TPOT` is the serial decode baseline used for decode speedup.

## Layer averages summary

_No per-layer events recorded._

## Per-layer breakdown

_No per-layer events recorded._

---
_Generated from `timing_attention_serial_b64_s1024_t20_comm-completion.json` + `timing_ffn_serial_b64_s1024_t20_comm-completion.json`._
