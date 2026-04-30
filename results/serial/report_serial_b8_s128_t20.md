# Experiment report — serial b8 s128 t20

## Configuration

- **Mode**: `serial`
- **Batch size**: 8
- **Prefill seq**: 128
- **Decode tokens**: 20
- **Layers**: ?
- **Micro-batches**: ?

## Serial timing (model-side total + decode fields)

| Metric | Attention rank view | FFN rank view |
|---|---:|---:|
| Model-side generation total | 10294.818 ms | 10294.300 ms |
| Prefill / TTFT-path | 2082.007 ms | 2070.397 ms |
| Decode loop total | 7990.129 ms | 7990.117 ms |
| Decode steps | 19 | 19 |
| Decode TPOT | 420.533 ms | 420.532 ms |
| Legacy decode step (not exact TPOT) | - | - |
| Compute | - | - |
| Recv wait | - | - |
| MoE router | - | - |
| MoE experts | - | - |
| MoE shared/dense | - | - |
| Compute ratio | - | - |
| Tokens/sec | 1.94 | - |

- `Model-side generation total` is `total_time_ms` for the full generation call.
- The Attention/FFN columns are rank-level wall-clock views of the same serial run; they are not per-role compute decomposition.
- `Decode TPOT` is the serial decode baseline used for decode speedup.

## Layer averages summary

_No per-layer events recorded._

## Per-layer breakdown

_No per-layer events recorded._

---
_Generated from `timing_attention_serial_b8_s128_t20.json` + `timing_ffn_serial_b8_s128_t20.json`._
