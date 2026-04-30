# Experiment report — serial b2 s128 t5

## Configuration

- **Mode**: `serial`
- **Batch size**: 2
- **Prefill seq**: 128
- **Decode tokens**: 5
- **Layers**: ?
- **Micro-batches**: ?

## Serial timing (model-side total + decode fields)

| Metric | Attention rank view | FFN rank view |
|---|---:|---:|
| Model-side generation total | 3202.660 ms | 3202.251 ms |
| Prefill / TTFT-path | 2065.471 ms | 2061.191 ms |
| Decode loop total | 912.954 ms | 912.998 ms |
| Decode steps | 4 | 4 |
| Decode TPOT | 228.238 ms | 228.250 ms |
| Legacy decode step (not exact TPOT) | - | - |
| Compute | - | - |
| Recv wait | - | - |
| MoE router | - | - |
| MoE experts | - | - |
| MoE shared/dense | - | - |
| Compute ratio | - | - |
| Tokens/sec | 1.56 | - |

- `Model-side generation total` is `total_time_ms` for the full generation call.
- The Attention/FFN columns are rank-level wall-clock views of the same serial run; they are not per-role compute decomposition.
- `Decode TPOT` is the serial decode baseline used for decode speedup.

## Layer averages summary

_No per-layer events recorded._

## Per-layer breakdown

_No per-layer events recorded._

---
_Generated from `timing_attention_serial_b2_s128_t5.json` + `timing_ffn_serial_b2_s128_t5.json`._
