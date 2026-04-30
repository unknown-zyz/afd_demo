# Experiment report — serial b16 s128 t20

## Configuration

- **Mode**: `serial`
- **Batch size**: 16
- **Prefill seq**: 128
- **Decode tokens**: 20
- **Layers**: ?
- **Micro-batches**: ?

## Serial timing (model-side total + decode fields)

| Metric | Attention rank view | FFN rank view |
|---|---:|---:|
| Model-side generation total | 14273.253 ms | 14272.690 ms |
| Prefill / TTFT-path | 2142.734 ms | 2123.876 ms |
| Decode loop total | 11905.823 ms | 11905.859 ms |
| Decode steps | 19 | 19 |
| Decode TPOT | 626.622 ms | 626.624 ms |
| Legacy decode step (not exact TPOT) | - | - |
| Compute | - | - |
| Recv wait | - | - |
| MoE router | - | - |
| MoE experts | - | - |
| MoE shared/dense | - | - |
| Compute ratio | - | - |
| Tokens/sec | 1.40 | - |

- `Model-side generation total` is `total_time_ms` for the full generation call.
- The Attention/FFN columns are rank-level wall-clock views of the same serial run; they are not per-role compute decomposition.
- `Decode TPOT` is the serial decode baseline used for decode speedup.

## Layer averages summary

_No per-layer events recorded._

## Per-layer breakdown

_No per-layer events recorded._

---
_Generated from `timing_attention_serial_b16_s128_t20.json` + `timing_ffn_serial_b16_s128_t20.json`._
