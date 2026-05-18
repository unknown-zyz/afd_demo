# Experiment report — serial b4 s128 t20

## Configuration

- **Mode**: `serial`
- **Batch size**: 4
- **Prefill seq**: 128
- **Decode tokens**: 20
- **Layers**: ?
- **Micro-batches**: ?

## Serial timing (model-side total + decode fields)

| Metric | Attention rank view | FFN rank view |
|---|---:|---:|
| Model-side generation total | 6706.048 ms | 6705.896 ms |
| Prefill / TTFT-path | 1810.319 ms | 1809.817 ms |
| Decode loop total | 4801.719 ms | 4801.722 ms |
| Decode steps | 19 | 19 |
| Decode TPOT | 252.722 ms | 252.722 ms |
| Legacy decode step (not exact TPOT) | - | - |
| Compute | - | - |
| Recv wait | - | - |
| MoE router | - | - |
| MoE experts | - | - |
| MoE shared/dense | - | - |
| Compute ratio | - | - |
| Tokens/sec | 2.98 | - |

- `Model-side generation total` is `total_time_ms` for the full generation call.
- The Attention/FFN columns are rank-level wall-clock views of the same serial run; they are not per-role compute decomposition.
- `Decode TPOT` is the serial decode baseline used for decode speedup.

## Layer averages summary

_No per-layer events recorded._

## Per-layer breakdown

_No per-layer events recorded._

---
_Generated from `timing_attention_serial_b4_s128_t20.json` + `timing_ffn_serial_b4_s128_t20.json`._
