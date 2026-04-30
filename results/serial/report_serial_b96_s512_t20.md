# Experiment report — serial b96 s512 t20

## Configuration

- **Mode**: `serial`
- **Batch size**: 96
- **Prefill seq**: 512
- **Decode tokens**: 20
- **Layers**: ?
- **Micro-batches**: ?

## Serial timing (model-side total + decode fields)

| Metric | Attention rank view | FFN rank view |
|---|---:|---:|
| Model-side generation total | 23899.679 ms | 23899.599 ms |
| Prefill / TTFT-path | 2863.771 ms | 2770.831 ms |
| Decode loop total | 20794.684 ms | 20794.668 ms |
| Decode steps | 19 | 19 |
| Decode TPOT | 1094.457 ms | 1094.456 ms |
| Legacy decode step (not exact TPOT) | - | - |
| Compute | - | - |
| Recv wait | - | - |
| MoE router | - | - |
| MoE experts | - | - |
| MoE shared/dense | - | - |
| Compute ratio | - | - |
| Tokens/sec | 0.84 | - |

- `Model-side generation total` is `total_time_ms` for the full generation call.
- The Attention/FFN columns are rank-level wall-clock views of the same serial run; they are not per-role compute decomposition.
- `Decode TPOT` is the serial decode baseline used for decode speedup.

## Layer averages summary

_No per-layer events recorded._

## Per-layer breakdown

_No per-layer events recorded._

---
_Generated from `timing_attention_serial_b96_s512_t20.json` + `timing_ffn_serial_b96_s512_t20.json`._
