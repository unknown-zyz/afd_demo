# Experiment report — serial b512 s512 t20

## Configuration

- **Mode**: `serial`
- **Batch size**: 512
- **Prefill seq**: 512
- **Decode tokens**: 20
- **Layers**: ?
- **Micro-batches**: ?

## Serial timing (model-side total + decode fields)

| Metric | Attention rank view | FFN rank view |
|---|---:|---:|
| Model-side generation total | 50540.170 ms | 50540.164 ms |
| Prefill / TTFT-path | 2407.139 ms | 2399.734 ms |
| Decode loop total | 46594.828 ms | 46594.880 ms |
| Decode steps | 19 | 19 |
| Decode TPOT | 2452.359 ms | 2452.362 ms |
| Legacy decode step (not exact TPOT) | - | - |
| Compute | - | - |
| Recv wait | - | - |
| MoE router | - | - |
| MoE experts | - | - |
| MoE shared/dense | - | - |
| Compute ratio | - | - |
| Tokens/sec | 0.40 | - |

- `Model-side generation total` is `total_time_ms` for the full generation call.
- The Attention/FFN columns are rank-level wall-clock views of the same serial run; they are not per-role compute decomposition.
- `Decode TPOT` is the serial decode baseline used for decode speedup.

## Layer averages summary

_No per-layer events recorded._

## Per-layer breakdown

_No per-layer events recorded._

---
_Generated from `timing_attention_serial_b512_s512_t20.json` + `timing_ffn_serial_b512_s512_t20.json`._
