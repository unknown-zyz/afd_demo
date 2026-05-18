# Experiment report — serial b256 s512 t20

## Configuration

- **Mode**: `serial`
- **Batch size**: 256
- **Prefill seq**: 512
- **Decode tokens**: 20
- **Layers**: ?
- **Micro-batches**: ?

## Serial timing (model-side total + decode fields)

| Metric | Attention rank view | FFN rank view |
|---|---:|---:|
| Model-side generation total | 31386.921 ms | 31386.805 ms |
| Prefill / TTFT-path | 2118.891 ms | 2115.092 ms |
| Decode loop total | 28461.236 ms | 28461.198 ms |
| Decode steps | 19 | 19 |
| Decode TPOT | 1497.960 ms | 1497.958 ms |
| Legacy decode step (not exact TPOT) | - | - |
| Compute | - | - |
| Recv wait | - | - |
| MoE router | - | - |
| MoE experts | - | - |
| MoE shared/dense | - | - |
| Compute ratio | - | - |
| Tokens/sec | 0.64 | - |

- `Model-side generation total` is `total_time_ms` for the full generation call.
- The Attention/FFN columns are rank-level wall-clock views of the same serial run; they are not per-role compute decomposition.
- `Decode TPOT` is the serial decode baseline used for decode speedup.

## Layer averages summary

_No per-layer events recorded._

## Per-layer breakdown

_No per-layer events recorded._

---
_Generated from `timing_attention_serial_b256_s512_t20.json` + `timing_ffn_serial_b256_s512_t20.json`._
