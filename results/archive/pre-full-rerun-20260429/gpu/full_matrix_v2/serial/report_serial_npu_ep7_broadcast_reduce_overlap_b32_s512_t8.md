# Experiment report — serial b32 s512 t8

## Configuration

- **Mode**: `serial`
- **Batch size**: 32
- **Prefill seq**: 512
- **Decode tokens**: 8
- **Layers**: ?
- **Micro-batches**: ?
- **Comm timing mode**: `enqueue`
- **Requested prefill seq**: 512
- **Actual prompt len**: 512

## Serial timing (model-side total + decode fields)

| Metric | Attention rank view | FFN rank view |
|---|---:|---:|
| Model-side generation total | 5230.651 ms | 5230.575 ms |
| Prefill / TTFT-path | 3902.721 ms | 3901.987 ms |
| Decode loop total | 1323.408 ms | 1323.413 ms |
| Decode steps | 7 | 7 |
| Decode TPOT | 189.058 ms | 189.059 ms |
| Legacy decode step (not exact TPOT) | - | - |
| Compute | - | - |
| Recv wait | - | - |
| MoE router | - | - |
| MoE experts | - | - |
| MoE shared/dense | - | - |
| Compute ratio | - | - |
| Tokens/sec | 1.53 | - |

- `Model-side generation total` is `total_time_ms` for the full generation call.
- The Attention/FFN columns are rank-level wall-clock views of the same serial run; they are not per-role compute decomposition.
- `Decode TPOT` is the serial decode baseline used for decode speedup.

## Layer averages summary

_No per-layer events recorded._

## Per-layer breakdown

_No per-layer events recorded._

---
_Generated from `timing_attention_serial_npu_ep7_broadcast_reduce_overlap_b32_s512_t8.json` + `timing_ffn_coordinator_serial_npu_ep7_broadcast_reduce_overlap_b32_s512_t8.json`._
