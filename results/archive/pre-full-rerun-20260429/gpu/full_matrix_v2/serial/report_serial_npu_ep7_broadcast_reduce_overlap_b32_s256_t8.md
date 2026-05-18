# Experiment report — serial b32 s256 t8

## Configuration

- **Mode**: `serial`
- **Batch size**: 32
- **Prefill seq**: 256
- **Decode tokens**: 8
- **Layers**: ?
- **Micro-batches**: ?
- **Comm timing mode**: `enqueue`
- **Requested prefill seq**: 256
- **Actual prompt len**: 256

## Serial timing (model-side total + decode fields)

| Metric | Attention rank view | FFN rank view |
|---|---:|---:|
| Model-side generation total | 4706.499 ms | 4706.495 ms |
| Prefill / TTFT-path | 3579.865 ms | 3579.158 ms |
| Decode loop total | 1122.447 ms | 1122.452 ms |
| Decode steps | 7 | 7 |
| Decode TPOT | 160.350 ms | 160.350 ms |
| Legacy decode step (not exact TPOT) | - | - |
| Compute | - | - |
| Recv wait | - | - |
| MoE router | - | - |
| MoE experts | - | - |
| MoE shared/dense | - | - |
| Compute ratio | - | - |
| Tokens/sec | 1.70 | - |

- `Model-side generation total` is `total_time_ms` for the full generation call.
- The Attention/FFN columns are rank-level wall-clock views of the same serial run; they are not per-role compute decomposition.
- `Decode TPOT` is the serial decode baseline used for decode speedup.

## Layer averages summary

_No per-layer events recorded._

## Per-layer breakdown

_No per-layer events recorded._

---
_Generated from `timing_attention_serial_npu_ep7_broadcast_reduce_overlap_b32_s256_t8.json` + `timing_ffn_coordinator_serial_npu_ep7_broadcast_reduce_overlap_b32_s256_t8.json`._
