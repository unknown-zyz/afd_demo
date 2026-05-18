# Experiment report — serial b256 s128 t8

## Configuration

- **Mode**: `serial`
- **Batch size**: 256
- **Prefill seq**: 128
- **Decode tokens**: 8
- **Layers**: ?
- **Micro-batches**: ?
- **Comm timing mode**: `enqueue`
- **Requested prefill seq**: 128
- **Actual prompt len**: 128

## Serial timing (model-side total + decode fields)

| Metric | Attention rank view | FFN rank view |
|---|---:|---:|
| Model-side generation total | 6564.518 ms | 6564.408 ms |
| Prefill / TTFT-path | 4893.538 ms | 4892.226 ms |
| Decode loop total | 1666.668 ms | 1666.670 ms |
| Decode steps | 7 | 7 |
| Decode TPOT | 238.095 ms | 238.096 ms |
| Legacy decode step (not exact TPOT) | - | - |
| Compute | - | - |
| Recv wait | - | - |
| MoE router | - | - |
| MoE experts | - | - |
| MoE shared/dense | - | - |
| Compute ratio | - | - |
| Tokens/sec | 1.22 | - |

- `Model-side generation total` is `total_time_ms` for the full generation call.
- The Attention/FFN columns are rank-level wall-clock views of the same serial run; they are not per-role compute decomposition.
- `Decode TPOT` is the serial decode baseline used for decode speedup.

## Layer averages summary

_No per-layer events recorded._

## Per-layer breakdown

_No per-layer events recorded._

---
_Generated from `timing_attention_serial_npu_ep7_broadcast_reduce_overlap_b256_s128_t8.json` + `timing_ffn_coordinator_serial_npu_ep7_broadcast_reduce_overlap_b256_s128_t8.json`._
