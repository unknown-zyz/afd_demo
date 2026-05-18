# Experiment report — serial b256 s256 t8

## Configuration

- **Mode**: `serial`
- **Batch size**: 256
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
| Model-side generation total | 9031.694 ms | 9031.581 ms |
| Prefill / TTFT-path | 6899.915 ms | 6898.497 ms |
| Decode loop total | 2126.999 ms | 2127.000 ms |
| Decode steps | 7 | 7 |
| Decode TPOT | 303.857 ms | 303.857 ms |
| Legacy decode step (not exact TPOT) | - | - |
| Compute | - | - |
| Recv wait | - | - |
| MoE router | - | - |
| MoE experts | - | - |
| MoE shared/dense | - | - |
| Compute ratio | - | - |
| Tokens/sec | 0.89 | - |

- `Model-side generation total` is `total_time_ms` for the full generation call.
- The Attention/FFN columns are rank-level wall-clock views of the same serial run; they are not per-role compute decomposition.
- `Decode TPOT` is the serial decode baseline used for decode speedup.

## Layer averages summary

_No per-layer events recorded._

## Per-layer breakdown

_No per-layer events recorded._

---
_Generated from `timing_attention_serial_npu_ep7_broadcast_reduce_overlap_b256_s256_t8.json` + `timing_ffn_coordinator_serial_npu_ep7_broadcast_reduce_overlap_b256_s256_t8.json`._
