# Experiment report — serial b512 s256 t8

## Configuration

- **Mode**: `serial`
- **Batch size**: 512
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
| Model-side generation total | 13953.056 ms | 13953.047 ms |
| Prefill / TTFT-path | 11045.798 ms | 11043.447 ms |
| Decode loop total | 2902.777 ms | 2902.793 ms |
| Decode steps | 7 | 7 |
| Decode TPOT | 414.682 ms | 414.685 ms |
| Legacy decode step (not exact TPOT) | - | - |
| Compute | - | - |
| Recv wait | - | - |
| MoE router | - | - |
| MoE experts | - | - |
| MoE shared/dense | - | - |
| Compute ratio | - | - |
| Tokens/sec | 0.57 | - |

- `Model-side generation total` is `total_time_ms` for the full generation call.
- The Attention/FFN columns are rank-level wall-clock views of the same serial run; they are not per-role compute decomposition.
- `Decode TPOT` is the serial decode baseline used for decode speedup.

## Layer averages summary

_No per-layer events recorded._

## Per-layer breakdown

_No per-layer events recorded._

---
_Generated from `timing_attention_serial_npu_ep7_broadcast_reduce_overlap_b512_s256_t8.json` + `timing_ffn_coordinator_serial_npu_ep7_broadcast_reduce_overlap_b512_s256_t8.json`._
