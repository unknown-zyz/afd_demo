# Experiment report — decode-dbo b2 s64 t2

## Configuration

- **Mode**: `decode-dbo`
- **Batch size**: 2
- **Prefill seq**: 64
- **Decode tokens**: 2
- **Layers**: ?
- **Micro-batches**: ?
- **Comm timing mode**: `completion`
- **Requested prefill seq**: 64
- **Actual prompt len**: 64

## Decode timing (exact TPOT + decode step detail)

| Metric | Attention rank view | FFN rank view |
|---|---:|---:|
| Decode step 1 timing total (0-based; 2nd decode-loop iteration) | 3739.921 ms | 3739.860 ms |
| Prefill / TTFT-path | 2608.389 ms | 2607.885 ms |
| Decode loop total | 233.519 ms | 233.469 ms |
| Decode steps | 1 | 1 |
| Decode TPOT | 233.519 ms | 233.469 ms |
| Legacy decode step (not exact TPOT) | - | - |
| Compute | - | - |
| Recv wait | - | - |
| MoE router | - | - |
| MoE experts | - | - |
| MoE shared/dense | - | - |
| Compute ratio | - | - |
| Tokens/sec | 0.53 | - |

## Layer averages summary

_No per-layer events recorded._

## Per-layer breakdown

_No per-layer events recorded._

---
_Generated from `timing_attention_xhost_static_decode-dbo_ep8_broadcast_reduce_overlap_b2_s64_t2.json` + `timing_ffn_coordinator_xhost_static_decode-dbo_ep8_broadcast_reduce_overlap_b2_s64_t2.json`._
