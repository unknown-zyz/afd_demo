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
| Decode step 1 timing total (0-based; 2nd decode-loop iteration) | 4713.452 ms | 4713.472 ms |
| Prefill / TTFT-path | 3647.898 ms | 3647.349 ms |
| Decode loop total | 243.832 ms | 243.811 ms |
| Decode steps | 1 | 1 |
| Decode TPOT | 243.832 ms | 243.811 ms |
| Legacy decode step (not exact TPOT) | - | - |
| Compute | - | - |
| Recv wait | - | - |
| MoE router | - | - |
| MoE experts | - | - |
| MoE shared/dense | - | - |
| Compute ratio | - | - |
| Tokens/sec | 0.42 | - |

## Layer averages summary

_No per-layer events recorded._

## Per-layer breakdown

_No per-layer events recorded._

---
_Generated from `timing_attention_xhost_static_decode-dbo_ep12_broadcast_reduce_sync_b2_s64_t2.json` + `timing_ffn_coordinator_xhost_static_decode-dbo_ep12_broadcast_reduce_sync_b2_s64_t2.json`._
