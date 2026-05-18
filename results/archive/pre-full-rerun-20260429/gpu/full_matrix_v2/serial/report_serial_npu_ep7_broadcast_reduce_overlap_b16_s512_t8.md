# Experiment report — serial b16 s512 t8

## Configuration

- **Mode**: `serial`
- **Batch size**: 16
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
| Model-side generation total | 4642.018 ms | 4641.972 ms |
| Prefill / TTFT-path | 3488.792 ms | 3488.111 ms |
| Decode loop total | 1148.900 ms | 1148.924 ms |
| Decode steps | 7 | 7 |
| Decode TPOT | 164.129 ms | 164.132 ms |
| Legacy decode step (not exact TPOT) | - | - |
| Compute | - | - |
| Recv wait | - | - |
| MoE router | - | - |
| MoE experts | - | - |
| MoE shared/dense | - | - |
| Compute ratio | - | - |
| Tokens/sec | 1.72 | - |

- `Model-side generation total` is `total_time_ms` for the full generation call.
- The Attention/FFN columns are rank-level wall-clock views of the same serial run; they are not per-role compute decomposition.
- `Decode TPOT` is the serial decode baseline used for decode speedup.

## Layer averages summary

_No per-layer events recorded._

## Per-layer breakdown

_No per-layer events recorded._

---
_Generated from `timing_attention_serial_npu_ep7_broadcast_reduce_overlap_b16_s512_t8.json` + `timing_ffn_coordinator_serial_npu_ep7_broadcast_reduce_overlap_b16_s512_t8.json`._
