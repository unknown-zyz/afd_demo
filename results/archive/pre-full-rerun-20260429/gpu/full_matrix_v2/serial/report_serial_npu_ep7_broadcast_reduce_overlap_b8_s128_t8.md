# Experiment report — serial b8 s128 t8

## Configuration

- **Mode**: `serial`
- **Batch size**: 8
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
| Model-side generation total | 3275.967 ms | 3275.864 ms |
| Prefill / TTFT-path | 2365.788 ms | 2365.220 ms |
| Decode loop total | 905.987 ms | 905.971 ms |
| Decode steps | 7 | 7 |
| Decode TPOT | 129.427 ms | 129.424 ms |
| Legacy decode step (not exact TPOT) | - | - |
| Compute | - | - |
| Recv wait | - | - |
| MoE router | - | - |
| MoE experts | - | - |
| MoE shared/dense | - | - |
| Compute ratio | - | - |
| Tokens/sec | 2.44 | - |

- `Model-side generation total` is `total_time_ms` for the full generation call.
- The Attention/FFN columns are rank-level wall-clock views of the same serial run; they are not per-role compute decomposition.
- `Decode TPOT` is the serial decode baseline used for decode speedup.

## Layer averages summary

_No per-layer events recorded._

## Per-layer breakdown

_No per-layer events recorded._

---
_Generated from `timing_attention_serial_npu_ep7_broadcast_reduce_overlap_b8_s128_t8.json` + `timing_ffn_coordinator_serial_npu_ep7_broadcast_reduce_overlap_b8_s128_t8.json`._
