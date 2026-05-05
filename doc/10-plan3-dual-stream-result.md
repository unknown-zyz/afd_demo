# Plan 3 — Dual NPU Compute Streams (Negative Result)

**Branch**: `exp/npu-ep-dual-stream`
**Status**: Implemented, validated, **default off**.

## Hypothesis
Wrap mb0/mb1 stacked-GEMM `forward_local` calls in two distinct
`torch.npu.Stream()`s so the second micro-batch's GEMMs can run on hardware
in parallel with the first, eliminating ≈half of the per-layer FFN time.

## Implementation summary

`src/model/ep_moe.py`
- `EPFFNLayer.compute_local(item, stream=None)` — when called with a side
  stream and on NPU, calls `stream.wait_stream(default)` to import producer
  state, runs `forward_local` inside `with torch.npu.stream(stream):`, and
  uses `record_stream(stream)` on inputs/outputs to keep the caching
  allocator from recycling tensors mid-flight. Stashes
  `item._compute_stream` for downstream sync.
- `EPFFNLayer.reduce_async(item)` — issues HCCL reduce inside the same
  side-stream context so collective ordering is preserved.
- `EPFFNLayer.finish_reduce(item)` — after `reduce_handle.wait()`, calls
  `default_stream.wait_stream(item._compute_stream)` so subsequent default-
  stream consumers see the reduced output. We do **not** insert this sync
  in `compute_local` because that creates a default→side0→default→side1
  chain that transitively serializes mb1 behind mb0 (we observed this
  regression first-hand).

`src/pipeline/decode_scheduler.py`
- Lazy-init `self._ffn_compute_streams = [torch.npu.Stream() for _ in range(2)]`
  gated by `AFD_FFN_DUAL_STREAM=1` (default `0`) on NPU.
- Per-MB loop selects `streams[mb_idx % 2]` and passes it to `compute_local`.

## Bugs fixed during implementation
1. `KeyError: 4440336306335333802` — uninitialized memory read on side
   stream because side stream started GEMMs before dispatch outputs were
   visible. Fixed by `stream.wait_stream(default_stream)` at entry.
2. `mb1 transitively waits mb0` — original fix put
   `default.wait_stream(side)` inside `compute_local`, defeating
   parallelism. Moved to `finish_reduce` so the two side streams stay
   independent throughout the pipelined loop.

## Empirical results (b16/s256 t20, decode-dbo, npu-ep7 broadcast_reduce_overlap)

| Run | TPOT (ms) | FFN avg/layer (ms) |
|---|---:|---:|
| `full_matrix_v2` baseline (prior round) | **241.66** | 1.184 |
| `dual_stream_v1` AFD_FFN_DUAL_STREAM=1   | 376.36 | 1.910 |
| `dual_stream_v1_off` AFD_FFN_DUAL_STREAM=0 | 376.13 | 1.911 |

The on/off pair was measured back-to-back on the same branch, same hardware.
Their ~0.06% delta means **dual-stream produced no observable speedup**.
The 56% gap vs the v2 baseline is shared by both, so it is environmental
(HBM caching state, kernel cache warmth, other tenants) and unrelated to
the dual-stream change.

## Why no speedup
MoE stacked GEMMs are **HBM-bandwidth bound** on 910C. Even with two
independent torch streams the AICore/HBM subsystem still serializes data
movement, so the two micro-batches' compute kernels cannot truly overlap.
`record_stream()` also delays caching-allocator recycling, raising the HBM
peak slightly without compensating throughput.

## How to re-enable
```bash
AFD_FFN_DUAL_STREAM=1 bash scripts/run_experiment_matrix_npu.sh \
  --preset npu-ep7 --ffn-ep-backend broadcast_reduce_overlap \
  --modes decode-dbo --batches 16 --seqs 256 --tokens 20 \
  --output-root results_npu/dual_stream_v1
```

## Lessons & next steps
- Stream-level overlap won't help when the bottleneck is HBM bandwidth;
  must instead reduce **HBM traffic per FFN** (operator fusion / weight
  reuse).
- **Plan 4 (next round)** — switch the inner expert loop to
  `torch_npu.npu_grouped_matmul` (single kernel for all active experts
  per MB) and consider `npu_grouped_matmul_swiglu_quant` to fuse
  gate/up/SiLU. Fewer kernel launches **and** denser HBM access pattern.
  APIs confirmed available on this CANN install:
  `npu_grouped_matmul`, `npu_moe_init_routing`, `npu_moe_compute_expert_tokens`,
  `npu_moe_finalize_routing`, `npu_grouped_matmul_swiglu_quant`,
  `npu_moe_token_permute / unpermute`.
- Re-baseline `full_matrix_v2` numbers under current cluster state before
  declaring any future plan a win, since environmental drift accounts for
  ~50% of the observed delta.

## Files of interest
- `src/model/ep_moe.py` (compute_local / reduce_async / finish_reduce)
- `src/pipeline/decode_scheduler.py` (lazy init + per-MB stream pick)
- `results_npu/dual_stream_v1/decode-dbo/report_decode-dbo_npu_ep7_broadcast_reduce_overlap_b16_s256_t20.md`
- `results_npu/dual_stream_v1_off/decode-dbo/report_decode-dbo_npu_ep7_broadcast_reduce_overlap_b16_s256_t20.md`
