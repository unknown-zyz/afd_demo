# D4 — Fallback cross-host RT bench (torch.dist all_to_all_single over HCCL)

Date: 2026-05-20
Branch: `feat/coordinator-arch-skeleton` @ b2147cc
Script: `scripts/cross_host_fallback_rt_bench.py`

## Topology

Same as D3: Host1 NPU0 (rank 0, 192.168.0.125) ↔ Host2 NPU0 (rank 1, 192.168.0.192).
backend=hccl, init_method=env://.

## Result: PASS (cross-host all_to_all_single works on HCCL)

Round-trip = dispatch(`all_to_all_single`) + combine(`all_to_all_single`).
Each rank starts with `[num_tokens, hidden]` bf16, scatters to peer, peer echoes back.

| Config                                | Payload  | mean (µs) | p50 | p99 | min | max  |
|---------------------------------------|----------|-----------|-----|-----|-----|------|
| 64 tokens × 4096 hidden × bf16, n=50  | 512 KiB | **379.2** | 371 | 453 | 347 | 495  |
| 8 tokens × 4096 hidden × bf16, n=100  | 64 KiB  | **330.3** | 327 | 414 | 296 | 462  |

(See `rt_rank0.log`, `rt_rank1.log`, `rt_small_rank0.log`, `rt_small_rank1.log`.)

## Interpretation

- Cross-host RoCE RTT was 0.27 ms (D2). HCCL `all_to_all_single` adds ~60 µs minimum.
- Two back-to-back all_to_all_single ≈ 296-495 µs round-trip; **above the 300 µs target**
  set by the design doc but represents the floor for this RoCE+HCCL stack without
  RDMA-optimized custom kernels (DeepEP path).
- Payload size has limited impact (64 KiB vs 512 KiB only +50 µs) → latency-bound,
  not bandwidth-bound.
- Verdict: `FallbackMoECommunicator` over HCCL is **functionally correct cross-host**
  but cannot meet the 300 µs target. Hitting the target requires either:
  1. Rebuilding DeepEP-Ascend against current CANN (unblocks IBGDA path)
  2. Co-locating attn+ffn on same node (intra-node HCCS, expected ~50-100 µs RT)

## Warnings observed

- `Cannot create tensor with internal format while allow_internel_format=False` →
  benign torch_npu format hint, doesn't affect result.

## Status of D-phase tasks

| Task | Status |
|------|--------|
| D1 — Host2 container | ✅ |
| D2 — Cross-host control plane | ✅ |
| D3 — DeepEP Buffer construction | ✅ |
| D3 — DeepEP dispatch+combine RT | ❌ BLOCKED (aclnnDispatchLayout missing in libopapi.so) |
| D4 — Fallback all_to_all_single RT | ✅ (functional, latency above target) |
| D5 — End-to-end Qwen3-30B decode | pending |
