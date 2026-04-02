# AFD + DBO Performance Experiment Summary

## Experiment Setup

- **Model**: Qwen3-30B-A3B (48 layers, hidden_size=2048, MoE sparse)
- **Precision**: bfloat16
- **Deployment**:
  - **Local**: Single node, 4 GPUs (GPU 0,1 = Attention, GPU 2,3 = FFN), NVLink interconnect
  - **Multinode**: 2 nodes (Attention on local, FFN on remote), TCP/NCCL interconnect
- **DBO**: 2 micro-batches, prefill phase
- **Profiling**: CUDA-synchronized timing with background polling for transfer measurement

## Key Results

### Batch Scaling (seq=128, 48 layers)

| Batch | Local DBO | Local Serial | Speedup | Multi DBO | Multi Serial | Speedup |
|------:|----------:|-------------:|--------:|----------:|-------------:|--------:|
|     4 |   3683 ms|      3642 ms |   0.99x |   3708 ms|      3716 ms |   1.00x |
|     8 |   3869 ms|      3877 ms |   1.00x |   3902 ms|      3935 ms |   1.01x |
|    16 |   4477 ms|      4590 ms |   1.03x |   4585 ms|      4696 ms |   1.02x |
|    32 |   5609 ms|      5853 ms |   1.04x |   5726 ms|      5997 ms |   1.05x |
|    64 |   7881 ms|           — |      — |   8148 ms|           — |      — |

### Sequence Scaling (batch=8, 48 layers)

| Seq   | Local DBO | Multi DBO | Multi/Local |
|------:|----------:|----------:|------------:|
|    32 |   3542 ms|   3586 ms|       1.01x |
|    64 |   3607 ms|   3701 ms|       1.03x |
|   128 |   3869 ms|   3946 ms|       1.02x |
|   256 |   4642 ms|   4731 ms|       1.02x |
|   512 |   6003 ms|   6139 ms|       1.02x |

## Analysis

### DBO Speedup
- DBO achieves **1.00x–1.05x** speedup over serial baseline
- Speedup increases with batch size: larger batches have proportionally more compute, giving DBO more opportunity to overlap communication
- At small batches (b4), DBO overhead roughly cancels the overlap benefit

### Local vs Multinode
- Multinode overhead is only **1–3%** over local, indicating efficient NCCL communication
- Transfer times are ~0.3 ms/layer on both local (NVLink) and multinode
- The small difference suggests the network may be using RDMA or high-bandwidth interconnect

### Per-Layer Time Breakdown (Local, L1+ avg)
- **Attention compute**: 1.1–7.0 ms/layer (scales with batch×seq)
- **FFN compute**: 59–116 ms/layer (dominates; MoE expert dispatch is expensive)
- **A→F transfer**: 0.2–0.6 ms/layer
- **F→A transfer**: 0.2–0.6 ms/layer
- Communication is <1% of per-layer time → overlap benefit is minimal

### Why DBO Speedup is Modest
1. **FFN dominates**: FFN compute is 10–50x attention compute due to MoE experts
2. **Transfer is tiny**: After halving A2F data (residual pre-addition), transfer is ~0.3ms vs ~60ms FFN
3. **Asymmetric pipeline**: The Attn→FFN pipeline is heavily FFN-bound; DBO overlap of communication only saves ~0.3ms × 48 layers ≈ 14ms out of 3600ms+ total

## Profiling Improvements Made

1. **CUDA synchronization**: Added `torch.cuda.synchronize()` before/after GPU compute to get accurate timing (was measuring only kernel dispatch time)
2. **A2F data halving**: Pre-combine `attn_output + residual` before sending (was sending both separately)
3. **Background polling**: Transfer time measured via polling thread (`is_completed()` at ~100μs intervals) instead of `wait()` duration
4. **Pre-compute sync**: Drain pending GPU work before starting measurement to avoid contamination from prior async operations

## Files

- `results/prefill_dbo/batch_scaling/` — Batch size scaling experiments (b4–b64)
- `results/prefill_dbo/seq_scaling/` — Sequence length scaling experiments (s32–s512)
- `results/prefill_dbo/performance_summary.png` — Combined summary chart
- `results/prefill_dbo/batch_scaling_comparison.png` — Batch scaling comparison chart
- `results/prefill_dbo/seq_scaling_comparison.png` — Seq scaling comparison chart
- `results/prefill_dbo/logs/` — Experiment logs
- `results/prefill_dbo/archive/` — Previous experiment results (pre-profiling-fix)
