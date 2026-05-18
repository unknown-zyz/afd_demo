# Experiment archive

This document keeps the high-level conclusions from experiment branches that
should not remain as long-lived development branches.

## Current positive mainline

### NPU EP7 + overlap

The current NPU recommendation is EP7 with `broadcast_reduce_overlap`.
`results_npu_ep7/` is the canonical full-matrix result directory. It contains:

- 45 serial baselines.
- 42 decode-dbo successful configurations.
- 35 prefill-dbo successful configurations.
- 34 decode-dbo-crosslayer configurations.

The current README in `results_npu_ep7/` explains the speedup heatmaps,
pipeline figures, OOM boundaries, and why crosslayer does not help on EP7.

### Fused dispatch

`exp/mb4-experiment` added a fused EP dispatch path that packs
`hidden_2d`, `selected_experts`, and `routing_weights` into one broadcast
instead of three. In `results_npu_ep7_mb4_v2/`, MB2 fused improves decode TPOT
by about 5-8% for small/mid batches (`b=8..32`, `seq=512`, `t=20`).

This is merged as a positive mainline optimization.

## Negative or non-mainline experiments

### MB4 micro-batches

MB4 is supported by `--num-micro-batches 4`, but current EP7 results show it is
slower than MB2:

- Decode MB4 is 9-48% slower than MB2 in the tested `seq=512` sweep.
- Prefill MB4 remains faster than serial but slower than MB2.
- The bottleneck is HCCL/dispatch serialization and smaller MoE GEMMs.

Keep MB4 as an experiment knob, not as the default recommendation.

### Controller CPU baseline

Branch: `exp/controller-baseline`.

The centralized CPU relay path was a deliberate data-plane baseline:

```text
Attention NPU -> D2H -> TCP controller -> H2D -> FFN coordinator
```

It is 40-56x slower than direct HCCL in representative decode configurations.
This proves the current direct device-to-device HCCL path is the correct
mainline direction. Do not merge the controller relay runtime path.

### Dual NPU compute streams

Branch: `exp/npu-ep-dual-stream`.

The hypothesis was to run MB0/MB1 MoE expert GEMMs on separate NPU streams. It
did not produce observable speedup. The likely reason is that Qwen3 MoE expert
compute is HBM-bandwidth-bound on 910C, so multiple streams compete for the same
memory subsystem.

Do not merge this runtime path.

### NPU MoE backend probes

Branch: `exp/npu-moe-backend`.

Findings:

- `torch_npu.npu_moe_gating_top_k_softmax` and token permute/unpermute are
  correct and usable.
- `torch_npu.npu_grouped_matmul(group_type=-1)` is correct for Qwen3 experts
  but 4.9-11.1x slower than the HF expert loop in the tested forms.
- HF Qwen3 MoE expert code cannot be captured by NPUGraph/aclgraph on 910C;
  capture fails with ACL stream synchronization errors.

Do not wire these probes into the main pipeline.

### FFN balance / layer placement

Branch: `exp/npu-ffn-balance`.

This explored configurable NPU layer placement and conservative D2D copies.
It is not merged in this consolidation because the current EP7 branch already
has a validated topology and the placement branch needs an isolated review.

### NVSHMEM

Branch: `feat/nvshmem`.

NVSHMEM was a CUDA-side single-sided communication design/prototype. It is not
part of the current Ascend 910C/HCCL path and is not merged.
