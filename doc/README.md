# Documentation index

This directory contains the maintained documentation for AFD Demo. Read the docs
in this order if you are new to the project.

| Document | Purpose |
|---|---|
| [01-architecture.md](01-architecture.md) | Current Attention/FFN split, schedulers, backend abstraction, KV cache, and timing semantics. |
| [02-usage.md](02-usage.md) | Command reference for serial, prefill DBO, decode DBO/crosslayer, matrix runs, plotting, and audits. |
| [03-api-reference.md](03-api-reference.md) | Public code and script surfaces that are still part of the current path. |
| [04-deployment.md](04-deployment.md) | GPU local/multinode setup and Ascend 910C container workflow. |
| [05-code-review-guide.md](05-code-review-guide.md) | Review checklist for scheduler, timing, baseline, OOM, and result-generation changes. |
| [npu_910c_adaptation.md](npu_910c_adaptation.md) | NPU/HCCL adaptation details and known 910C limitations. |
| [npu_vs_gpu_experiment_analysis.md](npu_vs_gpu_experiment_analysis.md) | TTFT/TPOT metric interpretation and why representative ITL is not speedup. |
| [gpu_npu_experiment_summary.md](gpu_npu_experiment_summary.md) | Latest GPU/NPU matrix coverage, OOM boundaries, baseline audit, and speedup summary. |

## Quick navigation

### I want to run one experiment

Read [02-usage.md](02-usage.md). The short version:

```bash
# serial baseline
./scripts/run_single.sh local 4 128 --tokens 20 --no-dbo --generate

# prefill DBO
./scripts/run_single.sh local 4 128 --tokens 20

# decode DBO
./scripts/run_single.sh local 4 128 --tokens 20 --generate

# decode crosslayer
./scripts/run_single.sh local 4 128 --tokens 20 --generate --crosslayer
```

### I want to understand the design

Start with [01-architecture.md](01-architecture.md), then use
[05-code-review-guide.md](05-code-review-guide.md) as the implementation review
checklist.

### I want to reproduce the latest figures

Read:

1. [02-usage.md](02-usage.md#5-matrix-experiments)
2. [gpu_npu_experiment_summary.md](gpu_npu_experiment_summary.md)
3. [npu_vs_gpu_experiment_analysis.md](npu_vs_gpu_experiment_analysis.md)

### I want to run on Ascend 910C

Read [04-deployment.md](04-deployment.md#4-ascend-910c-npu-workflow) and
[npu_910c_adaptation.md](npu_910c_adaptation.md).

## Maintenance rules

- Keep command examples synchronized with `scripts/*.sh`.
- Do not publish speedups unless `baseline_audit.csv` reports `ok`.
- Distinguish model-side TTFT-path from online end-to-end TTFT.
- Distinguish exact TPOT from representative ITL used in pipeline figures.
- Keep GPU and NPU result conclusions in
  [gpu_npu_experiment_summary.md](gpu_npu_experiment_summary.md), not scattered
  across older notes.
