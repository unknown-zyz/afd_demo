# Batch & Sequence Scaling Experiment Summary

**Model**: Qwen3-30B-A3B (48 layers, MoE)
**System**: AFD (Attention-FFN Disaggregation) + DBO (Dual Batch Overlap)
**Deployments**: Single-node (4× GPU, NVLink) and Multi-node (2 machines, TCP)

## Experiment Matrix

| Phase | Variable | Fixed Params | Configs Tested |
|-------|----------|-------------|----------------|
| 1. Batch Scaling | batch=4,8,16,32,64 | seq=128, tokens=5 | 10 (5×2 deploy) |
| 2. Seq Scaling | seq=32,64,128,256,512 | batch=8, tokens=5 | 8 (4×2, seq=128 overlap) |
| 3. High Load | b8s256, b16s128, b32s64 | tokens=5 | 6 (3×2 deploy) |
| **Total** | | | **20 experiments (0 OOM)** |

## Phase 1: Batch Size Scaling (seq=128)

### Single-Node (Local)

| Batch | Total(ms) | Attn/L(ms) | FFN/L(ms) | A2F/L(ms) | F2A/L(ms) | Compute% |
|------:|----------:|----------:|----------:|----------:|----------:|---------:|
| 4 | 3491 | 6.95 | 26.03 | 44.11 | 4.70 | 24.1% |
| 8 | 3622 | 5.58 | 27.15 | 48.29 | 5.14 | 19.7% |
| 16 | 4169 | 6.84 | 31.76 | 55.66 | 6.58 | 19.9% |
| 32 | 5088 | 7.44 | 40.00 | 70.97 | 9.02 | 17.5% |
| 64 | 6843 | 7.50 | 56.15 | 103.98 | 14.60 | 13.1% |

### Multi-Node

| Batch | Total(ms) | Attn/L(ms) | FFN/L(ms) | A2F/L(ms) | F2A/L(ms) | Compute% |
|------:|----------:|----------:|----------:|----------:|----------:|---------:|
| 4 | 3530 | 5.41 | 26.42 | 46.06 | 4.76 | 21.2% |
| 8 | 3793 | 5.40 | 27.93 | 50.34 | 5.05 | 19.7% |
| 16 | 4250 | 6.42 | 31.96 | 56.48 | 6.75 | 19.8% |
| 32 | 5120 | 7.06 | 40.12 | 71.28 | 9.43 | 17.2% |
| 64 | 6907 | 7.58 | 56.49 | 104.18 | 14.63 | 13.1% |

### Key Findings
- **Attention compute is nearly constant** (~5-7ms/layer) across all batch sizes — memory-bound
- **FFN compute scales linearly**: 26ms → 56ms (2.2× for 16× batch increase)
- **A2F communication dominates**: 44ms → 104ms/layer, growing faster than compute
- **Multi-node overhead is minimal**: only 1-5% slower than single-node
- **Compute efficiency degrades**: 24% → 13% as batch increases (comm grows faster)

## Phase 2: Sequence Length Scaling (batch=8)

### Single-Node (Local)

| Seq | Total(ms) | Attn/L(ms) | FFN/L(ms) | A2F/L(ms) | F2A/L(ms) | Compute% |
|----:|----------:|----------:|----------:|----------:|----------:|---------:|
| 32 | 3348 | 7.23 | 24.48 | 40.47 | 4.29 | 26.0% |
| 64 | 3460 | 5.84 | 25.59 | 44.34 | 4.12 | 21.4% |
| 128 | 3622 | 5.58 | 27.15 | 48.29 | 5.14 | 19.7% |
| 256 | 4225 | 4.94 | 32.47 | 60.20 | 6.62 | 15.4% |
| 512 | 5337 | 6.72 | 42.51 | 76.83 | 9.53 | 15.4% |

### Key Findings
- **Similar scaling pattern to batch size** — both increase total data volume
- **seq=512 is feasible** without OOM at batch=8
- **Attention compute stays flat** even at seq=512 (still ~7ms)
- **A2F scales**: 40ms → 77ms (1.9× for 16× seq increase) — sub-linear, better than batch scaling

## Phase 3: High Load Combinations (batch×seq ≈ 2048)

| Config | Local Total(ms) | Multinode Total(ms) | Overhead |
|--------|---------------:|-------------------:|---------:|
| b8, s256 | 4225 | 4392 | +4.0% |
| b16, s128 | 4169 | 4250 | +1.9% |
| b32, s64 | 4478 | 4533 | +1.2% |

### Key Findings
- **Same total tokens, different performance**: b16s128 is fastest despite same data volume
- **Multi-node overhead decreases with larger batch**: communication better amortized

## Analysis

### Communication Bottleneck
A2F communication (40-104ms/layer) far exceeds compute time (5-56ms/layer combined). Root causes:
1. **NCCL overhead**: Small tensor sends have high fixed overhead (~2-3ms minimum)
2. **Receiver readiness**: A2F sender often waits for FFN to post irecv (~2.5ms)
3. **Serialization**: Even with DBO, communication is sequential per micro-batch

### A2F vs F2A Asymmetry
- A2F: 40-104ms (sender waits for receiver)
- F2A: 4-15ms (receiver pre-posts irecv, ready when sender arrives)
- **8-10× difference** due to synchronization pattern

### DBO Effectiveness
DBO overlaps computation of micro-batch N+1 with communication of micro-batch N. However:
- Communication time >> Compute time → overlap benefit is limited
- Real speedup comes from pipelining, not from hiding latency
- Future optimization: reduce A2F overhead via receiver pre-posting or RDMA

## File Organization

```
results/prefill_dbo/
├── batch_scaling/b{4,8,16,32,64}/   # Phase 1: timing JSON + pipeline PNG
├── seq_scaling/s{32,64,128,256,512}/ # Phase 2: timing JSON + pipeline PNG
├── archive/                           # Old baseline experiments
├── *_comparison.png                   # Comparison analysis charts
└── README.md                          # Directory documentation
```

## Comparison Charts
- `batch_scaling_comparison.png` — Total time, per-layer compute/comm, efficiency vs batch size
- `seq_scaling_comparison.png` — Same metrics vs sequence length
- `combined_comparison.png` — High-load combinations comparison
