#!/usr/bin/env python3
"""NCCL vs NVSHMEM latency comparison chart."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Load NCCL benchmark data
with open('results/experiments_warmup/nvshmem/latency_benchmark.json') as f:
    nccl_data = json.load(f)

# NVSHMEM data from earlier test (test-nvshmem agent results)
# These are sender-side latencies from the 2-PE MPI test
nvshmem_data = {
    '64': {'cold_ms': 0.73, 'warm_avg_ms': 0.257},
    '4096': {'cold_ms': 0.73, 'warm_avg_ms': 0.123},
    '65536': {'cold_ms': 0.73, 'warm_avg_ms': 0.124},
    '1048576': {'cold_ms': 0.73, 'warm_avg_ms': 0.147},
}

# Common sizes for comparison
sizes = [64, 512, 4096, 32768, 131072, 524288, 1048576, 4194304, 8388608]
size_labels = ['64B', '512B', '4KB', '32KB', '128KB', '512KB', '1MB', '4MB', '8MB']

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# --- Plot 1: Cold start comparison ---
ax1 = axes[0]
nccl_cold = [nccl_data['nccl'].get(str(s), {}).get('cold_ms', 0) for s in sizes]
nvshmem_cold_val = 0.73  # Constant from test (init overhead, not size-dependent)
nvshmem_cold = [nvshmem_cold_val] * len(sizes)

x = np.arange(len(sizes))
width = 0.35
bars1 = ax1.bar(x - width/2, nccl_cold, width, label='NCCL', color='#e74c3c', alpha=0.8)
bars2 = ax1.bar(x + width/2, nvshmem_cold[:len(sizes)], width, label='NVSHMEM', color='#2ecc71', alpha=0.8)

ax1.set_ylabel('Latency (ms)')
ax1.set_title('Cold Start Latency: NCCL vs NVSHMEM')
ax1.set_xticks(x)
ax1.set_xticklabels(size_labels, rotation=45, ha='right')
ax1.legend()
ax1.set_yscale('log')
ax1.grid(axis='y', alpha=0.3)

# Add value labels
for bar in bars1:
    if bar.get_height() > 0.1:
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=8)
for bar in bars2:
    if bar.get_height() > 0.1:
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=8)

# --- Plot 2: Warm latency comparison ---
ax2 = axes[1]
nccl_warm = [nccl_data['nccl'].get(str(s), {}).get('warm_avg_ms', 0) for s in sizes]

# NVSHMEM warm data (only have some sizes)
nvshmem_warm_map = {64: 0.257, 4096: 0.123, 65536: 0.124, 1048576: 0.147}
nvshmem_warm = []
for s in sizes:
    if s in nvshmem_warm_map:
        nvshmem_warm.append(nvshmem_warm_map[s])
    else:
        nvshmem_warm.append(None)

# Plot NCCL warm
ax2.bar(x - width/2, nccl_warm, width, label='NCCL', color='#e74c3c', alpha=0.8)

# Plot NVSHMEM warm (only where we have data)
nvshmem_x = [i for i, v in enumerate(nvshmem_warm) if v is not None]
nvshmem_y = [v for v in nvshmem_warm if v is not None]
ax2.bar([xi + width/2 for xi in nvshmem_x], nvshmem_y, width, label='NVSHMEM', color='#2ecc71', alpha=0.8)

ax2.set_ylabel('Latency (ms)')
ax2.set_title('Warm Latency: NCCL vs NVSHMEM')
ax2.set_xticks(x)
ax2.set_xticklabels(size_labels, rotation=45, ha='right')
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('results/experiments_warmup/images/nccl_vs_nvshmem_latency.png', dpi=150, bbox_inches='tight')
print("Saved: results/experiments_warmup/images/nccl_vs_nvshmem_latency.png")
