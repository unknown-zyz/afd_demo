#!/usr/bin/env python3
"""Summary: MB0 cold-start fix comparison."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# MB0 A2F send latency for prefill b4s128
approaches = ['NCCL\nBaseline', 'NCCL +\nWarmup', 'NVSHMEM']
cold_latency = [47.4, 0.58, 0.73]  # ms
warm_latency = [0.15, 0.15, 0.13]  # ms (approximate)
colors_cold = ['#e74c3c', '#27ae60', '#3498db']
colors_warm = ['#e74c3c', '#27ae60', '#3498db']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Cold start
bars = ax1.bar(approaches, cold_latency, color=colors_cold, alpha=0.85, edgecolor='black', linewidth=0.5)
ax1.set_ylabel('Latency (ms)')
ax1.set_title('MB0 A→F Cold Start Latency\n(Prefill b4 s128, L0)')
ax1.set_yscale('log')
ax1.set_ylim(0.1, 100)
ax1.grid(axis='y', alpha=0.3)
for bar, val in zip(bars, cold_latency):
    ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() * 1.2,
            f'{val:.2f}ms', ha='center', va='bottom', fontsize=12, fontweight='bold')
# Add speedup annotations
ax1.annotate('81× ↓', xy=(1, 0.58), fontsize=14, color='green', fontweight='bold',
            ha='center', va='bottom', xytext=(1, 3), arrowprops=dict(arrowstyle='->', color='green'))
ax1.annotate('65× ↓', xy=(2, 0.73), fontsize=14, color='blue', fontweight='bold',
            ha='center', va='bottom', xytext=(2, 5), arrowprops=dict(arrowstyle='->', color='blue'))

# Warm (steady-state)
bars2 = ax2.bar(approaches, warm_latency, color=colors_warm, alpha=0.85, edgecolor='black', linewidth=0.5)
ax2.set_ylabel('Latency (ms)')
ax2.set_title('Warm (Steady-State) Latency\n(MB1+ avg)')
ax2.grid(axis='y', alpha=0.3)
for bar, val in zip(bars2, warm_latency):
    ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
            f'{val:.2f}ms', ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.suptitle('NCCL P2P Cold-Start Solutions Comparison (Qwen3-30B-A3B, 4×V100-SXM2)', 
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('results/experiments_warmup/images/coldstart_comparison.png', dpi=150, bbox_inches='tight')
print("Saved: results/experiments_warmup/images/coldstart_comparison.png")
