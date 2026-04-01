#!/usr/bin/env python3
"""
Batch Scaling 实验对比图表生成工具

生成多个子图对比不同配置下的性能指标：
- 总推理时间 vs batch/seq
- 每层计算时间 vs batch/seq
- 通信时间 vs batch/seq
- 单机 vs 多机对比
"""

import json
import os
import sys

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    print("Error: matplotlib and numpy required")
    sys.exit(1)


def load_experiment(suffix):
    attn_f = f"results/prefill_dbo/timing_attention_{suffix}.json"
    ffn_f = f"results/prefill_dbo/timing_ffn_{suffix}.json"
    if not os.path.exists(attn_f) or not os.path.exists(ffn_f):
        return None
    with open(attn_f) as f:
        attn = json.load(f)
    with open(ffn_f) as f:
        ffn = json.load(f)

    attn_comp = [e for e in attn['events'] if e['type'] == 'attn_compute' and e['layer'] > 0]
    ffn_comp = [e for e in ffn['events'] if e['type'] == 'ffn_compute' and e['layer'] > 0]
    a2f = [e for e in attn['events'] if e['type'] == 'send_transfer' and e['layer'] > 0]
    f2a = [e for e in ffn['events'] if e['type'] == 'send_transfer' and e['layer'] > 0]

    return {
        'total_ms': attn['total_time_ms'],
        'attn_total': attn['total_compute_ms'],
        'ffn_total': ffn['total_compute_ms'],
        'compute_ratio': attn['compute_ratio'] * 100,
        'attn_per_layer': sum(e['duration_ms'] for e in attn_comp) / max(len(attn_comp), 1),
        'ffn_per_layer': sum(e['duration_ms'] for e in ffn_comp) / max(len(ffn_comp), 1),
        'a2f_per_layer': sum(e['duration_ms'] for e in a2f) / max(len(a2f), 1),
        'f2a_per_layer': sum(e['duration_ms'] for e in f2a) / max(len(f2a), 1),
    }


def plot_batch_scaling():
    """Phase 1: Batch size scaling comparison."""
    batches = [4, 8, 16, 32, 64]
    local_data = []
    multi_data = []

    for b in batches:
        ld = load_experiment(f"local_b{b}_s128_t5")
        md = load_experiment(f"multinode_b{b}_s128_t5")
        local_data.append(ld)
        multi_data.append(md)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Phase 1: Batch Size Scaling (seq=128, tokens=5)', fontsize=14, fontweight='bold')

    x = np.arange(len(batches))
    w = 0.35

    # 1. Total inference time
    ax = axes[0, 0]
    ax.bar(x - w/2, [d['total_ms'] for d in local_data], w, label='Local', color='#4CAF50')
    ax.bar(x + w/2, [d['total_ms'] for d in multi_data], w, label='Multinode', color='#2196F3')
    ax.set_xlabel('Batch Size')
    ax.set_ylabel('Time (ms)')
    ax.set_title('Total Inference Time')
    ax.set_xticks(x)
    ax.set_xticklabels(batches)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # 2. Per-layer compute time
    ax = axes[0, 1]
    ax.bar(x - w/2, [d['attn_per_layer'] for d in local_data], w/2, label='Attn (Local)', color='#4CAF50')
    ax.bar(x, [d['ffn_per_layer'] for d in local_data], w/2, label='FFN (Local)', color='#81C784')
    ax.bar(x + w/2, [d['ffn_per_layer'] for d in multi_data], w/2, label='FFN (Multi)', color='#64B5F6')
    ax.set_xlabel('Batch Size')
    ax.set_ylabel('Time per Layer (ms)')
    ax.set_title('Per-Layer Compute Time')
    ax.set_xticks(x)
    ax.set_xticklabels(batches)
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)

    # 3. Communication time per layer
    ax = axes[1, 0]
    ax.bar(x - w/2, [d['a2f_per_layer'] for d in local_data], w/2, label='A2F (Local)', color='#FF9800')
    ax.bar(x - w/2 + w/2, [d['a2f_per_layer'] for d in multi_data], w/2, label='A2F (Multi)', color='#FFB74D')
    ax.bar(x + w/2, [d['f2a_per_layer'] for d in local_data], w/2, label='F2A (Local)', color='#F44336')
    ax.bar(x + w/2 + w/2, [d['f2a_per_layer'] for d in multi_data], w/2, label='F2A (Multi)', color='#EF9A9A')
    ax.set_xlabel('Batch Size')
    ax.set_ylabel('Time per Layer (ms)')
    ax.set_title('Communication Time per Layer (A2F vs F2A)')
    ax.set_xticks(x)
    ax.set_xticklabels(batches)
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)

    # 4. Compute ratio
    ax = axes[1, 1]
    ax.plot(batches, [d['compute_ratio'] for d in local_data], 'o-', label='Local', color='#4CAF50', linewidth=2)
    ax.plot(batches, [d['compute_ratio'] for d in multi_data], 's-', label='Multinode', color='#2196F3', linewidth=2)
    ax.set_xlabel('Batch Size')
    ax.set_ylabel('Compute Ratio (%)')
    ax.set_title('Compute Efficiency (higher = better)')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 35)

    plt.tight_layout()
    out = 'results/prefill_dbo/batch_scaling_comparison.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ {out}")


def plot_seq_scaling():
    """Phase 2: Sequence length scaling comparison."""
    seqs = [32, 64, 128, 256, 512]
    local_data = []
    multi_data = []

    for s in seqs:
        ld = load_experiment(f"local_b8_s{s}_t5")
        md = load_experiment(f"multinode_b8_s{s}_t5")
        local_data.append(ld)
        multi_data.append(md)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Phase 2: Sequence Length Scaling (batch=8, tokens=5)', fontsize=14, fontweight='bold')

    x = np.arange(len(seqs))
    w = 0.35

    # 1. Total time
    ax = axes[0, 0]
    ax.bar(x - w/2, [d['total_ms'] for d in local_data], w, label='Local', color='#4CAF50')
    ax.bar(x + w/2, [d['total_ms'] for d in multi_data], w, label='Multinode', color='#2196F3')
    ax.set_xlabel('Sequence Length')
    ax.set_ylabel('Time (ms)')
    ax.set_title('Total Inference Time')
    ax.set_xticks(x)
    ax.set_xticklabels(seqs)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # 2. Per-layer compute
    ax = axes[0, 1]
    ax.plot(seqs, [d['attn_per_layer'] for d in local_data], 'o-', label='Attn (Local)', color='#4CAF50', linewidth=2)
    ax.plot(seqs, [d['ffn_per_layer'] for d in local_data], 's-', label='FFN (Local)', color='#81C784', linewidth=2)
    ax.plot(seqs, [d['ffn_per_layer'] for d in multi_data], '^-', label='FFN (Multi)', color='#64B5F6', linewidth=2)
    ax.set_xlabel('Sequence Length')
    ax.set_ylabel('Time per Layer (ms)')
    ax.set_title('Per-Layer Compute Time')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # 3. Communication
    ax = axes[1, 0]
    ax.plot(seqs, [d['a2f_per_layer'] for d in local_data], 'o-', label='A2F (Local)', color='#FF9800', linewidth=2)
    ax.plot(seqs, [d['a2f_per_layer'] for d in multi_data], 's-', label='A2F (Multi)', color='#FFB74D', linewidth=2)
    ax.plot(seqs, [d['f2a_per_layer'] for d in local_data], '^-', label='F2A (Local)', color='#F44336', linewidth=2)
    ax.plot(seqs, [d['f2a_per_layer'] for d in multi_data], 'v-', label='F2A (Multi)', color='#EF9A9A', linewidth=2)
    ax.set_xlabel('Sequence Length')
    ax.set_ylabel('Time per Layer (ms)')
    ax.set_title('Communication Time per Layer')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # 4. Compute ratio
    ax = axes[1, 1]
    ax.plot(seqs, [d['compute_ratio'] for d in local_data], 'o-', label='Local', color='#4CAF50', linewidth=2)
    ax.plot(seqs, [d['compute_ratio'] for d in multi_data], 's-', label='Multinode', color='#2196F3', linewidth=2)
    ax.set_xlabel('Sequence Length')
    ax.set_ylabel('Compute Ratio (%)')
    ax.set_title('Compute Efficiency')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 35)

    plt.tight_layout()
    out = 'results/prefill_dbo/seq_scaling_comparison.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ {out}")


def plot_combined():
    """Phase 3: Combined high load comparison."""
    configs = [('b8s256', 8, 256), ('b16s128', 16, 128), ('b32s64', 32, 64)]
    local_data = []
    multi_data = []

    for label, b, s in configs:
        ld = load_experiment(f"local_b{b}_s{s}_t5")
        md = load_experiment(f"multinode_b{b}_s{s}_t5")
        local_data.append(ld)
        multi_data.append(md)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Phase 3: High Load Combinations (batch×seq ≈ 2048)', fontsize=14, fontweight='bold')

    labels = [c[0] for c in configs]
    x = np.arange(len(labels))
    w = 0.35

    # 1. Total time
    ax = axes[0]
    ax.bar(x - w/2, [d['total_ms'] for d in local_data], w, label='Local', color='#4CAF50')
    ax.bar(x + w/2, [d['total_ms'] for d in multi_data], w, label='Multinode', color='#2196F3')
    ax.set_xlabel('Configuration')
    ax.set_ylabel('Time (ms)')
    ax.set_title('Total Inference Time')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # 2. Per-layer breakdown (stacked)
    ax = axes[1]
    attn_l = [d['attn_per_layer'] for d in local_data]
    ffn_l = [d['ffn_per_layer'] for d in local_data]
    a2f_l = [d['a2f_per_layer'] for d in local_data]
    f2a_l = [d['f2a_per_layer'] for d in local_data]
    ax.bar(x, attn_l, w, label='Attn', color='#4CAF50')
    ax.bar(x, ffn_l, w, bottom=attn_l, label='FFN', color='#2196F3')
    ax.bar(x, a2f_l, w, bottom=[a+f for a, f in zip(attn_l, ffn_l)], label='A2F', color='#FF9800')
    ax.bar(x, f2a_l, w, bottom=[a+f+c for a, f, c in zip(attn_l, ffn_l, a2f_l)], label='F2A', color='#F44336')
    ax.set_xlabel('Configuration (Local)')
    ax.set_ylabel('Time per Layer (ms)')
    ax.set_title('Per-Layer Time Breakdown')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)

    # 3. Compute ratio
    ax = axes[2]
    ax.bar(x - w/2, [d['compute_ratio'] for d in local_data], w, label='Local', color='#4CAF50')
    ax.bar(x + w/2, [d['compute_ratio'] for d in multi_data], w, label='Multinode', color='#2196F3')
    ax.set_xlabel('Configuration')
    ax.set_ylabel('Compute Ratio (%)')
    ax.set_title('Compute Efficiency')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 30)

    plt.tight_layout()
    out = 'results/prefill_dbo/combined_comparison.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ {out}")


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)) + '/..')
    print("Generating comparison plots...")
    plot_batch_scaling()
    plot_seq_scaling()
    plot_combined()
    print("Done!")
