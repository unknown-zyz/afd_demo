#!/usr/bin/env python3
"""
Scaling 实验对比图表生成工具

生成多个子图对比不同配置下的性能指标：
- 总推理时间 (DBO vs Serial) vs batch/seq
- 每层计算时间 vs batch/seq
- 通信时间 vs batch/seq
- DBO speedup vs batch/seq
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
    """Load DBO experiment data. Returns None if files missing."""
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


def load_serial(suffix):
    """Load serial baseline total time. Returns None if missing."""
    f = f"results/prefill_dbo/timing_attention_{suffix}.json"
    if not os.path.exists(f):
        return None
    with open(f) as fh:
        return json.load(fh).get('total_time_ms')


def safe_get(data_list, key, default=0):
    """Extract values from list of dicts, using default for None entries."""
    return [d[key] if d else default for d in data_list]


def plot_batch_scaling():
    """Batch size scaling: DBO vs Serial, Local vs Multinode."""
    batches = [4, 8, 16, 32, 64]
    local_dbo, multi_dbo = [], []
    local_serial, multi_serial = [], []

    for b in batches:
        local_dbo.append(load_experiment(f"local_b{b}_s128_t5"))
        multi_dbo.append(load_experiment(f"multinode_b{b}_s128_t5"))
        local_serial.append(load_serial(f"serial_local_b{b}_s128_t5"))
        multi_serial.append(load_serial(f"serial_multinode_b{b}_s128_t5"))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Batch Size Scaling — Qwen3-30B-A3B (seq=128, 48 layers)',
                 fontsize=14, fontweight='bold')

    x = np.arange(len(batches))
    w = 0.2

    # 1. Total inference time: DBO vs Serial
    ax = axes[0, 0]
    dbo_local_t = safe_get(local_dbo, 'total_ms')
    dbo_multi_t = safe_get(multi_dbo, 'total_ms')
    ax.bar(x - 1.5*w, dbo_local_t, w, label='DBO Local', color='#4CAF50')
    ax.bar(x - 0.5*w, [s or 0 for s in local_serial], w, label='Serial Local', color='#81C784', alpha=0.7)
    ax.bar(x + 0.5*w, dbo_multi_t, w, label='DBO Multi', color='#2196F3')
    ax.bar(x + 1.5*w, [s or 0 for s in multi_serial], w, label='Serial Multi', color='#64B5F6', alpha=0.7)
    ax.set_xlabel('Batch Size')
    ax.set_ylabel('Time (ms)')
    ax.set_title('Total Inference Time (DBO vs Serial)')
    ax.set_xticks(x)
    ax.set_xticklabels(batches)
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)

    # 2. Per-layer compute time
    ax = axes[0, 1]
    ax.bar(x - w, safe_get(local_dbo, 'attn_per_layer'), w, label='Attn (Local)', color='#4CAF50')
    ax.bar(x, safe_get(local_dbo, 'ffn_per_layer'), w, label='FFN (Local)', color='#81C784')
    ax.bar(x + w, safe_get(multi_dbo, 'ffn_per_layer'), w, label='FFN (Multi)', color='#64B5F6')
    ax.set_xlabel('Batch Size')
    ax.set_ylabel('Time per Layer (ms)')
    ax.set_title('Per-Layer Compute Time')
    ax.set_xticks(x)
    ax.set_xticklabels(batches)
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)

    # 3. Communication time per layer
    ax = axes[1, 0]
    ax.bar(x - 1.5*w, safe_get(local_dbo, 'a2f_per_layer'), w, label='A→F Local', color='#FF9800')
    ax.bar(x - 0.5*w, safe_get(multi_dbo, 'a2f_per_layer'), w, label='A→F Multi', color='#FFB74D')
    ax.bar(x + 0.5*w, safe_get(local_dbo, 'f2a_per_layer'), w, label='F→A Local', color='#F44336')
    ax.bar(x + 1.5*w, safe_get(multi_dbo, 'f2a_per_layer'), w, label='F→A Multi', color='#EF9A9A')
    ax.set_xlabel('Batch Size')
    ax.set_ylabel('Time per Layer (ms)')
    ax.set_title('Communication Time per Layer')
    ax.set_xticks(x)
    ax.set_xticklabels(batches)
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)

    # 4. DBO Speedup vs Serial
    ax = axes[1, 1]
    local_speedup, multi_speedup = [], []
    valid_batches_l, valid_batches_m = [], []
    for i, b in enumerate(batches):
        if local_dbo[i] and local_serial[i]:
            local_speedup.append(local_serial[i] / local_dbo[i]['total_ms'])
            valid_batches_l.append(b)
        if multi_dbo[i] and multi_serial[i]:
            multi_speedup.append(multi_serial[i] / multi_dbo[i]['total_ms'])
            valid_batches_m.append(b)
    ax.plot(valid_batches_l, local_speedup, 'o-', label='Local', color='#4CAF50', linewidth=2, markersize=8)
    ax.plot(valid_batches_m, multi_speedup, 's-', label='Multinode', color='#2196F3', linewidth=2, markersize=8)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Break-even')
    ax.set_xlabel('Batch Size')
    ax.set_ylabel('Speedup (Serial / DBO)')
    ax.set_title('DBO Speedup vs Serial')
    ax.legend()
    ax.grid(alpha=0.3)
    for i, (b, s) in enumerate(zip(valid_batches_l, local_speedup)):
        ax.annotate(f'{s:.2f}x', (b, s), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9)
    for i, (b, s) in enumerate(zip(valid_batches_m, multi_speedup)):
        ax.annotate(f'{s:.2f}x', (b, s), textcoords="offset points", xytext=(0, -15), ha='center', fontsize=9, color='#2196F3')

    plt.tight_layout()
    out = 'results/prefill_dbo/batch_scaling_comparison.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ {out}")


def plot_seq_scaling():
    """Sequence length scaling: Local vs Multinode."""
    seqs = [32, 64, 128, 256, 512]
    local_data, multi_data = [], []

    for s in seqs:
        local_data.append(load_experiment(f"local_b8_s{s}_t5"))
        multi_data.append(load_experiment(f"multinode_b8_s{s}_t5"))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Sequence Length Scaling — Qwen3-30B-A3B (batch=8, 48 layers)',
                 fontsize=14, fontweight='bold')

    x = np.arange(len(seqs))
    w = 0.35

    # 1. Total time
    ax = axes[0, 0]
    ax.bar(x - w/2, safe_get(local_data, 'total_ms'), w, label='Local', color='#4CAF50')
    ax.bar(x + w/2, safe_get(multi_data, 'total_ms'), w, label='Multinode', color='#2196F3')
    ax.set_xlabel('Sequence Length')
    ax.set_ylabel('Time (ms)')
    ax.set_title('Total Inference Time')
    ax.set_xticks(x)
    ax.set_xticklabels(seqs)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # 2. Per-layer compute
    ax = axes[0, 1]
    valid_l = [(s, d) for s, d in zip(seqs, local_data) if d]
    valid_m = [(s, d) for s, d in zip(seqs, multi_data) if d]
    if valid_l:
        ax.plot([s for s, _ in valid_l], [d['attn_per_layer'] for _, d in valid_l],
                'o-', label='Attn (Local)', color='#4CAF50', linewidth=2)
        ax.plot([s for s, _ in valid_l], [d['ffn_per_layer'] for _, d in valid_l],
                's-', label='FFN (Local)', color='#81C784', linewidth=2)
    if valid_m:
        ax.plot([s for s, _ in valid_m], [d['ffn_per_layer'] for _, d in valid_m],
                '^-', label='FFN (Multi)', color='#64B5F6', linewidth=2)
    ax.set_xlabel('Sequence Length')
    ax.set_ylabel('Time per Layer (ms)')
    ax.set_title('Per-Layer Compute Time')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # 3. Communication
    ax = axes[1, 0]
    if valid_l:
        ax.plot([s for s, _ in valid_l], [d['a2f_per_layer'] for _, d in valid_l],
                'o-', label='A→F Local', color='#FF9800', linewidth=2)
        ax.plot([s for s, _ in valid_l], [d['f2a_per_layer'] for _, d in valid_l],
                '^-', label='F→A Local', color='#F44336', linewidth=2)
    if valid_m:
        ax.plot([s for s, _ in valid_m], [d['a2f_per_layer'] for _, d in valid_m],
                's-', label='A→F Multi', color='#FFB74D', linewidth=2)
        ax.plot([s for s, _ in valid_m], [d['f2a_per_layer'] for _, d in valid_m],
                'v-', label='F→A Multi', color='#EF9A9A', linewidth=2)
    ax.set_xlabel('Sequence Length')
    ax.set_ylabel('Time per Layer (ms)')
    ax.set_title('Communication Time per Layer')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # 4. Multinode overhead
    ax = axes[1, 1]
    overhead = []
    valid_seqs = []
    for i, s in enumerate(seqs):
        if local_data[i] and multi_data[i]:
            oh = (multi_data[i]['total_ms'] / local_data[i]['total_ms'] - 1) * 100
            overhead.append(oh)
            valid_seqs.append(s)
    ax.bar(range(len(valid_seqs)), overhead, color='#9C27B0', alpha=0.7)
    ax.set_xlabel('Sequence Length')
    ax.set_ylabel('Overhead (%)')
    ax.set_title('Multinode Overhead vs Local')
    ax.set_xticks(range(len(valid_seqs)))
    ax.set_xticklabels(valid_seqs)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.grid(axis='y', alpha=0.3)
    for i, oh in enumerate(overhead):
        ax.annotate(f'{oh:.1f}%', (i, oh), textcoords="offset points",
                    xytext=(0, 5), ha='center', fontsize=10)

    plt.tight_layout()
    out = 'results/prefill_dbo/seq_scaling_comparison.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ {out}")


def plot_summary():
    """Summary: DBO speedup table + per-layer breakdown for key configs."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('AFD + DBO Performance Summary — Qwen3-30B-A3B (48 layers)',
                 fontsize=14, fontweight='bold')

    # Left: Per-layer time breakdown for key configurations
    configs = [
        ('b4\ns128', 'local_b4_s128_t5'),
        ('b8\ns128', 'local_b8_s128_t5'),
        ('b16\ns128', 'local_b16_s128_t5'),
        ('b32\ns128', 'local_b32_s128_t5'),
        ('b8\ns256', 'local_b8_s256_t5'),
        ('b8\ns512', 'local_b8_s512_t5'),
    ]

    labels, attn_vals, ffn_vals, a2f_vals, f2a_vals = [], [], [], [], []
    for label, suffix in configs:
        d = load_experiment(suffix)
        if d:
            labels.append(label)
            attn_vals.append(d['attn_per_layer'])
            ffn_vals.append(d['ffn_per_layer'])
            a2f_vals.append(d['a2f_per_layer'])
            f2a_vals.append(d['f2a_per_layer'])

    ax = axes[0]
    x = np.arange(len(labels))
    w = 0.6
    p1 = ax.bar(x, attn_vals, w, label='Attention', color='#4CAF50')
    p2 = ax.bar(x, ffn_vals, w, bottom=attn_vals, label='FFN', color='#2196F3')
    bottom2 = [a + f for a, f in zip(attn_vals, ffn_vals)]
    p3 = ax.bar(x, a2f_vals, w, bottom=bottom2, label='A→F Transfer', color='#FF9800')
    bottom3 = [b + c for b, c in zip(bottom2, a2f_vals)]
    p4 = ax.bar(x, f2a_vals, w, bottom=bottom3, label='F→A Transfer', color='#F44336')
    ax.set_xlabel('Configuration')
    ax.set_ylabel('Time per Layer (ms)')
    ax.set_title('Per-Layer Time Breakdown (Local, L1+ avg)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(axis='y', alpha=0.3)

    # Right: DBO speedup across batch sizes
    ax = axes[1]
    batches = [4, 8, 16, 32]
    local_sp, multi_sp = [], []
    for b in batches:
        dbo_l = load_experiment(f"local_b{b}_s128_t5")
        ser_l = load_serial(f"serial_local_b{b}_s128_t5")
        dbo_m = load_experiment(f"multinode_b{b}_s128_t5")
        ser_m = load_serial(f"serial_multinode_b{b}_s128_t5")
        local_sp.append(ser_l / dbo_l['total_ms'] if dbo_l and ser_l else 0)
        multi_sp.append(ser_m / dbo_m['total_ms'] if dbo_m and ser_m else 0)

    x2 = np.arange(len(batches))
    w2 = 0.3
    ax.bar(x2 - w2/2, local_sp, w2, label='Local (NVLink)', color='#4CAF50')
    ax.bar(x2 + w2/2, multi_sp, w2, label='Multinode (TCP)', color='#2196F3')
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Batch Size (seq=128)')
    ax.set_ylabel('Speedup (Serial / DBO)')
    ax.set_title('DBO Speedup vs Serial Baseline')
    ax.set_xticks(x2)
    ax.set_xticklabels(batches)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    for i, (sl, sm) in enumerate(zip(local_sp, multi_sp)):
        if sl > 0:
            ax.annotate(f'{sl:.2f}x', (i - w2/2, sl), textcoords="offset points",
                        xytext=(0, 5), ha='center', fontsize=10, fontweight='bold')
        if sm > 0:
            ax.annotate(f'{sm:.2f}x', (i + w2/2, sm), textcoords="offset points",
                        xytext=(0, 5), ha='center', fontsize=10, fontweight='bold', color='#1565C0')

    plt.tight_layout()
    out = 'results/prefill_dbo/performance_summary.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ {out}")


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)) + '/..')
    print("Generating comparison plots...")
    plot_batch_scaling()
    plot_seq_scaling()
    plot_summary()
    print("Done!")
