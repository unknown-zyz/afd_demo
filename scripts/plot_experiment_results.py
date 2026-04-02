#!/usr/bin/env python3
"""
DBO 实验结果可视化脚本
从 summary.csv 生成 Prefill/Decode 性能对比图
"""

import csv
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.size'] = 11
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.family'] = 'DejaVu Sans'

def load_summary(path):
    """Loaded实验汇总数据"""
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['time_ms'] != 'OOM' and row['time_ms'] != 'N/A':
                row['time_ms'] = float(row['time_ms'])
                row['batch_size'] = int(row['batch_size'])
                row['seq_len'] = int(row['seq_len'])
                row['tok_s'] = float(row['tok_s']) if row.get('tok_s') and row['tok_s'] not in ('', 'N/A') else None
                rows.append(row)
    return rows


def plot_prefill_batch_scaling(rows, output_dir):
    """Prefill batch 扩展对比图"""
    prefill = [r for r in rows if r['phase'] == 'prefill' and r['seq_len'] == 128]
    
    batches = sorted(set(r['batch_size'] for r in prefill))
    dbo_on = {r['batch_size']: r['time_ms'] for r in prefill if r['dbo'] == 'on'}
    dbo_off = {r['batch_size']: r['time_ms'] for r in prefill if r['dbo'] == 'off'}
    
    batches = [b for b in batches if b in dbo_on and b in dbo_off]
    on_times = [dbo_on[b] for b in batches]
    off_times = [dbo_off[b] for b in batches]
    speedups = [off_times[i] / on_times[i] for i in range(len(batches))]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    x = np.arange(len(batches))
    w = 0.35
    ax1.bar(x - w/2, off_times, w, label='Serial (DBO OFF)', color='#4A90D9', alpha=0.85)
    ax1.bar(x + w/2, on_times, w, label='DBO ON', color='#E6553A', alpha=0.85)
    ax1.set_xlabel('Batch Size')
    ax1.set_ylabel('Prefill Time (ms)')
    ax1.set_title('Prefill: Batch Scaling (seq=128)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(batches)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # 在柱上标注时间
    for i, (t_off, t_on) in enumerate(zip(off_times, on_times)):
        ax1.text(i - w/2, t_off + 30, f'{t_off:.0f}', ha='center', va='bottom', fontsize=8)
        ax1.text(i + w/2, t_on + 30, f'{t_on:.0f}', ha='center', va='bottom', fontsize=8)
    
    # Speedup 图
    colors = ['#2ECC71' if s >= 1.0 else '#E74C3C' for s in speedups]
    bars = ax2.bar(x, speedups, color=colors, alpha=0.85)
    ax2.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='Baseline (1.0x)')
    ax2.set_xlabel('Batch Size')
    ax2.set_ylabel('DBO Speedup (Serial/DBO)')
    ax2.set_title('DBO Speedup (>1 = DBO faster)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(batches)
    ax2.set_ylim(0.85, 1.15)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    for i, s in enumerate(speedups):
        ax2.text(i, s + 0.005, f'{s:.3f}x', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    path = os.path.join(output_dir, 'prefill_batch_scaling.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f'✓ {path}')


def plot_prefill_seq_scaling(rows, output_dir):
    """Prefill seq 扩展对比图"""
    prefill = [r for r in rows if r['phase'] == 'prefill' and r['batch_size'] == 8]
    
    seqs = sorted(set(r['seq_len'] for r in prefill))
    dbo_on = {r['seq_len']: r['time_ms'] for r in prefill if r['dbo'] == 'on'}
    dbo_off = {r['seq_len']: r['time_ms'] for r in prefill if r['dbo'] == 'off'}
    
    seqs = [s for s in seqs if s in dbo_on and s in dbo_off]
    on_times = [dbo_on[s] for s in seqs]
    off_times = [dbo_off[s] for s in seqs]
    speedups = [off_times[i] / on_times[i] for i in range(len(seqs))]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    x = np.arange(len(seqs))
    w = 0.35
    ax1.bar(x - w/2, off_times, w, label='Serial (DBO OFF)', color='#4A90D9', alpha=0.85)
    ax1.bar(x + w/2, on_times, w, label='DBO ON', color='#E6553A', alpha=0.85)
    ax1.set_xlabel('Sequence Length')
    ax1.set_ylabel('Prefill Time (ms)')
    ax1.set_title('Prefill: Seq Scaling (batch=8)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(seqs)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    for i, (t_off, t_on) in enumerate(zip(off_times, on_times)):
        ax1.text(i - w/2, t_off + 30, f'{t_off:.0f}', ha='center', va='bottom', fontsize=8)
        ax1.text(i + w/2, t_on + 30, f'{t_on:.0f}', ha='center', va='bottom', fontsize=8)
    
    colors = ['#2ECC71' if s >= 1.0 else '#E74C3C' for s in speedups]
    ax2.bar(x, speedups, color=colors, alpha=0.85)
    ax2.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='Baseline (1.0x)')
    ax2.set_xlabel('Sequence Length')
    ax2.set_ylabel('DBO Speedup')
    ax2.set_title('DBO Speedup (>1 = DBO faster)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(seqs)
    ax2.set_ylim(0.85, 1.15)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    for i, s in enumerate(speedups):
        ax2.text(i, s + 0.005, f'{s:.3f}x', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    path = os.path.join(output_dir, 'prefill_seq_scaling.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f'✓ {path}')


def plot_decode_scaling(rows, output_dir):
    """Decode batch 扩展对比图"""
    decode = [r for r in rows if r['phase'] == 'decode']
    
    batches = sorted(set(r['batch_size'] for r in decode))
    dbo_on = {r['batch_size']: r for r in decode if r['dbo'] == 'on'}
    dbo_off = {r['batch_size']: r for r in decode if r['dbo'] == 'off'}
    
    batches = [b for b in batches if b in dbo_on and b in dbo_off]
    on_times = [dbo_on[b]['time_ms'] for b in batches]
    off_times = [dbo_off[b]['time_ms'] for b in batches]
    on_toks = [dbo_on[b]['tok_s'] for b in batches]
    off_toks = [dbo_off[b]['tok_s'] for b in batches]
    speedups = [off_times[i] / on_times[i] for i in range(len(batches))]
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    x = np.arange(len(batches))
    w = 0.35
    
    # 总时间
    ax1.bar(x - w/2, off_times, w, label='Serial (DBO OFF)', color='#4A90D9', alpha=0.85)
    ax1.bar(x + w/2, on_times, w, label='DBO ON', color='#E6553A', alpha=0.85)
    ax1.set_xlabel('Batch Size')
    ax1.set_ylabel('Generation Time (ms)')
    ax1.set_title('Decode: Total Time (seq=128, tokens=20)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(batches)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # 吞吐量
    ax2.bar(x - w/2, off_toks, w, label='Serial (DBO OFF)', color='#4A90D9', alpha=0.85)
    ax2.bar(x + w/2, on_toks, w, label='DBO ON', color='#E6553A', alpha=0.85)
    ax2.set_xlabel('Batch Size')
    ax2.set_ylabel('Throughput (tok/s)')
    ax2.set_title('Decode Throughput Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels(batches)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    for i, (t_off, t_on) in enumerate(zip(off_toks, on_toks)):
        if t_off and t_on:
            ax2.text(i - w/2, t_off + 0.1, f'{t_off:.1f}', ha='center', va='bottom', fontsize=8)
            ax2.text(i + w/2, t_on + 0.1, f'{t_on:.1f}', ha='center', va='bottom', fontsize=8)
    
    # Speedup
    colors = ['#2ECC71' if s >= 1.0 else '#E74C3C' for s in speedups]
    ax3.bar(x, speedups, color=colors, alpha=0.85)
    ax3.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='Baseline (1.0x)')
    ax3.set_xlabel('Batch Size')
    ax3.set_ylabel('DBO Speedup')
    ax3.set_title('Decode DBO Speedup')
    ax3.set_xticks(x)
    ax3.set_xticklabels(batches)
    ax3.set_ylim(0.9, 1.1)
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    
    for i, s in enumerate(speedups):
        ax3.text(i, s + 0.003, f'{s:.3f}x', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    path = os.path.join(output_dir, 'decode_batch_scaling.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f'✓ {path}')


def plot_combined_summary(rows, output_dir):
    """综合汇总图"""
    # Prefill batch (seq=128)
    prefill_b = [r for r in rows if r['phase'] == 'prefill' and r['seq_len'] == 128]
    batches_p = sorted(set(r['batch_size'] for r in prefill_b))
    p_on = {r['batch_size']: r['time_ms'] for r in prefill_b if r['dbo'] == 'on'}
    p_off = {r['batch_size']: r['time_ms'] for r in prefill_b if r['dbo'] == 'off'}
    batches_p = [b for b in batches_p if b in p_on and b in p_off]
    p_speedup = [p_off[b] / p_on[b] for b in batches_p]
    
    # Prefill seq (batch=8)
    prefill_s = [r for r in rows if r['phase'] == 'prefill' and r['batch_size'] == 8]
    seqs = sorted(set(r['seq_len'] for r in prefill_s))
    s_on = {r['seq_len']: r['time_ms'] for r in prefill_s if r['dbo'] == 'on'}
    s_off = {r['seq_len']: r['time_ms'] for r in prefill_s if r['dbo'] == 'off'}
    seqs = [s for s in seqs if s in s_on and s in s_off]
    s_speedup = [s_off[s] / s_on[s] for s in seqs]
    
    # Decode
    decode = [r for r in rows if r['phase'] == 'decode']
    batches_d = sorted(set(r['batch_size'] for r in decode))
    d_on = {r['batch_size']: r['time_ms'] for r in decode if r['dbo'] == 'on'}
    d_off = {r['batch_size']: r['time_ms'] for r in decode if r['dbo'] == 'off'}
    batches_d = [b for b in batches_d if b in d_on and b in d_off]
    d_speedup = [d_off[b] / d_on[b] for b in batches_d]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('DBO Experiment Summary - Qwen2-1.5B, 4xV100-32GB (NVLink)', fontsize=14, fontweight='bold')
    
    # Prefill batch speedup
    colors = ['#2ECC71' if s >= 1.0 else '#E74C3C' for s in p_speedup]
    axes[0].bar(range(len(batches_p)), p_speedup, color=colors, alpha=0.85)
    axes[0].axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
    axes[0].set_xticks(range(len(batches_p)))
    axes[0].set_xticklabels(batches_p)
    axes[0].set_xlabel('Batch Size')
    axes[0].set_ylabel('Speedup')
    axes[0].set_title('Prefill DBO: Batch Scaling\n(seq=128)')
    axes[0].set_ylim(0.85, 1.15)
    axes[0].grid(axis='y', alpha=0.3)
    for i, s in enumerate(p_speedup):
        axes[0].text(i, s + 0.005, f'{s:.2f}x', ha='center', fontsize=8, fontweight='bold')
    
    # Prefill seq speedup
    colors = ['#2ECC71' if s >= 1.0 else '#E74C3C' for s in s_speedup]
    axes[1].bar(range(len(seqs)), s_speedup, color=colors, alpha=0.85)
    axes[1].axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
    axes[1].set_xticks(range(len(seqs)))
    axes[1].set_xticklabels(seqs)
    axes[1].set_xlabel('Sequence Length')
    axes[1].set_ylabel('Speedup')
    axes[1].set_title('Prefill DBO: Seq Scaling\n(batch=8)')
    axes[1].set_ylim(0.85, 1.15)
    axes[1].grid(axis='y', alpha=0.3)
    for i, s in enumerate(s_speedup):
        axes[1].text(i, s + 0.005, f'{s:.2f}x', ha='center', fontsize=8, fontweight='bold')
    
    # Decode speedup
    colors = ['#2ECC71' if s >= 1.0 else '#E74C3C' for s in d_speedup]
    axes[2].bar(range(len(batches_d)), d_speedup, color=colors, alpha=0.85)
    axes[2].axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
    axes[2].set_xticks(range(len(batches_d)))
    axes[2].set_xticklabels(batches_d)
    axes[2].set_xlabel('Batch Size')
    axes[2].set_ylabel('Speedup')
    axes[2].set_title('Decode DBO: Batch Scaling\n(seq=128, tokens=20)')
    axes[2].set_ylim(0.9, 1.1)
    axes[2].grid(axis='y', alpha=0.3)
    for i, s in enumerate(d_speedup):
        axes[2].text(i, s + 0.003, f'{s:.2f}x', ha='center', fontsize=8, fontweight='bold')
    
    plt.tight_layout()
    path = os.path.join(output_dir, 'dbo_summary.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f'✓ {path}')


if __name__ == '__main__':
    summary_path = 'results/experiments/summary.csv'
    output_dir = 'results/experiments'
    
    rows = load_summary(summary_path)
    print(f'Loaded {len(rows)} experiment records\n')
    
    plot_prefill_batch_scaling(rows, output_dir)
    plot_prefill_seq_scaling(rows, output_dir)
    plot_decode_scaling(rows, output_dir)
    plot_combined_summary(rows, output_dir)
    
    print(f'\nAll visualizations saved to {output_dir}/')
