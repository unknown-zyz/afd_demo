#!/usr/bin/env python3
"""
Create summary visualization of DBO performance across Prefill and Decode phases.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Set non-interactive backend
plt.switch_backend('Agg')

# Data from tests
prefill_data = {
    'attention': {
        'total': 1278.5,
        'compute': 701.1,
        'wait': 345.6,
        'comm': 231.8,
    },
    'ffn': {
        'total': 1550.1,
        'compute': 1103.1,
        'wait': 407.1,
        'comm': 39.9,
    }
}

decode_data = {
    'dbo_on': {'time': 3384.49, 'throughput': 8.9},
    'dbo_off': {'time': 3411.87, 'throughput': 8.8},
}

# Create figure with subplots
fig = plt.figure(figsize=(16, 10))

# 1. Prefill Phase - Time Breakdown
ax1 = plt.subplot(2, 3, 1)
nodes = ['Attention', 'FFN']
compute = [prefill_data['attention']['compute'], prefill_data['ffn']['compute']]
wait = [prefill_data['attention']['wait'], prefill_data['ffn']['wait']]
comm = [prefill_data['attention']['comm'], prefill_data['ffn']['comm']]

x = np.arange(len(nodes))
width = 0.5

p1 = ax1.bar(x, compute, width, label='Compute', color='#4CAF50')
p2 = ax1.bar(x, wait, width, bottom=compute, label='Wait', color='#FF9800')
p3 = ax1.bar(x, comm, width, bottom=np.array(compute)+np.array(wait), label='Comm', color='#9C27B0')

ax1.set_ylabel('Time (ms)', fontsize=12)
ax1.set_title('Prefill Phase - Time Breakdown', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(nodes)
ax1.legend(loc='upper right')
ax1.grid(axis='y', alpha=0.3)

# Add total time labels
for i, node in enumerate(nodes):
    total = prefill_data[node.lower()]['total']
    ax1.text(i, total + 50, f'{total:.0f}ms', ha='center', fontsize=10, fontweight='bold')

# 2. Prefill Phase - Compute Efficiency
ax2 = plt.subplot(2, 3, 2)
efficiency_attn = (prefill_data['attention']['compute'] / prefill_data['attention']['total']) * 100
efficiency_ffn = (prefill_data['ffn']['compute'] / prefill_data['ffn']['total']) * 100

bars = ax2.bar(nodes, [efficiency_attn, efficiency_ffn], width=0.5, color=['#2196F3', '#00BCD4'])
ax2.set_ylabel('Compute Efficiency (%)', fontsize=12)
ax2.set_title('Prefill Phase - Compute Utilization', fontsize=14, fontweight='bold')
ax2.set_ylim(0, 100)
ax2.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50% baseline')
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

# Add percentage labels
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 2,
            f'{height:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

# 3. Decode Phase - Throughput Comparison
ax3 = plt.subplot(2, 3, 3)
configs = ['DBO ON', 'DBO OFF']
throughputs = [decode_data['dbo_on']['throughput'], decode_data['dbo_off']['throughput']]
colors_decode = ['#4CAF50', '#FFC107']

bars = ax3.bar(configs, throughputs, width=0.5, color=colors_decode)
ax3.set_ylabel('Throughput (tok/s)', fontsize=12)
ax3.set_title('Decode Phase - Throughput', fontsize=14, fontweight='bold')
ax3.set_ylim(0, 12)
ax3.grid(axis='y', alpha=0.3)

# Add value labels
for bar in bars:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + 0.2,
            f'{height:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Add diff annotation
diff_pct = ((decode_data['dbo_on']['throughput'] - decode_data['dbo_off']['throughput']) / 
            decode_data['dbo_off']['throughput'] * 100)
ax3.text(0.5, 10.5, f'Diff = +{diff_pct:.1f}%', ha='center', fontsize=10, 
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 4. Overall Pipeline - Attention Node
ax4 = plt.subplot(2, 3, 4)
ax4.text(0.5, 0.9, 'Attention Node Pipeline', ha='center', fontsize=14, fontweight='bold', transform=ax4.transAxes)

# Draw simplified pipeline diagram
colors_pipeline = {'compute': '#4CAF50', 'wait': '#FF9800', 'send': '#9C27B0'}
timeline = [
    ('Compute', 0, 701.1, colors_pipeline['compute']),
    ('Wait', 701.1, 345.6, colors_pipeline['wait']),
    ('Send', 1046.7, 231.8, colors_pipeline['send']),
]

y_pos = 0.5
for label, start, duration, color in timeline:
    width_norm = duration / prefill_data['attention']['total']
    start_norm = start / prefill_data['attention']['total']
    rect = patches.Rectangle((start_norm, y_pos - 0.1), width_norm, 0.2, 
                            linewidth=1, edgecolor='black', facecolor=color, transform=ax4.transAxes)
    ax4.add_patch(rect)
    
    # Add label in the middle
    mid_x = start_norm + width_norm / 2
    ax4.text(mid_x, y_pos, f'{label}\n{duration:.0f}ms', ha='center', va='center', 
            fontsize=9, transform=ax4.transAxes)

ax4.set_xlim(0, 1)
ax4.set_ylim(0, 1)
ax4.axis('off')

# Add timeline axis
ax4.text(0, 0.3, '0ms', ha='center', fontsize=9, transform=ax4.transAxes)
ax4.text(1, 0.3, f'{prefill_data["attention"]["total"]:.0f}ms', ha='center', fontsize=9, transform=ax4.transAxes)

# 5. Overall Pipeline - FFN Node
ax5 = plt.subplot(2, 3, 5)
ax5.text(0.5, 0.9, 'FFN Node Pipeline', ha='center', fontsize=14, fontweight='bold', transform=ax5.transAxes)

timeline_ffn = [
    ('Compute', 0, 1103.1, colors_pipeline['compute']),
    ('Wait', 1103.1, 407.1, colors_pipeline['wait']),
    ('Send', 1510.2, 39.9, colors_pipeline['send']),
]

y_pos = 0.5
for label, start, duration, color in timeline_ffn:
    width_norm = duration / prefill_data['ffn']['total']
    start_norm = start / prefill_data['ffn']['total']
    rect = patches.Rectangle((start_norm, y_pos - 0.1), width_norm, 0.2, 
                            linewidth=1, edgecolor='black', facecolor=color, transform=ax5.transAxes)
    ax5.add_patch(rect)
    
    mid_x = start_norm + width_norm / 2
    ax5.text(mid_x, y_pos, f'{label}\n{duration:.0f}ms', ha='center', va='center', 
            fontsize=9, transform=ax5.transAxes)

ax5.set_xlim(0, 1)
ax5.set_ylim(0, 1)
ax5.axis('off')

ax5.text(0, 0.3, '0ms', ha='center', fontsize=9, transform=ax5.transAxes)
ax5.text(1, 0.3, f'{prefill_data["ffn"]["total"]:.0f}ms', ha='center', fontsize=9, transform=ax5.transAxes)

# 6. Key Insights
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')

insights = [
    "Key Insights:",
    "",
    "Prefill DBO Efficiency:",
    "  - Attention: 54.8% compute util",
    "  - FFN: 71.2% compute util",
    "  - FFN is bottleneck (1.57x slower)",
    "",
    "Decode DBO Impact:",
    "  - Single-node: < 1% improvement",
    "  - Low comm latency (<1ms NVLink)",
    "  - Batch=1 limits DBO benefits",
    "",
    "Limitations:",
    "  - Batch > 1 KV cache bug exists",
    "  - Single-node comm latency minimal",
    "",
    "Next Steps:",
    "  - Fix KV cache slicing bug",
    "  - Multi-node test (10-100ms latency)",
    "  - Large batch size test",
]

y_start = 0.95
for i, line in enumerate(insights):
    if 'Prefill' in line or 'Decode' in line or 'Limitations' in line or 'Next Steps' in line:
        weight = 'bold'
        if 'Next' in line:
            color = '#2196F3'
        elif 'Limitations' in line:
            color = '#FF9800'
        else:
            color = '#4CAF50'
    elif line == "Key Insights:":
        weight = 'bold'
        color = 'black'
    else:
        weight = 'normal'
        color = 'black'
    
    ax6.text(0.05, y_start - i*0.042, line, fontsize=10, 
            verticalalignment='top', fontweight=weight, color=color, transform=ax6.transAxes)

plt.tight_layout()
plt.savefig('results/dbo_summary.png', dpi=150, bbox_inches='tight')
print("Saved: results/dbo_summary.png")

plt.close()
