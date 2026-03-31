#!/usr/bin/env python3
"""
改进的 DBO Pipeline 可视化工具

创建 4 泳道的 Gantt 图，清晰展示 Attention、通信、FFN 模块的重叠关系。
只显示前 2 层，便于理解 DBO 的工作原理。

用法:
    python scripts/visualize_dbo_pipeline.py
    python scripts/visualize_dbo_pipeline.py --output custom_output.png
"""

import argparse
import json
import sys
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import Rectangle
except ImportError:
    print("Error: matplotlib required. Install with: pip install matplotlib")
    sys.exit(1)


# 颜色方案 - 为不同的 micro-batch 使用不同颜色
MB_COLORS = {
    0: '#4CAF50',  # Green - MB 0
    1: '#2196F3',  # Blue - MB 1
}

# 泳道定义
LANES = {
    'A': {'index': 3, 'label': 'A (Attention)', 'height': 0.8},
    'A2F': {'index': 2, 'label': 'A→F (Comm)', 'height': 0.8},
    'F': {'index': 1, 'label': 'F (FFN)', 'height': 0.8},
    'F2A': {'index': 0, 'label': 'F→A (Comm)', 'height': 0.8},
}


def load_timing_data(attn_path: str, ffn_path: str, max_layers: int = 2):
    """
    加载并组织 timing 数据，提取前 N 层。
    
    Returns:
        dict: 按泳道组织的事件数据
        {
            'A': [(start_ms, duration_ms, layer, mb), ...],
            'A2F': [...],
            'F': [...],
            'F2A': [...],
        }
    """
    with open(attn_path) as f:
        attn_data = json.load(f)
    
    with open(ffn_path) as f:
        ffn_data = json.load(f)
    
    # 组织数据到各个泳道
    lanes_data = {
        'A': [],      # Attention compute
        'A2F': [],    # Attention send
        'F': [],      # FFN compute
        'F2A': [],    # FFN send
    }
    
    # 提取 Attention 节点事件
    for event in attn_data['events']:
        if event['layer'] >= max_layers:
            continue
        
        start_ms = event['start'] * 1000  # 转换为 ms
        duration_ms = event['duration_ms']
        layer = event['layer']
        mb = event['mb']
        
        if event['type'] == 'attn_compute':
            lanes_data['A'].append((start_ms, duration_ms, layer, mb))
        elif event['type'] == 'send_wait':
            lanes_data['A2F'].append((start_ms, duration_ms, layer, mb))
    
    # 提取 FFN 节点事件
    for event in ffn_data['events']:
        if event['layer'] >= max_layers:
            continue
        
        start_ms = event['start'] * 1000
        duration_ms = event['duration_ms']
        layer = event['layer']
        mb = event['mb']
        
        if event['type'] == 'ffn_compute':
            lanes_data['F'].append((start_ms, duration_ms, layer, mb))
        elif event['type'] == 'send_wait':
            lanes_data['F2A'].append((start_ms, duration_ms, layer, mb))
    
    return lanes_data, attn_data, ffn_data


def plot_pipeline(lanes_data: dict, attn_data: dict, ffn_data: dict, 
                  output_path: str, max_layers: int = 2):
    """
    绘制 4 泳道 pipeline 图。
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # 设置 y 轴 - 4 个泳道
    lane_positions = [LANES[lane]['index'] for lane in ['F2A', 'F', 'A2F', 'A']]
    lane_labels = [LANES[lane]['label'] for lane in ['F2A', 'F', 'A2F', 'A']]
    
    ax.set_yticks(lane_positions)
    ax.set_yticklabels(lane_labels, fontsize=11)
    ax.set_ylim(-0.5, 4)
    
    # 绘制每个泳道的事件
    for lane_name, events in lanes_data.items():
        lane_info = LANES[lane_name]
        y_pos = lane_info['index']
        height = lane_info['height']
        
        for start_ms, duration_ms, layer, mb in events:
            # 选择颜色
            color = MB_COLORS.get(mb, '#999999')
            
            # 绘制矩形
            rect = Rectangle(
                (start_ms, y_pos - height/2),
                duration_ms,
                height,
                facecolor=color,
                edgecolor='black',
                linewidth=0.5,
                alpha=0.8
            )
            ax.add_patch(rect)
            
            # 添加文本标注
            label = f'L{layer}'
            duration_label = f'{duration_ms:.1f}ms'
            
            # 标注位置
            text_x = start_ms + duration_ms / 2
            text_y = y_pos
            
            # Layer 标签
            ax.text(text_x, text_y + 0.15, label,
                   ha='center', va='center',
                   fontsize=8, fontweight='bold',
                   color='white' if mb == 1 else 'black')
            
            # Duration 标签
            ax.text(text_x, text_y - 0.15, duration_label,
                   ha='center', va='center',
                   fontsize=7,
                   color='white' if mb == 1 else 'black')
    
    # 设置 x 轴
    ax.set_xlabel('Time (ms)', fontsize=11)
    ax.set_xlim(0, None)
    ax.autoscale(axis='x')
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    # 标题
    num_mb = attn_data.get('num_micro_batches', 2)
    
    # 添加性能统计文本
    attn_total = attn_data.get('total_time_ms', 0)
    ffn_total = ffn_data.get('total_time_ms', 0)
    attn_ratio = attn_data.get('compute_ratio', 0) * 100
    ffn_ratio = ffn_data.get('compute_ratio', 0) * 100
    
    summary = (f"E2E: Attn={attn_total:.1f}ms, FFN={ffn_total:.1f}ms | "
               f"Compute Ratio: Attn={attn_ratio:.1f}%, FFN={ffn_ratio:.1f}%")
    
    title = f'DBO Pipeline (First {max_layers} Layers, {num_mb} Micro-batches)\n{summary}'
    ax.set_title(title, fontsize=12, pad=10)
    
    # 图例
    legend_elements = [
        mpatches.Patch(facecolor=MB_COLORS[0], edgecolor='black', 
                      label=f'Micro-batch 0', alpha=0.8),
        mpatches.Patch(facecolor=MB_COLORS[1], edgecolor='black', 
                      label=f'Micro-batch 1', alpha=0.8),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Pipeline visualization saved to: {output_path}")
    
    return fig


def main():
    parser = argparse.ArgumentParser(
        description='Visualize DBO pipeline with 4-lane Gantt chart'
    )
    parser.add_argument(
        '--attn-timing',
        default='results/prefill_dbo/timing_attention.json',
        help='Path to attention timing JSON'
    )
    parser.add_argument(
        '--ffn-timing',
        default='results/prefill_dbo/timing_ffn.json',
        help='Path to FFN timing JSON'
    )
    parser.add_argument(
        '--output',
        default='results/prefill_dbo/dbo_pipeline_4lanes.png',
        help='Output PNG file path'
    )
    parser.add_argument(
        '--max-layers',
        type=int,
        default=2,
        help='Number of layers to visualize (default: 2)'
    )
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not Path(args.attn_timing).exists():
        print(f"Error: Attention timing file not found: {args.attn_timing}")
        sys.exit(1)
    
    if not Path(args.ffn_timing).exists():
        print(f"Error: FFN timing file not found: {args.ffn_timing}")
        sys.exit(1)
    
    # 加载数据
    print(f"Loading timing data...")
    print(f"  Attention: {args.attn_timing}")
    print(f"  FFN: {args.ffn_timing}")
    
    lanes_data, attn_data, ffn_data = load_timing_data(
        args.attn_timing,
        args.ffn_timing,
        args.max_layers
    )
    
    # 打印数据统计
    print(f"\nData summary (first {args.max_layers} layers):")
    for lane_name, events in lanes_data.items():
        print(f"  {LANES[lane_name]['label']}: {len(events)} events")
    
    # 生成可视化
    print(f"\nGenerating visualization...")
    plot_pipeline(lanes_data, attn_data, ffn_data, args.output, args.max_layers)
    
    print(f"\n✓ Done! View the result at: {args.output}")


if __name__ == '__main__':
    main()
