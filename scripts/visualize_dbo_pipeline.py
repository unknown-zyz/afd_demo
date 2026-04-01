#!/usr/bin/env python3
"""
改进的 DBO Pipeline 可视化工具

创建 4 泳道的 Gantt 图，清晰展示 Attention、通信、FFN 模块的重叠关系。
使用 send_transfer 事件测量真实传输时间（通过后台轮询检测完成）。

特性:
  - 自动跳过 Layer 0 的初始化开销（~91ms vs 其他层 ~1.7ms）
  - 使用 --start-layer 0 可以显示 Layer 0
  - 支持自定义层范围和输出路径

用法:
    # 默认：显示 Layer 1-2（跳过 Layer 0）
    python scripts/visualize_dbo_pipeline.py
    
    # 显示 Layer 1-5
    python scripts/visualize_dbo_pipeline.py --start-layer 1 --num-layers 5
    
    # 包含 Layer 0
    python scripts/visualize_dbo_pipeline.py --start-layer 0 --num-layers 3
    
    # 自定义输入输出
    python scripts/visualize_dbo_pipeline.py \
        --attn-timing results/prefill_dbo/timing_attention_local_b8_s128_t5.json \
        --ffn-timing results/prefill_dbo/timing_ffn_local_b8_s128_t5.json \
        --output results/prefill_dbo/dbo_pipeline_local_b8_s128_t5.png
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


def load_timing_data(attn_path: str, ffn_path: str, start_layer: int = 1, num_layers: int = 2):
    """
    加载并组织 timing 数据，提取指定范围的层。
    
    Args:
        attn_path: Attention timing JSON 文件路径
        ffn_path: FFN timing JSON 文件路径
        start_layer: 起始层 (默认: 1, 跳过 Layer 0 的初始化开销)
        num_layers: 要显示的层数 (默认: 2)
    
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
    
    end_layer = start_layer + num_layers
    
    # 组织数据到各个泳道
    lanes_data = {
        'A': [],      # Attention compute
        'A2F': [],    # Attention send
        'F': [],      # FFN compute
        'F2A': [],    # FFN send
    }
    
    # 找到起始时间偏移 (使图从 0 开始)
    min_start = float('inf')
    for event in attn_data['events']:
        if start_layer <= event['layer'] < end_layer:
            min_start = min(min_start, event['start'] * 1000)
    for event in ffn_data['events']:
        if start_layer <= event['layer'] < end_layer:
            min_start = min(min_start, event['start'] * 1000)
    
    if min_start == float('inf'):
        min_start = 0
    
    # 提取 Attention 节点事件
    for event in attn_data['events']:
        layer = event['layer']
        if layer < start_layer or layer >= end_layer:
            continue
        
        start_ms = event['start'] * 1000 - min_start  # 相对时间
        duration_ms = event['duration_ms']
        mb = event['mb']
        display_layer = layer - start_layer  # 显示用的层号 (从 0 开始)
        
        if event['type'] == 'attn_compute':
            lanes_data['A'].append((start_ms, duration_ms, display_layer, mb))
        elif event['type'] == 'send_transfer':
            lanes_data['A2F'].append((start_ms, duration_ms, display_layer, mb))
    
    # 提取 FFN 节点事件
    for event in ffn_data['events']:
        layer = event['layer']
        if layer < start_layer or layer >= end_layer:
            continue
        
        start_ms = event['start'] * 1000 - min_start
        duration_ms = event['duration_ms']
        mb = event['mb']
        display_layer = layer - start_layer
        
        if event['type'] == 'ffn_compute':
            lanes_data['F'].append((start_ms, duration_ms, display_layer, mb))
        elif event['type'] == 'send_transfer':
            lanes_data['F2A'].append((start_ms, duration_ms, display_layer, mb))
    
    return lanes_data, attn_data, ffn_data, start_layer


def plot_pipeline(lanes_data: dict, attn_data: dict, ffn_data: dict, 
                  output_path: str, num_layers: int = 2, start_layer: int = 1):
    """
    绘制 4 泳道 pipeline 图。
    
    Args:
        lanes_data: 按泳道组织的事件数据
        attn_data: Attention 节点完整数据
        ffn_data: FFN 节点完整数据
        output_path: 输出图片路径
        num_layers: 显示的层数
        start_layer: 原始数据中的起始层号
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
            
            # 添加文本标注 - 显示实际层号
            actual_layer = layer + start_layer
            label = f'L{actual_layer}'
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
    
    # 计算改进的性能指标
    num_mb = attn_data.get('num_micro_batches', 2)
    attn_total = attn_data.get('total_time_ms', 0)
    ffn_total = ffn_data.get('total_time_ms', 0)
    
    # 总推理时间 = 两个节点中较大的
    total_inference_time = max(attn_total, ffn_total)
    
    # 计算时间
    attn_compute = attn_data.get('total_compute_ms', 0)
    ffn_compute = ffn_data.get('total_compute_ms', 0)
    
    # 计算各 MB 的真实传输时间 (从 send_transfer 事件中提取)
    attn_events = attn_data.get('events', [])
    ffn_events = ffn_data.get('events', [])
    
    # A→F 传输时间 (来自 attention 节点)
    a2f_mb0_times = [e['duration_ms'] for e in attn_events 
                    if e['type'] == 'send_transfer' and e['mb'] == 0 and start_layer <= e['layer'] < start_layer + num_layers]
    a2f_mb1_times = [e['duration_ms'] for e in attn_events
                    if e['type'] == 'send_transfer' and e['mb'] == 1 and start_layer <= e['layer'] < start_layer + num_layers]
    
    # F→A 传输时间 (来自 FFN 节点)
    f2a_mb0_times = [e['duration_ms'] for e in ffn_events 
                    if e['type'] == 'send_transfer' and e['mb'] == 0 and start_layer <= e['layer'] < start_layer + num_layers]
    f2a_mb1_times = [e['duration_ms'] for e in ffn_events
                    if e['type'] == 'send_transfer' and e['mb'] == 1 and start_layer <= e['layer'] < start_layer + num_layers]
    
    a2f_mb0_avg = sum(a2f_mb0_times) / len(a2f_mb0_times) if a2f_mb0_times else 0
    a2f_mb1_avg = sum(a2f_mb1_times) / len(a2f_mb1_times) if a2f_mb1_times else 0
    f2a_mb0_avg = sum(f2a_mb0_times) / len(f2a_mb0_times) if f2a_mb0_times else 0
    f2a_mb1_avg = sum(f2a_mb1_times) / len(f2a_mb1_times) if f2a_mb1_times else 0
    
    # 总通信时间
    total_comm = sum(a2f_mb0_times + a2f_mb1_times + f2a_mb0_times + f2a_mb1_times)
    
    # 理论串行时间 = 计算 + 通信 (无重叠)
    serial_time = attn_compute + ffn_compute + total_comm
    
    # Overlap 率 = (理论串行 - 实际) / 理论串行
    if serial_time > 0:
        overlap_ratio = (serial_time - total_inference_time) / serial_time * 100
        speedup = serial_time / total_inference_time if total_inference_time > 0 else 1.0
    else:
        overlap_ratio = 0
        speedup = 1.0
    
    # 构建标题和统计信息
    end_layer = start_layer + num_layers - 1
    layer_note = " (Layer 0 skipped)" if start_layer > 0 else ""
    title = f'DBO Pipeline - Layers {start_layer}-{end_layer}{layer_note}, {num_mb} Micro-batches'
    
    stats_line1 = f"Total: {total_inference_time:.1f}ms | Attn: {attn_compute:.1f}ms | FFN: {ffn_compute:.1f}ms"
    stats_line2 = f"Transfer: A→F (MB0={a2f_mb0_avg:.2f}ms, MB1={a2f_mb1_avg:.2f}ms) | F→A (MB0={f2a_mb0_avg:.2f}ms, MB1={f2a_mb1_avg:.2f}ms)"
    stats_line3 = f"Overlap: {overlap_ratio:.1f}% | Speedup vs Serial: {speedup:.2f}x"
    
    ax.set_title(f'{title}\n{stats_line1}\n{stats_line2}\n{stats_line3}', fontsize=10, pad=10)
    
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
        '--start-layer',
        type=int,
        default=1,
        help='Starting layer to visualize (default: 1, skip Layer 0 warmup)'
    )
    parser.add_argument(
        '--num-layers',
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
    print(f"  Layers: {args.start_layer} to {args.start_layer + args.num_layers - 1}")
    
    lanes_data, attn_data, ffn_data, start_layer = load_timing_data(
        args.attn_timing,
        args.ffn_timing,
        args.start_layer,
        args.num_layers
    )
    
    # 打印数据统计
    print(f"\nData summary (layer {args.start_layer}-{args.start_layer + args.num_layers - 1}):")
    for lane_name, events in lanes_data.items():
        print(f"  {LANES[lane_name]['label']}: {len(events)} events")
    
    # 生成可视化
    print(f"\nGenerating visualization...")
    plot_pipeline(lanes_data, attn_data, ffn_data, args.output, 
                  args.num_layers, start_layer)
    
    print(f"\n✓ Done! View the result at: {args.output}")


if __name__ == '__main__':
    main()
