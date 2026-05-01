#!/usr/bin/env python3
"""
改进的 DBO Pipeline 可视化工具

创建 4 泳道的 Gantt 图，清晰展示 Attention、send event、FFN 模块的重叠关系。
send_transfer 的含义由 timing JSON 的 comm_timing_mode 决定：
enqueue 表示 isend 返回/排队开销；completion 表示有效 Work 完成跨度。
completion span 包含真实数据搬运，但也可能包含 backend 排队、接收端 readiness、
通信流调度和完成通知，因此不是纯硬件链路传输时间。

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

from experiment_baselines import infer_mode_from_path, resolve_serial_baseline

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
    'A2F': {'index': 2, 'label': 'A→F Send', 'height': 0.8},
    'F': {'index': 1, 'label': 'F (FFN)', 'height': 0.8},
    'F2A': {'index': 0, 'label': 'F→A Send', 'height': 0.8},
}


def load_timing_data(attn_path: str, ffn_path: str, start_layer: int = 1, num_layers: int = 2):
    """
    加载并组织 timing 数据，提取指定范围的层。
    
    自动对齐 ATT 和 FFN 进程的时钟差异（不同进程 perf_counter 基准不同）。
    使用 A→F send 边界作为锚点：ATT send_transfer 结束 ≈ FFN recv_wait 结束。
    
    Args:
        attn_path: Attention timing JSON 文件路径
        ffn_path: FFN timing JSON 文件路径
        start_layer: 起始层 (默认: 1, 跳过 Layer 0 的初始化开销)
        num_layers: 要显示的层数 (默认: 2)
    
    Returns:
        dict: 按泳道组织的事件数据
    """
    with open(attn_path) as f:
        attn_data = json.load(f)
    
    with open(ffn_path) as f:
        ffn_data = json.load(f)
    
    end_layer = start_layer + num_layers
    
    # === Step 1: 计算 ATT-FFN 时钟偏移 ===
    # 对齐原理：
    #   观测量 = ffn_recv_end - attn_send_end (ffn 时钟 - att 时钟)
    #   enqueue 模式下 send_transfer 结束是 isend 返回，不是数据到达。
    #   completion 模式下 send_transfer 结束是 Work 完成或已有 wait 点观测到的完成上界。
    #   相比旧方法（依赖 recv_dur≈0 的假设）对 crosslayer 模式更鲁棒
    
    # Build lookup: (layer, mb) -> ATT send_transfer end time
    attn_send_ends = {}
    for event in attn_data['events']:
        if event['type'] == 'send_transfer':
            key = (event['layer'], event['mb'])
            attn_send_ends[key] = event['start'] * 1000 + event['duration_ms']
    
    # Build lookup: (layer, mb) -> FFN recv_wait end time (data actually arrived)
    ffn_recv_ends = {}
    for event in ffn_data['events']:
        if event['type'] == 'recv_wait':
            key = (event['layer'], event['mb'])
            ffn_recv_ends[key] = event['start'] * 1000 + event['duration_ms']
    
    # Collect offsets from ALL anchors in visualization range (no recv_dur filter needed)
    offsets = []
    for layer in range(start_layer, end_layer):
        for mb in range(2):  # MB 0 and 1
            key = (layer, mb)
            if key in attn_send_ends and key in ffn_recv_ends:
                offsets.append(ffn_recv_ends[key] - attn_send_ends[key])
    
    if offsets:
        # Use MIN (tightest upper bound on true clock offset)
        clock_offset = min(offsets)
        print(f"  Clock offset (FFN-ATT): {clock_offset:.2f}ms (min of {len(offsets)} anchors, range [{min(offsets):.2f}, {max(offsets):.2f}])")
    else:
        clock_offset = 0
        print(f"  Warning: No clock anchors found, using offset=0")
    
    # === Step 2: 提取事件到各泳道 ===
    lanes_data = {
        'A': [],      # Attention compute
        'A2F': [],    # Attention send (A→F)
        'F': [],      # FFN compute
        'F2A': [],    # FFN send (F→A)
    }
    
    # 找到起始时间偏移 (使图从 0 开始) — 只用 ATT 事件确定 (单一时钟)
    min_start = float('inf')
    for event in attn_data['events']:
        if start_layer <= event['layer'] < end_layer:
            min_start = min(min_start, event['start'] * 1000)
    # Also consider aligned FFN events
    for event in ffn_data['events']:
        if start_layer <= event['layer'] < end_layer:
            aligned_start = event['start'] * 1000 - clock_offset
            min_start = min(min_start, aligned_start)
    
    if min_start == float('inf'):
        min_start = 0
    
    # 提取 Attention 节点事件 (使用 ATT 时钟，无需调整)
    for event in attn_data['events']:
        layer = event['layer']
        if layer < start_layer or layer >= end_layer:
            continue
        
        start_ms = event['start'] * 1000 - min_start
        duration_ms = event['duration_ms']
        mb = event['mb']
        display_layer = layer - start_layer
        
        if event['type'] == 'attn_compute':
            lanes_data['A'].append((start_ms, duration_ms, display_layer, mb))
        elif event['type'] == 'send_transfer':
            lanes_data['A2F'].append((start_ms, duration_ms, display_layer, mb))
    
    # 提取 FFN 节点事件 (应用时钟偏移对齐到 ATT 时间线)
    for event in ffn_data['events']:
        layer = event['layer']
        if layer < start_layer or layer >= end_layer:
            continue
        
        start_ms = event['start'] * 1000 - clock_offset - min_start  # Align to ATT clock
        duration_ms = event['duration_ms']
        mb = event['mb']
        display_layer = layer - start_layer
        
        if event['type'] == 'ffn_compute':
            lanes_data['F'].append((start_ms, duration_ms, display_layer, mb))
        elif event['type'] == 'send_transfer':
            lanes_data['F2A'].append((start_ms, duration_ms, display_layer, mb))
    
    return lanes_data, attn_data, ffn_data, start_layer


def plot_pipeline(lanes_data: dict, attn_data: dict, ffn_data: dict, 
                  output_path: str, num_layers: int = 2, start_layer: int = 1,
                  serial_baseline_ms: float = None, serial_baseline_label: str = None,
                  dbo_total_ms: float = None, mode: str = None):
    """
    绘制 4 泳道 pipeline 图。
    
    Args:
        lanes_data: 按泳道组织的事件数据
        attn_data: Attention 节点完整数据
        ffn_data: FFN 节点完整数据
        output_path: 输出图片路径
        num_layers: 显示的层数
        start_layer: 原始数据中的起始层号
        serial_baseline_ms: 与 DBO 对比的串行基线 (mode-matched: prefill_ms or decode_tpot_ms)
        serial_baseline_label: 可选标签 (如 "prefill" / "decode_step")
        dbo_total_ms: DBO 端到端时间 (全模型, mode-matched); 默认取 attn JSON total_time_ms
        mode: "prefill" 或 "decode" (仅用于标题显示)
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

    # Mode-matched DBO latency from timing JSON. Decode speedup uses full
    # decode-loop TPOT; the Gantt bars visualize the recorded decode step.
    dbo_attn_total = attn_data.get('total_time_ms', 0)
    dbo_ffn_total = ffn_data.get('total_time_ms', 0)
    # For decode, total_time_ms is the recorded decode step timing.
    total_inference_time = max(dbo_attn_total, dbo_ffn_total)
    
    # 计算时间 (从事件中精确计算, 仅统计可视化范围内的层)
    attn_events = attn_data.get('events', [])
    ffn_events = ffn_data.get('events', [])
    
    attn_compute_times = [e['duration_ms'] for e in attn_events
                          if e['type'] == 'attn_compute' and start_layer <= e['layer'] < start_layer + num_layers]
    ffn_compute_times = [e['duration_ms'] for e in ffn_events
                         if e['type'] == 'ffn_compute' and start_layer <= e['layer'] < start_layer + num_layers]
    attn_compute = sum(attn_compute_times)
    ffn_compute = sum(ffn_compute_times)
    
    comm_mode = attn_data.get('comm_timing_mode') or ffn_data.get('comm_timing_mode') or 'enqueue'

    # A→F send event time (来自 attention 节点)
    a2f_times = [e['duration_ms'] for e in attn_events 
                 if e['type'] == 'send_transfer' and start_layer <= e['layer'] < start_layer + num_layers]
    
    # F→A send event time (来自 FFN 节点)
    f2a_times = [e['duration_ms'] for e in ffn_events 
                 if e['type'] == 'send_transfer' and start_layer <= e['layer'] < start_layer + num_layers]
    
    a2f_avg = sum(a2f_times) / len(a2f_times) if a2f_times else 0
    f2a_avg = sum(f2a_times) / len(f2a_times) if f2a_times else 0
    
    # Total send event time. In enqueue mode this is enqueue overhead; in
    # completion mode it is effective Work completion span, not pure wire time.
    total_comm = sum(a2f_times + f2a_times)
    
    # ─── Speedup: mode-matched baseline vs DBO full-model time ───────────────
    # DBO full-model time (one full prefill pass OR exact decode TPOT)
    if dbo_total_ms is not None and dbo_total_ms > 0:
        dbo_full = dbo_total_ms
    else:
        if mode == "decode":
            dbo_full = max(attn_data.get('decode_tpot_ms') or 0, ffn_data.get('decode_tpot_ms') or 0)
        else:
            dbo_full = max(attn_data.get('total_time_ms', 0), ffn_data.get('total_time_ms', 0))

    if mode == "prefill":
        unit = "TTFT"
        dbo_label = f"DBO TTFT-path: {dbo_full:.1f}ms"
        speedup_name = "TTFT"
    elif mode == "decode":
        unit = "TPOT"
        dbo_label = f"DBO TPOT: {dbo_full:.1f}ms"
        speedup_name = "TPOT"
    else:
        unit = "run"
        dbo_label = f"DBO: {dbo_full:.1f}ms"
        speedup_name = "Speed"

    if serial_baseline_ms and serial_baseline_ms > 0:
        tag = serial_baseline_label or unit
        serial_label = f"Serial {tag}: {serial_baseline_ms:.1f}ms"
        speedup_str = f"{speedup_name} Speedup: {serial_baseline_ms / dbo_full:.2f}x" if dbo_full > 0 else "Speedup: N/A"
    else:
        # No mode-matched baseline available → refuse to show a misleading number.
        serial_label = "Serial: N/A"
        speedup_str = "Speedup: N/A"

    # Keep legacy vars for per-layer display block below.
    avg_attn = attn_compute / num_layers if num_layers > 0 else 0
    avg_ffn = ffn_compute / num_layers if num_layers > 0 else 0
    
    # 构建标题和统计信息 (3 行)
    end_layer = start_layer + num_layers - 1
    if start_layer == 1:
        layer_note = " (L0 skipped)"
    elif start_layer > 1:
        layer_note = f" (L0–L{start_layer - 1} skipped)"
    else:
        layer_note = ""
    mode_tag = f" [{mode}]" if mode else ""
    comm_label = "send completion" if comm_mode == "completion" else "send enqueue"
    line1 = f'DBO Pipeline{mode_tag} — L{start_layer}–{end_layer}{layer_note}, {num_mb} Micro-batches, {comm_label}'
    line2 = f"{dbo_label} | {serial_label} | {speedup_str}"
    ax.set_title(f'{line1}\n{line2}', fontsize=10, pad=10)
    
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
        default=0,
        help='Starting layer to visualize (default: 0, include Layer 0). '
            'Use --auto-skip-warmup to drop warmup-skewed layers automatically.'
    )
    parser.add_argument(
        '--auto-skip-warmup',
        action='store_true',
        default=True,
        help='Auto-detect and skip warmup layers whose mb0 duration is >5× the '
            'median of later layers.'
    )
    parser.add_argument(
        '--no-auto-skip-warmup',
        action='store_false',
        dest='auto_skip_warmup',
        help='Disable auto warmup-layer skipping.'
    )
    parser.add_argument(
        '--num-layers',
        type=int,
        default=2,
        help='Number of layers to visualize (default: 2)'
    )
    parser.add_argument(
        '--serial-time',
        type=float,
        default=None,
        help='Measured serial baseline in ms (bypasses cache JSON lookup)'
    )
    parser.add_argument(
        '--serial-timing',
        default=None,
        help='Path to serial cache JSON (must contain prefill_ms and/or decode_tpot_ms)'
    )
    parser.add_argument(
        '--mode',
        choices=['prefill', 'decode', 'auto'],
        default='auto',
        help='Comparison mode. auto: infer from --attn-timing path (prefill-dbo/decode-dbo).'
    )
    
    args = parser.parse_args()

    # ── Auto-detect mode from attn-timing path ──────────────────────────────
    if args.mode == 'auto':
        args.mode = infer_mode_from_path(args.attn_timing)
        if args.mode is None:
            print("  Warning: could not infer mode from path; pass --mode explicitly.")

    # ── Resolve serial baseline from cache JSON (mode-matched) ──────────────
    serial_baseline_ms = args.serial_time
    serial_baseline_label = None
    if serial_baseline_ms is None and args.serial_timing and Path(args.serial_timing).exists():
        try:
            with open(args.serial_timing) as f:
                cache = json.load(f)
            baseline = resolve_serial_baseline(cache, args.mode)
            serial_baseline_ms = baseline.value_ms
            serial_baseline_label = baseline.unit
            if baseline.warning:
                print(f"  Warning: {baseline.warning}")
            if serial_baseline_ms is None:
                print(f"  Warning: '{args.mode}' baseline missing from {args.serial_timing}; "
                      f"Speedup will be N/A. Keys present: {list(cache.keys())}")
            else:
                metric = serial_baseline_label or args.mode
                print(f"  Serial {metric} baseline: {serial_baseline_ms:.1f}ms  ({args.serial_timing})")
        except Exception as e:
            print(f"  Warning: Failed to read serial timing: {e}")

    # ── DBO comparison time (TTFT for prefill, TPOT for decode) from attn JSON ─
    dbo_total_ms = None
    try:
        with open(args.attn_timing) as f:
            _attn = json.load(f)
        dbo_total_ms = _attn.get('decode_tpot_ms') if args.mode == 'decode' else _attn.get('total_time_ms')
    except Exception:
        pass
    
    # 检查输入文件
    if not Path(args.attn_timing).exists():
        print(f"Error: Attention timing file not found: {args.attn_timing}")
        sys.exit(1)
    
    if not Path(args.ffn_timing).exists():
        print(f"Error: FFN timing file not found: {args.ffn_timing}")
        sys.exit(1)

    # Auto-detect warmup-skewed layers: if mb0 attn durations
    # for the first few layers are >5× the median of later layers, bump start_layer.
    auto_skipped = 0
    if args.auto_skip_warmup:
        try:
            with open(args.attn_timing) as f:
                _attn = json.load(f)
            mb0_attn = {}
            for ev in _attn.get('events', []):
                if ev.get('type') == 'attn_compute' and ev.get('mb') == 0:
                    mb0_attn.setdefault(ev['layer'], ev['duration_ms'])
            if mb0_attn:
                layers_sorted = sorted(mb0_attn)
                # Reference median: use layers from max(start_layer+3, 4) onwards
                tail = [mb0_attn[l] for l in layers_sorted if l >= max(args.start_layer + 3, 4)]
                if not tail:
                    tail = list(mb0_attn.values())[-max(len(mb0_attn)//2, 1):]
                tail_sorted = sorted(tail)
                median = tail_sorted[len(tail_sorted)//2]
                threshold = 5.0 * median
                # Walk forward from current start_layer while that layer is warmup
                new_start = args.start_layer
                while new_start in mb0_attn and mb0_attn[new_start] > threshold:
                    print(f"  Auto-skip warmup layer L{new_start}: "
                          f"attn_mb0={mb0_attn[new_start]:.1f}ms > 5×median({median:.2f}ms)")
                    new_start += 1
                auto_skipped = new_start - args.start_layer
                if auto_skipped > 0:
                    args.start_layer = new_start
                    args.num_layers = max(1, args.num_layers - auto_skipped)
        except Exception as e:
            print(f"  Warning: auto-skip-warmup detection failed: {e}")
    
    # 加载数据
    print(f"Loading timing data...")
    print(f"  Attention: {args.attn_timing}")
    print(f"  FFN: {args.ffn_timing}")
    print(f"  Layers: {args.start_layer} to {args.start_layer + args.num_layers - 1}"
          + (f" (auto-skipped {auto_skipped} warmup layer(s))" if auto_skipped else ""))
    
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
                  args.num_layers, start_layer,
                  serial_baseline_ms=serial_baseline_ms,
                  serial_baseline_label=serial_baseline_label,
                  dbo_total_ms=dbo_total_ms,
                  mode=args.mode)
    
    print(f"\n✓ Done! View the result at: {args.output}")


if __name__ == '__main__':
    main()
