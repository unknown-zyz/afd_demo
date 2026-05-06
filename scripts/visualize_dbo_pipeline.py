#!/usr/bin/env python3
"""
改进的 DBO Pipeline 可视化工具

创建 Gantt 图，清晰展示 Attention、send event、FFN 子阶段的重叠关系。
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
import re
import sys
from pathlib import Path

from experiment_baselines import infer_mode_from_path, resolve_serial_baseline


def autodetect_serial_baseline(attn_timing_path: str) -> str | None:
    """Infer serial baseline JSON from attn_timing path.

    Parses ``b{B}_s{S}_t{T}`` from filename and looks up a matching
    ``results_npu/serial/timing_attention_serial_b{B}_s{S}_t{T}.json``.
    Returns the path string if found, else None.
    """
    name = Path(attn_timing_path).name
    m = re.search(r'_b(\d+)_s(\d+)_t(\d+)', name)
    if not m:
        return None
    b, s, t = m.groups()
    candidates = [
        Path('results_npu') / 'serial' / f'timing_attention_serial_b{b}_s{s}_t{t}.json',
        Path('results') / 'serial' / f'timing_attention_serial_b{b}_s{s}_t{t}.json',
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return None

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
LEGACY_LANES = {
    'A': {'index': 3, 'label': 'A (Attention)', 'height': 0.8},
    'A2F': {'index': 2, 'label': 'A→F Send', 'height': 0.8},
    'F': {'index': 1, 'label': 'F (FFN)', 'height': 0.8},
    'F2A': {'index': 0, 'label': 'F→A Send', 'height': 0.8},
}

STAGED_LANES = {
    'A': {'index': 6, 'label': 'A (Attention)', 'height': 0.65},
    'A2F': {'index': 5, 'label': 'A→F Send', 'height': 0.65},
    'F/router': {'index': 4, 'label': 'F/router', 'height': 0.65},
    'F/dispatch': {'index': 3, 'label': 'F/dispatch', 'height': 0.65},
    'F/local_experts': {'index': 2, 'label': 'F/local_experts', 'height': 0.65},
    'F/combine': {'index': 1, 'label': 'F/combine', 'height': 0.65},
    'F2A': {'index': 0, 'label': 'F→A Send', 'height': 0.65},
}

FOURLANE_LANES = {
    'A':   {'index': 3, 'label': 'Attention',           'height': 0.75},
    'A2F': {'index': 2, 'label': 'A2F (send+recv+router+dispatch)', 'height': 0.75},
    'F':   {'index': 1, 'label': 'FFN (local experts)', 'height': 0.75},
    'F2A': {'index': 0, 'label': 'F2A (combine+send+recv)', 'height': 0.75},
}

LANE_SETS = {
    'legacy': LEGACY_LANES,
    'staged': STAGED_LANES,
    'fourlane': FOURLANE_LANES,
}


def _build_fourlane(
    attn_data, ffn_data, start_layer, end_layer, min_start,
    clock_offset, ffn_recv_ends_aligned, attn_recv_ends_aligned,
):
    """Build 4-lane bars (Attention / A2F / FFN / F2A).

    Semantics (per user spec — bar = transmission start → receiver completion):

    - **A** (`attn_compute`): ATT GPU compute.
    - **A2F**: start = ATT.send_transfer.start (sender enqueue);
      end = FFN.ep_local_experts.start (receiver finished recv + router +
      dispatch + dispatch_wait, ready to start GEMM).
      Fallback: FFN.recv_wait.end → ATT.send_transfer enqueue duration.
    - **FFN**: start = FFN.ep_local_experts.start (or moe_experts /
      moe_shared_or_dense / ffn_compute); end = same event's end.
    - **F2A**: start = FFN.ep_reduce.start (combine begin); end =
      ATT.recv_wait.end (receiver actually consumed the data — includes
      ATT-side serial recv queueing for that MB).
      Fallback: FFN.send_transfer.start as start, recv_wait.end as end.
    """
    lanes = {n: [] for n in FOURLANE_LANES}

    # ATT-side per-(layer, mb) lookups in aligned ms
    attn_compute = {}
    attn_send_starts_aligned = {}
    for ev in attn_data['events']:
        layer = ev['layer']
        if layer < start_layer or layer >= end_layer:
            continue
        mb = ev['mb']
        s = ev['start'] * 1000 - min_start
        d = ev['duration_ms']
        if ev['type'] == 'attn_compute':
            attn_compute[(layer, mb)] = (s, d)
        elif ev['type'] == 'send_transfer':
            attn_send_starts_aligned[(layer, mb)] = s

    # FFN-side per-(layer, mb) lookups (already in aligned ATT ms via offset)
    def _aligned(t_ms): return t_ms - clock_offset - min_start
    ffn_phase = {}  # (layer, mb) -> {phase: (start_aligned, dur)}
    for ev in ffn_data['events']:
        layer = ev['layer']
        if layer < start_layer or layer >= end_layer:
            continue
        mb = ev['mb']
        s = ev['start'] * 1000
        d = ev['duration_ms']
        ffn_phase.setdefault((layer, mb), {})[ev['type']] = (_aligned(s), d)

    fallback_used_a2f = 0
    fallback_used_f2a = 0
    fallback_used_ffn = 0

    for (layer, mb), (a_start, a_dur) in attn_compute.items():
        dl = layer - start_layer
        lanes['A'].append((a_start, a_dur, dl, mb))

        send_s = attn_send_starts_aligned.get((layer, mb))
        phases = ffn_phase.get((layer, mb), {})

        # ── FFN body bar ────────────────────────────────────────────────
        ffn_seg = (
            phases.get('ep_local_experts')
            or phases.get('moe_experts')
            or phases.get('moe_shared_or_dense')
            or phases.get('ffn_compute')
        )
        if ffn_seg is not None:
            f_start, f_dur = ffn_seg
            if not phases.get('ep_local_experts'):
                fallback_used_ffn += 1
            lanes['F'].append((f_start, f_dur, dl, mb))

        # ── A2F bar: sender enqueue → FFN ready for GEMM ────────────────
        if send_s is not None:
            # Endpoint preference: ep_local_experts.start (FFN truly ready
            # for compute) → recv_wait.end → just the send-enqueue duration.
            le_seg = phases.get('ep_local_experts')
            if le_seg is not None:
                a2f_end = le_seg[0]
            else:
                rw_end = ffn_recv_ends_aligned.get((layer, mb))
                if rw_end is not None and rw_end > send_s:
                    a2f_end = rw_end
                    fallback_used_a2f += 1
                else:
                    # Pure ATT enqueue duration (no FFN observation).
                    st_seg = next(
                        (ev for ev in attn_data['events']
                         if ev['layer'] == layer and ev['mb'] == mb
                         and ev['type'] == 'send_transfer'),
                        None,
                    )
                    if st_seg is None:
                        continue
                    a2f_end = send_s + st_seg['duration_ms']
                    fallback_used_a2f += 1
            a2f_dur = max(0.0, a2f_end - send_s)
            lanes['A2F'].append((send_s, a2f_dur, dl, mb))

        # ── F2A bar: combine begin → ATT recv_wait end ──────────────────
        f2a_start = None
        red_seg = phases.get('ep_reduce')
        if red_seg is not None:
            f2a_start = red_seg[0]
        else:
            ffn_send_seg = phases.get('send_transfer')
            if ffn_send_seg is not None:
                f2a_start = ffn_send_seg[0]
                fallback_used_f2a += 1
        if f2a_start is None:
            continue
        f2a_end = attn_recv_ends_aligned.get((layer, mb))
        if f2a_end is None or f2a_end <= f2a_start:
            # Fallback: end at FFN send_transfer.end + small propagation
            ffn_send_seg = phases.get('send_transfer')
            if ffn_send_seg is not None:
                f2a_end = ffn_send_seg[0] + ffn_send_seg[1]
                fallback_used_f2a += 1
            else:
                continue
        f2a_dur = max(0.0, f2a_end - f2a_start)
        lanes['F2A'].append((f2a_start, f2a_dur, dl, mb))

    if fallback_used_a2f:
        print(f"  fourlane: {fallback_used_a2f} A2F bar(s) used fallback endpoint "
              f"(no ep_local_experts in JSON).")
    if fallback_used_f2a:
        print(f"  fourlane: {fallback_used_f2a} F2A bar(s) used fallback endpoint "
              f"(no ep_reduce or no ATT recv_wait).")
    if fallback_used_ffn:
        print(f"  fourlane: {fallback_used_ffn} FFN bar(s) used legacy ffn_compute/moe_experts.")

    return lanes, attn_data, ffn_data, start_layer


def load_timing_data(
    attn_path: str,
    ffn_path: str,
    start_layer: int = 1,
    num_layers: int = 2,
    ffn_view: str = 'staged',
):
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
    
    # Build lookup: (layer, mb) -> ATT send_transfer start (sender enqueue) for A2F
    attn_send_starts = {}
    for event in attn_data['events']:
        if event['type'] == 'send_transfer':
            key = (event['layer'], event['mb'])
            attn_send_starts[key] = event['start'] * 1000

    # Build lookup: (layer, mb) -> FFN recv_wait end time (data actually arrived, FFN clock)
    ffn_recv_ends_raw = {}
    for event in ffn_data['events']:
        if event['type'] == 'recv_wait':
            key = (event['layer'], event['mb'])
            ffn_recv_ends_raw[key] = event['start'] * 1000 + event['duration_ms']

    # Build lookup: (layer, mb) -> FFN send_transfer start (sender enqueue) for F2A
    ffn_send_starts = {}
    for event in ffn_data['events']:
        if event['type'] == 'send_transfer':
            key = (event['layer'], event['mb'])
            ffn_send_starts[key] = event['start'] * 1000

    # Build lookup: (layer, mb) -> ATT recv_wait end time (data actually arrived at ATT, ATT clock)
    attn_recv_ends = {}
    for event in attn_data['events']:
        if event['type'] == 'recv_wait':
            key = (event['layer'], event['mb'])
            attn_recv_ends[key] = event['start'] * 1000 + event['duration_ms']
    
    # Collect offsets from ALL anchors in visualization range (no recv_dur filter needed)
    offsets = []
    for layer in range(start_layer, end_layer):
        for mb in range(2):  # MB 0 and 1
            key = (layer, mb)
            if key in attn_send_ends and key in ffn_recv_ends_raw:
                offsets.append(ffn_recv_ends_raw[key] - attn_send_ends[key])
    
    if offsets:
        # Use MIN (tightest upper bound on true clock offset)
        clock_offset = min(offsets)
        print(f"  Clock offset (FFN-ATT): {clock_offset:.2f}ms (min of {len(offsets)} anchors, range [{min(offsets):.2f}, {max(offsets):.2f}])")
    else:
        clock_offset = 0
        print(f"  Warning: No clock anchors found, using offset=0")
    
    # === Step 2: 提取事件到各泳道 ===
    if ffn_view not in LANE_SETS:
        raise ValueError(f"Unsupported ffn_view: {ffn_view}")

    lanes_data = {lane_name: [] for lane_name in LANE_SETS[ffn_view]}
    
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
    
    # Build aligned per-(layer,mb) lookups (in ATT-clock, min_start-shifted ms)
    def _att_ms(t_ms): return t_ms - min_start
    def _ffn_ms(t_ms): return t_ms - clock_offset - min_start

    # FFN recv_wait end aligned to ATT clock (for A2F bar end = data arrived at FFN)
    ffn_recv_ends_aligned = {k: _ffn_ms(v) for k, v in ffn_recv_ends_raw.items()}
    # ATT recv_wait end (for F2A bar end = data arrived at ATT)
    attn_recv_ends_aligned = {k: _att_ms(v) for k, v in attn_recv_ends.items()}

    # ========================================================================
    # 4-lane view: collapse all FFN sub-stages into 4 lanes that mirror
    # "compute vs comm" resource utilisation. A2F/F2A endpoints use the
    # receiver's wait-end time (data has been consumed by app).
    # ========================================================================
    if ffn_view == 'fourlane':
        return _build_fourlane(
            attn_data, ffn_data, start_layer, end_layer, min_start,
            clock_offset, ffn_recv_ends_aligned, attn_recv_ends_aligned,
        )

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
            # A2F bar: span sender enqueue start -> receiver recv_wait end
            # (= data has arrived at FFN). Falls back to enqueue duration if
            # no matching FFN recv_wait is recorded.
            recv_end = ffn_recv_ends_aligned.get((layer, mb))
            if recv_end is not None and recv_end > start_ms:
                bar_dur = recv_end - start_ms
            else:
                bar_dur = duration_ms
            lanes_data['A2F'].append((start_ms, bar_dur, display_layer, mb))
    
    # 提取 FFN 节点事件 (应用时钟偏移对齐到 ATT 时间线)
    ffn_phase_keys = {
        'ffn_compute',          # router + dispatch enqueue (host)
        'ep_local_experts',     # body GEMM
        'ep_reduce',            # combine
        'ep_overlap_hidden',    # overlapped hidden compute (if any)
        'ep_dispatch_wait',     # NCCL/HCCL dispatch wait
        'ep_reduce_wait',       # NCCL/HCCL reduce wait
        'moe_router',           # router subset of ffn_compute
        'ep_dispatch',          # dispatch enqueue subset
        'moe_experts',          # non-EP MoE experts
        'moe_shared_or_dense',  # dense FFN body
    }
    # phase_events[(layer, mb)][phase] = (start_ms_aligned, duration_ms)
    phase_events: dict = {}
    for event in ffn_data['events']:
        layer = event['layer']
        if layer < start_layer or layer >= end_layer:
            continue

        start_ms = event['start'] * 1000 - clock_offset - min_start
        duration_ms = event['duration_ms']
        mb = event['mb']
        display_layer = layer - start_layer
        etype = event['type']

        if etype == 'send_transfer':
            # F2A bar: FFN sender enqueue start -> ATT recv_wait end
            recv_end_att = attn_recv_ends_aligned.get((layer, mb))
            if recv_end_att is not None and recv_end_att > start_ms:
                bar_dur = recv_end_att - start_ms
            else:
                bar_dur = duration_ms
            lanes_data['F2A'].append((start_ms, bar_dur, display_layer, mb))
            continue
        if etype == 'recv_wait':
            # No longer overlaid on A2F lane (round 3 cleanup).
            continue
        if etype not in ffn_phase_keys:
            continue

        if ffn_view == 'staged':
            if etype == 'moe_router':
                lanes_data['F/router'].append((start_ms, duration_ms, display_layer, mb))
            elif etype == 'ep_dispatch':
                lanes_data['F/dispatch'].append((start_ms, duration_ms, display_layer, mb))
            elif etype in ('ep_local_experts', 'moe_experts', 'moe_shared_or_dense'):
                lanes_data['F/local_experts'].append((start_ms, duration_ms, display_layer, mb))
            elif etype in ('ep_reduce', 'ep_overlap_hidden', 'ep_reduce_wait'):
                phase_events.setdefault((display_layer, mb), {})[etype] = (start_ms, duration_ms)
            elif etype == 'ffn_compute':
                # Fallback for older JSON that lacks fine-grained MoE/EP events.
                key = (display_layer, mb)
                phase_events.setdefault(key, {})[etype] = (start_ms, duration_ms)
        else:
            phase_events.setdefault((display_layer, mb), {})[etype] = (start_ms, duration_ms)

    if ffn_view == 'staged':
        for (dl, mb), phases in phase_events.items():
            reduce_phase = phases.get('ep_reduce')
            if reduce_phase:
                total_start, total_dur = reduce_phase
                hidden = phases.get('ep_overlap_hidden')
                hidden_dur = hidden[1] if hidden else 0.0
                # Effective combine = wall span minus the time that was hidden
                # under the next MB's compute. Both MBs end up similar (~0.1ms).
                effective_dur = max(0.0, total_dur - hidden_dur)
                # Anchor effective bar at total_start; this places it at the
                # start of the async reduce kick (visually at its real start).
                segs = [(total_start, effective_dur, 'combine_effective')]
                if hidden is not None:
                    segs.append((hidden[0], hidden[1], 'overlap_hidden'))
                # Outer tuple uses effective_dur so the duration label reflects
                # actual non-hidden combine time (close between MBs).
                lanes_data['F/combine'].append((total_start, effective_dur, dl, mb, segs))

        fine_grained_keys = ('F/router', 'F/dispatch', 'F/local_experts', 'F/combine')
        if not any(lanes_data[key] for key in fine_grained_keys):
            print(
                "  Warning: no fine-grained FFN stage events found; "
                "falling back to ffn_compute on F/local_experts."
            )
            for (dl, mb), phases in phase_events.items():
                ffn_c = phases.get('ffn_compute')
                if ffn_c:
                    lanes_data['F/local_experts'].append((ffn_c[0], ffn_c[1], dl, mb))
        else:
            missing = [
                LANE_SETS[ffn_view][key]['label']
                for key in fine_grained_keys
                if not lanes_data[key]
            ]
            if missing:
                print(
                    "  Note: timing JSON does not contain events for "
                    f"{', '.join(missing)}; those lanes will be empty."
                )
        return lanes_data, attn_data, ffn_data, start_layer

    # Build composite F lane records keyed by (display_layer, mb)
    # Each record: (start_ms, duration_ms, display_layer, mb, segments)
    # where segments = [(seg_start, seg_dur, label)] in chronological order.
    f_send_starts = {(dl, mb): s for s, _, dl, mb in lanes_data['F2A']}
    for (dl, mb), phases in phase_events.items():
        ffn_c = phases.get('ffn_compute')
        if not ffn_c:
            continue
        ffn_start = ffn_c[0]
        # End of FFN work = start of F2A send (per-MB early send is enqueued
        # immediately when this MB's reduce completes). Fallback to max of
        # known phase ends if send not recorded.
        send_start = f_send_starts.get((dl, mb))
        candidate_ends = [s + d for s, d in phases.values()]
        if send_start is not None:
            f_end = send_start
        else:
            f_end = max(candidate_ends) if candidate_ends else ffn_start + ffn_c[1]
        f_dur = max(0.0, f_end - ffn_start)

        # Build internal segments for the composite bar.
        # We pick: router/dispatch (= ffn_compute span), local_experts, reduce.
        segs = []
        segs.append((ffn_c[0], ffn_c[1], 'router/dispatch'))
        if 'ep_local_experts' in phases:
            ls, ld = phases['ep_local_experts']
            segs.append((ls, ld, 'local_experts'))
        if 'ep_reduce' in phases:
            rs, rd = phases['ep_reduce']
            segs.append((rs, rd, 'reduce'))
        lanes_data['F'].append((ffn_start, f_dur, dl, mb, segs))

    return lanes_data, attn_data, ffn_data, start_layer


def plot_pipeline(lanes_data: dict, attn_data: dict, ffn_data: dict, 
                  output_path: str, num_layers: int = 2, start_layer: int = 1,
                  serial_baseline_ms: float = None, serial_baseline_label: str = None,
                  dbo_total_ms: float = None, mode: str = None,
                  ffn_view: str = 'staged',
                  pipeline_variant: str | None = None):
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
    lane_defs = LANE_SETS[ffn_view]
    if ffn_view == 'staged':
        fig_height = 7.2
    elif ffn_view == 'fourlane':
        fig_height = 5.0
    else:
        fig_height = 6
    fig, ax = plt.subplots(figsize=(14, fig_height))
    
    # 设置 y 轴
    lanes_by_y = sorted(lane_defs, key=lambda name: lane_defs[name]['index'])
    lane_positions = [lane_defs[lane]['index'] for lane in lanes_by_y]
    lane_labels = [lane_defs[lane]['label'] for lane in lanes_by_y]
    
    ax.set_yticks(lane_positions)
    ax.set_yticklabels(lane_labels, fontsize=11)
    ax.set_ylim(-0.5, max(lane_positions) + 1)
    
    # 绘制每个泳道的事件
    # Sub-segment styles for F lane composite bar
    F_SEG_STYLES = {
        'router/dispatch':    {'alpha': 0.40, 'hatch': '//'},
        'local_experts':      {'alpha': 0.95, 'hatch': None},
        'reduce':             {'alpha': 0.65, 'hatch': '\\\\'},
        'combine_effective':  {'alpha': 0.92, 'hatch': None},
        'overlap_hidden':     {'alpha': 0.22, 'hatch': 'xx'},
    }
    for lane_name, events in lanes_data.items():
        lane_info = lane_defs[lane_name]
        y_pos = lane_info['index']
        height = lane_info['height']

        for event_tuple in events:
            event_kind = None
            # Composite tuple: (start, dur, layer, mb, segments)
            if len(event_tuple) == 5 and isinstance(event_tuple[4], list):
                start_ms, duration_ms, layer, mb, segs = event_tuple
            else:
                start_ms, duration_ms, layer, mb = event_tuple[:4]
                if len(event_tuple) == 5:
                    event_kind = event_tuple[4]
                segs = None

            color = MB_COLORS.get(mb, '#999999')
            rect_height = height
            rect_y = y_pos - height/2
            facecolor = color
            edgecolor = 'black'
            linewidth = 0.5
            alpha = 0.8
            hatch = None

            if lane_name == 'A2F' and event_kind == 'send_enqueue':
                rect_height = height * 0.42
                rect_y = y_pos + height * 0.04
                hatch = None

            if segs is not None:
                # Draw composite outer frame (transparent fill, just outline)
                outer = Rectangle(
                    (start_ms, y_pos - height/2),
                    duration_ms,
                    height,
                    facecolor='none',
                    edgecolor=color,
                    linewidth=1.4,
                )
                ax.add_patch(outer)
                # Draw each phase segment with distinct alpha/hatch
                for seg_start, seg_dur, seg_label in segs:
                    style = F_SEG_STYLES.get(seg_label, {'alpha': 0.6, 'hatch': None})
                    seg_rect = Rectangle(
                        (seg_start, y_pos - height/2),
                        seg_dur,
                        rect_height,
                        facecolor=color,
                        edgecolor=color,
                        linewidth=0.3,
                        alpha=style['alpha'],
                        hatch=style['hatch'],
                    )
                    ax.add_patch(seg_rect)
            else:
                rect = Rectangle(
                    (start_ms, rect_y),
                    duration_ms,
                    rect_height,
                    facecolor=facecolor,
                    edgecolor=edgecolor,
                    linewidth=linewidth,
                    alpha=alpha,
                    hatch=hatch,
                )
                ax.add_patch(rect)

            # 添加文本标注 - 显示实际层号
            actual_layer = layer + start_layer
            label = f'L{actual_layer}'
            duration_label = f'{duration_ms:.1f}ms'
            text_x = start_ms + duration_ms / 2
            text_y = y_pos

            if event_kind != 'recv_wait':
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

            # On send lanes (A2F/F2A), add an enqueue marker (triangle) to mark
            # the sender's isend() return time (start of bar).
            if lane_name in ('A2F', 'F2A'):
                ax.plot([start_ms], [y_pos + height/2 + 0.05], marker='v',
                        markersize=5, color=color, zorder=5)

    # Layer boundary vertical lines: between display_layers (use earliest FFN-stage
    # start of each layer).
    layer_boundary_xs = []
    layer_min_starts = {}  # display_layer -> earliest start across F lane
    for lane_name, events in lanes_data.items():
        if not lane_name.startswith('F') or lane_name == 'F2A':
            continue
        for ev in events:
            s = ev[0]
            dl = ev[2]
            layer_min_starts[dl] = min(s, layer_min_starts.get(dl, float('inf')))
    for dl in sorted(layer_min_starts):
        if dl > 0:
            ax.axvline(x=layer_min_starts[dl], color='gray',
                       linestyle=':', alpha=0.5, linewidth=0.8, zorder=1)
            ax.text(layer_min_starts[dl], max(lane_positions) + 0.5 - 0.05,
                    f'L{dl + start_layer}', ha='left', va='top',
                    fontsize=7, color='gray')
    
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
    view_label = {'staged': 'staged FFN',
                  'legacy': 'legacy FFN',
                  'fourlane': '4-lane (compute/comm)'}[ffn_view]
    variant_label = f", {pipeline_variant}" if pipeline_variant else ""
    line1 = (
        f'DBO Pipeline{mode_tag} — L{start_layer}–{end_layer}{layer_note}, '
        f'{num_mb} Micro-batches, {comm_label}, {view_label}{variant_label}'
    )
    line2 = f"{dbo_label} | {serial_label} | {speedup_str}"
    ax.set_title(f'{line1}\n{line2}', fontsize=10, pad=10)
    
    # 图例
    legend_elements = [
        mpatches.Patch(facecolor=MB_COLORS[0], edgecolor='black',
                       label='Micro-batch 0', alpha=0.8),
        mpatches.Patch(facecolor=MB_COLORS[1], edgecolor='black',
                       label='Micro-batch 1', alpha=0.8),
    ]
    if ffn_view == 'legacy':
        legend_elements.extend([
            mpatches.Patch(facecolor='gray', edgecolor='gray', alpha=0.40, hatch='//',
                           label='F seg: router/dispatch'),
            mpatches.Patch(facecolor='gray', edgecolor='gray', alpha=0.95,
                           label='F seg: local_experts (GEMM)'),
            mpatches.Patch(facecolor='gray', edgecolor='gray', alpha=0.65, hatch='\\\\',
                           label='F seg: reduce (combine)'),
        ])
    elif ffn_view == 'staged':
        legend_elements.extend([
            mpatches.Patch(facecolor='gray', edgecolor='black', alpha=0.8,
                           label='F/local_experts = actual FFN GEMM body'),
            mpatches.Patch(facecolor='gray', edgecolor='gray', alpha=0.92,
                           label='combine: effective (non-hidden)'),
            mpatches.Patch(facecolor='gray', edgecolor='gray', alpha=0.22, hatch='xx',
                           label='combine: hidden in-flight (under next MB compute)'),
        ])
    # fourlane: keep legend MB0/MB1 only (bar semantics documented in QA.md / README).
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9, ncol=2)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Pipeline visualization saved to: {output_path}")
    
    return fig


def main():
    parser = argparse.ArgumentParser(
        description='Visualize DBO pipeline with staged FFN Gantt chart'
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
             'Use --auto-skip-warmup to drop backend JIT warmup layers automatically.'
    )
    parser.add_argument(
        '--auto-skip-warmup',
        action='store_true',
        default=True,
        help='Auto-detect and skip warmup layers whose mb0 duration is >5× the '
             'median of later layers (for NPU prefill without --prefill-warmup-rounds).'
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
    parser.add_argument(
        '--ffn-view',
        choices=['staged', 'legacy', 'fourlane'],
        default='fourlane',
        help='FFN visualization. fourlane (default): collapse to '
             'Attention/A2F/FFN/F2A — show compute vs comm overlap. '
             'staged: split router/dispatch/local_experts/combine. '
             'legacy: old composite F lane.'
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
    serial_timing_path = args.serial_timing
    if serial_baseline_ms is None and not serial_timing_path:
        auto = autodetect_serial_baseline(args.attn_timing)
        if auto:
            serial_timing_path = auto
            print(f"  Auto-detected serial baseline: {serial_timing_path}")
    if serial_baseline_ms is None and serial_timing_path and Path(serial_timing_path).exists():
        try:
            with open(serial_timing_path) as f:
                cache = json.load(f)
            baseline = resolve_serial_baseline(cache, args.mode)
            serial_baseline_ms = baseline.value_ms
            serial_baseline_label = baseline.unit
            if baseline.warning:
                print(f"  Warning: {baseline.warning}")
            if serial_baseline_ms is None:
                print(f"  Warning: '{args.mode}' baseline missing from {serial_timing_path}; "
                      f"Speedup will be N/A. Keys present: {list(cache.keys())}")
            else:
                metric = serial_baseline_label or args.mode
                print(f"  Serial {metric} baseline: {serial_baseline_ms:.1f}ms  ({serial_timing_path})")
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

    # Auto-detect warmup layers (NPU per-shape JIT compile): if mb0 attn durations
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
    path_text = f"{args.attn_timing} {args.ffn_timing}"
    if args.mode == 'decode':
        pipeline_variant = 'crosslayer' if 'crosslayer' in path_text else 'no crosslayer'
    else:
        pipeline_variant = None
    
    lanes_data, attn_data, ffn_data, start_layer = load_timing_data(
        args.attn_timing,
        args.ffn_timing,
        args.start_layer,
        args.num_layers,
        args.ffn_view,
    )
    lane_defs = LANE_SETS[args.ffn_view]
    
    # 打印数据统计
    print(f"\nData summary (layer {args.start_layer}-{args.start_layer + args.num_layers - 1}):")
    for lane_name, events in lanes_data.items():
        print(f"  {lane_defs[lane_name]['label']}: {len(events)} events")
    
    # 生成可视化
    print(f"\nGenerating visualization...")
    plot_pipeline(lanes_data, attn_data, ffn_data, args.output, 
                  args.num_layers, start_layer,
                  serial_baseline_ms=serial_baseline_ms,
                  serial_baseline_label=serial_baseline_label,
                  dbo_total_ms=dbo_total_ms,
                  mode=args.mode,
                  ffn_view=args.ffn_view,
                  pipeline_variant=pipeline_variant)
    
    print(f"\n✓ Done! View the result at: {args.output}")


if __name__ == '__main__':
    main()
