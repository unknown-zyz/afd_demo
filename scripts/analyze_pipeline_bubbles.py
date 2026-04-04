#!/usr/bin/env python3
"""
分析 Prefill DBO pipeline 中的空泡（bubble）。

空泡来源：MB0 的下一层 Attention 必须等 MB1 的 FFN 结果也返回后才能开始，
因为 for layer_idx 循环要求所有 MB 完成当前层才能进入下一层。

本脚本量化这个空泡的大小，并估算跨层流水线优化的理论上限。
"""

import json
import sys
import os
from pathlib import Path
from collections import defaultdict


def load_timing(filepath):
    with open(filepath) as f:
        return json.load(f)


def analyze_config(config_name, attn_json_path):
    """分析单个配置的空泡。"""
    data = load_timing(attn_json_path)
    events = data["events"]

    # 按 layer 和 type 组织 events
    by_layer = defaultdict(lambda: defaultdict(list))
    for e in events:
        by_layer[e["layer"]][(e["type"], e["mb"])].append(e)

    layers = sorted(by_layer.keys())
    if len(layers) < 2:
        return None

    bubble_times = []
    layer_times = []

    # 跳过 layer 0（warmup），分析 layer 1 到 layer N-1
    for i in range(1, len(layers) - 1):
        layer = layers[i]
        next_layer = layers[i + 1]

        # MB0 的 recv_wait 结束时刻
        mb0_recv = by_layer[layer].get(("recv_wait", 0), [])
        # MB1 的 recv_wait 结束时刻
        mb1_recv = by_layer[layer].get(("recv_wait", 1), [])
        # MB0 下一层的 attn_compute 开始时刻
        mb0_next_attn = by_layer[next_layer].get(("attn_compute", 0), [])

        if not mb0_recv or not mb1_recv or not mb0_next_attn:
            continue

        mb0_recv_end = mb0_recv[0]["end"]
        mb1_recv_end = mb1_recv[0]["end"]
        next_attn_start = mb0_next_attn[0]["start"]

        # 空泡 = MB0 下一层开始 - MB0 当前层 recv 结束
        # 这包括：等 MB1 recv 完成 + 其他开销
        bubble = next_attn_start - mb0_recv_end

        # MB0 等 MB1 的时间 = MB1 recv 结束 - MB0 recv 结束
        wait_for_mb1 = max(0, mb1_recv_end - mb0_recv_end)

        # 当前层 attn_compute MB0 开始
        mb0_attn = by_layer[layer].get(("attn_compute", 0), [])
        current_attn_start = mb0_attn[0]["start"] if mb0_attn else None

        # 层时间 = 下一层 MB0 attn 开始 - 当前层 MB0 attn 开始
        if current_attn_start is not None:
            layer_time = next_attn_start - current_attn_start
        else:
            layer_time = None

        bubble_times.append({
            "layer": layer,
            "bubble_s": bubble,
            "wait_mb1_s": wait_for_mb1,
            "layer_time_s": layer_time,
        })

    if not bubble_times:
        return None

    avg_bubble = sum(b["bubble_s"] for b in bubble_times) / len(bubble_times)
    avg_wait_mb1 = sum(b["wait_mb1_s"] for b in bubble_times) / len(bubble_times)
    avg_layer_time = sum(b["layer_time_s"] for b in bubble_times if b["layer_time_s"]) / len(bubble_times)
    total_bubble = sum(b["bubble_s"] for b in bubble_times)
    total_wait_mb1 = sum(b["wait_mb1_s"] for b in bubble_times)
    num_layers_analyzed = len(bubble_times)

    bubble_ratio = avg_bubble / avg_layer_time if avg_layer_time > 0 else 0
    wait_ratio = avg_wait_mb1 / avg_layer_time if avg_layer_time > 0 else 0

    # E2E 时间（从 events 中推算）
    all_starts = [e["start"] for e in events]
    all_ends = [e["end"] for e in events]
    e2e_time = max(all_ends) - min(all_starts)

    # 理论省时 = 消除所有等 MB1 的时间
    theoretical_saving = total_wait_mb1
    theoretical_speedup = theoretical_saving / e2e_time if e2e_time > 0 else 0

    return {
        "config": config_name,
        "num_layers": num_layers_analyzed,
        "avg_bubble_ms": avg_bubble * 1000,
        "avg_wait_mb1_ms": avg_wait_mb1 * 1000,
        "avg_layer_time_ms": avg_layer_time * 1000,
        "bubble_ratio": bubble_ratio,
        "wait_ratio": wait_ratio,
        "total_bubble_ms": total_bubble * 1000,
        "total_wait_mb1_ms": total_wait_mb1 * 1000,
        "e2e_time_ms": e2e_time * 1000,
        "theoretical_saving_ms": theoretical_saving * 1000,
        "theoretical_speedup": theoretical_speedup,
        "per_layer": bubble_times,
    }


def main():
    results_dir = Path("results/prefill_dbo")

    configs = [
        ("b32_s128", "timing_attention_qwen3_dbo_b32_s128.json"),
        ("b64_s128", "timing_attention_qwen3_dbo_b64_s128.json"),
        ("b4_s256", "timing_attention_qwen3_dbo_b4_s256.json"),
        ("b4_s512", "timing_attention_qwen3_dbo_b4_s512.json"),
        ("b4_s1024", "timing_attention_qwen3_dbo_b4_s1024.json"),
        ("b4_s2048", "timing_attention_qwen3_dbo_b4_s2048.json"),
    ]

    results = []
    for name, filename in configs:
        filepath = results_dir / filename
        if not filepath.exists():
            print(f"[SKIP] {filepath} 不存在")
            continue
        r = analyze_config(name, filepath)
        if r:
            results.append(r)
        else:
            print(f"[SKIP] {name}: 数据不足")

    if not results:
        print("无可分析的数据")
        return

    # 打印汇总表格
    print("\n" + "=" * 100)
    print("Prefill DBO Pipeline 空泡分析")
    print("=" * 100)
    print(f"{'配置':<12} {'层数':>4} {'层时间(ms)':>10} {'空泡(ms)':>9} {'等MB1(ms)':>9} "
          f"{'空泡比':>7} {'等MB1比':>7} {'E2E(ms)':>9} {'理论省时(ms)':>12} {'理论加速':>8}")
    print("-" * 100)

    for r in results:
        print(f"{r['config']:<12} {r['num_layers']:>4} "
              f"{r['avg_layer_time_ms']:>10.2f} {r['avg_bubble_ms']:>9.2f} "
              f"{r['avg_wait_mb1_ms']:>9.2f} "
              f"{r['bubble_ratio']:>6.1%} {r['wait_ratio']:>6.1%} "
              f"{r['e2e_time_ms']:>9.1f} "
              f"{r['theoretical_saving_ms']:>12.1f} "
              f"{r['theoretical_speedup']:>7.1%}")

    print("-" * 100)
    print("\n说明:")
    print("  空泡 = MB0 下一层 attn 开始 - MB0 当前层 recv 结束（包含等 MB1 + 其他开销）")
    print("  等MB1 = MB1 recv 结束 - MB0 recv 结束（纯粹因等 MB1 浪费的时间）")
    print("  理论省时 = 消除所有等 MB1 的时间（跨层流水线优化上限）")
    print("  理论加速 = 理论省时 / E2E 时间")

    # 输出详细的逐层数据（选一个代表配置）
    best = max(results, key=lambda r: r["theoretical_speedup"])
    print(f"\n逐层详细数据（{best['config']}，理论加速最高）:")
    print(f"{'Layer':>6} {'空泡(ms)':>9} {'等MB1(ms)':>9} {'层时间(ms)':>10}")
    print("-" * 40)
    for pl in best["per_layer"][:10]:
        lt = pl["layer_time_s"] * 1000 if pl["layer_time_s"] else 0
        print(f"{pl['layer']:>6} {pl['bubble_s']*1000:>9.2f} {pl['wait_mb1_s']*1000:>9.2f} {lt:>10.2f}")
    if len(best["per_layer"]) > 10:
        print(f"  ... ({len(best['per_layer'])} layers total)")

    return results


if __name__ == "__main__":
    main()
