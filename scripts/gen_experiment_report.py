#!/usr/bin/env python3
"""Generate a Markdown report for one experiment run.

Reads the pair of timing JSON files (attention + FFN) produced by src.main and
emits a detailed markdown report with:
  - Metadata (mode, batch, seq, tokens, dtype, model)
  - Model-side TTFT timing for prefill runs, or exact TPOT timing for decode runs
  - Per-step timing table (if events span multiple steps)
  - Per-layer × {Attention, A2F, FFN, F2A} mean/min/max table
  - Optional comparison vs a cached serial baseline

Event type mapping (from timing_tracker):
  attention node: attn_compute -> A;   send_transfer -> A2F;   recv_wait -> F2A wait
  ffn       node: ffn_compute  -> FFN; send_transfer -> F2A;   recv_wait -> A2F wait

Decode DBO still records ONE representative inter-token-latency (ITL) sample for
pipeline/per-layer analysis. Speedup uses exact full decode-loop TPOT from
``decode_tpot_ms``.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from experiment_baselines import normalize_mode, resolve_serial_baseline


def _load(path: str) -> Optional[Dict[str, Any]]:
    if not path or not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


def _group_by_layer(events: List[Dict[str, Any]]) -> Dict[int, Dict[str, List[float]]]:
    """Group event durations by (layer, type) -> [durations]."""
    by_layer: Dict[int, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for e in events:
        layer = e.get("layer")
        t = e.get("type")
        d = e.get("duration_ms")
        if layer is None or t is None or d is None:
            continue
        by_layer[layer][t].append(float(d))
    return by_layer


def _fmt_stats(vals: List[float]) -> str:
    if not vals:
        return "-"
    if len(vals) == 1:
        return f"{vals[0]:.3f}"
    return f"{statistics.mean(vals):.3f} / {min(vals):.3f} / {max(vals):.3f}"


def _per_layer_table(attn: Optional[Dict], ffn: Optional[Dict]) -> str:
    """Build per-layer A / A2F / FFN / F2A table (mean over micro-batches)."""
    a_events = attn.get("events", []) if attn else []
    f_events = ffn.get("events", []) if ffn else []
    a_by = _group_by_layer(a_events)
    f_by = _group_by_layer(f_events)
    layers = sorted(set(a_by) | set(f_by))
    if not layers:
        return "_No per-layer events recorded._"

    rows = ["| Layer | Attention (ms) | A2F send (ms) | FFN (ms) | F2A send (ms) | F2A recv-wait (ms) |",
            "|------:|---------------:|--------------:|---------:|--------------:|-------------------:|"]
    for l in layers:
        a = a_by.get(l, {})
        f = f_by.get(l, {})
        att = _fmt_stats(a.get("attn_compute", []))
        a2f = _fmt_stats(a.get("send_transfer", []))
        ffn_c = _fmt_stats(f.get("ffn_compute", []))
        f2a = _fmt_stats(f.get("send_transfer", []))
        f2a_wait = _fmt_stats(a.get("recv_wait", []))
        rows.append(f"| {l} | {att} | {a2f} | {ffn_c} | {f2a} | {f2a_wait} |")

    # Aggregate totals (skip L0 because warmup step is atypically long)
    def _sum_mean(by, typ, skip_first=True):
        total = 0.0
        for l, m in by.items():
            if skip_first and l == 0:
                continue
            if m.get(typ):
                total += statistics.mean(m[typ])
        return total

    totals = [
        ("Σ (excl. L0)", _sum_mean(a_by, "attn_compute"), _sum_mean(a_by, "send_transfer"),
         _sum_mean(f_by, "ffn_compute"), _sum_mean(f_by, "send_transfer"), _sum_mean(a_by, "recv_wait")),
    ]
    rows.append(f"| **{totals[0][0]}** | **{totals[0][1]:.3f}** | **{totals[0][2]:.3f}** | "
                f"**{totals[0][3]:.3f}** | **{totals[0][4]:.3f}** | **{totals[0][5]:.3f}** |")
    rows.append("")
    rows.append("_Cells with three values are **mean / min / max across micro-batches**, not repeated runs._")
    rows.append("_L0 is skipped in the Σ row because layer-0 contains pipeline warmup._")
    if len(layers) == 48 and 21 in layers:
        rows.append(
            "_Layer 21 is the default 2-GPU shard boundary for 48-layer Qwen3 "
            "(21 layers on role GPU0, remaining layers on role GPU1). A large max "
            "with a normal min usually means only the first micro-batch paid "
            "cross-device/lazy CUDA warmup cost._"
        )
    return "\n".join(rows)


def _layer_average_summary(attn: Optional[Dict], ffn: Optional[Dict]) -> str:
    """Summarize mean per-layer timing across all layers."""
    a_events = attn.get("events", []) if attn else []
    f_events = ffn.get("events", []) if ffn else []
    a_by = _group_by_layer(a_events)
    f_by = _group_by_layer(f_events)
    layers = sorted(set(a_by) | set(f_by))
    if not layers:
        return "_No per-layer events recorded._"

    def mean_for_layer(layer: int, by_layer: Dict[int, Dict[str, List[float]]], event_type: str) -> float:
        values = by_layer.get(layer, {}).get(event_type, [])
        return statistics.mean(values) if values else 0.0

    def avg_across_layers(selected_layers: List[int], by_layer: Dict[int, Dict[str, List[float]]], event_type: str) -> float:
        if not selected_layers:
            return 0.0
        return statistics.mean(mean_for_layer(layer, by_layer, event_type) for layer in selected_layers)

    scopes = [("All layers", layers)]
    layers_without_l0 = [layer for layer in layers if layer != 0]
    if layers_without_l0:
        scopes.append(("Excl. L0", layers_without_l0))

    rows = [
        "| Scope | Layers | Attention avg/layer (ms) | A2F avg/layer (ms) | FFN avg/layer (ms) | F2A avg/layer (ms) | F2A recv-wait avg/layer (ms) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, selected_layers in scopes:
        rows.append(
            f"| {label} | {len(selected_layers)} | "
            f"{avg_across_layers(selected_layers, a_by, 'attn_compute'):.3f} | "
            f"{avg_across_layers(selected_layers, a_by, 'send_transfer'):.3f} | "
            f"{avg_across_layers(selected_layers, f_by, 'ffn_compute'):.3f} | "
            f"{avg_across_layers(selected_layers, f_by, 'send_transfer'):.3f} | "
            f"{avg_across_layers(selected_layers, a_by, 'recv_wait'):.3f} |"
        )
    rows.append("")
    rows.append("_Each value first averages across micro-batches within a layer, then averages those layer means across the selected layer set._")
    return "\n".join(rows)


def _metadata_block(attn: Optional[Dict], ffn: Optional[Dict], args) -> str:
    md = attn or ffn or {}
    mode_str = args.mode or "unknown"
    lines = [
        f"- **Mode**: `{mode_str}`",
        f"- **Batch size**: {args.batch}",
        f"- **Prefill seq**: {args.seq}",
        f"- **Decode tokens**: {args.tokens}",
        f"- **Layers**: {md.get('num_layers', '?')}",
        f"- **Micro-batches**: {md.get('num_micro_batches', '?')}",
    ]
    if args.model:
        lines.append(f"- **Model**: `{args.model}`")
    if args.dtype:
        lines.append(f"- **Dtype**: `{args.dtype}`")
    return "\n".join(lines)


def _e2e_block(attn: Optional[Dict], ffn: Optional[Dict], mode: str) -> str:
    lines = ["| Metric | Attention | FFN |", "|---|---:|---:|"]

    def g(d, k):
        return d.get(k) if d else None

    def fmt(v, unit="ms"):
        if v is None:
            return "-"
        if isinstance(v, float):
            return f"{v:.3f} {unit}".strip()
        return str(v)

    normalized_mode = normalize_mode(mode)
    total_label = (
        "Model-side TTFT / prefill total"
        if normalized_mode == "prefill"
        else "Representative ITL sample total"
    )
    keys = [
        (total_label, "total_time_ms", "ms"),
        ("Decode loop total", "decode_loop_ms", "ms"),
        ("Decode steps", "decode_steps", ""),
        ("Decode TPOT", "decode_tpot_ms", "ms"),
        ("Compute", "total_compute_ms", "ms"),
        ("Recv wait", "total_recv_wait_ms", "ms"),
        ("MoE router", "total_moe_router_ms", "ms"),
        ("MoE experts", "total_moe_experts_ms", "ms"),
        ("MoE shared/dense", "total_moe_shared_or_dense_ms", "ms"),
        ("Compute ratio", "compute_ratio", ""),
    ]
    for label, key, unit in keys:
        av = g(attn, key)
        fv = g(ffn, key)
        lines.append(f"| {label} | {fmt(av, unit)} | {fmt(fv, unit)} |")

    # Serial path writes gen_time_ms & tokens_per_sec
    if attn and ("gen_time_ms" in attn or "tokens_per_sec" in attn):
        if "gen_time_ms" in attn:
            lines.append(f"| Full-generate wall time | {attn['gen_time_ms']:.3f} ms | - |")
        if "tokens_per_sec" in attn:
            lines.append(f"| Tokens/sec | {attn['tokens_per_sec']:.2f} | - |")
    return "\n".join(lines)


def _compare_vs_serial(cur_attn: Optional[Dict], serial_attn: Optional[Dict], mode: str) -> str:
    if not cur_attn or not serial_attn:
        return ""
    normalized_mode = normalize_mode(mode)
    cur_value = cur_attn.get("total_time_ms") if normalized_mode == "prefill" else cur_attn.get("decode_tpot_ms")
    if cur_value is None:
        metric = "TTFT" if normalized_mode == "prefill" else "exact TPOT"
        return (f"\n## Compared to serial baseline\n\n"
                f"- This run {metric}: **N/A** (missing required timing field)\n"
                f"- Speedup: **N/A**\n")
    cur = float(cur_value)
    if cur <= 0:
        return ""

    baseline = resolve_serial_baseline(serial_attn, normalized_mode)
    if not baseline.available:
        warning = baseline.warning or "mode-matched serial baseline unavailable"
        run_metric = "TTFT" if normalized_mode == "prefill" else "exact TPOT"
        return (f"\n## Compared to serial baseline\n\n"
                f"- Serial baseline: **N/A** ({warning})\n"
                f"- This run {run_metric}: **{cur:.3f} ms**\n"
                f"- Speedup: **N/A**\n")

    assert baseline.value_ms is not None
    assert baseline.unit is not None
    speedup = baseline.value_ms / cur
    delta = cur - baseline.value_ms
    cur_metric = "TTFT" if normalized_mode == "prefill" else "exact TPOT"
    speedup_metric = "TTFT" if normalized_mode == "prefill" else "TPOT"
    source_note = baseline.source
    if baseline.warning:
        source_note += f"; {baseline.warning}"
    return (f"\n## Compared to serial baseline\n\n"
            f"- Serial {baseline.unit}: **{baseline.value_ms:.3f} ms**  ({source_note})\n"
            f"- This run {cur_metric}: **{cur:.3f} ms**\n"
            f"- Δ: {delta:+.3f} ms   |   {speedup_metric} speedup: **{speedup:.3f}×**\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attn-timing", required=True)
    ap.add_argument("--ffn-timing", required=True)
    ap.add_argument("--output", required=True, help="Output markdown path")
    ap.add_argument("--mode", default="", help="serial | dbo | dbo_crosslayer | prefill_dbo …")
    ap.add_argument("--batch", type=int, default=0)
    ap.add_argument("--seq", type=int, default=0)
    ap.add_argument("--tokens", type=int, default=0)
    ap.add_argument("--model", default="")
    ap.add_argument("--dtype", default="")
    ap.add_argument("--serial-baseline", default="",
                    help="Optional path to serial attention timing JSON for comparison")
    args = ap.parse_args()

    attn = _load(args.attn_timing)
    ffn = _load(args.ffn_timing)
    if attn is None and ffn is None:
        raise SystemExit(f"ERROR: neither {args.attn_timing} nor {args.ffn_timing} exist")

    serial_attn = _load(args.serial_baseline) if args.serial_baseline else None

    out = []
    title = f"Experiment report — {args.mode or 'run'} b{args.batch} s{args.seq} t{args.tokens}"
    out.append(f"# {title}\n")
    out.append("## Configuration\n")
    out.append(_metadata_block(attn, ffn, args))
    out.append("")
    timing_title = (
        "Model-side TTFT timing (prefill path)"
        if normalize_mode(args.mode) == "prefill"
        else "Decode timing (exact TPOT + representative ITL detail)"
    )
    out.append(f"## {timing_title}\n")
    out.append(_e2e_block(attn, ffn, args.mode))
    out.append("")
    cmp_block = _compare_vs_serial(attn, serial_attn, args.mode)
    if cmp_block:
        out.append(cmp_block)
    out.append("## Layer averages summary\n")
    out.append(_layer_average_summary(attn, ffn))
    out.append("")
    out.append("## Per-layer breakdown\n")
    out.append(_per_layer_table(attn, ffn))
    out.append("")
    out.append("---")
    out.append(f"_Generated from `{os.path.basename(args.attn_timing)}` + "
               f"`{os.path.basename(args.ffn_timing)}`._")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        f.write("\n".join(out) + "\n")
    print(f"✅ Report written: {args.output}")


if __name__ == "__main__":
    main()
