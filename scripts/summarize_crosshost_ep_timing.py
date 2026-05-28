#!/usr/bin/env python3
"""Summarize cross-host EP timing JSONs into CSV and Markdown."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
TAG_RE = re.compile(
    r"xhost_static_(?P<mode>decode-dbo(?:-crosslayer)?)_ep(?P<ep>\d+)_"
    r"(?P<backend>broadcast_reduce_(?:sync|overlap))(?P<mb>_mb\d+)?"
    r"_b(?P<batch>\d+)_s(?P<seq>\d+)_t(?P<tokens>\d+)"
)


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def group_by_layer(events: list[dict[str, Any]]) -> dict[int, dict[str, list[float]]]:
    result: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for event in events:
        layer = event.get("layer")
        typ = event.get("type")
        duration = event.get("duration_ms")
        if layer is None or typ is None or duration is None:
            continue
        result[int(layer)][str(typ)].append(float(duration))
    return result


def avg_layer(by_layer: dict[int, dict[str, list[float]]], event_type: str, *, skip_l0: bool = True) -> float:
    values: list[float] = []
    for layer, grouped in by_layer.items():
        if skip_l0 and layer == 0:
            continue
        durations = grouped.get(event_type, [])
        if durations:
            values.append(statistics.mean(durations))
    return statistics.mean(values) if values else 0.0


def sum_layer(by_layer: dict[int, dict[str, list[float]]], event_type: str, *, skip_l0: bool = True) -> float:
    total = 0.0
    for layer, grouped in by_layer.items():
        if skip_l0 and layer == 0:
            continue
        durations = grouped.get(event_type, [])
        if durations:
            total += statistics.mean(durations)
    return total


def summarize_pair(attn_path: Path, ffn_path: Path) -> dict[str, Any] | None:
    match = TAG_RE.search(attn_path.stem)
    if not match:
        return None
    attn = load_json(attn_path)
    ffn = load_json(ffn_path)
    attn_by = group_by_layer(attn.get("events", []))
    ffn_by = group_by_layer(ffn.get("events", []))
    mb_tag = match.group("mb") or ""
    num_mb = int(mb_tag.removeprefix("_mb")) if mb_tag else 2

    attn_avg = avg_layer(attn_by, "attn_compute")
    ffn_avg = avg_layer(ffn_by, "ffn_compute")
    a2f_avg = avg_layer(attn_by, "send_transfer")
    f2a_avg = avg_layer(ffn_by, "send_transfer")
    recv_wait_avg = avg_layer(attn_by, "recv_wait")
    router_avg = avg_layer(ffn_by, "moe_router")
    dispatch_avg = avg_layer(ffn_by, "ep_dispatch")
    local_experts_avg = avg_layer(ffn_by, "ep_local_experts")
    reduce_avg = avg_layer(ffn_by, "ep_reduce")
    stage_sum = attn_avg + ffn_avg + recv_wait_avg
    overlap_eff = max(attn_avg, ffn_avg) / stage_sum if stage_sum > 0 else 0.0

    return {
        "mode": match.group("mode"),
        "ep_size": int(match.group("ep")),
        "backend": match.group("backend"),
        "attn_kernel": attn.get("attention_optimizations", {}).get("attn_kernel", ""),
        "attn_fused_rmsnorm": attn.get("attention_optimizations", {}).get("attn_fused_rmsnorm", ""),
        "attn_fused_rope": attn.get("attention_optimizations", {}).get("attn_fused_rope", ""),
        "num_micro_batches": num_mb,
        "batch": int(match.group("batch")),
        "seq": int(match.group("seq")),
        "tokens": int(match.group("tokens")),
        "decode_tpot_ms": attn.get("decode_tpot_ms", ""),
        "prefill_ms": attn.get("prefill_ms", ""),
        "attention_avg_layer_ms_excl_l0": attn_avg,
        "ffn_avg_layer_ms_excl_l0": ffn_avg,
        "a2f_send_avg_layer_ms_excl_l0": a2f_avg,
        "f2a_send_avg_layer_ms_excl_l0": f2a_avg,
        "attention_recv_wait_avg_layer_ms_excl_l0": recv_wait_avg,
        "moe_router_avg_layer_ms_excl_l0": router_avg,
        "ep_dispatch_avg_layer_ms_excl_l0": dispatch_avg,
        "ep_local_experts_avg_layer_ms_excl_l0": local_experts_avg,
        "ep_reduce_avg_layer_ms_excl_l0": reduce_avg,
        "attention_sum_ms_excl_l0": sum_layer(attn_by, "attn_compute"),
        "ffn_sum_ms_excl_l0": sum_layer(ffn_by, "ffn_compute"),
        "overlap_efficiency_proxy": overlap_eff,
        "attn_timing": str(attn_path),
        "ffn_timing": str(ffn_path),
    }


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def write_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# Cross-host static EP timing summary",
        "",
        "| EP | Backend | Attn | Fusion | Mode | MB | B | S | T | TPOT ms | A avg/layer | F avg/layer | F/A | recv-wait | dispatch | local experts | reduce | overlap proxy |",
        "|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        attn = float(row["attention_avg_layer_ms_excl_l0"] or 0)
        ffn = float(row["ffn_avg_layer_ms_excl_l0"] or 0)
        ratio = ffn / attn if attn > 0 else 0.0
        fusion = "rms+rope" if row.get("attn_fused_rmsnorm") and row.get("attn_fused_rope") else "-"
        template = (
            "| {ep_size} | {backend} | {attn_kernel} | " + fusion + " | {mode} | "
            "{num_micro_batches} | {batch} | {seq} | {tokens} | "
            "{decode_tpot_ms} | {attention_avg_layer_ms_excl_l0:.3f} | "
            "{ffn_avg_layer_ms_excl_l0:.3f} | "
            f"{ratio:.2f} | "
            "{attention_recv_wait_avg_layer_ms_excl_l0:.3f} | "
            "{ep_dispatch_avg_layer_ms_excl_l0:.3f} | "
            "{ep_local_experts_avg_layer_ms_excl_l0:.3f} | "
            "{ep_reduce_avg_layer_ms_excl_l0:.3f} | "
            "{overlap_efficiency_proxy:.3f} |"
        )
        lines.append(template.format(**row))
    lines.append("")
    lines.append("说明：均值默认跳过 L0，以避免 pipeline/JIT warmup 干扰。`F/A` 越接近 1，FFN 与 Attention 单层耗时越对齐。")
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="results_npu/crosshost_static_ep")
    parser.add_argument("--output-csv", default="")
    parser.add_argument("--output-md", default="")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.is_absolute():
        root = ROOT / root
    rows: list[dict[str, Any]] = []
    for attn_path in sorted(root.rglob("timing_attention_xhost_static_*.json")):
        suffix = attn_path.name.removeprefix("timing_attention_").removesuffix(".json")
        ffn_path = attn_path.with_name(f"timing_ffn_coordinator_{suffix}.json")
        if not ffn_path.exists():
            continue
        row = summarize_pair(attn_path, ffn_path)
        if row is not None:
            rows.append(row)
    rows.sort(key=lambda r: (-r["batch"], -r["seq"], -r["ep_size"], r["backend"], r["mode"], r["num_micro_batches"]))

    csv_path = Path(args.output_csv) if args.output_csv else root / "crosshost_ep_timing_summary.csv"
    md_path = Path(args.output_md) if args.output_md else root / "crosshost_ep_timing_summary.md"
    if not csv_path.is_absolute():
        csv_path = ROOT / csv_path
    if not md_path.is_absolute():
        md_path = ROOT / md_path
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else [
        "mode",
        "ep_size",
        "backend",
        "num_micro_batches",
        "batch",
        "seq",
        "tokens",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    write_markdown(rows, md_path)
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
