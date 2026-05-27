#!/usr/bin/env python3
"""Align single-layer Attention benchmarks with decode-DBO timing JSON.

The single-layer benchmark measures either one full batch or one synthetic
microbatch. Decode-DBO timing records per-microbatch ``attn_compute`` events.
This script keeps those scopes explicit and reports both comparisons.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


PHASE_CASES = {
    "prefill_core": ("hf_sdpa", "official_pfa"),
    "prefill_full_layer": ("hf", "official", "official_fused_both"),
    "decode_core": ("hf_sdpa", "official_ifa"),
    "decode_full_layer": ("hf", "official", "official_fused_both"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare single-layer Attention bench with pipeline timing.")
    parser.add_argument("--single-layer-json", action="append", default=[], help="Single-layer benchmark JSON.")
    parser.add_argument("--mb-json", action="append", default=[], help="MB-equivalent single-layer benchmark JSON.")
    parser.add_argument("--pipeline-attn", action="append", default=[], help="Decode-DBO attention timing JSON.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--exclude-layer0", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pipeline-case", default="official_fused_both")
    return parser.parse_args()


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def shape_seq(row: dict[str, Any]) -> int:
    return int(row.get("cache_len") or row.get("seq") or 0)


def load_aggregate(paths: list[str]) -> dict[tuple[str, str, int, int], dict[str, Any]]:
    index: dict[tuple[str, str, int, int], dict[str, Any]] = {}
    for path in paths:
        data = read_json(path)
        for row in data.get("aggregate_results", []):
            if not row.get("ok_count", 0):
                continue
            key = (str(row["phase"]), str(row["case"]), int(row["batch"]), shape_seq(row))
            existing = index.get(key)
            if existing is None or int(row.get("ok_count", 0)) > int(existing.get("ok_count", 0)):
                index[key] = dict(row, source=str(path))
    return index


def parse_bst_from_name(path: str | Path) -> tuple[int | None, int | None, int | None]:
    match = re.search(r"_b(\d+)_s(\d+)_t(\d+)", Path(path).name)
    if not match:
        return None, None, None
    return tuple(int(v) for v in match.groups())  # type: ignore[return-value]


def pct(values: list[float], q: float) -> float:
    if not values:
        return math.nan
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - pos) + values[hi] * (pos - lo)


def mb_sizes(batch: int, num_mb: int) -> list[int]:
    num_mb = min(num_mb, batch)
    base = batch // num_mb
    rem = batch % num_mb
    return [base + (1 if i < rem else 0) for i in range(num_mb)]


def summarize_pipeline(path: str, exclude_layer0: bool) -> dict[str, Any]:
    data = read_json(path)
    b_name, s_name, t_name = parse_bst_from_name(path)
    batch = int(data.get("batch_size") or b_name or 0)
    seq = int(data.get("prefill_seq_len") or s_name or data.get("actual_prompt_len") or 0)
    tokens = int(data.get("decode_steps") or t_name or 0)
    num_mb = int(data.get("num_micro_batches") or 1)

    per_layer_mb: dict[int, list[float]] = defaultdict(list)
    for ev in data.get("events", []):
        if ev.get("type") != "attn_compute":
            continue
        layer = int(ev.get("layer", -1))
        if exclude_layer0 and layer == 0:
            continue
        per_layer_mb[layer].append(float(ev.get("duration_ms", 0.0)))

    per_layer_sum = [sum(v) for _, v in sorted(per_layer_mb.items()) if v]
    flat_mb = [d for vals in per_layer_mb.values() for d in vals]
    return {
        "pipeline_file": path,
        "batch": batch,
        "seq": seq,
        "tokens": tokens,
        "num_micro_batches": num_mb,
        "mb_sizes": ",".join(str(v) for v in mb_sizes(batch, num_mb)),
        "decode_tpot_ms": data.get("decode_tpot_ms"),
        "prefill_ms": data.get("prefill_ms"),
        "decode_loop_ms": data.get("decode_loop_ms"),
        "tbt_mean_ms": data.get("tbt_mean_ms"),
        "tbt_p50_ms": data.get("tbt_p50_ms"),
        "tbt_p99_ms": data.get("tbt_p99_ms"),
        "pipeline_layer_count": len(per_layer_sum),
        "pipeline_attn_mb_event_count": len(flat_mb),
        "pipeline_attn_per_mb_median_ms": statistics.median(flat_mb) if flat_mb else math.nan,
        "pipeline_attn_per_layer_sum_median_ms": statistics.median(per_layer_sum) if per_layer_sum else math.nan,
        "pipeline_attn_per_layer_sum_p90_ms": pct(per_layer_sum, 0.90),
        "pipeline_attn_per_layer_sum_p99_ms": pct(per_layer_sum, 0.99),
    }


def latency(index: dict[tuple[str, str, int, int], dict[str, Any]], phase: str, case: str, batch: int, seq: int) -> float | None:
    row = index.get((phase, case, batch, seq))
    if not row:
        return None
    return float(row.get("latency_median_ms", math.nan))


def single_layer_rows(index: dict[tuple[str, str, int, int], dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for (phase, case, batch, seq), row in sorted(index.items()):
        if case not in PHASE_CASES.get(phase, ()):
            continue
        ms = float(row.get("latency_median_ms", math.nan))
        out = {
            "phase": phase,
            "case": case,
            "batch": batch,
            "seq_or_cache": seq,
            "latency_median_ms": ms,
            "latency_mean_ms": row.get("latency_mean_ms"),
            "speedup_vs_hf": row.get("aggregate_speedup_vs_hf"),
            "ok_count": row.get("ok_count"),
            "source": row.get("source"),
        }
        if phase.startswith("prefill"):
            out["ms_per_b_s"] = ms / max(batch * seq, 1)
            out["ms_per_b_s2"] = ms / max(batch * seq * seq, 1)
        else:
            out["ms_per_b_cache"] = ms / max(batch * seq, 1)
            out["ms_per_b"] = ms / max(batch, 1)
        rows.append(out)
    return rows


def alignment_rows(
    single_index: dict[tuple[str, str, int, int], dict[str, Any]],
    mb_index: dict[tuple[str, str, int, int], dict[str, Any]],
    pipeline_paths: list[str],
    pipeline_case: str,
    exclude_layer0: bool,
) -> list[dict[str, Any]]:
    rows = []
    for path in pipeline_paths:
        pipe = summarize_pipeline(path, exclude_layer0)
        batch = int(pipe["batch"])
        seq = int(pipe["seq"])
        num_mb = int(pipe["num_micro_batches"])
        global_ms = latency(single_index, "decode_full_layer", pipeline_case, batch, seq)

        mb_sum = 0.0
        mb_missing = []
        for size in mb_sizes(batch, num_mb):
            value = latency(mb_index, "decode_full_layer", pipeline_case, size, seq)
            if value is None:
                mb_missing.append(size)
            else:
                mb_sum += value
        mb_sum_value = None if mb_missing else mb_sum
        pipeline_ms = float(pipe["pipeline_attn_per_layer_sum_median_ms"])
        row = dict(pipe)
        row.update(
            {
                "single_layer_case": pipeline_case,
                "single_layer_global_ms": global_ms,
                "single_layer_mb_sum_ms": mb_sum_value,
                "missing_mb_sizes": ",".join(str(v) for v in mb_missing),
                "pipeline_vs_single_global": pipeline_ms / global_ms if global_ms else None,
                "pipeline_vs_single_mb_sum": pipeline_ms / mb_sum_value if mb_sum_value else None,
            }
        )
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.6g}"
    return str(value)


def write_md(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("| " + " | ".join(fields) + " |\n")
        f.write("| " + " | ".join(["---"] * len(fields)) + " |\n")
        for row in rows:
            f.write("| " + " | ".join(fmt(row.get(field)) for field in fields) + " |\n")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    single_index = load_aggregate(args.single_layer_json)
    mb_index = load_aggregate(args.mb_json or args.single_layer_json)

    scaling = single_layer_rows(single_index)
    write_csv(output_dir / "single_layer_scaling_summary.csv", scaling)
    write_md(
        output_dir / "single_layer_scaling_summary.md",
        scaling,
        [
            "phase",
            "case",
            "batch",
            "seq_or_cache",
            "latency_median_ms",
            "speedup_vs_hf",
            "ms_per_b_s",
            "ms_per_b_s2",
            "ms_per_b_cache",
            "ms_per_b",
        ],
    )

    aligned = alignment_rows(single_index, mb_index, args.pipeline_attn, args.pipeline_case, args.exclude_layer0)
    write_csv(output_dir / "alignment_summary.csv", aligned)
    write_md(
        output_dir / "alignment_summary.md",
        aligned,
        [
            "batch",
            "seq",
            "tokens",
            "num_micro_batches",
            "mb_sizes",
            "decode_tpot_ms",
            "pipeline_attn_per_layer_sum_median_ms",
            "single_layer_global_ms",
            "single_layer_mb_sum_ms",
            "pipeline_vs_single_global",
            "pipeline_vs_single_mb_sum",
            "pipeline_file",
        ],
    )


if __name__ == "__main__":
    main()
