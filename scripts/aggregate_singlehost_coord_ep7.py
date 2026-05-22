#!/usr/bin/env python3
"""Aggregate single-host coordinator EP7 decode-DBO results against static EP7.

The static side may be either a fresh matrix output root or the historical
``results_npu_ep7`` baseline. Metrics are read from timing JSONs so the script
does not depend on a particular ``experiment_matrix_summary.csv`` schema.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
CONFIG_RE = re.compile(r"_b(?P<b>\d+)_s(?P<s>\d+)_t(?P<t>\d+)")


def parse_list(value: str) -> list[int]:
    return [int(x) for x in value.split(",") if x]


def parse_configs(value: str, default_batches: list[int], default_seqs: list[int]) -> list[tuple[int, int]]:
    if not value:
        return [(b, s) for s in default_seqs for b in default_batches]
    out: list[tuple[int, int]] = []
    for item in value.split(","):
        if not item:
            continue
        b_s = item.replace("x", ":").split(":")
        if len(b_s) != 2:
            raise ValueError(f"Invalid config '{item}', expected B:S")
        out.append((int(b_s[0]), int(b_s[1])))
    return out


def load_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def find_timing(root: Path, mode: str, b: int, s: int, t: int, role: str = "attention") -> Path | None:
    decode_dir = root / mode
    if not decode_dir.is_dir():
        return None
    prefix = "timing_attention_" if role == "attention" else "timing_ffn_"
    patterns = [
        f"{prefix}{mode}_*_b{b}_s{s}_t{t}.json",
        f"{prefix}coordinator_{mode}_*_b{b}_s{s}_t{t}.json",
    ]
    matches: list[Path] = []
    for pattern in patterns:
        matches.extend(sorted(decode_dir.glob(pattern)))
    return matches[0] if matches else None


def timing_row(root: Path, label: str, b: int, s: int, t: int) -> dict[str, Any]:
    attn_path = find_timing(root, "decode-dbo", b, s, t, "attention")
    ffn_path = find_timing(root, "decode-dbo", b, s, t, "ffn")
    attn = load_json(attn_path)
    ffn = load_json(ffn_path)
    tpot = attn.get("decode_tpot_ms")
    throughput = (1000.0 * b / float(tpot)) if tpot else None
    return {
        f"{label}_status": "ok" if attn else "missing",
        f"{label}_attn_timing": str(attn_path) if attn_path else "",
        f"{label}_ffn_timing": str(ffn_path) if ffn_path else "",
        f"{label}_tpot_ms": tpot,
        f"{label}_throughput_tok_s": throughput,
        f"{label}_prefill_ms": attn.get("prefill_ms"),
        f"{label}_decode_loop_ms": attn.get("decode_loop_ms"),
        f"{label}_tbt_mean_ms": attn.get("tbt_mean_ms"),
        f"{label}_tbt_p50_ms": attn.get("tbt_p50_ms"),
        f"{label}_tbt_p99_ms": attn.get("tbt_p99_ms"),
        f"{label}_routing_backend": attn.get("routing_backend") or "",
        f"{label}_routing_table_version": attn.get("routing_table_version") or "",
        f"{label}_routing_update_mode": attn.get("routing_update_mode") or "",
        f"{label}_routing_poll_count": attn.get("routing_poll_count") or "",
        f"{label}_total_ep_dispatch_ms": ffn.get("total_ep_dispatch_ms"),
        f"{label}_total_ep_local_experts_ms": ffn.get("total_ep_local_experts_ms"),
        f"{label}_total_ep_reduce_ms": ffn.get("total_ep_reduce_ms"),
        f"{label}_total_ep_overlap_hidden_ms": ffn.get("total_ep_overlap_hidden_ms"),
    }


def build_rows(coord_root: Path, static_root: Path, batches: list[int], seqs: list[int], tokens: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for s in seqs:
        for b in batches:
            row: dict[str, Any] = {"batch": b, "seq": s, "tokens": tokens}
            row.update(timing_row(static_root, "static", b, s, tokens))
            row.update(timing_row(coord_root, "coord", b, s, tokens))
            st = row.get("static_tpot_ms")
            ct = row.get("coord_tpot_ms")
            if st and ct:
                row["coord_vs_static_tpot_ratio"] = float(ct) / float(st)
                row["coord_vs_static_tpot_delta_pct"] = (float(ct) / float(st) - 1.0) * 100.0
            else:
                row["coord_vs_static_tpot_ratio"] = None
                row["coord_vs_static_tpot_delta_pct"] = None
            sth = row.get("static_throughput_tok_s")
            cth = row.get("coord_throughput_tok_s")
            if sth and cth:
                row["coord_vs_static_throughput_ratio"] = float(cth) / float(sth)
                row["coord_vs_static_throughput_delta_pct"] = (float(cth) / float(sth) - 1.0) * 100.0
            else:
                row["coord_vs_static_throughput_ratio"] = None
                row["coord_vs_static_throughput_delta_pct"] = None
            rows.append(row)
    return rows


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.6f}"
    return str(value)


def write_csv(rows: list[dict[str, Any]], out: Path) -> None:
    fields = [
        "batch", "seq", "tokens",
        "static_status", "coord_status",
        "static_tpot_ms", "coord_tpot_ms", "coord_vs_static_tpot_ratio", "coord_vs_static_tpot_delta_pct",
        "static_throughput_tok_s", "coord_throughput_tok_s",
        "coord_vs_static_throughput_ratio", "coord_vs_static_throughput_delta_pct",
        "static_prefill_ms", "coord_prefill_ms",
        "static_decode_loop_ms", "coord_decode_loop_ms",
        "static_tbt_mean_ms", "coord_tbt_mean_ms",
        "static_tbt_p50_ms", "coord_tbt_p50_ms",
        "static_tbt_p99_ms", "coord_tbt_p99_ms",
        "coord_routing_backend", "coord_routing_update_mode", "coord_routing_table_version",
        "coord_routing_poll_count",
        "static_total_ep_dispatch_ms", "coord_total_ep_dispatch_ms",
        "static_total_ep_local_experts_ms", "coord_total_ep_local_experts_ms",
        "static_total_ep_reduce_ms", "coord_total_ep_reduce_ms",
        "static_total_ep_overlap_hidden_ms", "coord_total_ep_overlap_hidden_ms",
        "static_attn_timing", "static_ffn_timing", "coord_attn_timing", "coord_ffn_timing",
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: fmt(row.get(k)) for k in fields})
    print(f"Wrote {out}")


def make_heatmap(rows: list[dict[str, Any]], field: str, title: str, label: str, out: Path, cmap: str = "RdYlGn") -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib unavailable; skipping figures")
        return
    data = {(int(r["batch"]), int(r["seq"])): r.get(field) for r in rows if r.get(field) is not None}
    if not data:
        print(f"No data for {field}; skipping {out}")
        return
    batches = sorted({b for b, _ in data})
    seqs = sorted({s for _, s in data})
    z = np.full((len(seqs), len(batches)), np.nan)
    for i, s in enumerate(seqs):
        for j, b in enumerate(batches):
            if (b, s) in data:
                z[i, j] = float(data[(b, s)])
    fig, ax = plt.subplots(figsize=(10, 4.8))
    im = ax.imshow(z, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(batches)))
    ax.set_xticklabels(batches)
    ax.set_yticks(range(len(seqs)))
    ax.set_yticklabels(seqs)
    ax.set_xlabel("Batch")
    ax.set_ylabel("Prefill seq len")
    ax.set_title(title)
    for i, _s in enumerate(seqs):
        for j, _b in enumerate(batches):
            if np.isnan(z[i, j]):
                ax.text(j, i, "NA", ha="center", va="center", color="gray", fontsize=8)
            else:
                ax.text(j, i, f"{z[i, j]:.2f}", ha="center", va="center", color="black", fontsize=8)
    plt.colorbar(im, ax=ax, label=label)
    plt.tight_layout()
    plt.savefig(out, dpi=130)
    plt.close()
    print(f"Wrote {out}")


def make_s512_lines(rows: list[dict[str, Any]], out_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    s512 = sorted([r for r in rows if int(r["seq"]) == 512 and r.get("coord_tpot_ms") and r.get("static_tpot_ms")],
                  key=lambda r: int(r["batch"]))
    if not s512:
        print("No complete seq=512 rows; skipping line plots")
        return
    batches = [int(r["batch"]) for r in s512]
    series = [
        ("static", [float(r["static_throughput_tok_s"]) for r in s512], "Throughput (tok/s)",
         out_dir / "static_vs_coord_throughput_vs_batch_s512.png"),
        ("static", [float(r["static_tpot_ms"]) for r in s512], "TPOT (ms)",
         out_dir / "static_vs_coord_tpot_vs_batch_s512.png"),
    ]
    coord_throughput = [float(r["coord_throughput_tok_s"]) for r in s512]
    coord_tpot = [float(r["coord_tpot_ms"]) for r in s512]
    for label, static_values, ylabel, out in series:
        coord_values = coord_throughput if "Throughput" in ylabel else coord_tpot
        fig, ax = plt.subplots(figsize=(8, 4.8))
        ax.plot(batches, static_values, marker="o", label="static EP7")
        ax.plot(batches, coord_values, marker="s", label="coordinator EP7")
        ax.set_xscale("log", base=2)
        ax.set_xticks(batches)
        ax.set_xticklabels(batches)
        ax.set_xlabel("Batch (log2)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Static vs Coordinator EP7 — {ylabel}, seq=512, t=20")
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        plt.savefig(out, dpi=130)
        plt.close()
        print(f"Wrote {out}")


def build_validation_rows(static_root: Path, current_static_root: Path, configs: list[tuple[int, int]], tokens: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for b, s in configs:
        old = timing_row(static_root, "historical", b, s, tokens)
        cur = timing_row(current_static_root, "current", b, s, tokens)
        row: dict[str, Any] = {"batch": b, "seq": s, "tokens": tokens, **old, **cur}
        old_tpot = row.get("historical_tpot_ms")
        cur_tpot = row.get("current_tpot_ms")
        if old_tpot and cur_tpot:
            row["current_vs_historical_tpot_delta_pct"] = (float(cur_tpot) / float(old_tpot) - 1.0) * 100.0
        else:
            row["current_vs_historical_tpot_delta_pct"] = None
        rows.append(row)
    return rows


def write_validation_csv(rows: list[dict[str, Any]], out: Path) -> None:
    fields = [
        "batch", "seq", "tokens", "historical_status", "current_status",
        "historical_tpot_ms", "current_tpot_ms", "current_vs_historical_tpot_delta_pct",
        "historical_attn_timing", "current_attn_timing",
    ]
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: fmt(row.get(k)) for k in fields})
    print(f"Wrote {out}")


def write_markdown(
    rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    static_root: Path,
    coord_root: Path,
    out: Path,
) -> None:
    complete = [r for r in rows if r.get("static_tpot_ms") and r.get("coord_tpot_ms")]
    lines = [
        "# Single-host EP7 coordinator vs static summary",
        "",
        f"- Static baseline root: `{static_root}`",
        f"- Coordinator root: `{coord_root}`",
        "- Metrics are read directly from `decode-dbo/timing_attention_*.json` and paired FFN timing JSONs.",
        "- Throughput is computed as `1000 * batch / decode_tpot_ms`.",
        "",
        "## Source data",
        "",
        f"- Static summary/table source: `{static_root / 'experiment_matrix_summary.csv'}`",
        f"- Static timing source: `{static_root / 'decode-dbo'}/timing_attention_*.json`",
        f"- Coordinator timing source: `{coord_root / 'decode-dbo'}/timing_attention_*.json`",
        f"- Aggregated CSV: `{out.with_name('coord_vs_static_summary.csv')}`",
        "",
    ]
    if validation_rows:
        ok = [
            r for r in validation_rows
            if r.get("current_vs_historical_tpot_delta_pct") is not None
            and abs(float(r["current_vs_historical_tpot_delta_pct"])) <= 5.0
        ]
        lines += [
            "## Static reuse validation",
            "",
            f"- Representative current-code static runs within ±5%: {len(ok)}/{len(validation_rows)}.",
            f"- Validation CSV: `{out.with_name('static_reuse_validation.csv')}`",
            "",
            "| batch | seq | historical TPOT ms | current TPOT ms | delta % |",
            "|---:|---:|---:|---:|---:|",
        ]
        for r in validation_rows:
            lines.append(
                f"| {r['batch']} | {r['seq']} | {fmt(r.get('historical_tpot_ms'))} | "
                f"{fmt(r.get('current_tpot_ms'))} | {fmt(r.get('current_vs_historical_tpot_delta_pct'))} |"
            )
        lines.append("")
    lines += [
        "## Coordinator vs static TPOT / throughput",
        "",
        "| batch | seq | static TPOT ms | coord TPOT ms | coord TPOT delta % | static tok/s | coord tok/s | coord throughput delta % |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in complete:
        lines.append(
            f"| {r['batch']} | {r['seq']} | {fmt(r.get('static_tpot_ms'))} | {fmt(r.get('coord_tpot_ms'))} | "
            f"{fmt(r.get('coord_vs_static_tpot_delta_pct'))} | {fmt(r.get('static_throughput_tok_s'))} | "
            f"{fmt(r.get('coord_throughput_tok_s'))} | {fmt(r.get('coord_vs_static_throughput_delta_pct'))} |"
        )
    lines += [
        "",
        "## Figure outputs",
        "",
        f"- Throughput ratio heatmap: `{out.with_name('coord_vs_static_throughput_ratio_heatmap.png')}`",
        f"- TPOT delta heatmap: `{out.with_name('coord_vs_static_tpot_delta_heatmap.png')}`",
        f"- seq=512 throughput line plot: `{out.with_name('static_vs_coord_throughput_vs_batch_s512.png')}`",
        f"- seq=512 TPOT line plot: `{out.with_name('static_vs_coord_tpot_vs_batch_s512.png')}`",
        "",
        "## Notes",
        "",
        "- Current coordinator matrix uses one-shot routing; a uniform routing table is expected to be close to static round-robin EP ownership.",
        "- Any performance gain from load-aware routing requires a later poll/dynamic-routing experiment.",
    ]
    out.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coord-root", default="results_npu/coordinator_arch/singlehost_ep7/coordinator")
    parser.add_argument("--static-root", default="results_npu_ep7")
    parser.add_argument("--current-static-root", default="",
                        help="Optional current-code static root for historical baseline drift validation")
    parser.add_argument("--out-dir", default="results_npu/coordinator_arch/singlehost_ep7")
    parser.add_argument("--batches", default="2,4,8,16,32,64,128,256")
    parser.add_argument("--seqs", default="128,256,512")
    parser.add_argument("--validation-configs", default="",
                        help="Comma-separated B:S pairs for current static reuse validation, e.g. 8:128,32:512")
    parser.add_argument("--tokens", type=int, default=20)
    args = parser.parse_args()

    coord_root = (ROOT / args.coord_root).resolve() if not Path(args.coord_root).is_absolute() else Path(args.coord_root)
    static_root = (ROOT / args.static_root).resolve() if not Path(args.static_root).is_absolute() else Path(args.static_root)
    out_dir = (ROOT / args.out_dir).resolve() if not Path(args.out_dir).is_absolute() else Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    batches = parse_list(args.batches)
    seqs = parse_list(args.seqs)

    rows = build_rows(coord_root, static_root, batches, seqs, args.tokens)
    write_csv(rows, out_dir / "coord_vs_static_summary.csv")
    make_heatmap(
        rows,
        "coord_vs_static_throughput_ratio",
        "Coordinator / static throughput ratio (EP7 decode-DBO)",
        "ratio",
        out_dir / "coord_vs_static_throughput_ratio_heatmap.png",
    )
    make_heatmap(
        rows,
        "coord_vs_static_tpot_delta_pct",
        "Coordinator TPOT delta vs static (%, lower is better)",
        "%",
        out_dir / "coord_vs_static_tpot_delta_heatmap.png",
        cmap="RdYlGn_r",
    )
    make_s512_lines(rows, out_dir)

    validation_rows: list[dict[str, Any]] = []
    if args.current_static_root:
        current_static_root = (
            (ROOT / args.current_static_root).resolve()
            if not Path(args.current_static_root).is_absolute()
            else Path(args.current_static_root)
        )
        validation_configs = parse_configs(args.validation_configs, batches, seqs)
        validation_rows = build_validation_rows(static_root, current_static_root, validation_configs, args.tokens)
        write_validation_csv(validation_rows, out_dir / "static_reuse_validation.csv")

    write_markdown(rows, validation_rows, static_root, coord_root, out_dir / "comparison_summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
