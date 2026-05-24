#!/usr/bin/env python3
"""Analyze decode-dbo layer-0 warmup artifacts and optionally redraw without L0.

The script scans decode-dbo timing JSONs under a result root, compares layer-0
micro-batch-0 durations with later-layer medians, and writes a CSV/Markdown
summary. With --write-no-l0-figs it also regenerates decode pipeline figures
starting from L1 into a sibling directory of pipeline_figs.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
VIS = ROOT / "scripts" / "visualize_dbo_pipeline.py"
CONFIG_RE = re.compile(r"_b(?P<b>\d+)_s(?P<s>\d+)_t(?P<t>\d+)")


def _load_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def _mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def _fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.6f}"


def _event_values(data: dict, event_type: str, layer: int | None = None,
                  mb: int | None = None, min_layer: int | None = None) -> list[float]:
    out = []
    for ev in data.get("events", []):
        if ev.get("type") != event_type:
            continue
        if layer is not None and ev.get("layer") != layer:
            continue
        if min_layer is not None and ev.get("layer", -1) < min_layer:
            continue
        if mb is not None and ev.get("mb") != mb:
            continue
        out.append(float(ev.get("duration_ms", 0.0)))
    return out


def _ratio(numer: float | None, denom: float | None) -> float | None:
    if numer is None or denom is None or denom <= 0:
        return None
    return numer / denom


def _matching_ffn(attn_path: Path) -> Path | None:
    candidates = [
        attn_path.with_name(attn_path.name.replace("timing_attention_", "timing_ffn_coordinator_")),
        attn_path.with_name(attn_path.name.replace("timing_attention_", "timing_ffn_")),
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def analyze_one(attn_path: Path) -> dict:
    data = _load_json(attn_path)
    m = CONFIG_RE.search(attn_path.name)
    b = int(m["b"]) if m else None
    s = int(m["s"]) if m else None
    t = int(m["t"]) if m else None
    tail_min_layer = 4

    row: dict[str, object] = {
        "file": str(attn_path),
        "batch": b,
        "seq": s,
        "tokens": t,
        "decode_tpot_ms": data.get("decode_tpot_ms"),
        "prefill_ms": data.get("prefill_ms"),
        "total_time_ms": data.get("total_time_ms"),
    }

    for event_type in ("attn_compute", "send_transfer", "recv_wait"):
        l0_mb0 = _mean(_event_values(data, event_type, layer=0, mb=0))
        l0_all = _mean(_event_values(data, event_type, layer=0))
        tail_mb0 = _median(_event_values(data, event_type, mb=0, min_layer=tail_min_layer))
        tail_all = _median(_event_values(data, event_type, min_layer=tail_min_layer))
        row[f"{event_type}_l0_mb0_ms"] = l0_mb0
        row[f"{event_type}_l0_all_ms"] = l0_all
        row[f"{event_type}_tail_mb0_median_ms"] = tail_mb0
        row[f"{event_type}_tail_all_median_ms"] = tail_all
        row[f"{event_type}_l0_mb0_ratio"] = _ratio(l0_mb0, tail_mb0)
        row[f"{event_type}_l0_all_ratio"] = _ratio(l0_all, tail_all)

    ratios = [
        row.get("attn_compute_l0_mb0_ratio"),
        row.get("send_transfer_l0_mb0_ratio"),
        row.get("recv_wait_l0_mb0_ratio"),
    ]
    row["max_l0_mb0_ratio"] = max((float(r) for r in ratios if r is not None), default=None)
    row["likely_l0_cold_start"] = bool(row["max_l0_mb0_ratio"] and row["max_l0_mb0_ratio"] >= 5.0)
    return row


def write_csv(rows: list[dict], out: Path) -> None:
    fields = [
        "batch", "seq", "tokens", "decode_tpot_ms", "prefill_ms", "total_time_ms",
        "attn_compute_l0_mb0_ms", "attn_compute_tail_mb0_median_ms", "attn_compute_l0_mb0_ratio",
        "send_transfer_l0_mb0_ms", "send_transfer_tail_mb0_median_ms", "send_transfer_l0_mb0_ratio",
        "recv_wait_l0_mb0_ms", "recv_wait_tail_mb0_median_ms", "recv_wait_l0_mb0_ratio",
        "max_l0_mb0_ratio", "likely_l0_cold_start", "file",
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _fmt(row.get(k)) if isinstance(row.get(k), float) else row.get(k, "")
                             for k in fields})
    print(f"Wrote {out}")


def write_markdown(rows: list[dict], out: Path, scope: Path) -> None:
    rows_sorted = sorted(
        rows,
        key=lambda r: (float(r.get("max_l0_mb0_ratio") or 0), int(r.get("batch") or 0)),
        reverse=True,
    )
    lines = [
        "# Decode-DBO L0 warmup analysis",
        "",
        f"Scope: `{scope}/timing_attention_*.json`.",
        "",
        "Warmup interpretation:",
        "",
        "- `--warmup-p2p` warms distributed P2P communication paths only.",
        "- `--prefill-warmup-rounds` runs untimed prefill forward passes and absorbs prefill-shape JIT/graph compile cost.",
        "- Decode L0 can still be slow if the first timed decode layer/micro-batch pays decode-path lazy init that neither warmup covers.",
        "",
        "| batch | seq | tpot ms | attn L0 mb0/tail | send L0 mb0/tail | recv L0 mb0/tail | max ratio | cold-start? |",
        "|---:|---:|---:|---:|---:|---:|---:|:---|",
    ]
    for row in rows_sorted[:30]:
        lines.append(
            f"| {row.get('batch')} | {row.get('seq')} | {_fmt(row.get('decode_tpot_ms'))} | "
            f"{_fmt(row.get('attn_compute_l0_mb0_ratio'))} | "
            f"{_fmt(row.get('send_transfer_l0_mb0_ratio'))} | "
            f"{_fmt(row.get('recv_wait_l0_mb0_ratio'))} | "
            f"{_fmt(row.get('max_l0_mb0_ratio'))} | {row.get('likely_l0_cold_start')} |"
        )
    out.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out}")


def write_no_l0_figs(rows: list[dict], root: Path, out_dir: Path, num_layers: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    ok = failed = 0
    for row in sorted(rows, key=lambda r: (int(r.get("seq") or 0), int(r.get("batch") or 0))):
        attn = Path(str(row["file"]))
        ffn = _matching_ffn(attn)
        if ffn is None:
            print(f"missing FFN timing for {attn.name}")
            failed += 1
            continue
        tag = attn.name.removeprefix("timing_attention_").removesuffix(".json")
        safe_tag = tag.replace("-", "_")
        out = out_dir / f"{safe_tag}_no_l0.png"
        serial = root / "serial" / "cache" / f"b{row['batch']}_s{row['seq']}_t{row['tokens']}.json"
        cmd = [
            sys.executable, str(VIS),
            "--attn-timing", str(attn),
            "--ffn-timing", str(ffn),
            "--output", str(out),
            "--start-layer", "1",
            "--num-layers", str(num_layers),
            "--mode", "decode",
            "--ffn-view", "fourlane",
            "--no-auto-skip-warmup",
        ]
        if serial.exists():
            cmd += ["--serial-timing", str(serial)]
        result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        if result.returncode == 0:
            ok += 1
        else:
            failed += 1
            err = (result.stderr or result.stdout).strip().splitlines()
            print(f"failed {out.name}: {err[-1] if err else 'unknown error'}")
    print(f"Wrote no-L0 figures: {ok} ok, {failed} failed -> {out_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(ROOT / "results_npu_ep7"))
    parser.add_argument("--write-no-l0-figs", action="store_true")
    parser.add_argument("--num-layers", type=int, default=3,
                        help="Number of layers to draw starting at L1 for no-L0 figures")
    args = parser.parse_args()

    root = Path(args.root)
    decode_dir = root / "decode-dbo"
    rows = [analyze_one(path) for path in sorted(decode_dir.glob("timing_attention_*.json"))]
    if not rows:
        print(f"No decode-dbo timing files found under {decode_dir}", file=sys.stderr)
        return 1
    write_csv(rows, root / "decode_dbo_l0_warmup_analysis.csv")
    write_markdown(rows, root / "decode_dbo_l0_warmup_analysis.md", decode_dir)
    if args.write_no_l0_figs:
        write_no_l0_figs(rows, root, root / "pipeline_figs_no_l0", args.num_layers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
