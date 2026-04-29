#!/usr/bin/env python3
"""Audit whether DBO timing files have mode-matched serial baselines."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from experiment_baselines import infer_mode_from_path, resolve_serial_baseline


ROOT = Path(__file__).resolve().parent.parent
TARGET_SUBDIRS = ("decode-dbo", "decode-dbo-crosslayer", "prefill-dbo")
NAME_RE = re.compile(r"timing_attention_(?P<tag>.+)\.json$")
CONFIG_RE = re.compile(r"b(?P<batch>\d+)_s(?P<seq>\d+)_t(?P<tokens>\d+)")


@dataclass
class Row:
    root: str
    mode: str
    tag: str
    batch: str
    seq: str
    tokens: str
    status: str
    serial_cache: str
    serial_baseline_ms: str
    dbo_ms: str
    speedup: str
    note: str


def _load_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def _fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.6f}"


def audit_root(root: Path) -> list[Row]:
    rows: list[Row] = []
    cache_dir = root / "serial" / "cache"

    for subdir in TARGET_SUBDIRS:
        timing_dir = root / subdir
        if not timing_dir.is_dir():
            continue

        mode = infer_mode_from_path(subdir) or "unknown"
        for attn_path in sorted(timing_dir.glob("timing_attention_*.json")):
            name_match = NAME_RE.match(attn_path.name)
            if not name_match:
                continue
            tag = name_match.group("tag")
            config_match = CONFIG_RE.search(tag)
            batch = seq = tokens = ""
            serial_path = None
            if config_match:
                batch = config_match.group("batch")
                seq = config_match.group("seq")
                tokens = config_match.group("tokens")
                serial_path = cache_dir / f"b{batch}_s{seq}_t{tokens}.json"

            dbo_ms = None
            try:
                attn_data = _load_json(attn_path)
                dbo_key = "decode_tpot_ms" if mode == "decode" else "total_time_ms"
                dbo_ms = float(attn_data.get(dbo_key) or 0)
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                pass

            if serial_path is None:
                rows.append(Row(root.name, mode, tag, batch, seq, tokens, "unparseable-tag", "", "", _fmt(dbo_ms), "", "could not parse b/s/t"))
                continue

            if not serial_path.exists():
                rows.append(Row(root.name, mode, tag, batch, seq, tokens, "serial-cache-missing", str(serial_path), "", _fmt(dbo_ms), "", "matching serial cache not found"))
                continue

            try:
                baseline = resolve_serial_baseline(_load_json(serial_path), mode)
            except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
                rows.append(Row(root.name, mode, tag, batch, seq, tokens, "serial-cache-invalid", str(serial_path), "", _fmt(dbo_ms), "", str(exc)))
                continue

            if not baseline.available:
                rows.append(Row(root.name, mode, tag, batch, seq, tokens, "baseline-missing", str(serial_path), "", _fmt(dbo_ms), "", baseline.warning or "mode-matched baseline missing"))
                continue

            assert baseline.value_ms is not None
            speedup = baseline.value_ms / dbo_ms if dbo_ms and dbo_ms > 0 else None
            status = "ok" if baseline.source in {"prefill_ms", "decode_tpot_ms"} else "baseline-missing"
            rows.append(Row(root.name, mode, tag, batch, seq, tokens, status, str(serial_path), _fmt(baseline.value_ms), _fmt(dbo_ms), _fmt(speedup), baseline.source))

    return rows


def write_csv(rows: list[Row], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(Row.__dataclass_fields__), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def print_summary(rows: list[Row]) -> None:
    counts: dict[tuple[str, str, str], int] = {}
    for row in rows:
        key = (row.root, row.mode, row.status)
        counts[key] = counts.get(key, 0) + 1

    print("| Root | Mode | Status | Count |")
    print("|---|---|---|---:|")
    for (root, mode, status), count in sorted(counts.items()):
        print(f"| {root} | {mode} | {status} | {count} |")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", action="append", default=[],
                        help="Results root to audit. Can be passed more than once. Defaults to results.")
    parser.add_argument("--output-csv", default="", help="Optional CSV path for detailed rows")
    args = parser.parse_args()

    roots = []
    for raw_root in args.root:
        path = Path(raw_root)
        roots.append(path if path.is_absolute() else ROOT / path)
    if not roots:
        roots = [ROOT / "results"]
    rows: list[Row] = []
    for root in roots:
        if root.is_dir():
            rows.extend(audit_root(root))
        else:
            print(f"Warning: results root not found: {root}", file=sys.stderr)

    print_summary(rows)
    if args.output_csv:
        write_csv(rows, Path(args.output_csv))
        print(f"\nWrote detail CSV: {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
