#!/usr/bin/env python3
"""Summarize EP dispatch/reduce payload and effective bandwidth from timing JSONs."""

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


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def relpath(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def infer_config(path: Path) -> tuple[int | None, int | None, int | None]:
    match = CONFIG_RE.search(path.name)
    if match is None:
        return None, None, None
    return int(match["b"]), int(match["s"]), int(match["t"])


def gib(value: int | float | None) -> float | None:
    if value in (None, ""):
        return None
    return float(value) / (1024 ** 3)


def gib_per_s(bytes_value: int | float | None, ms_value: int | float | None) -> float | None:
    if bytes_value in (None, "") or ms_value in (None, "", 0):
        return None
    ms = float(ms_value)
    if ms <= 0:
        return None
    return float(bytes_value) / (1024 ** 3) / (ms / 1000.0)


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.6f}"
    return str(value)


def build_row(path: Path) -> dict[str, Any]:
    data = load_json(path)
    batch, seq, tokens = infer_config(path)
    dispatch_bytes = data.get("total_ep_dispatch_bytes")
    reduce_bytes = data.get("total_ep_reduce_bytes")
    dispatch_ms = data.get("total_ep_dispatch_ms")
    reduce_ms = data.get("total_ep_reduce_ms")
    return {
        "file": relpath(path),
        "batch": batch,
        "seq": seq,
        "tokens": tokens,
        "routing_backend": data.get("routing_backend"),
        "routing_update_mode": data.get("routing_update_mode"),
        "routing_table_version": data.get("routing_table_version"),
        "routing_poll_count": data.get("routing_poll_count"),
        "decode_tpot_ms": data.get("decode_tpot_ms"),
        "compute_ratio": data.get("compute_ratio"),
        "total_ep_dispatch_ms": dispatch_ms,
        "total_ep_dispatch_bytes": dispatch_bytes,
        "total_ep_dispatch_gib": gib(dispatch_bytes),
        "dispatch_effective_gib_s": gib_per_s(dispatch_bytes, dispatch_ms),
        "total_ep_reduce_ms": reduce_ms,
        "total_ep_reduce_bytes": reduce_bytes,
        "total_ep_reduce_gib": gib(reduce_bytes),
        "reduce_effective_gib_s": gib_per_s(reduce_bytes, reduce_ms),
        "total_ep_overlap_hidden_ms": data.get("total_ep_overlap_hidden_ms"),
    }


def write_csv(rows: list[dict[str, Any]], out: Path) -> None:
    fields = [
        "file",
        "batch",
        "seq",
        "tokens",
        "routing_backend",
        "routing_update_mode",
        "routing_table_version",
        "routing_poll_count",
        "decode_tpot_ms",
        "compute_ratio",
        "total_ep_dispatch_ms",
        "total_ep_dispatch_bytes",
        "total_ep_dispatch_gib",
        "dispatch_effective_gib_s",
        "total_ep_reduce_ms",
        "total_ep_reduce_bytes",
        "total_ep_reduce_gib",
        "reduce_effective_gib_s",
        "total_ep_overlap_hidden_ms",
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: fmt(row.get(k)) for k in fields})
    print(f"Wrote {out}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default="",
        help="Result root containing decode-dbo/ and timing_ffn*.json, or a decode-dbo directory itself.",
    )
    parser.add_argument(
        "--glob",
        default="timing_ffn*.json",
        help="Glob under decode-dbo/ (or --root if it already points there). Default: timing_ffn*.json",
    )
    parser.add_argument("--out", default="", help="Optional CSV path. Defaults to <root>/ep_bandwidth_summary.csv")
    args = parser.parse_args()

    if not args.root:
        raise SystemExit("--root is required")

    root = Path(args.root)
    if not root.is_absolute():
        root = (ROOT / root).resolve()
    decode_dir = root if root.name == "decode-dbo" else root / "decode-dbo"
    if not decode_dir.is_dir():
        raise SystemExit(f"decode-dbo directory not found: {decode_dir}")

    rows = [build_row(path) for path in sorted(decode_dir.glob(args.glob))]
    if not rows:
        raise SystemExit(f"No timing files matched {args.glob} under {decode_dir}")

    out = Path(args.out) if args.out else decode_dir.parent / "ep_bandwidth_summary.csv"
    if not out.is_absolute():
        out = (ROOT / out).resolve()
    write_csv(rows, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
