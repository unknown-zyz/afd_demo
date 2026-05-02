#!/usr/bin/env python3
"""Plot NPUGraph FFN probe outputs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

try:
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover
    raise SystemExit("matplotlib is required to plot graph probe results") from exc


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def label(row: dict[str, str]) -> str:
    return f"{row['probe']}\n{row['shape_label']} {row['batch']}x{row['seq']}"


def ok_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in rows if row.get("status") == "ok" and row.get("graph_median_ms")]


def plot_latency(rows: list[dict[str, str]], out_path: Path) -> None:
    rows = ok_rows(rows)
    labels = [label(row) for row in rows]
    eager = [float(row["eager_median_ms"]) for row in rows]
    graph = [float(row["graph_median_ms"]) for row in rows]
    x = range(len(rows))

    fig, ax = plt.subplots(figsize=(max(10, len(rows) * 1.1), 5))
    ax.bar([pos - 0.18 for pos in x], eager, width=0.36, label="eager")
    ax.bar([pos + 0.18 for pos in x], graph, width=0.36, label="graph replay")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Median latency (ms)")
    ax.set_title("Qwen3 FFN NPUGraph latency probe")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_speedup(rows: list[dict[str, str]], out_path: Path) -> None:
    rows = ok_rows(rows)
    labels = [label(row) for row in rows]
    speedup = [float(row["speedup"]) for row in rows]
    fig, ax = plt.subplots(figsize=(max(10, len(rows) * 1.1), 4.5))
    x = range(len(rows))
    ax.bar(x, speedup)
    ax.axhline(1.0, color="black", linewidth=1, linestyle="--")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Eager / graph")
    ax.set_title("Qwen3 FFN NPUGraph speedup")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="results_npu/moe_backend_probe/graph_probe/decode_graph_probe.csv")
    parser.add_argument("--output-dir", default="results_npu/moe_backend_probe/graph_probe")
    args = parser.parse_args()

    rows = load_rows(Path(args.csv))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_latency(rows, out_dir / "graph_latency_comparison.png")
    plot_speedup(rows, out_dir / "graph_speedup.png")
    print(f"Wrote plots to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
