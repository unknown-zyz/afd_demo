#!/usr/bin/env python3
"""Plot NPU MoE microbenchmark CSV outputs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

try:
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - environment dependent
    raise SystemExit("matplotlib is required to plot microbenchmark results") from exc


STAGES = [
    ("norm_median_ms", "Norm"),
    ("router_median_ms", "Router"),
    ("experts_median_ms", "Experts"),
    ("residual_add_median_ms", "Residual add"),
]


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def row_label(row: dict[str, str]) -> str:
    return f"L{row['layer']} {row['shape_label']} {row['batch']}x{row['seq']}"


def plot_total(rows: list[dict[str, str]], out_path: Path) -> None:
    labels = [row_label(row) for row in rows]
    totals = [float(row["total_median_ms"]) for row in rows]
    fig, ax = plt.subplots(figsize=(max(8, len(rows) * 1.3), 4.8))
    ax.bar(labels, totals, color="#4C78A8")
    ax.set_ylabel("Median total FFN latency (ms)")
    ax.set_title("Qwen3 MoE FFN layer latency")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_breakdown(rows: list[dict[str, str]], out_path: Path) -> None:
    labels = [row_label(row) for row in rows]
    bottoms = [0.0] * len(rows)
    fig, ax = plt.subplots(figsize=(max(9, len(rows) * 1.4), 5.2))
    for key, name in STAGES:
        values = [float(row[key]) for row in rows]
        ax.bar(labels, values, bottom=bottoms, label=name)
        bottoms = [bottom + value for bottom, value in zip(bottoms, values)]
    ax.set_ylabel("Median latency (ms)")
    ax.set_title("Qwen3 MoE FFN stage breakdown")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_scaling(rows: list[dict[str, str]], out_path: Path) -> None:
    by_label: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_label.setdefault(row["shape_label"], []).append(row)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    for label, label_rows in sorted(by_label.items()):
        label_rows = sorted(label_rows, key=lambda row: int(row["tokens"]))
        ax.plot(
            [int(row["tokens"]) for row in label_rows],
            [float(row["total_median_ms"]) for row in label_rows],
            marker="o",
            label=label,
        )
    ax.set_xlabel("Tokens per FFN call (batch x seq)")
    ax.set_ylabel("Median total FFN latency (ms)")
    ax.set_title("Qwen3 MoE FFN token scaling")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="results_npu/moe_backend_probe/microbench/moe_layer_baseline.csv")
    parser.add_argument("--output-dir", default="results_npu/moe_backend_probe/microbench")
    args = parser.parse_args()

    rows = load_rows(Path(args.csv))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_total(rows, out_dir / "moe_layer_baseline.png")
    plot_breakdown(rows, out_dir / "moe_stage_breakdown.png")
    plot_scaling(rows, out_dir / "moe_shape_scaling.png")
    print(f"Wrote plots to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
