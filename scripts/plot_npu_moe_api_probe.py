#!/usr/bin/env python3
"""Plot torch_npu MoE API probe outputs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

try:
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover
    raise SystemExit("matplotlib is required to plot API probe results") from exc


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def ok_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in rows if row.get("status") == "ok" and row.get("latency_median_ms")]


def label(row: dict[str, str]) -> str:
    return f"{row['shape_label']} {row['batch']}x{row['seq']}"


def plot_latency(rows: list[dict[str, str]], out_path: Path) -> None:
    rows = ok_rows(rows)
    shapes = sorted({label(row) for row in rows}, key=lambda value: (value.split()[0], int(value.split()[1].split("x")[0])))
    probes = ["hf_experts", "router_topk", "token_permute_roundtrip", "grouped_experts"]
    x = range(len(shapes))
    width = 0.18
    fig, ax = plt.subplots(figsize=(max(9, len(shapes) * 1.5), 5))
    for idx, probe in enumerate(probes):
        values = []
        for shape in shapes:
            match = next((row for row in rows if label(row) == shape and row["probe"] == probe), None)
            values.append(float(match["latency_median_ms"]) if match else 0.0)
        offsets = [pos + (idx - 1.5) * width for pos in x]
        ax.bar(offsets, values, width=width, label=probe)
    ax.set_xticks(list(x))
    ax.set_xticklabels(shapes, rotation=30)
    ax.set_ylabel("Median latency (ms)")
    ax.set_title("torch_npu MoE API latency probe")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_error(rows: list[dict[str, str]], out_path: Path) -> None:
    rows = [
        row for row in ok_rows(rows)
        if row["probe"] in {"router_topk", "token_permute_roundtrip", "grouped_experts"}
    ]
    labels = [f"{row['probe']}\n{label(row)}" for row in rows]
    mean_abs = [float(row.get("mean_abs_error") or 0.0) for row in rows]
    max_abs = [float(row.get("max_abs_error") or 0.0) for row in rows]
    fig, ax = plt.subplots(figsize=(max(10, len(rows) * 0.75), 5))
    x = range(len(rows))
    ax.bar([pos - 0.18 for pos in x], mean_abs, width=0.36, label="mean abs")
    ax.bar([pos + 0.18 for pos in x], max_abs, width=0.36, label="max abs")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=55, ha="right")
    ax.set_ylabel("Error")
    ax.set_title("torch_npu MoE API correctness error")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="results_npu/moe_backend_probe/api_smoke/correctness_summary.csv")
    parser.add_argument("--output-dir", default="results_npu/moe_backend_probe/api_smoke")
    args = parser.parse_args()

    rows = load_rows(Path(args.csv))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_latency(rows, out_dir / "api_latency_comparison.png")
    plot_error(rows, out_dir / "correctness_error.png")
    print(f"Wrote plots to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
