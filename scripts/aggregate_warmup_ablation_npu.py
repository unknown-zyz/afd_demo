#!/usr/bin/env python3
"""Aggregate NPU warmup-ablation timing directories.

Expected layout:
  results_npu_ep7/warmup_ablation/{both_on,p2p_only,prefill_only,both_off}/decode-dbo/
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
CONFIG_RE = re.compile(r"_b(?P<b>\d+)_s(?P<s>\d+)_t(?P<t>\d+)")
VARIANT_ORDER = ["both_on", "p2p_only", "prefill_only", "both_off"]


def _load(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def _durations(data: dict, event_type: str, *, layer: int | None = None,
               mb: int | None = None, min_layer: int | None = None) -> list[float]:
    values = []
    for ev in data.get("events", []):
        if ev.get("type") != event_type:
            continue
        if layer is not None and ev.get("layer") != layer:
            continue
        if mb is not None and ev.get("mb") != mb:
            continue
        if min_layer is not None and ev.get("layer", -1) < min_layer:
            continue
        values.append(float(ev.get("duration_ms", 0.0)))
    return values


def _mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def _ratio(numer: float | None, denom: float | None) -> float | None:
    if numer is None or denom is None or denom <= 0:
        return None
    return numer / denom


def _fmt(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return value


def variant_flags(variant: str) -> tuple[bool | None, int | None]:
    return {
        "both_on": (True, 1),
        "p2p_only": (True, 0),
        "prefill_only": (False, 1),
        "both_off": (False, 0),
    }.get(variant, (None, None))


def collect_rows(root: Path) -> list[dict]:
    rows: list[dict] = []
    if not root.exists():
        return rows
    for variant_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        variant = variant_dir.name
        p2p, prefill_rounds = variant_flags(variant)
        for mode in ("decode-dbo", "serial"):
            mode_dir = variant_dir / mode
            if not mode_dir.exists():
                continue
            for path in sorted(mode_dir.glob("timing_attention_*.json")):
                match = CONFIG_RE.search(path.name)
                if not match:
                    continue
                data = _load(path)
                b, s, t = int(match["b"]), int(match["s"]), int(match["t"])
                tpot = data.get("decode_tpot_ms")
                l0_attn = _mean(_durations(data, "attn_compute", layer=0, mb=0))
                tail_attn = _median(_durations(data, "attn_compute", mb=0, min_layer=4))
                l0_send = _mean(_durations(data, "send_transfer", layer=0, mb=0))
                tail_send = _median(_durations(data, "send_transfer", mb=0, min_layer=4))
                l0_recv = _mean(_durations(data, "recv_wait", layer=0, mb=0))
                tail_recv = _median(_durations(data, "recv_wait", mb=0, min_layer=4))
                rows.append({
                    "variant": variant,
                    "mode": mode,
                    "warmup_p2p": p2p,
                    "prefill_warmup_rounds": prefill_rounds,
                    "batch": b,
                    "seq": s,
                    "tokens": t,
                    "tpot_ms": tpot,
                    "throughput": (b * 1000.0 / tpot) if tpot else None,
                    "prefill_ms": data.get("prefill_ms"),
                    "total_time_ms": data.get("total_time_ms"),
                    "attn_l0_mb0_ms": l0_attn,
                    "attn_tail_mb0_median_ms": tail_attn,
                    "attn_l0_mb0_ratio": _ratio(l0_attn, tail_attn),
                    "send_l0_mb0_ms": l0_send,
                    "send_tail_mb0_median_ms": tail_send,
                    "send_l0_mb0_ratio": _ratio(l0_send, tail_send),
                    "recv_l0_mb0_ms": l0_recv,
                    "recv_tail_mb0_median_ms": tail_recv,
                    "recv_l0_mb0_ratio": _ratio(l0_recv, tail_recv),
                    "file": str(path),
                })
    return rows


def write_csv(rows: list[dict], out: Path) -> None:
    fields = [
        "variant", "mode", "warmup_p2p", "prefill_warmup_rounds",
        "batch", "seq", "tokens", "tpot_ms", "throughput", "prefill_ms", "total_time_ms",
        "attn_l0_mb0_ms", "attn_tail_mb0_median_ms", "attn_l0_mb0_ratio",
        "send_l0_mb0_ms", "send_tail_mb0_median_ms", "send_l0_mb0_ratio",
        "recv_l0_mb0_ms", "recv_tail_mb0_median_ms", "recv_l0_mb0_ratio",
        "file",
    ]
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _fmt(row.get(field)) for field in fields})
    print(f"Wrote {out}")


def make_s512_plots(rows: list[dict], out_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable; skipping warmup ablation plots")
        return

    decode_rows = [
        r for r in rows
        if r["mode"] == "decode-dbo" and r["seq"] == 512 and r.get("tpot_ms")
    ]
    if not decode_rows:
        print("No seq=512 decode-dbo warmup rows; skipping plots")
        return

    def plot_metric(metric: str, ylabel: str, title: str, out_name: str, marker: str) -> None:
        fig, ax = plt.subplots(figsize=(8, 5))
        for variant in VARIANT_ORDER:
            pts = sorted(
                [(r["batch"], r.get(metric)) for r in decode_rows
                 if r["variant"] == variant and r.get(metric) is not None]
            )
            if not pts:
                continue
            xs, ys = zip(*pts)
            ax.plot(xs, ys, marker=marker, linewidth=2, label=variant)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Batch (log2)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        out = out_dir / out_name
        plt.savefig(out, dpi=120)
        plt.close()
        print(f"Wrote {out}")

    plot_metric(
        "tpot_ms",
        "TPOT (ms)",
        "Warmup ablation: Decode-DBO TPOT vs Batch (seq=512)",
        "warmup_ablation_tpot_vs_batch_s512.png",
        "o",
    )
    plot_metric(
        "throughput",
        "Throughput (token/s)",
        "Warmup ablation: Decode-DBO Throughput vs Batch (seq=512)",
        "warmup_ablation_throughput_vs_batch_s512.png",
        "s",
    )
    plot_metric(
        "attn_l0_mb0_ratio",
        "L0 MB0 attn / later-layer median",
        "Warmup ablation: L0 attention cold-start ratio (seq=512)",
        "warmup_ablation_l0_attn_ratio_vs_batch_s512.png",
        "^",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(ROOT / "results_npu_ep7" / "warmup_ablation"))
    args = parser.parse_args()

    root = Path(args.root)
    rows = collect_rows(root)
    if not rows:
        print(f"No warmup ablation timing JSONs found under {root}")
        return 1
    write_csv(rows, root / "warmup_ablation_summary.csv")
    make_s512_plots(rows, root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
