#!/usr/bin/env python3
"""Aggregate results_npu_ep7 (serial + decode-dbo + prefill-dbo + decode-dbo-crosslayer)
into experiment_matrix_summary.csv plus heatmap and crosslayer comparison figures.

This script supersedes scripts/aggregate_full_matrix_v2.py for the npu-ep7 dataset.
The DBO reports under results_npu_ep7 do NOT embed serial-baseline comparisons
(except crosslayer), so speedups are computed by joining DBO timing JSONs against
the cached serial JSONs in results_npu_ep7/serial/.
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "results_npu_ep7"
SERIAL_DIR = ROOT / "serial"
OUT_CSV = ROOT / "experiment_matrix_summary.csv"

CONFIG_RE = re.compile(r"b(?P<b>\d+)_s(?P<s>\d+)_t(?P<t>\d+)")
DBO_REPORT_RE = re.compile(
    r"report_(?P<mode>decode-dbo|prefill-dbo|decode-dbo-crosslayer)_npu_ep7_"
    r"broadcast_reduce_overlap_b(?P<b>\d+)_s(?P<s>\d+)_t(?P<t>\d+)\.md"
)


def load_serial_cache() -> dict[tuple[int, int, int], dict]:
    out: dict[tuple[int, int, int], dict] = {}
    for jf in SERIAL_DIR.glob("b*_s*_t*.json"):
        m = CONFIG_RE.search(jf.name)
        if not m:
            continue
        try:
            d = json.loads(jf.read_text())
        except Exception:
            continue
        out[(int(m["b"]), int(m["s"]), int(m["t"]))] = d
    return out


def load_dbo_timing(mode_dir: Path, mode: str, b: int, s: int, t: int) -> dict | None:
    p = mode_dir / (
        f"timing_attention_{mode}_npu_ep7_broadcast_reduce_overlap_"
        f"b{b}_s{s}_t{t}.json"
    )
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def build_rows():
    serial = load_serial_cache()
    rows: list[dict] = []

    # Serial rows
    for (b, s, t), d in sorted(serial.items()):
        rows.append({
            "mode": "serial",
            "batch": b, "seq": s, "tokens": t,
            "status": "cached",
            "tpot_ms": d.get("decode_tpot_ms"),
            "serial_tpot_ms": d.get("decode_tpot_ms"),
            "speedup": "",
            "prefill_ms": d.get("prefill_ms"),
            "serial_prefill_ms": d.get("prefill_ms"),
            "prefill_speedup": "",
            "total_time_ms": d.get("total_time_ms"),
            "tbt_mean_ms": d.get("tbt_mean_ms"),
            "tbt_p50_ms": d.get("tbt_p50_ms"),
            "tbt_p99_ms": d.get("tbt_p99_ms"),
            "correctness_tokens": d.get("correctness_tokens") or "",
        })

    # DBO modes
    for mode in ("decode-dbo", "prefill-dbo", "decode-dbo-crosslayer"):
        mode_dir = ROOT / mode
        if not mode_dir.exists():
            continue
        for rpath in sorted(mode_dir.glob("report_*.md")):
            m = DBO_REPORT_RE.match(rpath.name)
            if not m:
                continue
            b, s, t = int(m["b"]), int(m["s"]), int(m["t"])
            timing = load_dbo_timing(mode_dir, mode, b, s, t) or {}
            ser = serial.get((b, s, t), {})
            row = {
                "mode": mode,
                "batch": b, "seq": s, "tokens": t,
                "status": "ok",
                "tpot_ms": timing.get("decode_tpot_ms"),
                "serial_tpot_ms": ser.get("decode_tpot_ms"),
                "prefill_ms": timing.get("prefill_ms"),
                "serial_prefill_ms": ser.get("prefill_ms"),
                "total_time_ms": timing.get("total_time_ms"),
                "tbt_mean_ms": timing.get("tbt_mean_ms"),
                "tbt_p50_ms": timing.get("tbt_p50_ms"),
                "tbt_p99_ms": timing.get("tbt_p99_ms"),
                "correctness_tokens": timing.get("correctness_tokens") or "",
            }
            # Decode TPOT speedup (decode-dbo / crosslayer)
            if mode in ("decode-dbo", "decode-dbo-crosslayer"):
                if row["tpot_ms"] and row["serial_tpot_ms"]:
                    row["speedup"] = row["serial_tpot_ms"] / row["tpot_ms"]
                else:
                    row["speedup"] = ""
            else:
                row["speedup"] = ""
            # Prefill TTFT-path speedup (prefill-dbo): serial.prefill_ms / dbo.total_time_ms
            if mode == "prefill-dbo":
                if ser.get("prefill_ms") and timing.get("total_time_ms"):
                    row["prefill_speedup"] = ser["prefill_ms"] / timing["total_time_ms"]
                else:
                    row["prefill_speedup"] = ""
            else:
                row["prefill_speedup"] = ""
            rows.append(row)
    return rows


def write_csv(rows):
    fields = [
        "mode", "batch", "seq", "tokens", "status",
        "tpot_ms", "serial_tpot_ms", "speedup",
        "prefill_ms", "serial_prefill_ms", "prefill_speedup",
        "tbt_mean_ms", "tbt_p50_ms", "tbt_p99_ms",
        "total_time_ms", "correctness_tokens",
    ]
    if OUT_CSV.exists():
        bak = OUT_CSV.with_suffix(".csv.bak")
        if not bak.exists():
            bak.write_bytes(OUT_CSV.read_bytes())

    def fmt(v):
        if v is None or v == "":
            return ""
        if isinstance(v, float):
            return f"{v:.6f}"
        if isinstance(v, list):
            return ",".join(str(x) for x in v)
        return v

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: fmt(r.get(k, "")) for k in fields})
    print(f"Wrote {OUT_CSV} ({len(rows)} rows)")


def make_figures(rows):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib unavailable, skipping figures")
        return

    by_mode: dict[str, dict[tuple[int, int], float]] = {
        "decode-dbo": {},
        "prefill-dbo": {},
        "decode-dbo-crosslayer": {},
    }
    decode_tpot: dict[tuple[int, int], float] = {}
    serial_tpot: dict[tuple[int, int], float] = {}
    for r in rows:
        key = (r["batch"], r["seq"])
        if r["mode"] == "decode-dbo" and r.get("speedup"):
            by_mode["decode-dbo"][key] = float(r["speedup"])
            if r.get("tpot_ms"):
                decode_tpot[key] = float(r["tpot_ms"])
        if r["mode"] == "prefill-dbo" and r.get("prefill_speedup"):
            by_mode["prefill-dbo"][key] = float(r["prefill_speedup"])
        if r["mode"] == "decode-dbo-crosslayer" and r.get("speedup"):
            by_mode["decode-dbo-crosslayer"][key] = float(r["speedup"])
        if r["mode"] == "serial" and r.get("serial_tpot_ms"):
            serial_tpot[key] = float(r["serial_tpot_ms"])

    def heatmap(data, title, out, vmin, vmax, fmt="{:.2f}"):
        if not data:
            return
        batches = sorted({k[0] for k in data})
        seqs = sorted({k[1] for k in data})
        Z = np.full((len(seqs), len(batches)), np.nan)
        for i, s in enumerate(seqs):
            for j, b in enumerate(batches):
                if (b, s) in data:
                    Z[i, j] = data[(b, s)]
        fig, ax = plt.subplots(figsize=(10, 5))
        im = ax.imshow(Z, cmap="RdYlGn", vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(batches)))
        ax.set_xticklabels(batches)
        ax.set_yticks(range(len(seqs)))
        ax.set_yticklabels(seqs)
        ax.set_xlabel("Batch")
        ax.set_ylabel("Prefill seq len")
        ax.set_title(title)
        for i in range(len(seqs)):
            for j in range(len(batches)):
                if not np.isnan(Z[i, j]):
                    color = "black" if 0.9 < Z[i, j] < 1.6 else "white"
                    ax.text(j, i, fmt.format(Z[i, j]),
                            ha="center", va="center", color=color, fontsize=9)
                else:
                    ax.text(j, i, "OOM", ha="center", va="center",
                            color="gray", fontsize=8)
        plt.colorbar(im, ax=ax, label="speedup ×")
        plt.tight_layout()
        plt.savefig(out, dpi=120)
        plt.close()
        print(f"Wrote {out}")

    heatmap(by_mode["decode-dbo"],
            "Decode-DBO TPOT speedup vs serial (npu-ep7, t=20)",
            ROOT / "decode_dbo_speedup_heatmap.png", 0.5, 2.5)
    heatmap(by_mode["prefill-dbo"],
            "Prefill-DBO TTFT speedup vs serial (npu-ep7, t=20)",
            ROOT / "prefill_dbo_speedup_heatmap.png", 0.5, 2.0)
    heatmap(by_mode["decode-dbo-crosslayer"],
            "Decode-DBO-crosslayer TPOT speedup vs serial (npu-ep7, t=20)",
            ROOT / "decode_dbo_crosslayer_speedup_heatmap.png", 0.5, 2.5)

    # TPOT heatmap (raw decode-dbo TPOT in ms)
    if decode_tpot:
        batches = sorted({k[0] for k in decode_tpot})
        seqs = sorted({k[1] for k in decode_tpot})
        Z = np.full((len(seqs), len(batches)), np.nan)
        for i, s in enumerate(seqs):
            for j, b in enumerate(batches):
                if (b, s) in decode_tpot:
                    Z[i, j] = decode_tpot[(b, s)]
        fig, ax = plt.subplots(figsize=(10, 5))
        im = ax.imshow(Z, cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(batches))); ax.set_xticklabels(batches)
        ax.set_yticks(range(len(seqs))); ax.set_yticklabels(seqs)
        ax.set_xlabel("Batch"); ax.set_ylabel("Prefill seq len")
        ax.set_title("Decode-DBO TPOT (ms) — npu-ep7, t=20")
        for i in range(len(seqs)):
            for j in range(len(batches)):
                if not np.isnan(Z[i, j]):
                    ax.text(j, i, f"{Z[i,j]:.0f}", ha="center", va="center",
                            color="white", fontsize=8)
        plt.colorbar(im, ax=ax, label="TPOT ms")
        plt.tight_layout()
        plt.savefig(ROOT / "decode_dbo_tpot_heatmap.png", dpi=120)
        plt.close()
        print(f"Wrote {ROOT / 'decode_dbo_tpot_heatmap.png'}")

    # Crosslayer comparison: line plot per batch, x=seq, y=speedup
    base = by_mode["decode-dbo"]
    cross = by_mode["decode-dbo-crosslayer"]
    common_batches = sorted({k[0] for k in base} | {k[0] for k in cross})
    if common_batches:
        fig, ax = plt.subplots(figsize=(10, 6))
        cmap = plt.get_cmap("tab10")
        for idx, b in enumerate(common_batches):
            base_pts = sorted([(s, v) for (bb, s), v in base.items() if bb == b])
            cross_pts = sorted([(s, v) for (bb, s), v in cross.items() if bb == b])
            color = cmap(idx % 10)
            if base_pts:
                xs, ys = zip(*base_pts)
                ax.plot(xs, ys, marker="o", linestyle="-",
                        color=color, label=f"b={b} decode-dbo")
            if cross_pts:
                xs, ys = zip(*cross_pts)
                ax.plot(xs, ys, marker="s", linestyle="--",
                        color=color, label=f"b={b} crosslayer")
        ax.axhline(1.0, color="k", linestyle=":", alpha=0.5)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Prefill seq len")
        ax.set_ylabel("TPOT speedup ×")
        ax.set_title("Decode-DBO vs Decode-DBO-crosslayer TPOT speedup (npu-ep7)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, ncol=2, loc="best")
        plt.tight_layout()
        out = ROOT / "crosslayer_comparison.png"
        plt.savefig(out, dpi=120)
        plt.close()
        print(f"Wrote {out}")


def main():
    rows = build_rows()
    write_csv(rows)
    make_figures(rows)


if __name__ == "__main__":
    main()
