#!/usr/bin/env python3
"""Round-8 Track B: aggregate MB4 results and produce MB2 vs MB4 comparison.

Reads:
  - results_npu_ep7_mb4/decode-dbo/timing_attention_*_mb4_b{B}_s512_t20.json
    + timing_ffn_coordinator_*
  - results_npu_ep7_mb4/prefill-dbo/timing_*_mb4_b{B}_s512_t20.json
  - results_npu_ep7/decode-dbo/timing_attention_*_b{B}_s512_t20.json (MB2 baseline)
  - results_npu_ep7/serial/b{B}_s512_t20.json (serial)

Writes:
  - results_npu_ep7_mb4/mb2_vs_mb4_summary.csv
  - results_npu_ep7_mb4/mb2_vs_mb4_decode_tpot.png
  - results_npu_ep7_mb4/mb2_vs_mb4_decode_throughput.png
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MB4_DIR = ROOT / "results_npu_ep7_mb4"
MB2_DECODE = ROOT / "results_npu_ep7" / "decode-dbo"
SERIAL_DIR = ROOT / "results_npu_ep7" / "serial"

DECODE_BATCHES = [8, 16, 32, 64, 128, 256]
PREFILL_BATCHES = [8, 16]
SEQ = 512
TOKENS = 20


def load_decode(timing_dir: Path, mb_tag: str, b: int):
    if mb_tag:
        attn = timing_dir / f"timing_attention_decode-dbo_npu_ep7_broadcast_reduce_overlap_{mb_tag}_b{b}_s{SEQ}_t{TOKENS}.json"
        ffn = timing_dir / f"timing_ffn_coordinator_decode-dbo_npu_ep7_broadcast_reduce_overlap_{mb_tag}_b{b}_s{SEQ}_t{TOKENS}.json"
    else:
        attn = timing_dir / f"timing_attention_decode-dbo_npu_ep7_broadcast_reduce_overlap_b{b}_s{SEQ}_t{TOKENS}.json"
        ffn = timing_dir / f"timing_ffn_coordinator_decode-dbo_npu_ep7_broadcast_reduce_overlap_b{b}_s{SEQ}_t{TOKENS}.json"
    if not attn.exists() or not ffn.exists():
        return None
    return json.loads(attn.read_text()), json.loads(ffn.read_text())


def load_prefill(timing_dir: Path, mb_tag: str, b: int):
    if mb_tag:
        attn = timing_dir / f"timing_attention_prefill-dbo_npu_ep7_broadcast_reduce_overlap_{mb_tag}_b{b}_s{SEQ}_t{TOKENS}.json"
    else:
        attn = timing_dir / f"timing_attention_prefill-dbo_npu_ep7_broadcast_reduce_overlap_b{b}_s{SEQ}_t{TOKENS}.json"
    return json.loads(attn.read_text()) if attn.exists() else None


def load_serial(b: int):
    p = SERIAL_DIR / f"b{b}_s{SEQ}_t{TOKENS}.json"
    return json.loads(p.read_text()) if p.exists() else None


def main():
    rows = []
    for b in DECODE_BATCHES:
        ser = load_serial(b)
        ser_tpot = ser.get("decode_tpot_ms") if ser else None

        mb2 = load_decode(MB2_DECODE, "", b)
        mb2_tpot = mb2[0].get("decode_tpot_ms") if mb2 else None

        mb4 = load_decode(MB4_DIR / "decode-dbo", "mb4", b)
        mb4_tpot = mb4[0].get("decode_tpot_ms") if mb4 else None

        rows.append({
            "metric": "decode",
            "batch": b,
            "serial_tpot_ms": ser_tpot,
            "mb2_tpot_ms": mb2_tpot,
            "mb4_tpot_ms": mb4_tpot,
            "mb2_speedup": (ser_tpot / mb2_tpot) if (ser_tpot and mb2_tpot) else None,
            "mb4_speedup": (ser_tpot / mb4_tpot) if (ser_tpot and mb4_tpot) else None,
            "mb2_throughput": (b * 1000.0 / mb2_tpot) if mb2_tpot else None,
            "mb4_throughput": (b * 1000.0 / mb4_tpot) if mb4_tpot else None,
        })

    prefill_rows = []
    for b in PREFILL_BATCHES:
        ser = load_serial(b)
        ser_pf = ser.get("prefill_ms") if ser else None
        mb2 = load_prefill(ROOT / "results_npu_ep7" / "prefill-dbo", "", b)
        mb2_pf = mb2.get("total_time_ms") if mb2 else None
        mb4 = load_prefill(MB4_DIR / "prefill-dbo", "mb4", b)
        mb4_pf = mb4.get("total_time_ms") if mb4 else None
        prefill_rows.append({
            "metric": "prefill",
            "batch": b,
            "serial_prefill_ms": ser_pf,
            "mb2_total_ms": mb2_pf,
            "mb4_total_ms": mb4_pf,
            "mb2_speedup": (ser_pf / mb2_pf) if (ser_pf and mb2_pf) else None,
            "mb4_speedup": (ser_pf / mb4_pf) if (ser_pf and mb4_pf) else None,
        })

    out_csv = MB4_DIR / "mb2_vs_mb4_summary.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "batch", "serial_ms", "mb2_ms", "mb4_ms",
                    "mb2_speedup", "mb4_speedup", "mb2_throughput", "mb4_throughput"])
        for r in rows:
            w.writerow(["decode", r["batch"], r["serial_tpot_ms"],
                        r["mb2_tpot_ms"], r["mb4_tpot_ms"],
                        r["mb2_speedup"], r["mb4_speedup"],
                        r["mb2_throughput"], r["mb4_throughput"]])
        for r in prefill_rows:
            w.writerow(["prefill", r["batch"], r["serial_prefill_ms"],
                        r["mb2_total_ms"], r["mb4_total_ms"],
                        r["mb2_speedup"], r["mb4_speedup"], "", ""])
    print(f"Wrote {out_csv}")

    # Plots
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bs = [r["batch"] for r in rows]
    ser = [r["serial_tpot_ms"] for r in rows]
    m2 = [r["mb2_tpot_ms"] for r in rows]
    m4 = [r["mb4_tpot_ms"] for r in rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(bs, ser, marker="^", label="serial", color="gray", linewidth=2)
    ax.plot(bs, m2, marker="o", label="DBO MB=2", color="tab:blue", linewidth=2)
    ax.plot(bs, m4, marker="s", label="DBO MB=4", color="tab:orange", linewidth=2)
    ax.set_xscale("log", base=2); ax.set_xticks(bs); ax.set_xticklabels(bs)
    ax.set_xlabel("Batch (log2)")
    ax.set_ylabel("TPOT (ms)")
    ax.set_title("Decode TPOT: serial / MB=2 / MB=4 (npu-ep7, seq=512, t=20)")
    ax.grid(True, alpha=0.3); ax.legend()
    plt.tight_layout()
    out = MB4_DIR / "mb2_vs_mb4_decode_tpot.png"
    plt.savefig(out, dpi=120); plt.close()
    print(f"Wrote {out}")

    th2 = [r["mb2_throughput"] for r in rows]
    th4 = [r["mb4_throughput"] for r in rows]
    th_s = [b * 1000.0 / s if s else None for b, s in zip(bs, ser)]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(bs, th_s, marker="^", label="serial", color="gray", linewidth=2)
    ax.plot(bs, th2, marker="o", label="DBO MB=2", color="tab:blue", linewidth=2)
    ax.plot(bs, th4, marker="s", label="DBO MB=4", color="tab:orange", linewidth=2)
    ax.set_xscale("log", base=2); ax.set_xticks(bs); ax.set_xticklabels(bs)
    ax.set_xlabel("Batch (log2)")
    ax.set_ylabel("Decode throughput (token/s)")
    ax.set_title("Decode Throughput: serial / MB=2 / MB=4 (npu-ep7, seq=512, t=20)")
    ax.grid(True, alpha=0.3); ax.legend()
    plt.tight_layout()
    out = MB4_DIR / "mb2_vs_mb4_decode_throughput.png"
    plt.savefig(out, dpi=120); plt.close()
    print(f"Wrote {out}")

    # Speedup line
    sp2 = [r["mb2_speedup"] for r in rows]
    sp4 = [r["mb4_speedup"] for r in rows]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(bs, sp2, marker="o", label="MB=2", color="tab:blue", linewidth=2)
    ax.plot(bs, sp4, marker="s", label="MB=4", color="tab:orange", linewidth=2)
    ax.axhline(1.0, color="red", linestyle="--", alpha=0.5, label="serial baseline")
    ax.set_xscale("log", base=2); ax.set_xticks(bs); ax.set_xticklabels(bs)
    ax.set_xlabel("Batch (log2)")
    ax.set_ylabel("DBO speedup × (serial / DBO)")
    ax.set_title("Decode DBO Speedup vs Batch (npu-ep7, seq=512, t=20)")
    ax.grid(True, alpha=0.3); ax.legend()
    plt.tight_layout()
    out = MB4_DIR / "mb2_vs_mb4_decode_speedup.png"
    plt.savefig(out, dpi=120); plt.close()
    print(f"Wrote {out}")

    # Print summary table to stdout
    print("\n=== Decode summary (s=512) ===")
    print(f"{'batch':>6} {'serial':>10} {'mb2':>10} {'mb4':>10} {'mb2_x':>7} {'mb4_x':>7}")
    for r in rows:
        sx = f"{r['mb2_speedup']:.2f}" if r["mb2_speedup"] else "-"
        s4 = f"{r['mb4_speedup']:.2f}" if r["mb4_speedup"] else "-"
        print(f"{r['batch']:>6} {r['serial_tpot_ms']:>10.1f} {r['mb2_tpot_ms']:>10.1f} {r['mb4_tpot_ms']:>10.1f} {sx:>7} {s4:>7}")

    print("\n=== Prefill summary (s=512) ===")
    print(f"{'batch':>6} {'serial':>10} {'mb2':>10} {'mb4':>10} {'mb2_x':>7} {'mb4_x':>7}")
    for r in prefill_rows:
        sx = f"{r['mb2_speedup']:.2f}" if r["mb2_speedup"] else "-"
        s4 = f"{r['mb4_speedup']:.2f}" if r["mb4_speedup"] else "-"
        ser = f"{r['serial_prefill_ms']:.1f}" if r["serial_prefill_ms"] else "-"
        m2s = f"{r['mb2_total_ms']:.1f}" if r["mb2_total_ms"] else "-"
        m4s = f"{r['mb4_total_ms']:.1f}" if r["mb4_total_ms"] else "-"
        print(f"{r['batch']:>6} {ser:>10} {m2s:>10} {m4s:>10} {sx:>7} {s4:>7}")


if __name__ == "__main__":
    main()
