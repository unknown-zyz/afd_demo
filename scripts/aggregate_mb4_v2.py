#!/usr/bin/env python3
"""Round-9: 4-way comparison serial / MB2-orig / MB2-fused / MB4-fused.

Reads:
  - results_npu_ep7/serial/b{B}_s512_t20.json (serial baselines)
  - results_npu_ep7/decode-dbo/timing_attention_*broadcast_reduce_overlap_b{B}_s512_t20.json (MB2-orig)
  - results_npu_ep7_mb4_v2/decode-dbo/timing_attention_*_b{B}_s512_t20.json (MB2-fused)
  - results_npu_ep7_mb4_v2/decode-dbo/timing_attention_*_mb4_b{B}_s512_t20.json (MB4-fused)
  - prefill counterparts for prefill chart
"""
from __future__ import annotations
import csv, json, re
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
SERIAL = ROOT / "results_npu_ep7" / "serial"
ORIG_DIR = ROOT / "results_npu_ep7" / "decode-dbo"
ORIG_PRE = ROOT / "results_npu_ep7" / "prefill-dbo"
V2_DIR = ROOT / "results_npu_ep7_mb4_v2" / "decode-dbo"
V2_PRE = ROOT / "results_npu_ep7_mb4_v2" / "prefill-dbo"
OUT = ROOT / "results_npu_ep7_mb4_v2"

DECODE_BS = [8, 16, 32, 64, 128, 256, 512]
PREFILL_BS = [8, 16, 32]
SEQ = 512
T = 20


def _load_decode_tpot(path: Path):
    if not path.exists(): return None
    return json.load(open(path)).get("decode_tpot_ms")


def _load_prefill(path: Path):
    if not path.exists(): return None
    d = json.load(open(path))
    return d.get("total_time_ms") or d.get("prefill_ms")


def main():
    # Decode
    rows = []
    for b in DECODE_BS:
        ser = json.load(open(SERIAL / f"b{b}_s{SEQ}_t{T}.json")).get("decode_tpot_ms")
        mb2_orig = _load_decode_tpot(ORIG_DIR / f"timing_attention_decode-dbo_npu_ep7_broadcast_reduce_overlap_b{b}_s{SEQ}_t{T}.json")
        mb2_v2 = _load_decode_tpot(V2_DIR / f"timing_attention_decode-dbo_npu_ep7_broadcast_reduce_overlap_b{b}_s{SEQ}_t{T}.json")
        mb4_v2 = _load_decode_tpot(V2_DIR / f"timing_attention_decode-dbo_npu_ep7_broadcast_reduce_overlap_mb4_b{b}_s{SEQ}_t{T}.json")
        rows.append(dict(batch=b, serial=ser, mb2_orig=mb2_orig, mb2_v2=mb2_v2, mb4_v2=mb4_v2))

    pre_rows = []
    for b in PREFILL_BS:
        sj = SERIAL / f"b{b}_s{SEQ}_t{T}.json"
        ser = json.load(open(sj)).get("prefill_ms") if sj.exists() else None
        mb2_orig = _load_prefill(ORIG_PRE / f"timing_attention_prefill-dbo_npu_ep7_broadcast_reduce_overlap_b{b}_s{SEQ}_t{T}.json")
        mb2_v2 = _load_prefill(V2_PRE / f"timing_attention_prefill-dbo_npu_ep7_broadcast_reduce_overlap_b{b}_s{SEQ}_t{T}.json")
        mb4_v2 = _load_prefill(V2_PRE / f"timing_attention_prefill-dbo_npu_ep7_broadcast_reduce_overlap_mb4_b{b}_s{SEQ}_t{T}.json")
        pre_rows.append(dict(batch=b, serial=ser, mb2_orig=mb2_orig, mb2_v2=mb2_v2, mb4_v2=mb4_v2))

    # CSV
    csv_path = OUT / "mb2_vs_mb4_v2_summary.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mode","batch","serial_ms","mb2_orig_ms","mb2_v2_ms","mb4_v2_ms",
                    "mb2_orig_speedup","mb2_v2_speedup","mb4_v2_speedup",
                    "mb2_v2_vs_orig_pct"])
        for r in rows + pre_rows:
            mode = "decode" if r in rows else "prefill"
            ser = r["serial"]
            def sp(v): return (ser/v) if (ser and v) else None
            v2_gain = ((r["mb2_orig"]-r["mb2_v2"])/r["mb2_orig"]*100.0) if (r["mb2_orig"] and r["mb2_v2"]) else None
            w.writerow([mode, r["batch"], ser, r["mb2_orig"], r["mb2_v2"], r["mb4_v2"],
                        sp(r["mb2_orig"]), sp(r["mb2_v2"]), sp(r["mb4_v2"]),
                        f"{v2_gain:.2f}" if v2_gain is not None else ""])
    print(f"Wrote {csv_path}")

    # Decode TPOT plot
    bs = [r["batch"] for r in rows]
    fig, ax = plt.subplots(figsize=(9,5))
    ax.plot(bs, [r["serial"] for r in rows], "^-", color="gray", label="serial", lw=2)
    ax.plot(bs, [r["mb2_orig"] for r in rows], "o-", color="tab:blue", label="DBO MB=2 (orig)", lw=2)
    ax.plot(bs, [r["mb2_v2"] for r in rows], "o--", color="tab:cyan", label="DBO MB=2 (fused)", lw=2)
    ax.plot(bs, [r["mb4_v2"] for r in rows], "s-", color="tab:orange", label="DBO MB=4 (fused)", lw=2)
    ax.set_xscale("log", base=2); ax.set_xticks(bs); ax.set_xticklabels(bs)
    ax.set_xlabel("Batch (log2)"); ax.set_ylabel("TPOT (ms)")
    ax.set_title("Decode TPOT 4-way: serial / MB2-orig / MB2-fused / MB4-fused (npu-ep7, seq=512)")
    ax.grid(True, alpha=0.3); ax.legend()
    plt.tight_layout()
    p = OUT / "mb2_vs_mb4_v2_decode_tpot.png"; plt.savefig(p, dpi=120); plt.close(); print(f"Wrote {p}")

    # Throughput
    fig, ax = plt.subplots(figsize=(9,5))
    def th(r, k): v = r[k]; return (r["batch"]*1000.0/v) if v else None
    ax.plot(bs, [th(r,"serial") for r in rows], "^-", color="gray", label="serial", lw=2)
    ax.plot(bs, [th(r,"mb2_orig") for r in rows], "o-", color="tab:blue", label="DBO MB=2 (orig)", lw=2)
    ax.plot(bs, [th(r,"mb2_v2") for r in rows], "o--", color="tab:cyan", label="DBO MB=2 (fused)", lw=2)
    ax.plot(bs, [th(r,"mb4_v2") for r in rows], "s-", color="tab:orange", label="DBO MB=4 (fused)", lw=2)
    ax.set_xscale("log", base=2); ax.set_xticks(bs); ax.set_xticklabels(bs)
    ax.set_xlabel("Batch (log2)"); ax.set_ylabel("Throughput (token/s)")
    ax.set_title("Decode Throughput 4-way (npu-ep7, seq=512)")
    ax.grid(True, alpha=0.3); ax.legend()
    plt.tight_layout()
    p = OUT / "mb2_vs_mb4_v2_decode_throughput.png"; plt.savefig(p, dpi=120); plt.close(); print(f"Wrote {p}")

    # Speedup
    fig, ax = plt.subplots(figsize=(9,5))
    def sp(r, k): v=r[k]; return (r["serial"]/v) if (r["serial"] and v) else None
    ax.axhline(1.0, color="red", linestyle="--", alpha=0.5, label="serial")
    ax.plot(bs, [sp(r,"mb2_orig") for r in rows], "o-", color="tab:blue", label="MB=2 (orig)", lw=2)
    ax.plot(bs, [sp(r,"mb2_v2") for r in rows], "o--", color="tab:cyan", label="MB=2 (fused)", lw=2)
    ax.plot(bs, [sp(r,"mb4_v2") for r in rows], "s-", color="tab:orange", label="MB=4 (fused)", lw=2)
    ax.set_xscale("log", base=2); ax.set_xticks(bs); ax.set_xticklabels(bs)
    ax.set_xlabel("Batch (log2)"); ax.set_ylabel("Speedup (serial / DBO)")
    ax.set_title("Decode DBO Speedup 4-way (npu-ep7, seq=512)")
    ax.grid(True, alpha=0.3); ax.legend()
    plt.tight_layout()
    p = OUT / "mb2_vs_mb4_v2_decode_speedup.png"; plt.savefig(p, dpi=120); plt.close(); print(f"Wrote {p}")

    # Console
    print("\n=== Decode 4-way (s=512) ===")
    print(f"{'batch':>6} {'serial':>8} {'mb2-orig':>9} {'mb2-v2':>8} {'mb4-v2':>8} | {'mb2-orig×':>9} {'mb2-v2×':>8} {'mb4-v2×':>8} | {'fuse_gain%':>10}")
    for r in rows:
        def fmt(v): return f"{v:.1f}" if v else "-"
        s = r["serial"]
        sp_o = (s/r["mb2_orig"]) if r["mb2_orig"] else None
        sp_v = (s/r["mb2_v2"]) if r["mb2_v2"] else None
        sp_4 = (s/r["mb4_v2"]) if r["mb4_v2"] else None
        gain = ((r["mb2_orig"]-r["mb2_v2"])/r["mb2_orig"]*100.0) if (r["mb2_orig"] and r["mb2_v2"]) else None
        print(f"{r['batch']:>6} {fmt(s):>8} {fmt(r['mb2_orig']):>9} {fmt(r['mb2_v2']):>8} {fmt(r['mb4_v2']):>8} | "
              f"{(f'{sp_o:.2f}' if sp_o else '-'):>9} {(f'{sp_v:.2f}' if sp_v else '-'):>8} {(f'{sp_4:.2f}' if sp_4 else '-'):>8} | "
              f"{(f'{gain:+.1f}%' if gain is not None else '-'):>10}")

    print("\n=== Prefill (s=512) ===")
    print(f"{'batch':>6} {'serial':>8} {'mb2-orig':>9} {'mb2-v2':>8} {'mb4-v2':>8}")
    for r in pre_rows:
        def fmt(v): return f"{v:.1f}" if v else "-"
        print(f"{r['batch']:>6} {fmt(r['serial']):>8} {fmt(r['mb2_orig']):>9} {fmt(r['mb2_v2']):>8} {fmt(r['mb4_v2']):>8}")


if __name__ == "__main__":
    main()
