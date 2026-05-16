#!/usr/bin/env python3
"""Round-8 Track A:
  1) decode-dbo throughput heatmap (token/s = batch * 1000 / TPOT_ms)
  2) seq=512: decode-dbo and serial throughput vs batch + TPOT vs batch
  3) seq=512: attn / ffn compute time vs batch markdown table

Inputs:
  - results_npu_ep7/experiment_matrix_summary.csv (decode-dbo + serial rows)
  - results_npu_ep7/decode-dbo/timing_attention_*_b{B}_s512_t20.json
  - results_npu_ep7/decode-dbo/timing_ffn_coordinator_*_b{B}_s512_t20.json
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "results_npu_ep7"
CSV_PATH = ROOT / "experiment_matrix_summary.csv"
DECODE_DIR = ROOT / "decode-dbo"
DOC_OUT = Path(__file__).resolve().parent.parent / "doc" / "compute_time_vs_batch_s512.md"

CONFIG_RE = re.compile(r"_b(?P<b>\d+)_s(?P<s>\d+)_t(?P<t>\d+)\.json$")


def load_metric_rows(mode: str):
    rows = []
    with CSV_PATH.open() as f:
        for r in csv.DictReader(f):
            if r["mode"] != mode:
                continue
            try:
                b = int(r["batch"]); s = int(r["seq"]); t = int(r["tokens"])
                tpot = float(r["tpot_ms"])
            except (ValueError, KeyError):
                continue
            rows.append({"batch": b, "seq": s, "tokens": t, "tpot_ms": tpot,
                         "throughput": b * 1000.0 / tpot})
    return rows


def load_decode_dbo_rows():
    return load_metric_rows("decode-dbo")


def make_throughput_heatmap(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    data = {(r["batch"], r["seq"]): r["throughput"] for r in rows}
    if not data:
        print("no decode-dbo rows; skipping throughput heatmap")
        return
    batches = sorted({k[0] for k in data})
    seqs = sorted({k[1] for k in data})
    Z = np.full((len(seqs), len(batches)), np.nan)
    for i, s in enumerate(seqs):
        for j, b in enumerate(batches):
            if (b, s) in data:
                Z[i, j] = data[(b, s)]
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(Z, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(batches))); ax.set_xticklabels(batches)
    ax.set_yticks(range(len(seqs))); ax.set_yticklabels(seqs)
    ax.set_xlabel("Batch"); ax.set_ylabel("Prefill seq len")
    ax.set_title("Decode-DBO Throughput (token/s) — npu-ep7, t=20")
    for i in range(len(seqs)):
        for j in range(len(batches)):
            if not np.isnan(Z[i, j]):
                ax.text(j, i, f"{Z[i, j]:.0f}", ha="center", va="center",
                        color="white", fontsize=8)
            else:
                ax.text(j, i, "OOM", ha="center", va="center",
                        color="gray", fontsize=8)
    plt.colorbar(im, ax=ax, label="token/s")
    plt.tight_layout()
    out = ROOT / "decode_dbo_throughput_heatmap.png"
    plt.savefig(out, dpi=120); plt.close()
    print(f"Wrote {out}")


def make_s512_lineplots(rows, label: str, out_prefix: str, color_prefix: str = "tab"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s512 = sorted([r for r in rows if r["seq"] == 512], key=lambda r: r["batch"])
    if not s512:
        print(f"no {label} s=512 rows; skipping line plots")
        return
    bs = [r["batch"] for r in s512]
    thr = [r["throughput"] for r in s512]
    tpot = [r["tpot_ms"] for r in s512]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(bs, thr, marker="o", color=f"{color_prefix}:blue", linewidth=2)
    ax.set_xscale("log", base=2)
    ax.set_xticks(bs); ax.set_xticklabels(bs)
    ax.set_xlabel("Batch (log2)")
    ax.set_ylabel("Throughput (token/s)")
    ax.set_title(f"{label} Throughput vs Batch (seq=512, npu-ep7, t=20)")
    ax.grid(True, alpha=0.3)
    for x, y in zip(bs, thr):
        ax.annotate(f"{y:.0f}", (x, y), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8)
    plt.tight_layout()
    out = ROOT / f"{out_prefix}_throughput_vs_batch_s512.png"
    plt.savefig(out, dpi=120); plt.close()
    print(f"Wrote {out}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(bs, tpot, marker="s", color=f"{color_prefix}:red", linewidth=2)
    ax.set_xscale("log", base=2)
    ax.set_xticks(bs); ax.set_xticklabels(bs)
    ax.set_xlabel("Batch (log2)")
    ax.set_ylabel("TPOT (ms)")
    ax.set_title(f"{label} TPOT vs Batch (seq=512, npu-ep7, t=20)")
    ax.grid(True, alpha=0.3)
    for x, y in zip(bs, tpot):
        ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8)
    plt.tight_layout()
    out = ROOT / f"{out_prefix}_tpot_vs_batch_s512.png"
    plt.savefig(out, dpi=120); plt.close()
    print(f"Wrote {out}")


def aggregate_compute_time():
    """For each batch with timing_attention + timing_ffn at s=512, compute total
    attn_compute / ffn_compute across all events and per-layer-per-MB averages.
    """
    results = []
    for attn_path in sorted(DECODE_DIR.glob("timing_attention_*_s512_t20.json")):
        m = CONFIG_RE.search(attn_path.name)
        if not m:
            continue
        b = int(m["b"])
        ffn_path = DECODE_DIR / attn_path.name.replace("timing_attention_", "timing_ffn_coordinator_")
        if not ffn_path.exists():
            continue
        try:
            attn = json.loads(attn_path.read_text())
            ffn = json.loads(ffn_path.read_text())
        except Exception:
            continue

        num_layers = ffn.get("num_layers") or attn.get("num_layers")
        num_mb = ffn.get("num_micro_batches") or attn.get("num_micro_batches") or 2

        attn_total = sum(e.get("duration_ms", 0.0)
                         for e in attn.get("events", [])
                         if e.get("type") == "attn_compute")
        ffn_total = sum(e.get("duration_ms", 0.0)
                        for e in ffn.get("events", [])
                        if e.get("type") == "ffn_compute")
        ep_experts_total = sum(e.get("duration_ms", 0.0)
                               for e in ffn.get("events", [])
                               if e.get("type") == "ep_local_experts")

        denom = max(num_layers * num_mb, 1)
        results.append({
            "batch": b,
            "num_layers": num_layers,
            "num_mb": num_mb,
            "attn_total_ms": attn_total,
            "ffn_total_ms": ffn_total,
            "ep_experts_total_ms": ep_experts_total,
            "attn_per_layer_per_mb_ms": attn_total / denom,
            "ffn_per_layer_per_mb_ms": ffn_total / denom,
            "ep_experts_per_layer_per_mb_ms": ep_experts_total / denom,
            "ffn_to_attn_ratio": (ffn_total / attn_total) if attn_total else float("nan"),
        })
    results.sort(key=lambda r: r["batch"])
    return results


def write_compute_doc(rows):
    DOC_OUT.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("# Attn / FFN 计算时间 vs Batch（seq=512, decode-dbo, npu-ep7, t=20）\n")
    lines.append("数据源：`results_npu_ep7/decode-dbo/timing_{attention,ffn_coordinator}_*_s512_t20.json`\n")
    lines.append("- `*_total_ms`：整次 decode loop 中所有 layer × 所有 MB 的累计时间")
    lines.append("- `*_per_layer_per_mb_ms`：除以 `num_layers × num_mb` 得到的单层单 MB 平均")
    lines.append("- `ep_experts`：FFN 内 EP local experts（routed MoE）的实际计算时间，是 `ffn_compute` 的子集")
    lines.append("- `ffn/attn ratio`：FFN 总耗时 / Attn 总耗时\n")
    lines.append("| batch | layers×mb | attn_total ms | ffn_total ms | ep_experts ms | attn/层/MB ms | ffn/层/MB ms | ep_experts/层/MB ms | ffn/attn |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        lines.append(
            f"| {r['batch']} | {r['num_layers']}×{r['num_mb']} | "
            f"{r['attn_total_ms']:.2f} | {r['ffn_total_ms']:.2f} | {r['ep_experts_total_ms']:.2f} | "
            f"{r['attn_per_layer_per_mb_ms']:.3f} | {r['ffn_per_layer_per_mb_ms']:.3f} | "
            f"{r['ep_experts_per_layer_per_mb_ms']:.3f} | {r['ffn_to_attn_ratio']:.2f} |"
        )
    lines.append("\n## 关键观察\n")
    if rows:
        small = rows[0]; big = rows[-1]
        lines.append(
            f"- batch 从 {small['batch']} → {big['batch']}：attn 单层 MB "
            f"{small['attn_per_layer_per_mb_ms']:.2f} → {big['attn_per_layer_per_mb_ms']:.2f} ms，"
            f"ffn 单层 MB {small['ffn_per_layer_per_mb_ms']:.2f} → {big['ffn_per_layer_per_mb_ms']:.2f} ms。"
        )
        ratios = [r["ffn_to_attn_ratio"] for r in rows]
        lines.append(
            f"- FFN/Attn 计算时间比在 {min(ratios):.2f}× ~ {max(ratios):.2f}×。"
            "中 batch（8~64）时 FFN 显著占优（routed MoE 主导），DBO overlap 收益最大；"
            "极小 batch（2）和极大 batch（512）下两者接近或 attn 反超，DBO 收益相应减弱。"
        )
    DOC_OUT.write_text("\n".join(lines) + "\n")
    print(f"Wrote {DOC_OUT}")


def main():
    rows = load_decode_dbo_rows()
    serial_rows = load_metric_rows("serial")
    print(f"Loaded {len(rows)} decode-dbo CSV rows")
    print(f"Loaded {len(serial_rows)} serial CSV rows")
    make_throughput_heatmap(rows)
    make_s512_lineplots(rows, "Decode-DBO", "decode_dbo")
    make_s512_lineplots(serial_rows, "Serial", "serial")
    compute_rows = aggregate_compute_time()
    print(f"Aggregated {len(compute_rows)} compute-time configs at s=512")
    write_compute_doc(compute_rows)


if __name__ == "__main__":
    main()
