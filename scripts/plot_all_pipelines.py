#!/usr/bin/env python3
"""Batch-plot pipeline Gantt figures for every successful DBO run.

Iterates results/{decode-dbo, decode-dbo-crosslayer, prefill-dbo}, pairs
timing_attention_*.json with timing_ffn_*.json, locates a matching serial
baseline in results/serial/cache/, and calls scripts/visualize_dbo_pipeline.py
for each pair. Emits a per-directory pipelines_index.md embedding all PNGs.
"""
from __future__ import annotations
import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VIS = ROOT / "scripts" / "visualize_dbo_pipeline.py"

# (subdir, mode_tag, start_layer, num_layers)
TARGET_SUBDIRS = [
    ("decode-dbo",             "decode-dbo",             0, 3),
    ("decode-dbo-crosslayer",  "decode-dbo-crosslayer",  0, 3),
    ("prefill-dbo",            "prefill-dbo",            1, 4),
]

NAME_RE = re.compile(r"timing_attention_(?P<tag>.+)\.json$")


def plot_one(attn_path: Path, ffn_path: Path, out_png: Path,
             start_layer: int, num_layers: int,
             serial_json: Path | None) -> bool:
    cmd = [
        sys.executable, str(VIS),
        "--attn-timing", str(attn_path),
        "--ffn-timing", str(ffn_path),
        "--output", str(out_png),
        "--start-layer", str(start_layer),
        "--num-layers", str(num_layers),
    ]
    if serial_json and serial_json.exists():
        cmd += ["--serial-timing", str(serial_json)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  ✗ {out_png.name}\n    {r.stderr.strip().splitlines()[-1] if r.stderr else ''}")
        return False
    print(f"  ✓ {out_png.name}")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="results",
                        help="Root directory (default: 'results'; use 'results_npu' for NPU)")
    args = parser.parse_args()
    root = ROOT / args.root
    cache_dir = root / "serial" / "cache"

    total_ok = total_fail = 0
    for sub, mode_tag, sl, nl in TARGET_SUBDIRS:
        d = root / sub
        if not d.is_dir():
            continue
        print(f"\n[{d.relative_to(ROOT)}]")
        index_lines = [f"# Pipeline figures — {mode_tag}\n"]

        attn_files = sorted(d.glob("timing_attention_*.json"))
        for attn in attn_files:
            m = NAME_RE.search(attn.name)
            if not m:
                continue
            tag = m.group("tag")
            ffn = d / f"timing_ffn_{tag}.json"
            if not ffn.exists():
                print(f"  ! missing ffn for {tag}")
                continue
            bm = re.search(r"b(\d+)_s(\d+)_t(\d+)", tag)
            serial = None
            if bm:
                b, s, t = bm.groups()
                serial = cache_dir / f"b{b}_s{s}_t{t}.json"
            out = d / f"pipeline_{tag}.png"
            ok = plot_one(attn, ffn, out, sl, nl, serial)
            if ok:
                total_ok += 1
                index_lines.append(f"## `{tag}`\n\n![{tag}]({out.name})\n")
            else:
                total_fail += 1
        (d / "pipelines_index.md").write_text("\n".join(index_lines) + "\n")

    print(f"\n{'='*50}\nDone: {total_ok} ok, {total_fail} failed")
    return 0 if total_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
