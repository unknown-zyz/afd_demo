#!/usr/bin/env python3
"""Extract decode-dbo summary from ep16_speedup_summary.csv."""
import csv
from pathlib import Path

INPUT = Path(__file__).parent / "ep16_speedup_summary.csv"
OUTPUT = Path(__file__).parent / "decode_dbo_summary.csv"

rows = []
with open(INPUT) as f:
    reader = csv.DictReader(f)
    for r in reader:
        if r["mode"] != "decode-dbo":
            continue
        batch = int(r["batch"])
        if batch > 256:
            continue
        seq = int(r["seq"])
        prefill_ms = float(r["prefill_ms"])
        tpot_ms = float(r["decode_tpot_ms"])
        prefill_throughput = batch * 1000 / prefill_ms if prefill_ms > 0 else 0
        decode_throughput = batch * 1000 / tpot_ms if tpot_ms > 0 else 0
        rows.append({
            "batch_size": batch,
            "seq": seq,
            "TTFT_ms": round(prefill_ms, 2),
            "TPOT_ms": round(tpot_ms, 2),
            "prefill_throughput": round(prefill_throughput, 1),
            "decode_throughput": round(decode_throughput, 1),
        })

rows.sort(key=lambda x: (x["seq"], x["batch_size"]))

with open(OUTPUT, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["batch_size", "seq", "TTFT_ms", "TPOT_ms", "prefill_throughput", "decode_throughput"])
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote {len(rows)} rows to {OUTPUT}")
