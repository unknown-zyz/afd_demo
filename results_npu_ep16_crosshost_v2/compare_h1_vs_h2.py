#!/usr/bin/env python3
"""Compare 1A16F cross-host results: Host1 vs Host2."""
import csv
from pathlib import Path

ROOT = Path(__file__).parent
H1_CSV = ROOT.parent / "results_npu_ep16_crosshost" / "crosshost_static_ep16_sweep" / "matrix_summary.csv"
H2_CSV = ROOT / "matrix_summary.csv"

def load_csv(path):
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r.get("status") in ("ok", "OK", "cached"):
                rows.append(r)
    return rows

def key(r):
    return (r["mode"], int(r["batch"]), int(r["seq"]))

def main():
    h1 = {key(r): r for r in load_csv(H1_CSV)}
    h2 = {key(r): r for r in load_csv(H2_CSV)}

    common = sorted(set(h1.keys()) & set(h2.keys()))

    print("=" * 120)
    print("1A16F Cross-Host Comparison: Host1 vs Host2")
    print("=" * 120)

    # Serial TPOT comparison
    print("\n## Serial TPOT (ms) - lower is better")
    print(f"{'batch':>5} {'seq':>5} {'Host1':>10} {'Host2':>10} {'H2/H1':>8} {'Delta%':>8}")
    print("-" * 50)
    serial_keys = [k for k in common if k[0] == "serial"]
    for k in serial_keys:
        h1_tpot = float(h1[k].get("decode_tpot_ms", 0) or 0)
        h2_tpot = float(h2[k].get("decode_tpot_ms", 0) or 0)
        if h1_tpot > 0 and h2_tpot > 0:
            ratio = h2_tpot / h1_tpot
            delta_pct = (ratio - 1) * 100
            print(f"{k[1]:>5} {k[2]:>5} {h1_tpot:>10.1f} {h2_tpot:>10.1f} {ratio:>8.3f} {delta_pct:>+7.1f}%")

    # Decode-dbo TPOT comparison
    print("\n## Decode-DBO TPOT (ms) - lower is better")
    print(f"{'batch':>5} {'seq':>5} {'Host1':>10} {'Host2':>10} {'H2/H1':>8} {'Delta%':>8}")
    print("-" * 50)
    dbo_keys = [k for k in common if k[0] == "decode-dbo"]
    for k in dbo_keys:
        h1_tpot = float(h1[k].get("decode_tpot_ms", 0) or 0)
        h2_tpot = float(h2[k].get("decode_tpot_ms", 0) or 0)
        if h1_tpot > 0 and h2_tpot > 0:
            ratio = h2_tpot / h1_tpot
            delta_pct = (ratio - 1) * 100
            print(f"{k[1]:>5} {k[2]:>5} {h1_tpot:>10.1f} {h2_tpot:>10.1f} {ratio:>8.3f} {delta_pct:>+7.1f}%")

    # Speedup comparison
    print("\n## Decode-DBO Speedup vs Serial - higher is better")
    print(f"{'batch':>5} {'seq':>5} {'H1_speedup':>10} {'H2_speedup':>10} {'Delta':>8}")
    print("-" * 50)
    for k in dbo_keys:
        serial_k = ("serial", k[1], k[2])
        if serial_k not in h1 or serial_k not in h2:
            continue
        h1_serial = float(h1[serial_k].get("decode_tpot_ms", 0) or 0)
        h1_dbo = float(h1[k].get("decode_tpot_ms", 0) or 0)
        h2_serial = float(h2[serial_k].get("decode_tpot_ms", 0) or 0)
        h2_dbo = float(h2[k].get("decode_tpot_ms", 0) or 0)
        if h1_serial > 0 and h1_dbo > 0 and h2_serial > 0 and h2_dbo > 0:
            h1_speedup = h1_serial / h1_dbo
            h2_speedup = h2_serial / h2_dbo
            delta = h2_speedup - h1_speedup
            print(f"{k[1]:>5} {k[2]:>5} {h1_speedup:>10.3f} {h2_speedup:>10.3f} {delta:>+7.3f}")

    # Summary stats
    serial_h1 = [float(h1[k].get("decode_tpot_ms", 0) or 0) for k in serial_keys if k in h1 and float(h1[k].get("decode_tpot_ms", 0) or 0) > 0]
    serial_h2 = [float(h2[k].get("decode_tpot_ms", 0) or 0) for k in serial_keys if k in h2 and float(h2[k].get("decode_tpot_ms", 0) or 0) > 0]
    dbo_h1 = [float(h1[k].get("decode_tpot_ms", 0) or 0) for k in dbo_keys if k in h1 and float(h1[k].get("decode_tpot_ms", 0) or 0) > 0]
    dbo_h2 = [float(h2[k].get("decode_tpot_ms", 0) or 0) for k in dbo_keys if k in h2 and float(h2[k].get("decode_tpot_ms", 0) or 0) > 0]

    print("\n" + "=" * 80)
    print("Summary Statistics")
    print("=" * 80)
    if serial_h1 and serial_h2:
        print(f"Serial TPOT  - H1 avg: {sum(serial_h1)/len(serial_h1):.1f}ms  H2 avg: {sum(serial_h2)/len(serial_h2):.1f}ms  H2/H1: {sum(serial_h2)/sum(serial_h1):.3f}")
    if dbo_h1 and dbo_h2:
        print(f"DBO TPOT     - H1 avg: {sum(dbo_h1)/len(dbo_h1):.1f}ms  H2 avg: {sum(dbo_h2)/len(dbo_h2):.1f}ms  H2/H1: {sum(dbo_h2)/sum(dbo_h1):.3f}")

    serial_h2_better = sum(1 for k in serial_keys if k in h1 and k in h2 and float(h2[k].get("decode_tpot_ms", 0) or 0) < float(h1[k].get("decode_tpot_ms", 0) or 0))
    dbo_h2_better = sum(1 for k in dbo_keys if k in h1 and k in h2 and float(h2[k].get("decode_tpot_ms", 0) or 0) < float(h1[k].get("decode_tpot_ms", 0) or 0))
    print(f"\nH2 faster than H1: {serial_h2_better}/{len(serial_keys)} serial configs, {dbo_h2_better}/{len(dbo_keys)} decode-dbo configs")

if __name__ == "__main__":
    main()
