#!/usr/bin/env python3
"""Compare EP=7 results between Host1 and Host2."""
import csv
from pathlib import Path

ROOT = Path(__file__).parent
H1_CSV = ROOT.parent / "results_npu_ep7" / "experiment_matrix_summary.csv"
H2_CSV = ROOT / "experiment_matrix_summary.csv"

def load_csv(path):
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows

def key(r):
    return (r["mode"], int(r["batch"]), int(r["seq"]))

def main():
    h1 = {key(r): r for r in load_csv(H1_CSV) if r.get("status") in ("ok", "cached")}
    h2 = {key(r): r for r in load_csv(H2_CSV) if r.get("status") in ("ok", "cached")}

    common = sorted(set(h1.keys()) & set(h2.keys()))

    print("=" * 120)
    print("Host1 vs Host2 EP=7 Comparison (serial + decode-dbo, batch 2-64, seq 128-1024)")
    print("=" * 120)

    # Serial TPOT comparison
    print("\n## Serial TPOT (ms) - lower is better")
    print(f"{'mode':<12} {'batch':>5} {'seq':>5} {'Host1':>10} {'Host2':>10} {'H2/H1':>8} {'Delta%':>8}")
    print("-" * 60)
    serial_keys = [k for k in common if k[0] == "serial"]
    for k in serial_keys:
        h1_tpot = float(h1[k]["tpot_ms"])
        h2_tpot = float(h2[k]["tpot_ms"])
        ratio = h2_tpot / h1_tpot if h1_tpot else 0
        delta_pct = (ratio - 1) * 100
        print(f"{k[0]:<12} {k[1]:>5} {k[2]:>5} {h1_tpot:>10.1f} {h2_tpot:>10.1f} {ratio:>8.3f} {delta_pct:>+7.1f}%")

    # Decode-dbo TPOT comparison
    print("\n## Decode-DBO TPOT (ms) - lower is better")
    print(f"{'mode':<12} {'batch':>5} {'seq':>5} {'Host1':>10} {'Host2':>10} {'H2/H1':>8} {'Delta%':>8}")
    print("-" * 60)
    dbo_keys = [k for k in common if k[0] == "decode-dbo"]
    for k in dbo_keys:
        h1_tpot = float(h1[k]["tpot_ms"])
        h2_tpot = float(h2[k]["tpot_ms"])
        ratio = h2_tpot / h1_tpot if h1_tpot else 0
        delta_pct = (ratio - 1) * 100
        print(f"{k[0]:<12} {k[1]:>5} {k[2]:>5} {h1_tpot:>10.1f} {h2_tpot:>10.1f} {ratio:>8.3f} {delta_pct:>+7.1f}%")

    # Speedup comparison (decode-dbo speedup = serial_tpot / decode_dbo_tpot)
    print("\n## Decode-DBO Speedup vs Serial (serial_tpot / decode_dbo_tpot) - higher is better")
    print(f"{'batch':>5} {'seq':>5} {'H1_speedup':>10} {'H2_speedup':>10} {'Delta':>8}")
    print("-" * 50)
    for k in dbo_keys:
        serial_k = ("serial", k[1], k[2])
        if serial_k not in h1 or serial_k not in h2:
            continue
        h1_serial_tpot = float(h1[serial_k]["tpot_ms"])
        h1_dbo_tpot = float(h1[k]["tpot_ms"])
        h1_speedup = h1_serial_tpot / h1_dbo_tpot if h1_dbo_tpot else 0

        h2_serial_tpot = float(h2[serial_k]["tpot_ms"])
        h2_dbo_tpot = float(h2[k]["tpot_ms"])
        h2_speedup = h2_serial_tpot / h2_dbo_tpot if h2_dbo_tpot else 0

        delta = h2_speedup - h1_speedup
        print(f"{k[1]:>5} {k[2]:>5} {h1_speedup:>10.3f} {h2_speedup:>10.3f} {delta:>+7.3f}")

    # Prefill comparison
    print("\n## Serial Prefill (ms) - lower is better")
    print(f"{'batch':>5} {'seq':>5} {'Host1':>10} {'Host2':>10} {'H2/H1':>8} {'Delta%':>8}")
    print("-" * 50)
    for k in serial_keys:
        h1_pre = float(h1[k]["prefill_ms"])
        h2_pre = float(h2[k]["prefill_ms"])
        ratio = h2_pre / h1_pre if h1_pre else 0
        delta_pct = (ratio - 1) * 100
        print(f"{k[1]:>5} {k[2]:>5} {h1_pre:>10.1f} {h2_pre:>10.1f} {ratio:>8.3f} {delta_pct:>+7.1f}%")

    # Summary stats
    serial_h1_tpots = [float(h1[k]["tpot_ms"]) for k in serial_keys]
    serial_h2_tpots = [float(h2[k]["tpot_ms"]) for k in serial_keys]
    dbo_h1_tpots = [float(h1[k]["tpot_ms"]) for k in dbo_keys]
    dbo_h2_tpots = [float(h2[k]["tpot_ms"]) for k in dbo_keys]

    print("\n" + "=" * 80)
    print("Summary Statistics")
    print("=" * 80)
    print(f"Serial TPOT  - H1 avg: {sum(serial_h1_tpots)/len(serial_h1_tpots):.1f}ms  H2 avg: {sum(serial_h2_tpots)/len(serial_h2_tpots):.1f}ms  H2/H1: {sum(serial_h2_tpots)/sum(serial_h1_tpots):.3f}")
    print(f"DBO TPOT     - H1 avg: {sum(dbo_h1_tpots)/len(dbo_h1_tpots):.1f}ms  H2 avg: {sum(dbo_h2_tpots)/len(dbo_h2_tpots):.1f}ms  H2/H1: {sum(dbo_h2_tpots)/sum(dbo_h1_tpots):.3f}")

    # Count how many configs H2 is better
    serial_h2_better = sum(1 for k in serial_keys if float(h2[k]["tpot_ms"]) < float(h1[k]["tpot_ms"]))
    dbo_h2_better = sum(1 for k in dbo_keys if float(h2[k]["tpot_ms"]) < float(h1[k]["tpot_ms"]))
    print(f"\nH2 faster than H1: {serial_h2_better}/{len(serial_keys)} serial configs, {dbo_h2_better}/{len(dbo_keys)} decode-dbo configs")

if __name__ == "__main__":
    main()
