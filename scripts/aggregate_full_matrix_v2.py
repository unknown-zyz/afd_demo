#!/usr/bin/env python3
"""Aggregate results_npu/full_matrix_v2 into summary CSV + heatmap + curves."""
import re, csv, json, sys, os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "results_npu" / "full_matrix_v2"
OUT_CSV = ROOT / "summary.csv"

SPEEDUP_RE = re.compile(r"TPOT speedup: \*\*([\d.]+)×\*\*")
PRECISION_RE = re.compile(r"Speedup: \*\*([\d.]+)×\*\*")
TPOT_RE = re.compile(r"This run exact TPOT: \*\*([\d.]+) ms\*\*")
SERIAL_TPOT_RE = re.compile(r"Serial TPOT: \*\*([\d.]+) ms\*\*")
PREFILL_RE = re.compile(r"Prefill / TTFT-path \| ([\d.]+) ms")
PREFILL_SERIAL_TOTAL_RE = re.compile(r"\*\*Total wall time\*\* \| ([\d.]+) ms")  # serial reports
PREFILL_DBO_TOTAL_RE = re.compile(r"\| Total wall time \| \*\*([\d.]+) ms\*\*")

def parse_report(path: Path):
    text = path.read_text()
    out = {}
    m = SPEEDUP_RE.search(text)
    if m:
        out['speedup'] = float(m.group(1))
    m = TPOT_RE.search(text)
    if m:
        out['tpot_ms'] = float(m.group(1))
    m = SERIAL_TPOT_RE.search(text)
    if m:
        out['serial_tpot_ms'] = float(m.group(1))
    return out

def parse_correctness(timing_dir: Path, mode: str, batch: int, seq: int, tokens: int):
    """Read correctness_tokens from attention timing JSON."""
    pat = f"timing_attention_{mode}_npu_ep7_broadcast_reduce_overlap_b{batch}_s{seq}_t{tokens}.json"
    p = timing_dir / pat
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text())
        return d.get("correctness_tokens")
    except Exception:
        return None

def main():
    csv_in = ROOT / "experiment_matrix_summary.csv"
    rows = []
    with open(csv_in) as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            mode = r['mode']
            batch = int(r['batch'])
            seq = int(r['seq'])
            tokens = int(r['tokens'])
            status = r['status']
            row = {
                'mode': mode, 'batch': batch, 'seq': seq, 'tokens': tokens,
                'status': status,
            }
            if status == 'ok':
                report = ROOT / mode / f"report_{mode}_npu_ep7_broadcast_reduce_overlap_b{batch}_s{seq}_t{tokens}.md"
                if report.exists():
                    row.update(parse_report(report))
            # correctness tokens from attention timing (works for both ok and cached serial)
            row['correctness_tokens'] = parse_correctness(ROOT / mode, mode, batch, seq, tokens)
            rows.append(row)

    # write enriched summary
    fields = ['mode', 'batch', 'seq', 'tokens', 'status', 'tpot_ms',
              'serial_tpot_ms', 'speedup', 'correctness_tokens']
    with open(OUT_CSV, 'w') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            r.setdefault('tpot_ms', '')
            r.setdefault('serial_tpot_ms', '')
            r.setdefault('speedup', '')
            r.setdefault('correctness_tokens', '')
            if isinstance(r.get('correctness_tokens'), list):
                r['correctness_tokens'] = ','.join(str(x) for x in r['correctness_tokens'])
            w.writerow({k: r.get(k, '') for k in fields})
    print(f"Wrote {OUT_CSV}")

    # Build heatmaps
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib unavailable, skipping figures")
        return

    # Decode-dbo speedup heatmap
    decode_data = {}
    for r in rows:
        if r['mode'] == 'decode-dbo' and r['status'] == 'ok' and r.get('speedup'):
            decode_data[(r['batch'], r['seq'])] = r['speedup']

    if decode_data:
        batches = sorted({k[0] for k in decode_data.keys()})
        seqs = sorted({k[1] for k in decode_data.keys()})
        Z = np.full((len(seqs), len(batches)), np.nan)
        for i, s in enumerate(seqs):
            for j, b in enumerate(batches):
                if (b, s) in decode_data:
                    Z[i, j] = decode_data[(b, s)]

        fig, ax = plt.subplots(figsize=(10, 5))
        im = ax.imshow(Z, cmap='RdYlGn', vmin=0.5, vmax=2.5, aspect='auto')
        ax.set_xticks(range(len(batches)))
        ax.set_xticklabels(batches)
        ax.set_yticks(range(len(seqs)))
        ax.set_yticklabels(seqs)
        ax.set_xlabel('Batch')
        ax.set_ylabel('Prefill seq len')
        ax.set_title('Decode-DBO TPOT speedup vs serial (npu-ep7, t=20)')
        for i in range(len(seqs)):
            for j in range(len(batches)):
                if not np.isnan(Z[i, j]):
                    color = 'black' if 0.9 < Z[i, j] < 1.6 else 'white'
                    ax.text(j, i, f'{Z[i,j]:.2f}', ha='center', va='center', color=color, fontsize=9)
                else:
                    ax.text(j, i, 'OOM', ha='center', va='center', color='gray', fontsize=8)
        plt.colorbar(im, ax=ax, label='speedup ×')
        plt.tight_layout()
        plt.savefig(ROOT / 'fig_decode_speedup_heatmap.png', dpi=120)
        plt.close()
        print(f"Wrote {ROOT / 'fig_decode_speedup_heatmap.png'}")

        # Decode speedup curves
        fig, ax = plt.subplots(figsize=(10, 5))
        for s in seqs:
            xs, ys = [], []
            for b in batches:
                if (b, s) in decode_data:
                    xs.append(b)
                    ys.append(decode_data[(b, s)])
            if xs:
                ax.plot(xs, ys, marker='o', label=f'seq={s}')
        ax.axhline(1.0, color='k', linestyle='--', alpha=0.5)
        ax.axhline(1.2, color='g', linestyle=':', alpha=0.5, label='1.2× target')
        ax.set_xscale('log', base=2)
        ax.set_xlabel('Batch')
        ax.set_ylabel('Decode TPOT speedup ×')
        ax.set_title('Decode-DBO speedup curves (npu-ep7, t=20)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(ROOT / 'fig_decode_speedup_curves.png', dpi=120)
        plt.close()
        print(f"Wrote {ROOT / 'fig_decode_speedup_curves.png'}")

    # Prefill heatmap — scan all prefill-dbo report files directly (CSV may not have prefill rows)
    pf_speedup = {}
    TTFT_RE = re.compile(r"TTFT speedup: \*\*([\d.]+)×\*\*")
    pf_pat = re.compile(r"report_prefill-dbo_npu_ep7_broadcast_reduce_overlap_b(?P<b>\d+)_s(?P<s>\d+)_t(?P<t>\d+)\.md")
    for rpath in (ROOT / 'prefill-dbo').glob("report_*.md"):
        m = pf_pat.match(rpath.name)
        if not m:
            continue
        sp = TTFT_RE.search(rpath.read_text())
        if sp:
            pf_speedup[(int(m['b']), int(m['s']))] = float(sp.group(1))
    if pf_speedup:
        batches = sorted({k[0] for k in pf_speedup.keys()})
        seqs = sorted({k[1] for k in pf_speedup.keys()})
        Z = np.full((len(seqs), len(batches)), np.nan)
        for i, s in enumerate(seqs):
            for j, b in enumerate(batches):
                if (b, s) in pf_speedup:
                    Z[i, j] = pf_speedup[(b, s)]
        fig, ax = plt.subplots(figsize=(10, 5))
        im = ax.imshow(Z, cmap='RdYlGn', vmin=0.5, vmax=2.0, aspect='auto')
        ax.set_xticks(range(len(batches)))
        ax.set_xticklabels(batches)
        ax.set_yticks(range(len(seqs)))
        ax.set_yticklabels(seqs)
        ax.set_xlabel('Batch')
        ax.set_ylabel('Prefill seq len')
        ax.set_title('Prefill-DBO TTFT speedup vs serial (npu-ep7, t=8)')
        for i in range(len(seqs)):
            for j in range(len(batches)):
                if not np.isnan(Z[i, j]):
                    color = 'black' if 0.9 < Z[i, j] < 1.6 else 'white'
                    ax.text(j, i, f'{Z[i,j]:.2f}', ha='center', va='center', color=color, fontsize=9)
                else:
                    ax.text(j, i, 'OOM', ha='center', va='center', color='gray', fontsize=8)
        plt.colorbar(im, ax=ax, label='speedup ×')
        plt.tight_layout()
        plt.savefig(ROOT / 'fig_prefill_speedup_heatmap.png', dpi=120)
        plt.close()
        print(f"Wrote {ROOT / 'fig_prefill_speedup_heatmap.png'}")

    # Correctness comparison — scan all attention timing JSONs directly
    # (CSV may not contain t=8 rows from earlier run, but JSONs persist)
    print("\n=== Correctness check ===")
    cor = {}
    pat = re.compile(r"timing_attention_(?P<mode>serial|prefill-dbo|decode-dbo)_npu_ep7_broadcast_reduce_overlap_b(?P<b>\d+)_s(?P<s>\d+)_t(?P<t>\d+)\.json")
    for mode_dir in ('serial', 'prefill-dbo', 'decode-dbo'):
        for jf in (ROOT / mode_dir).glob("timing_attention_*.json"):
            m = pat.match(jf.name)
            if not m:
                continue
            try:
                d = json.loads(jf.read_text())
            except Exception:
                continue
            tok = d.get("correctness_tokens")
            if tok:
                cor[(m['mode'], int(m['b']), int(m['s']), int(m['t']))] = tok
    matches = 0
    diverges = 0
    diverge_examples = []
    for (mode, b, s, t), tok in cor.items():
        if mode in ('decode-dbo', 'prefill-dbo'):
            ser_key = ('serial', b, s, t)
            if ser_key in cor:
                if cor[ser_key] == tok:
                    matches += 1
                else:
                    diverges += 1
                    if len(diverge_examples) < 10:
                        diverge_examples.append((mode, b, s, t, cor[ser_key], tok))
    print(f"  total compared: {matches+diverges}; match={matches}, diverge={diverges}")
    for ex in diverge_examples:
        print(f"  diverge {ex[0]} b{ex[1]}/s{ex[2]}/t{ex[3]}: serial={ex[4]} dbo={ex[5]}")

if __name__ == '__main__':
    main()
