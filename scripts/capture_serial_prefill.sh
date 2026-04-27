#!/bin/bash
# For each serial cache config, run a --no-dbo --no-generate pass to capture
# pure prefill time, then merge prefill_ms + decode_step_ms into the cache JSON.
#
# Usage: ./scripts/capture_serial_prefill.sh [--skip-existing]
set -u
cd "$(dirname "$0")/.."

SKIP_EXISTING=false
[ "${1:-}" = "--skip-existing" ] && SKIP_EXISTING=true

SKIP_BATCHES="${SKIP_BATCHES:-128}"   # configs known to OOM on prefill-only path

for CACHE in results/serial/cache/b*_s*_t*.json; do
    name=$(basename "$CACHE" .json)                  # b8_s128_t20
    B=$(echo "$name" | sed -n 's/^b\([0-9]*\)_.*/\1/p')
    S=$(echo "$name" | sed -n 's/^b[0-9]*_s\([0-9]*\)_.*/\1/p')
    T=$(echo "$name" | sed -n 's/.*_t\([0-9]*\)$/\1/p')

    if [[ ",$SKIP_BATCHES," == *,"$B",* ]]; then
        echo "[skip-oom] $name (batch=$B in SKIP_BATCHES)"
        continue
    fi

    if [ "$SKIP_EXISTING" = true ] && python -c "import json,sys; d=json.load(open('$CACHE')); sys.exit(0 if 'prefill_ms' in d else 1)" 2>/dev/null; then
        echo "[skip] $name (has prefill_ms)"
        continue
    fi

    echo "════════════════════════════════════════════════════════════"
    echo "  Capturing prefill time: $name"
    echo "════════════════════════════════════════════════════════════"

    SFX="prefill_only_b${B}_s${S}_t${T}"
    PORT=$((29500 + RANDOM % 2000))
    MASTER_PORT=$PORT bash scripts/run_single.sh local "$B" "$S" \
        --tokens "$T" --no-dbo --warmup-p2p --warmup-rounds 3 \
        2>&1 | tail -20
    # run_single.sh writes to results/prefill_dbo/timing_attention_warmup_serial_local_bB_sS_tT.json
    # (our flags exclude --generate, so it stays in the serial-prefill branch)
    RAW="results/prefill_dbo/timing_attention_warmup_serial_local_b${B}_s${S}_t${T}.json"
    if [ ! -f "$RAW" ]; then
        echo "[warn] missing $RAW; skipping merge"
        continue
    fi
    python - <<PY
import json, pathlib
cache = pathlib.Path("$CACHE")
raw   = pathlib.Path("$RAW")
d = json.loads(cache.read_text())
p = json.loads(raw.read_text())
prefill_ms = float(p["total_time_ms"])
total_ms   = float(d["total_time_ms"])
n          = int(d["max_new_tokens"])
d["prefill_ms"] = prefill_ms
d["decode_step_ms"] = (total_ms - prefill_ms) / n if n > 0 else 0.0
cache.write_text(json.dumps(d, indent=2))
print(f"  merged: prefill={prefill_ms:.1f}ms  decode_step={(total_ms-prefill_ms)/n:.1f}ms  total={total_ms:.1f}ms")
PY
done

echo ""
echo "✅ Done."
