# GPU / NPU experiment metric notes

This document explains how to interpret GPU `results/` and NPU `results_npu/`
experiment outputs. The current final coverage and speedup conclusions are in
[gpu_npu_experiment_summary.md](gpu_npu_experiment_summary.md).

## 1. One speedup formula, mode-matched denominators

All reports and pipeline figures use:

```text
speedup = serial_baseline / DBO_time
```

Values above `1.0x` mean DBO is faster. The important rule is that the serial
baseline must match the mode:

| Mode | DBO metric | Serial baseline | Meaning |
|---|---|---|---|
| `prefill-dbo` | `total_time_ms` from prefill-only DBO | `prefill_ms` | model-side TTFT / TTFT-path |
| `decode-dbo` | `decode_tpot_ms` | `decode_tpot_ms` | exact TPOT |
| `decode-dbo-crosslayer` | `decode_tpot_ms` | `decode_tpot_ms` | exact TPOT |

Do not use `total_time_ms / max_new_tokens` as TPOT. It folds the prefill
first-token path into decode and uses the wrong denominator for exact decode
loop TPOT.

## 2. TTFT-path vs online TTFT

The project measures a **model-side TTFT-path** for prefill experiments. It
includes the model prefill path being timed by the scheduler, but it does not
include a production serving stack's queueing, tokenizer, network response,
streaming transport, or scheduler batching overhead.

Use this metric to compare DBO vs serial within this repo. Do not present it as
end-to-end online TTFT.

## 3. Exact TPOT

Decode timing records:

```text
decode_tpot_ms = decode_loop_ms / decode_steps
decode_steps = max_new_tokens - 1
```

The first token sampled after prefill is part of the TTFT-path. Subsequent
tokens are decode-loop steps.

## 4. Representative ITL

DBO decode pipeline figures visualize one representative decode step. This is
useful for checking overlap, bubbles, and per-layer timing, but it is not the
speedup denominator. Speedup always uses exact `decode_tpot_ms` when available.

Representative timing can be absent when:

- timing is disabled;
- DBO is disabled;
- generation is disabled;
- `max_new_tokens` is too small to reach the representative step;
- `batch_size < num_micro_batches`;
- the run fails or OOMs.

## 5. Baseline audit

Use:

```bash
python scripts/audit_experiment_baselines.py --root results --output-csv results/baseline_audit.csv
python scripts/audit_experiment_baselines.py --root results_npu --output-csv results_npu/baseline_audit.csv
```

Audit statuses:

| Status | Meaning |
|---|---|
| `ok` | The mode-matched serial baseline exists and the speedup is meaningful. |
| `serial-cache-missing` | The matching serial cache file does not exist. |
| `baseline-missing` | The cache exists but lacks `prefill_ms` or `decode_tpot_ms`. |
| `serial-cache-invalid` | The cache could not be parsed. |

The active GPU and NPU result roots from the fresh rerun audit cleanly:

| Root | Valid DBO rows |
|---|---:|
| `results/` | 110 / 110 |
| `results_npu/` | 115 / 115 |

## 6. Current interpretation boundaries

- OOM rows in `experiment_matrix_summary.csv` are explicit capacity boundaries,
  not missing data.
- GPU and NPU results should be compared only where both have mode-matched
  serial baselines and the same metric definition.
- The old “NPU decode 5x” conclusion should not be reused; exact TPOT reruns
  changed the conclusion.
- See [gpu_npu_experiment_summary.md](gpu_npu_experiment_summary.md) for the
  current speedup table and OOM boundaries.
