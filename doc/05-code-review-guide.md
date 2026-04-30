# Code review guide

This guide is for reviewing changes to the current AFD/DBO code path. It focuses
on correctness risks that can invalidate timing, speedup, or distributed
behavior.

## 1. Current execution modes

| Mode | User command | Scheduler | Main metric |
|---|---|---|---|
| Serial | `--no-dbo --generate` | `SimplePipelineScheduler` + generation path | `prefill_ms`, `decode_tpot_ms` |
| Prefill DBO | default `run_single.sh` prefill-only path | `AsyncPipelineScheduler` | model-side TTFT-path |
| Decode DBO | `--generate` | `DecodeDBOScheduler` | exact TPOT |
| Decode crosslayer | `--generate --crosslayer` | `DecodeDBOScheduler(use_crosslayer=True)` | exact TPOT |

Review any change by asking which mode it affects and whether serial and DBO are
still being compared with the same metric.

## 2. Files to inspect first

| Area | Files |
|---|---|
| CLI / run orchestration | `src/main.py`, `scripts/run_single.sh`, `scripts/run_npu.sh` |
| Distributed state | `src/distributed/__init__.py`, `src/distributed/warmup.py` |
| Workers | `src/model/attention_worker.py`, `src/model/ffn_worker.py` |
| Generation / KV cache | `src/model/disaggregated.py` |
| Schedulers | `src/pipeline/scheduler.py`, `src/pipeline/async_scheduler.py`, `src/pipeline/decode_scheduler.py` |
| Timing / reports | `src/utils/timing.py`, `scripts/gen_experiment_report.py`, `scripts/experiment_baselines.py` |
| Plotting / audits | `scripts/visualize_dbo_pipeline.py`, `scripts/plot_all_pipelines.py`, `scripts/audit_experiment_baselines.py` |

## 3. Distributed and tensor-lifetime checks

1. Async sends must keep tensor references alive until the send handle is waited.
2. Do not post NCCL/HCCL receives so early that they block later sends on the
   same internal stream. Crosslayer uses directional groups for this reason.
3. After OOM, one rank may wait forever for its peer. Logs must be inspected
   before killing only the matching stuck peer process.
4. `NCCL_BUFFSIZE` must cover the tensors being sent on GPU; otherwise `isend`
   can become flow-control blocked and destroy pipeline timing.

## 4. Timing checks

The valid speedup fields are:

| Mode | Required serial field | Required DBO field |
|---|---|---|
| Prefill | `prefill_ms` | `total_time_ms` |
| Decode | `decode_tpot_ms` | `decode_tpot_ms` |
| Crosslayer | `decode_tpot_ms` | `decode_tpot_ms` |

Do not reintroduce:

- `total_time_ms / max_new_tokens` fallback speedup;
- legacy `decode_step_ms` as a TPOT denominator;
- representative ITL as the report speedup denominator.

Representative ITL is allowed only for pipeline Gantt detail.

## 5. Scheduler-specific review points

### Serial

- Must remain a stable baseline, not optimized differently from DBO in ways that
  change semantics.
- Serial cache must be keyed by `(batch, seq, tokens)`.
- Serial generation must write exact decode loop fields.

### Prefill DBO

- Check peak memory and in-flight buffers when changing micro-batch logic.
- Prefill OOM boundaries are expected to differ from decode boundaries because
  activation memory scales with `batch * seq`.
- NPU prefill uses untimed warmup to absorb HCCL/JIT compile cost.

### Decode DBO

- KV cache slicing must preserve batch order and cache ownership on the
  attention role.
- `decode_steps` must match the actual number of decode-loop iterations.
- Crosslayer changes must be checked for deadlock risk and directional group use.

## 6. Script review points

### GPU matrix

`scripts/run_experiment_matrix.sh`:

- rewrites `results/experiment_matrix_summary.csv` unless phase outputs are
  saved manually;
- caches serial baselines unless `--no-cache` is set;
- adds `--warmup-p2p --warmup-rounds 5`;
- stops larger batch probes for a `(mode, seq)` after OOM.

### NPU matrix

`scripts/run_experiment_matrix_npu.sh`:

- writes to `results_npu/`;
- supports `--append` for multi-phase experiments;
- records visible chip pool and active world size;
- inspects rank logs for OOM.

Review NPU changes with the validated active topology in mind:

```text
attn_size=1, ffn_size=1, ffn_tp_size=1
```

## 7. Result review checklist

Before trusting a new result set:

```bash
python scripts/plot_all_pipelines.py --root results
python scripts/audit_experiment_baselines.py --root results --output-csv results/baseline_audit.csv

python scripts/plot_all_pipelines.py --root results_npu
python scripts/audit_experiment_baselines.py --root results_npu --output-csv results_npu/baseline_audit.csv
```

Check:

1. `baseline_audit.csv` has no missing/fallback rows for cells used in conclusions.
2. OOM rows are present in matrix summaries and not silently dropped.
3. Report speedups match `serial / DBO`.
4. Pipeline PNGs are used for overlap diagnosis, not as the source of final
   speedup claims.
5. GPU and NPU conclusions are taken from
   `doc/gpu_npu_experiment_summary.md`.

## 8. Validation before merge

For documentation-only changes:

```bash
git diff --check
```

For code or script changes:

```bash
source venv/bin/activate
python -m compileall -q src scripts tests
pytest tests/ -q
```

For experiment changes, also run a small serial/DBO smoke before large matrices.
