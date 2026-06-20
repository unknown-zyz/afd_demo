# AGENTS.md

Python research repo for Attention/FFN disaggregated inference and Dual Batch Overlap (DBO) pipeline scheduling on Qwen3-30B-A3B. Dual backend: CUDA/NCCL and Ascend 910C/HCCL.

## Verify changes

No linter or type checker configured. Validate with:

```bash
python -m compileall -q src scripts tests
pytest tests/ -q
```

Single test file or case:

```bash
pytest tests/test_pipeline.py -q
pytest tests/test_pipeline.py::TestMicroBatch::test_advance_layer -q
```

## Key gotchas

- **Serial baseline is still A/F disaggregated.** `--no-dbo` disables DBO but the model runs through the disaggregated path, not a monolithic model. Decode baselines need `--no-dbo --generate` for meaningful `decode_tpot_ms`.
- **Mode flag semantics are non-obvious.** Default (no flags) = prefill-only DBO. Add `--generate` for decode DBO. Add `--generate --crosslayer` for crosslayer decode. Add `--no-dbo --generate` for serial decode baseline.
- **Speedup = serial / DBO.** Values >1.0x mean DBO is faster. Do NOT use decode step 1 pipeline charts or `total_time_ms / max_new_tokens` as speedup denominators.
- **`init_backend()` must be called before `torch.distributed.init_process_group` and before any `.to(device)`.** It patches `torch.cuda.*` to `torch.npu.*` when `--backend=npu`. See `src/utils/device.py`.
- **HF_ENDPOINT defaults to hf-mirror.com.** Set `MODEL_PATH` (GPU) or `MODEL_NAME` (NPU) env vars for model weights.
- **NPU runs require the prepared 910C container/worktree** with model weights mounted read-only. Do not move/copy/redownload weights.
- **Matrix scripts stop at OOM** for a given `(mode, seq)` and record `ok`/`cached`/`OOM`/`FAIL` in CSV. Preserve OOM rows.
- **Async distributed sends must keep tensor references alive until `handle.wait()`.** All FFN EP ranks must enter HCCL collectives in identical layer-major, micro-batch-major order.
- **KV cache ownership stays on Attention role.** FFN ranks only process hidden states.
- **NPU prefill warmup rounds** (default 1 on NPU, 0 on CUDA) are required to absorb per-shape HCCL JIT compile cost. Without them, layers 0-1 of mb0 dominate timing.
- **Prefill warmup rounds** default to 1 on NPU, 0 on CUDA/CPU. The `--no-dbo` flag disables DBO but still runs the A/F disaggregated path.
- **NPU EP overlap** uses full hidden broadcast + dense reduce with reordered scheduling. Not token-aware sparse dispatch/combine.
- **Communication timing modes:** `enqueue` (default) records `isend()` return overhead; `completion` records effective Work completion span. Neither is pure hardware link latency.
- **Environment variable attention optimizations** (`AFD_ATTN_KERNEL`, `AFD_ATTN_PRECOPY_LAYER_INPUTS`, `AFD_ATTN_FUSED_RMSNORM`, `AFD_ATTN_FUSED_ROPE`) are A/B controls; stable ones should become defaults.
- **Do not use `pkill`/`killall`** for cleanup. Inspect processes and kill explicit PIDs only.
- **Cross-host work** needs fresh `MASTER_PORT`/`HCCL_IF_BASE_PORT` for each phase. `HCCL_CONNECT_TIMEOUT`/`HCCL_EXEC_TIMEOUT` commonly set to 600.
- **`decode_tpot_ms`** is batch-level per-step wall time (one step = one token per sequence in batch). Throughput = `1000 * batch / decode_tpot_ms`.

## Architecture at a glance

- `src/main.py` — CLI entry point, backend init, distributed topology, model load, scheduler selection
- `src/model/disaggregated.py` — Static A/F disaggregated Qwen model; `DisaggregatedQwenModel` orchestrates `AttentionWorker` + `FFNWorker`
- `src/model/ep_moe.py` — Expert-parallel MoE core (broadcast/reduce overlap, all-to-all, sparse P2P)
- `src/pipeline/` — `SimplePipelineScheduler` (serial), `AsyncPipelineScheduler` (prefill DBO), `DecodeDBOScheduler` (decode DBO/crosslayer/EP overlap)
- `src/distributed/` — Rank roles, process groups, P2P groups, FFN EP groups
- `src/coordinator_arch/` — Newer coordinator-based dynamic MoE subsystem (gRPC control plane, routing tables); complements static DBO
- `src/utils/device.py` — CUDA/NPU/CPU abstraction; `init_backend()` selects backend and patches torch
- `src/utils/timing.py` — Timing events, JSON output, profiling metadata

## Results directories

- `results/` — GPU experiment artifacts
- `results_npu*` — NPU experiment artifacts
- `*/serial/cache/` — Serial baseline timing caches (used by DBO report scripts for comparison)
- `*/experiment_matrix_summary.csv` — Matrix sweep status
- `*/baseline_audit.csv` — Whether each DBO result has a mode-matched serial baseline

## Documentation

See `doc/README.md` for reading order. Key docs: `doc/01-architecture.md`, `doc/08-gpu-npu-experiment-summary.md`, `doc/24-msprof-ep-communication-analysis.md`.
