# Copilot instructions for AFD Demo

## Build, test, and check commands

Use the existing Python environment workflow:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

There is no dedicated lint configuration in the repo. For documentation-only changes, run:

```bash
git diff --check
```

For Python code or script behavior changes, run:

```bash
python -m compileall -q src scripts tests
pytest tests/ -q
```

Run a single test file or test case with pytest:

```bash
pytest tests/test_pipeline.py -q
pytest tests/test_pipeline.py::test_name -q
```

Common experiment entry points:

```bash
# GPU single config
./scripts/run_single.sh local 4 128 --tokens 20 --no-dbo --generate
./scripts/run_single.sh local 4 128 --tokens 20
./scripts/run_single.sh local 4 128 --tokens 20 --generate --crosslayer

# GPU matrix
./scripts/run_experiment_matrix.sh \
  --modes serial,prefill-dbo,decode-dbo,decode-dbo-crosslayer \
  --batches 2,4,8,16,32,64 \
  --seqs 128,256,512 \
  --tokens 20

# NPU single config, inside the prepared 910C container/worktree
./scripts/run_npu.sh --attn-size 1 --ffn-size 1 --ffn-tp-size 1 \
  --batch 8 --seq 128 --tokens 20 --model-name "$MODEL_NAME"

# NPU matrix
./scripts/run_experiment_matrix_npu.sh \
  --modes serial,prefill-dbo,decode-dbo,decode-dbo-crosslayer \
  --batches 2,4,8,16,32,64,128,256 \
  --seqs 128,256,512 \
  --tokens 20
```

Post-process result directories with:

```bash
python scripts/plot_all_pipelines.py --root results
python scripts/audit_experiment_baselines.py --root results --output-csv results/baseline_audit.csv
python scripts/plot_all_pipelines.py --root results_npu
python scripts/audit_experiment_baselines.py --root results_npu --output-csv results_npu/baseline_audit.csv
```

## High-level architecture

This repository studies Attention/FFN disaggregation and Dual Batch Overlap (DBO) for Qwen3-30B-A3B inference on CUDA/NCCL and Ascend 910C/HCCL.

- `src/main.py` is the CLI entry point. It initializes the selected device backend, distributed topology, model, scheduler, generation mode, timing output, and optional coordinator routing.
- `src/model/` contains the static A/F disaggregated Qwen path. `DisaggregatedQwenModel` owns the orchestration, `AttentionWorker` owns embedding, attention, KV cache, lm_head, and sampling, and `FFNWorker` owns post-attention norm, MoE gate, experts, and combine.
- `src/pipeline/` contains scheduler variants: `SimplePipelineScheduler` for serial A/F baseline, `AsyncPipelineScheduler` for prefill DBO, and `DecodeDBOScheduler` for decode DBO, crosslayer decode, and EP overlap decode paths.
- `src/distributed/` and `src/utils/device.py` abstract rank roles, process groups, P2P groups, CUDA/NPU/CPU selection, and NCCL/HCCL/Gloo backend selection. Call `init_backend()` before distributed initialization or device placement.
- `src/utils/timing.py` plus scripts such as `gen_experiment_report.py`, `visualize_dbo_pipeline.py`, and `audit_experiment_baselines.py` define timing JSON, reports, Gantt charts, and baseline audits.
- `src/coordinator_arch/` is the newer coordinator-based dynamic MoE subsystem. It adds a gRPC control plane, routing tables, batching queues, worker skeletons, and MoE communicators. It complements, rather than replaces, the static DBO path.

## Repository-specific conventions

- Serial baseline is still an A/F disaggregated path with DBO disabled, not a monolithic model path. Decode baselines should use `--no-dbo --generate` so `decode_tpot_ms` is meaningful.
- Mode semantics are fixed: prefill DBO is the default no-generation path, decode DBO requires `--generate`, and crosslayer decode adds `--crosslayer`.
- Speedup is always `serial / DBO`; values greater than `1.0x` mean DBO is faster. Do not use decode step 1 timing, `decode_step_ms`, or `total_time_ms / max_new_tokens` as final decode speedup.
- For prefill DBO compare `serial_prefill_ms` to DBO `total_time_ms`; for decode and crosslayer compare serial `decode_tpot_ms` to DBO `decode_tpot_ms`.
- Decode pipeline visualizations intentionally use 0-based decode step 1 to explain overlap and bubbles. They are diagnostic only, not final performance denominators.
- Preserve OOM rows in matrix summaries. Matrix scripts stop larger batches for the same `(mode, seq)` after OOM and record `ok`, `cached`, `OOM`, or `FAIL`.
- Result directories are meaningful experiment artifacts: GPU results live under `results/`; NPU results live under `results_npu*`; serial caches live under `results/serial/cache/` or the NPU equivalent.
- The current production/validation communicator in coordinator architecture is `FallbackMoECommunicator` using `torch.distributed.all_to_all_single`; DeepEP is opt-in with `--use-deepep` and remains experimental.
- NPU runs must execute inside the prepared 910C containers/worktrees with model weights mounted read-only. Do not move, overwrite, redownload, or copy Qwen3 weights unless explicitly asked.
- For NPU/HCCL cross-host work, use fresh `MASTER_PORT` and `HCCL_IF_BASE_PORT` values for new phases and keep `HCCL_CONNECT_TIMEOUT`/`HCCL_EXEC_TIMEOUT` in the supported range, commonly `600`.
- Do not use `pkill` or `killall` for cleanup in this environment. If a distributed run leaves stale `python -m src.main` ranks, inspect processes and kill explicit PIDs only.
- Async distributed sends must keep tensor references alive until `handle.wait()` completes, and all FFN EP ranks must enter HCCL collectives in identical layer-major, micro-batch-major order.
- KV cache ownership stays on the Attention role; FFN roles process hidden states and do not own KV cache.
- NPU EP overlap uses full hidden broadcast plus dense reduce with reordered scheduling. Do not describe it as token-aware sparse dispatch/combine unless that path has actually been implemented.
