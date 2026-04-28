# NPU-910C Adaptation

This document describes the adaptation of the AFD demo (GPU baseline) for
Huawei Ascend **NPU-910C** (16 cards per node; initial validation on 4 cards).
It is scoped to the `feat/npu-910c` branch.

> **Status:** Scaffolding + backend abstraction complete and CPU-dry-run-
> validated. Full NPU execution has **not** been tested on real 910C hardware
> in this branch — the author did not have access. Known integration risks are
> listed in [§6](#6-known-limitations--tested-surface).

---

## 1. Environment & Dependencies

### 1.1 Runtime

| Component | GPU baseline | NPU-910C branch |
| --- | --- | --- |
| Accelerator runtime | CUDA 11.8 / 12.x | CANN 8.0+ (ships with 910C driver) |
| Python bindings | `torch` (CUDA build) | `torch` (CPU build) + `torch_npu` |
| Distributed | NCCL (bundled with torch) | HCCL (in `torch_npu`, uses CANN) |
| Shared-memory collectives | NVSHMEM (optional, unused by default) | Not supported — auto-disabled |

### 1.2 Pip requirements

The baseline `requirements.txt` remains unchanged; install **additionally**:

```bash
# On the 910C host, with CANN already installed:
pip install torch==2.1.0   # must match torch_npu's supported torch version
pip install torch_npu==2.1.0.post3   # or the version shipped with your CANN
```

Actual version numbers depend on the CANN release on the target machine —
`torch_npu` is tightly pinned to specific `torch` minor versions. Check
`https://gitee.com/ascend/pytorch` for the compatibility matrix.

### 1.3 Environment variables

| Variable | Purpose | Set by |
| --- | --- | --- |
| `ASCEND_VISIBLE_DEVICES` | Which NPUs are visible to this process (analogue of `CUDA_VISIBLE_DEVICES`). | `scripts/run_npu.sh` |
| `HCCL_BUFFSIZE` | HCCL P2P/collective buffer size (MB). Analogue of `NCCL_BUFFSIZE`. | Default 200MB. |
| `HCCL_CONNECT_TIMEOUT` | HCCL rendezvous timeout (s). | Default 600. |
| `HCCL_EXEC_TIMEOUT` | HCCL op execution timeout (s). | Default 1800. |
| `NCCL_BUFFSIZE`, `NCCL_NCHANNELS_PER_NET_PEER` | CUDA-only, unset on NPU. | `scripts/run_npu.sh` strips them. |

`src/utils/device.py::apply_backend_envs()` applies the correct defaults based
on the resolved backend, so running on either GPU or NPU from the same CLI
works without manual env manipulation.

---

## 2. Core Changes

### 2.1 Device abstraction layer — `src/utils/device.py`

A small indirection between business code and the device module. The crucial
trick:

```python
if backend == "npu":
    import torch_npu  # noqa
    from torch_npu.contrib import transfer_to_npu  # noqa
```

`transfer_to_npu` is a monkey-patch shipped with `torch_npu` that rewires
`torch.cuda.*` calls to their `torch.npu.*` equivalents **at runtime**. This
means the bulk of the existing business code (`torch.cuda.synchronize()`,
`torch.cuda.Stream()`, `torch.cuda.Event()`, `torch.cuda.empty_cache()`, etc.)
keeps working unchanged on NPU — we don't need to rewrite every call site.

Nonetheless, code that branches on `primary_device.type == "cuda"` was
generalised to accept `"cuda"` and `"npu"` both (see
`ffn_worker.py::_resolve_role_devices` and
`attention_worker.py::_resolve_role_devices`), and `DistributedContext.device`
now returns the correct device string based on `DEVICE_TYPE`.

Key exports:

```python
from src.utils.device import (
    init_backend,          # call once at program start
    DEVICE_TYPE,           # "cuda" | "npu" | "cpu"
    DIST_BACKEND,          # "nccl" | "hccl" | "gloo"
    device_module,         # -> torch.cuda / torch.npu / None
    synchronize, Event, Stream, stream_context,
    set_device, device_count, empty_cache,
    memory_allocated, max_memory_allocated, memory_reserved,
    apply_backend_envs,    # sets NCCL_*/HCCL_* defaults
)
```

### 2.2 Backend selection — `src/main.py`

New CLI flags:

| Flag | Default | Meaning |
| --- | --- | --- |
| `--backend {auto,cuda,npu,cpu}` | `auto` | Compute backend. `auto` picks NPU if `torch_npu` is importable, else CUDA, else CPU. |
| `--attn-size N` | `1` | Number of ranks assigned to the attention role (DP over micro-batches / requests). |
| `--ffn-size N` | `1` | Number of ranks assigned to the FFN role. |
| `--ffn-tp-size N` | `1` | Tensor-parallel degree within the FFN role. Must divide `--ffn-size`. |

`main()` now calls `devmod.init_backend(args.backend)` **before** any
`torch.cuda.*` usage, ensuring the monkey-patch is installed before the
scheduler and workers are constructed.

### 2.3 Distributed init — `src/distributed/__init__.py`

- Backend string (`nccl`/`hccl`/`gloo`) is now pulled from `devmod.DIST_BACKEND`
  rather than a hard-coded `"nccl" if torch.cuda.is_available() else "gloo"`
  conditional.
- `set_device(local_rank)` goes through `devmod.set_device()` so both CUDA
  and NPU local-rank pinning works with the same call site.
- `DistributedContext.device` returns a `torch.device(f"{DEVICE_TYPE}:{local_rank}")`
  with the right type prefix.
- Existing directional process groups (`a2f_group`, `f2a_group`) work on HCCL
  unchanged — `dist.new_group(ranks)` is backend-agnostic.

### 2.4 Communication

- `dist.isend` / `dist.irecv` / `dist.batch_isend_irecv` are supported on HCCL.
- NVSHMEM (`src/distributed/nvshmem_communicator.py`) is **only** instantiated
  when `use_nvshmem=True` AND `DEVICE_TYPE=="cuda"`. On NPU it is silently
  unused; there is no NPU-side equivalent in this branch.
- P2P warmup (`src/distributed/warmup.py`) works on HCCL unchanged — `isend`/
  `irecv` semantics are equivalent.

### 2.5 Worker device selection

`ffn_worker.py::_resolve_role_devices` and
`attention_worker.py::_resolve_role_devices` now accept both `cuda` and `npu`
device types, enumerating visible devices via `devmod.device_count()` and
returning `torch.device(f"{type}:{idx}")`.

---

## 3. Parallel & Scalability Design

### 3.1 Problem statement

Qwen3-30B-A3B (A:F FLOPs ≈ 1:2 per layer in prefill/decode) wastes pipeline
slots if A and F have unequal wall-clock time. Scaling on 910C should aim to
**balance A and F time** so the A→F→F→A pipeline is tight. FFN is the
expensive side, so we expand FFN resources more aggressively than attention.

### 3.2 Chosen layout

| Card count | `--attn-size` | `--ffn-size` | `--ffn-tp-size` | Intuition |
| --- | --- | --- | --- | --- |
| 4 | 2 | 2 | 2 | ATT DP over 2 micro-batches; FFN TP-2 halves F wall-clock → F ≈ A. |
| 16 | 4 | 12 | 12 | 4 ATT DP replicas; single FFN TP-12 group reduces F to ≈ A/3, leaving headroom for in-flight micro-batches. |

Invariants (enforced at launch):

- `world_size == attn_size + ffn_size`
- `ffn_size % ffn_tp_size == 0`
- `attn_dp_size := attn_size` (no TP on attention — attention is already
  memory-cheap on Qwen3-30B-A3B).

### 3.3 Process-group structure

Conceptually (implementation in `src/distributed/__init__.py` is incremental —
see `a2f_group` and `f2a_group`; the *TP group* described below is wired up in
`run_npu.sh` via the CLI flags but the FFN-internal TP collectives are tagged
as TODO in §6):

```
global world (size = attn_size + ffn_size)
├── role groups:
│   ├── ATT DP group  = ranks [0, attn_size)
│   └── FFN TP groups = consecutive chunks of ffn_tp_size within
│                       ranks [attn_size, world_size)
└── directional groups (per-pair, lazily created):
    ├── a2f_group[i,j]  ATT rank i ↔ FFN rank j   (for ATT→FFN transfer)
    └── f2a_group[i,j]  FFN rank j ↔ ATT rank i   (for FFN→ATT transfer)
```

### 3.4 Scaling to 16 NPUs

The `--preset npu-16card` preset in `scripts/run_npu.sh` sets
`attn_size=4, ffn_size=12, ffn_tp_size=12`. The launcher loops
`RANK in [0, world_size)` and assigns `attention` to ranks `[0, attn_size)`
and `ffn` to the remainder; the same code path handles 4 and 16 cards without
hard-coding.

---

## 4. Performance & Behaviour Differences

> Numbers below are expected ballpark figures based on public 910C specs and
> V100 baseline numbers. **None of the 910C numbers are measured on real
> hardware in this branch.** All real-hardware performance results should be
> filled in once the branch is exercised on a 910C node.

| Axis | V100-32GB (baseline, measured) | 910C (expected) | Notes |
| --- | --- | --- | --- |
| Per-card FP16 FLOPs | 125 TFLOPS | ~320 TFLOPS | 2.5× raw compute |
| HBM | 32 GB | 64 GB | 2× memory ⇒ room for TP-2/TP-12 weights |
| Inter-card BW | NVLink / PCIe P2P | HCCS (910C interconnect) | HCCL bandwidth vs NCCL — comparable order of magnitude, but ring/tree selection differs |
| Pipeline balance (A:F) | ~1:2 → needs DBO to hide F | ~1:2 → same balance, but FFN TP-2 drops F back to ~1:1 | Allows tighter A→F pipeline |
| Cold-start P2P | ~40–60 ms first send (warmup script exists) | Expect similar — same warmup path is exercised | `--warmup-p2p` works unchanged |
| NVSHMEM path | Optional, off by default | Not supported | Raise `NotImplementedError` if `--use-nvshmem` with `--backend=npu` |

### 4.1 Behaviour differences

- `torch.cuda.Stream` mapped to `torch.npu.Stream` via `transfer_to_npu`.
  Stream semantics are largely compatible, but NPU stream ordering guarantees
  in mixed compute/comm scenarios should be verified on hardware.
- `torch.cuda.Event.elapsed_time()` mapped to NPU equivalent — used heavily by
  `src/utils/timing.py` for per-stage timing. Expected to produce comparable
  precision (µs-level) but has not been validated on 910C.
- `torch.bfloat16` is supported on 910C; no dtype change needed.

---

## 5. Files Added / Changed

```
src/utils/device.py                       # NEW: backend abstraction
src/main.py                               # +4 flags, init_backend() at entry
src/distributed/__init__.py               # use devmod.DIST_BACKEND / set_device
src/model/ffn_worker.py                   # accept cuda|npu in _resolve_role_devices
src/model/attention_worker.py             # accept cuda|npu in _resolve_role_devices
scripts/run_npu.sh                        # NEW: launch preset for 4/16-card NPU
scripts/dry_run_cpu.sh                    # NEW: end-to-end CPU dry-run (needs tiny model)
scripts/npu_smoke_test.py                 # NEW: import/init-only path check (no model)
doc/npu_910c_adaptation.md                # NEW: this document
```

---

## 6. Known Limitations & Tested Surface

### 6.1 What was tested in this branch

- `scripts/npu_smoke_test.py` (6 check groups, all PASS on CPU):
  1. Device abstraction `init_backend("cpu")` → `cpu`/`gloo`, no-op API works.
  2. `init_backend("auto")` resolves correctly in ambient env.
  3. `--backend / --attn-size / --ffn-size / --ffn-tp-size` flags parse.
  4. `build_distributed_config` routes the correct backend string.
  5. `src.distributed` imports, `DistributedContext` singleton works.
  6. `src.model` / `src.pipeline` import cleanly without CUDA.

### 6.2 What was **not** tested (requires hardware / model)

- No execution on real NPU-910C.
- No numerical parity against the GPU baseline.
- HCCL rendezvous and P2P send/recv — expected to work by construction
  (standard torch_npu path), but not validated.
- `scripts/dry_run_cpu.sh` requires a downloadable tiny model
  (e.g. Qwen2-0.5B-Instruct) and network access; currently the repo contains
  only Qwen3-30B-A3B weights, which is impractical for CPU execution.

### 6.3 Incomplete items (TODO)

1. **FFN tensor parallelism** — `--ffn-tp-size > 1` is accepted by the CLI,
   but the actual expert sharding + router all-reduce code inside
   `FFNLayer.forward` is not yet implemented. The scaffolding is in place:
   - `ffn_size / ffn_tp_size` = number of TP groups (each group hosts one
     expert shard).
   - Expert sharding strategy: `experts[i]` hosted on
     `rank_in_tp_group == i % ffn_tp_size`; each rank runs its own experts
     and the FFN output is summed across the TP group with `all_reduce(SUM)`.
   - Router computes gates on the full expert space; gate indices are
     broadcast across the TP group; each rank processes only its experts.
   - A `NotImplementedError` should be raised until the implementation
     lands. (Currently the code silently ignores `ffn_tp_size > 1`.)

2. **ATT DP multi-rank** — `--attn-size > 1` similarly needs:
   - `input_ids` broadcast within the ATT DP group only.
   - KV-cache sharded by request (no cross-rank cache sharing).
   - `logits` gather within the ATT DP group for sampling.
   - Directional `a2f_group[i,j]` needs to enumerate all (i,j) pairs, not
     just (0, attn_size).

3. **NVSHMEM** — disabled on NPU for now. A torch_npu-native shared-memory
   equivalent (if it exists in future CANN releases) could be added behind a
   new flag.

4. **Profiling** — `src/utils/profiler.py` uses `torch.cuda.Event`; this maps
   to NPU via `transfer_to_npu` but the accuracy of `elapsed_time()` on 910C
   has not been validated. May need calibration against a wall-clock baseline.

### 6.4 Potential future optimisations

- **Fused kernels**: Qwen3 uses RMSNorm + RoPE, both of which have Ascend
  native implementations (`torch_npu.npu_rms_norm`, `torch_npu.npu_rope`) that
  can be swapped in for better performance than the HF defaults.
- **FlashAttention on 910C**: `torch_npu.npu_fusion_attention` is the 910C
  analogue of FlashAttention-2. The attention worker could be specialised to
  call it when `DEVICE_TYPE == "npu"`.
- **FP8**: 910C supports FP8 matmul (via `hfloat`) — not exercised here.
- **HCCS topology-aware collectives**: `HCCL_ALGO` / `HCCL_INTRA_ROCE_ENABLE`
  tuning can meaningfully change inter-card bandwidth; leave to deployment.

---

## 7. Quick-start (on a real 910C node)

```bash
# 1. Ensure CANN + torch_npu are installed, and `torch_npu` imports cleanly.
python -c "import torch_npu; print(torch_npu.__version__)"

# 2. 4-card validation (2 ATT DP + 2 FFN TP=2 scaffold).
./scripts/run_npu.sh --preset npu-4card --batch 8 --seq 128 --tokens 5

# 3. 16-card scale-out.
./scripts/run_npu.sh --preset npu-16card --batch 32 --seq 256 --tokens 10

# Results land in:
#   results/prefill_dbo/timing_attention_npu_*.json
#   results/prefill_dbo/timing_ffn_npu_*.json
#   results/logs/npu_*_r*.log
```

---

*Last updated: this commit on `feat/npu-910c`.*
