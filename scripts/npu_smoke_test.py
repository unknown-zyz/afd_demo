#!/usr/bin/env python
"""Lightweight CPU path smoke test for the NPU branch.

Validates that the backend abstraction + distributed path imports and initializes
correctly on a plain CPU host. Does NOT load any HF model (intentionally, so the
test is runnable on any machine regardless of model availability).

Exit status 0 = PASS.

What this covers:
  - src.utils.device.init_backend('cpu')      → device_type='cpu', dist='gloo'
  - src.utils.device.init_backend('auto')     → selects cuda/npu/cpu per env
  - src.utils.device module surface (synchronize/Event/stream_context/empty_cache
    are no-ops on CPU without exceptions).
  - src.distributed import chain and DistributedConfig construction.
  - src.main.parse_args accepts the new --backend/--attn-size/--ffn-size/
    --ffn-tp-size flags and build_distributed_config produces the right backend.

What this does NOT cover (requires real hardware / HF model):
  - init_process_group with gloo (needs 2 ranks; exercised in dry_run_cpu.sh)
  - FFN TP sharding math
  - HCCL / NCCL collective correctness
  - Numerical parity with GPU version
"""
from __future__ import annotations
import argparse
import os
import sys

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    tag = "OK  " if cond else "FAIL"
    print(f"  [{tag}] {name}" + (f" : {detail}" if detail else ""))
    if not cond:
        FAILURES.append(name)


def main() -> int:
    print("=== NPU-branch CPU path smoke test ===\n")

    print("[1] Device abstraction module")
    # Fresh import/reset: the real path runs in a fresh python process, but
    # here we toggle via direct state reset.
    from src.utils import device as d
    d._INITIALIZED = False  # type: ignore[attr-defined]
    d.init_backend("cpu")
    check("backend=cpu resolves to DEVICE_TYPE='cpu'", d.DEVICE_TYPE == "cpu",
          d.DEVICE_TYPE)
    check("backend=cpu resolves to DIST_BACKEND='gloo'", d.DIST_BACKEND == "gloo",
          d.DIST_BACKEND)
    check("device_module() is None on cpu", d.device_module() is None)
    check("is_available() is False on cpu", not d.is_available())
    check("device_count() == 0", d.device_count() == 0)
    check("current_device_str() == 'cpu'", d.current_device_str() == "cpu")
    # These should be callable without raising:
    try:
        d.synchronize()
        d.current_stream_synchronize()
        d.empty_cache()
        d.set_device(0)
        ev = d.Event()
        st = d.Stream()
        with d.stream_context(st):
            pass
        check("cpu no-op API does not raise", True)
    except Exception as e:
        check("cpu no-op API does not raise", False, repr(e))

    print("\n[2] Backend 'auto' resolution")
    d._INITIALIZED = False  # type: ignore[attr-defined]
    resolved = d.init_backend("auto")
    check("auto resolves to one of {cuda,npu,cpu}", resolved in {"cuda", "npu", "cpu"},
          resolved)

    print("\n[3] Argparse accepts new flags")
    sys.argv = [
        "main", "--backend", "cpu",
        "--attn-size", "2", "--ffn-size", "2", "--ffn-tp-size", "2",
        "--local-test", "--batch-size", "1",
    ]
    from src.main import parse_args, build_distributed_config
    args = parse_args()
    check("args.backend == 'cpu'", args.backend == "cpu")
    check("args.attn_size == 2", args.attn_size == 2)
    check("args.ffn_size == 2", args.ffn_size == 2)
    check("args.ffn_tp_size == 2", args.ffn_tp_size == 2)

    print("\n[4] build_distributed_config picks dist backend from devmod")
    # Reset + set to cpu explicitly so assertion reflects --backend=cpu path.
    d._INITIALIZED = False  # type: ignore[attr-defined]
    d.init_backend("cpu")
    # build_distributed_config uses devmod.DIST_BACKEND, which is 'gloo' here.
    os.environ.setdefault("RANK", "0"); os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "2")
    cfg = build_distributed_config(args)
    check("DistributedConfig.backend == 'gloo'", cfg.backend == "gloo", cfg.backend)

    print("\n[5] src.distributed import + context singleton")
    from src.distributed import get_distributed_context, DistributedConfig
    ctx = get_distributed_context()
    check("get_distributed_context() returns object", ctx is not None)
    _ = DistributedConfig(world_size=2)
    check("DistributedConfig dataclass constructs", True)

    print("\n[6] Core model imports succeed")
    try:
        from src.model import DisaggregatedQwenModel  # noqa: F401
        from src.pipeline import SimplePipelineScheduler, AsyncPipelineScheduler  # noqa: F401
        check("model + pipeline modules import without cuda", True)
    except Exception as e:
        check("model + pipeline modules import without cuda", False, repr(e))

    print("\n=== Summary ===")
    if FAILURES:
        print(f"FAIL: {len(FAILURES)} checks failed:")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("PASS: all smoke checks OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
