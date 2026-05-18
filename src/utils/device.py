"""Device backend abstraction for CUDA / NPU / CPU.

Usage:
    from src.utils.device import (
        init_backend, DEVICE_TYPE, device_module,
        synchronize, Event, empty_cache, device_count,
    )

    init_backend("auto")     # once at program start
    torch.Tensor([...]).to(DEVICE_TYPE)

Design:
    - "auto"  → try npu if torch_npu importable, else cuda, else cpu
    - "cuda"  → require torch.cuda
    - "npu"   → require torch_npu (imports torch_npu + transfer_to_npu so that
                existing torch.cuda.* calls redirect to torch.npu.*)
    - "cpu"   → gloo backend, timing Events fall back to perf_counter

Post-init guarantees:
    - DEVICE_TYPE ∈ {"cuda", "npu", "cpu"}
    - DIST_BACKEND ∈ {"nccl", "hccl", "gloo"}
    - device_module() returns torch.cuda / torch.npu / None
"""
from __future__ import annotations

import logging
import os
from typing import Optional

import torch

logger = logging.getLogger(__name__)

DEVICE_TYPE: str = "cpu"
DIST_BACKEND: str = "gloo"
_INITIALIZED = False


def _has_npu() -> bool:
    try:
        import torch_npu  # noqa: F401
        return torch.npu.is_available()
    except Exception:
        return False


def init_backend(backend: str = "auto") -> str:
    """Select compute backend; idempotent.

    Returns the resolved device type: "cuda" | "npu" | "cpu".
    Must be called BEFORE torch.distributed.init_process_group and before
    any .to(device) / torch.cuda.set_device calls.
    """
    global DEVICE_TYPE, DIST_BACKEND, _INITIALIZED
    if _INITIALIZED:
        return DEVICE_TYPE

    backend = backend.lower()
    if backend == "auto":
        if _has_npu():
            backend = "npu"
        elif torch.cuda.is_available():
            backend = "cuda"
        else:
            backend = "cpu"

    if backend == "npu":
        import torch_npu  # noqa: F401
        # transfer_to_npu monkey-patches torch.cuda.* to torch.npu.*,
        # so existing business code stays untouched on NPU.
        try:
            from torch_npu.contrib import transfer_to_npu  # noqa: F401
        except ImportError:
            logger.warning("torch_npu.contrib.transfer_to_npu not available; "
                           "torch.cuda.* calls may fail on NPU.")
        if not torch.npu.is_available():
            raise RuntimeError("backend=npu requested but torch.npu not available")
        DEVICE_TYPE = "npu"
        DIST_BACKEND = "hccl"
    elif backend == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("backend=cuda requested but torch.cuda not available")
        DEVICE_TYPE = "cuda"
        DIST_BACKEND = "nccl"
    elif backend == "cpu":
        DEVICE_TYPE = "cpu"
        DIST_BACKEND = "gloo"
    else:
        raise ValueError(f"Unknown backend: {backend!r}")

    _INITIALIZED = True
    logger.info(f"Device backend initialized: device_type={DEVICE_TYPE}, "
                f"dist_backend={DIST_BACKEND}")
    return DEVICE_TYPE


def device_module():
    """Return torch.cuda / torch.npu / None for cpu."""
    if DEVICE_TYPE == "cuda":
        return torch.cuda
    if DEVICE_TYPE == "npu":
        return torch.npu
    return None


def is_available() -> bool:
    return DEVICE_TYPE in ("cuda", "npu")


def device_count() -> int:
    m = device_module()
    return m.device_count() if m else 0


def current_device_str(local_rank: int = 0) -> str:
    if DEVICE_TYPE == "cuda":
        return f"cuda:{local_rank}"
    if DEVICE_TYPE == "npu":
        return f"npu:{local_rank}"
    return "cpu"


def set_device(local_rank: int) -> None:
    m = device_module()
    if m is not None:
        m.set_device(local_rank)


def synchronize(device: Optional[int] = None) -> None:
    m = device_module()
    if m is None:
        return
    if device is None:
        m.synchronize()
    else:
        m.synchronize(device)


def current_stream_synchronize() -> None:
    m = device_module()
    if m is None:
        return
    m.current_stream().synchronize()


def Event(enable_timing: bool = True):
    """Return a CUDA/NPU Event (for timing). None on CPU."""
    m = device_module()
    if m is None:
        return None
    return m.Event(enable_timing=enable_timing)


def Stream():
    m = device_module()
    if m is None:
        return None
    return m.Stream()


def stream_context(stream):
    """Context manager for stream scheduling; no-op on CPU."""
    m = device_module()
    if m is None or stream is None:
        import contextlib
        return contextlib.nullcontext()
    return m.stream(stream)


def empty_cache() -> None:
    m = device_module()
    if m is not None and hasattr(m, "empty_cache"):
        m.empty_cache()


def memory_allocated(device=None) -> int:
    m = device_module()
    if m is None:
        return 0
    return m.memory_allocated(device) if device is not None else m.memory_allocated()


def max_memory_allocated(device=None) -> int:
    m = device_module()
    if m is None:
        return 0
    return m.max_memory_allocated(device) if device is not None else m.max_memory_allocated()


def memory_reserved(device=None) -> int:
    m = device_module()
    if m is None:
        return 0
    return m.memory_reserved(device) if device is not None else m.memory_reserved()


def apply_backend_envs() -> None:
    """Set sensible backend-specific env vars if not already set."""
    if DEVICE_TYPE == "cuda":
        os.environ.setdefault("NCCL_BUFFSIZE", "33554432")
        os.environ.setdefault("NCCL_NCHANNELS_PER_NET_PEER", "1")
    elif DEVICE_TYPE == "npu":
        # HCCL counterparts; conservative defaults
        os.environ.setdefault("HCCL_BUFFSIZE", "200")  # in MB on 910C
        os.environ.setdefault("HCCL_EXEC_TIMEOUT", "1800")
        os.environ.setdefault("HCCL_CONNECT_TIMEOUT", "600")
    # cpu: no extra env needed
