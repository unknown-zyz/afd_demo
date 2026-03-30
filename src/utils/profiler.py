"""
Profiling utilities for AFD Demo.

Provides timing and performance measurement tools.
"""

import time
import functools
import logging
from typing import Optional, Callable, Any
from contextlib import contextmanager
from dataclasses import dataclass, field
from collections import defaultdict

import torch

logger = logging.getLogger(__name__)


@dataclass
class TimingResult:
    """Result of a timing measurement."""
    name: str
    elapsed_ms: float
    count: int = 1
    
    @property
    def avg_ms(self) -> float:
        return self.elapsed_ms / self.count if self.count > 0 else 0


class Timer:
    """Simple CPU timer context manager."""
    
    def __init__(self, name: str = ""):
        self.name = name
        self.start_time: Optional[float] = None
        self.elapsed: float = 0.0
    
    def __enter__(self) -> "Timer":
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, *args) -> None:
        assert self.start_time is not None
        self.elapsed = (time.perf_counter() - self.start_time) * 1000  # ms
        if self.name:
            logger.debug(f"{self.name}: {self.elapsed:.2f}ms")
    
    @property
    def elapsed_ms(self) -> float:
        return self.elapsed


class CUDATimer:
    """
    CUDA-aware timer using CUDA events.
    
    Provides accurate GPU timing by using CUDA events
    and synchronization.
    """
    
    def __init__(self, name: str = "", sync: bool = True):
        """
        Initialize CUDA timer.
        
        Args:
            name: Name for logging
            sync: Whether to synchronize before starting
        """
        self.name = name
        self.sync = sync
        self.elapsed: float = 0.0
        
        if torch.cuda.is_available():
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)
        else:
            self.start_event = None
            self.end_event = None
            self._cpu_start: Optional[float] = None
    
    def __enter__(self) -> "CUDATimer":
        if self.start_event is not None:
            if self.sync:
                torch.cuda.synchronize()
            self.start_event.record()
        else:
            self._cpu_start = time.perf_counter()
        return self
    
    def __exit__(self, *args) -> None:
        if self.end_event is not None:
            self.end_event.record()
            torch.cuda.synchronize()
            self.elapsed = self.start_event.elapsed_time(self.end_event)  # ms
        else:
            assert self._cpu_start is not None
            self.elapsed = (time.perf_counter() - self._cpu_start) * 1000  # ms
        
        if self.name:
            logger.debug(f"{self.name}: {self.elapsed:.2f}ms")
    
    @property
    def elapsed_ms(self) -> float:
        return self.elapsed


class Profiler:
    """
    Aggregate profiler for collecting multiple timing measurements.
    """
    
    def __init__(self):
        self.timings: dict = defaultdict(list)
        self._active_timers: dict = {}
    
    @contextmanager
    def timer(self, name: str, use_cuda: bool = True):
        """Context manager for timing a code block."""
        if use_cuda and torch.cuda.is_available():
            timer = CUDATimer(name)
        else:
            timer = Timer(name)
        
        with timer:
            yield timer
        
        self.timings[name].append(timer.elapsed_ms)
    
    def record(self, name: str, elapsed_ms: float) -> None:
        """Record a timing measurement."""
        self.timings[name].append(elapsed_ms)
    
    def get_stats(self, name: str) -> dict:
        """Get statistics for a timing category."""
        times = self.timings.get(name, [])
        if not times:
            return {"count": 0, "total_ms": 0, "avg_ms": 0, "min_ms": 0, "max_ms": 0}
        
        return {
            "count": len(times),
            "total_ms": sum(times),
            "avg_ms": sum(times) / len(times),
            "min_ms": min(times),
            "max_ms": max(times),
        }
    
    def report(self) -> str:
        """Generate a timing report."""
        lines = ["=" * 60, "Profiling Report", "=" * 60]
        
        for name in sorted(self.timings.keys()):
            stats = self.get_stats(name)
            lines.append(
                f"{name:30s} | "
                f"count: {stats['count']:4d} | "
                f"total: {stats['total_ms']:8.2f}ms | "
                f"avg: {stats['avg_ms']:8.2f}ms"
            )
        
        lines.append("=" * 60)
        return "\n".join(lines)
    
    def reset(self) -> None:
        """Reset all timings."""
        self.timings.clear()


# Global profiler instance
_global_profiler: Optional[Profiler] = None


def get_profiler() -> Profiler:
    """Get or create the global profiler."""
    global _global_profiler
    if _global_profiler is None:
        _global_profiler = Profiler()
    return _global_profiler


def profile_function(name: Optional[str] = None, use_cuda: bool = True):
    """
    Decorator to profile a function.
    
    Args:
        name: Name for the timing (defaults to function name)
        use_cuda: Whether to use CUDA timing
    """
    def decorator(func: Callable) -> Callable:
        timing_name = name or func.__name__
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            profiler = get_profiler()
            with profiler.timer(timing_name, use_cuda=use_cuda):
                return func(*args, **kwargs)
        
        return wrapper
    return decorator


def print_memory_stats(device: Optional[torch.device] = None) -> None:
    """Print current GPU memory statistics."""
    if not torch.cuda.is_available():
        logger.info("CUDA not available")
        return
    
    if device is None:
        device = torch.cuda.current_device()
    
    allocated = torch.cuda.memory_allocated(device) / 1024**3
    reserved = torch.cuda.memory_reserved(device) / 1024**3
    max_allocated = torch.cuda.max_memory_allocated(device) / 1024**3
    
    logger.info(
        f"GPU Memory - "
        f"Allocated: {allocated:.2f}GB, "
        f"Reserved: {reserved:.2f}GB, "
        f"Max Allocated: {max_allocated:.2f}GB"
    )
