"""
Detailed timing tracker for DBO pipeline analysis.

Records per-micro-batch timing for each stage:
- Attention compute
- FFN compute  
- Send (async)
- Recv wait

Supports two timing modes:
- "cuda_events" (default): Uses current_stream().synchronize() + perf_counter.
  Stream-level sync only blocks the default compute stream (not NCCL streams),
  preserving DBO overlap while giving consistent CPU timestamps for all events.
  This ensures compute and communication events share a single timeline.
- "sync": Legacy mode using torch.cuda.synchronize() (device-level) + perf_counter.
  Syncs ALL streams including NCCL — breaks DBO overlap. For debugging only.

Outputs JSON timeline data for visualization.
"""

import json
import time
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any
from enum import Enum

import torch


class EventType(Enum):
    ATTN_COMPUTE = "attn_compute"
    FFN_COMPUTE = "ffn_compute"
    MOE_ROUTER = "moe_router"
    MOE_EXPERTS = "moe_experts"
    MOE_SHARED_OR_DENSE = "moe_shared_or_dense"
    SEND_START = "send_start"
    SEND_TRANSFER = "send_transfer"   # 真实传输时间（isend 到传输完成，通过轮询检测）
    RECV_POST = "recv_post"
    RECV_WAIT = "recv_wait"


@dataclass
class TimingEvent:
    """Single timing event."""
    event_type: str
    layer_idx: int
    mb_idx: int
    start_time: float
    end_time: float
    node: str  # "attention" or "ffn"
    
    @property
    def duration_ms(self) -> float:
        return (self.end_time - self.start_time) * 1000
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.event_type,
            "layer": self.layer_idx,
            "mb": self.mb_idx,
            "start": self.start_time,
            "end": self.end_time,
            "duration_ms": self.duration_ms,
            "node": self.node,
        }


@dataclass
class LayerTiming:
    """Timing for a single layer."""
    layer_idx: int
    # Per-microbatch timings
    attn_compute: List[float] = field(default_factory=list)  # ms per MB
    ffn_compute: List[float] = field(default_factory=list)   # ms per MB
    send_time: List[float] = field(default_factory=list)     # ms per MB
    recv_wait: List[float] = field(default_factory=list)     # ms per MB


@dataclass 
class PipelineTiming:
    """Complete pipeline timing data."""
    node: str  # "attention" or "ffn"
    num_layers: int = 0
    num_micro_batches: int = 0
    total_time_ms: float = 0.0
    prefill_ms: float | None = None
    decode_loop_ms: float | None = None
    decode_steps: int | None = None
    decode_tpot_ms: float | None = None
    representative_itl_ms: float | None = None
    events: List[TimingEvent] = field(default_factory=list)
    
    # Aggregated stats
    total_compute_ms: float = 0.0
    total_recv_wait_ms: float = 0.0  # Only recv_wait matters (actual waiting)
    total_send_wait_ms: float = 0.0  # For reference only
    total_moe_router_ms: float = 0.0
    total_moe_experts_ms: float = 0.0
    total_moe_shared_or_dense_ms: float = 0.0
    
    def add_event(self, event: TimingEvent):
        self.events.append(event)
        
        # Update aggregates
        if event.event_type in (EventType.ATTN_COMPUTE.value, EventType.FFN_COMPUTE.value):
            self.total_compute_ms += event.duration_ms
        elif event.event_type == EventType.MOE_ROUTER.value:
            self.total_moe_router_ms += event.duration_ms
        elif event.event_type == EventType.MOE_EXPERTS.value:
            self.total_moe_experts_ms += event.duration_ms
        elif event.event_type == EventType.MOE_SHARED_OR_DENSE.value:
            self.total_moe_shared_or_dense_ms += event.duration_ms
        elif event.event_type == EventType.RECV_WAIT.value:
            self.total_recv_wait_ms += event.duration_ms
        elif event.event_type == EventType.SEND_TRANSFER.value:
            self.total_send_wait_ms += event.duration_ms
    
    @property
    def compute_ratio(self) -> float:
        """DBO efficiency: compute_time / e2e_time (higher is better)."""
        if self.total_time_ms == 0:
            return 0.0
        return self.total_compute_ms / self.total_time_ms
    
    def to_dict(self) -> Dict[str, Any]:
        data = {
            "node": self.node,
            "num_layers": self.num_layers,
            "num_micro_batches": self.num_micro_batches,
            "total_time_ms": self.total_time_ms,
            "total_compute_ms": self.total_compute_ms,
            "total_recv_wait_ms": self.total_recv_wait_ms,
            "total_moe_router_ms": self.total_moe_router_ms,
            "total_moe_experts_ms": self.total_moe_experts_ms,
            "total_moe_shared_or_dense_ms": self.total_moe_shared_or_dense_ms,
            "compute_ratio": self.compute_ratio,
            "events": [e.to_dict() for e in self.events],
        }
        if self.prefill_ms is not None:
            data["prefill_ms"] = self.prefill_ms
        if self.decode_loop_ms is not None:
            data["decode_loop_ms"] = self.decode_loop_ms
        if self.decode_steps is not None:
            data["decode_steps"] = self.decode_steps
        if self.decode_tpot_ms is not None:
            data["decode_tpot_ms"] = self.decode_tpot_ms
        if self.representative_itl_ms is not None:
            data["representative_itl_ms"] = self.representative_itl_ms
        return data
    
    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)
    
    def save(self, path: str):
        with open(path, 'w') as f:
            f.write(self.to_json())
    
    def summary(self) -> str:
        """Return a concise summary string."""
        lines = [
            f"=== {self.node.upper()} 节点 ===",
            f"端到端: {self.total_time_ms:.2f}ms",
            f"计算: {self.total_compute_ms:.2f}ms ({self.compute_ratio:.1%})",
            f"等待(recv): {self.total_recv_wait_ms:.2f}ms",
        ]
        if self.total_moe_router_ms > 0 or self.total_moe_experts_ms > 0:
            lines.append(
                f"MoE: router={self.total_moe_router_ms:.2f}ms, "
                f"experts={self.total_moe_experts_ms:.2f}ms, "
                f"shared/dense={self.total_moe_shared_or_dense_ms:.2f}ms"
            )
        return "\n".join(lines)


class TimingTracker:
    """
    Tracker for recording pipeline timing events.
    
    Supports two modes:
    - "cuda_events" (default): Stream-level sync + CPU timestamps.
      current_stream().synchronize() only blocks the default compute stream,
      not NCCL streams, so DBO overlap is preserved. All events use CPU
      perf_counter for a single consistent timeline.
    - "sync": Device-level sync (torch.cuda.synchronize()) + CPU timestamps.
      Syncs ALL streams including NCCL — breaks DBO overlap. For debugging.
    
    Usage:
        tracker = TimingTracker("attention", num_layers=48, num_mb=2)
        
        for layer_idx in range(num_layers):
            for mb_idx in range(num_mb):
                tracker.mark_start(EventType.ATTN_COMPUTE, layer_idx, mb_idx)
                # ... GPU compute ...
                tracker.mark_end(EventType.ATTN_COMPUTE, layer_idx, mb_idx)
        
        timing = tracker.finish()
        timing.save("results/timing.json")
    """
    
    def __init__(self, node: str, num_layers: int, num_micro_batches: int,
                 mode: str = "cuda_events"):
        """
        Args:
            node: "attention" or "ffn"
            num_layers: Number of transformer layers
            num_micro_batches: Number of micro-batches
            mode: "cuda_events" (default, stream-level sync) or "sync" (device-level sync)
        """
        self.mode = mode
        self.timing = PipelineTiming(
            node=node,
            num_layers=num_layers,
            num_micro_batches=num_micro_batches,
        )
        self.start_time = time.perf_counter()
        self._sync_start_time: float = 0.0
        self._current_event: Optional[Dict] = None
        
        if mode == "cuda_events" and torch.cuda.is_available():
            # Verify CUDA is available for stream sync
            pass
        elif mode == "cuda_events":
            # Fallback to sync if CUDA not available
            self.mode = "sync"
    
    def mark_start(self, event_type: EventType, layer_idx: int, mb_idx: int):
        """Record start of a GPU compute event.
        
        - cuda_events mode: current_stream().synchronize() + perf_counter.
          Only syncs the default compute stream (not NCCL), preserving DBO overlap.
        - sync mode: torch.cuda.synchronize() + perf_counter (syncs ALL streams).
        """
        if self.mode == "cuda_events":
            torch.cuda.current_stream().synchronize()
        else:
            torch.cuda.synchronize()
        self._sync_start_time = time.perf_counter()
    
    def mark_end(self, event_type: EventType, layer_idx: int, mb_idx: int):
        """Record end of a GPU compute event.
        
        - cuda_events mode: current_stream().synchronize() + perf_counter.
        - sync mode: torch.cuda.synchronize() + perf_counter.
        Both record the event immediately via record_event().
        """
        if self.mode == "cuda_events":
            torch.cuda.current_stream().synchronize()
        else:
            torch.cuda.synchronize()
        end_time = time.perf_counter()
        self.record_event(event_type, layer_idx, mb_idx,
                         self._sync_start_time, end_time)
    
    def record_event(self, event_type: EventType, layer_idx: int, mb_idx: int,
                     start_time: float, end_time: float):
        """Directly record a timing event with CPU timestamps (sync mode or comm events)."""
        event = TimingEvent(
            event_type=event_type.value,
            layer_idx=layer_idx,
            mb_idx=mb_idx,
            start_time=start_time - self.start_time,  # Relative to pipeline start
            end_time=end_time - self.start_time,
            node=self.timing.node,
        )
        self.timing.add_event(event)
    
    def track(self, event_type: EventType, layer_idx: int, mb_idx: int):
        """Return a context manager to track an event (sync mode only)."""
        return _TimingContext(self, event_type, layer_idx, mb_idx)
    
    def finish(self) -> PipelineTiming:
        """Finalize timing and return results."""
        end_time = time.perf_counter()
        self.timing.total_time_ms = (end_time - self.start_time) * 1000
        return self.timing


class _TimingContext:
    """Context manager for a single timing event."""
    
    def __init__(self, tracker: TimingTracker, event_type: EventType, 
                 layer_idx: int, mb_idx: int):
        self.tracker = tracker
        self.event_type = event_type
        self.layer_idx = layer_idx
        self.mb_idx = mb_idx
        self.start_time: float = 0.0
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.perf_counter()
        self.tracker.record_event(
            self.event_type,
            self.layer_idx,
            self.mb_idx,
            self.start_time,
            end_time,
        )
        return False


def merge_timings(attn_timing: PipelineTiming, ffn_timing: PipelineTiming) -> Dict[str, Any]:
    """Merge timing data from both nodes for visualization."""
    return {
        "attention": attn_timing.to_dict(),
        "ffn": ffn_timing.to_dict(),
        "combined": {
            "num_layers": attn_timing.num_layers,
            "num_micro_batches": attn_timing.num_micro_batches,
            "attn_e2e_ms": attn_timing.total_time_ms,
            "ffn_e2e_ms": ffn_timing.total_time_ms,
            "attn_compute_ms": attn_timing.total_compute_ms,
            "ffn_compute_ms": ffn_timing.total_compute_ms,
            "attn_recv_wait_ms": attn_timing.total_recv_wait_ms,
            "ffn_recv_wait_ms": ffn_timing.total_recv_wait_ms,
            "attn_moe_router_ms": attn_timing.total_moe_router_ms,
            "attn_moe_experts_ms": attn_timing.total_moe_experts_ms,
            "attn_moe_shared_or_dense_ms": attn_timing.total_moe_shared_or_dense_ms,
            "ffn_moe_router_ms": ffn_timing.total_moe_router_ms,
            "ffn_moe_experts_ms": ffn_timing.total_moe_experts_ms,
            "ffn_moe_shared_or_dense_ms": ffn_timing.total_moe_shared_or_dense_ms,
            "attn_compute_ratio": attn_timing.compute_ratio,
            "ffn_compute_ratio": ffn_timing.compute_ratio,
        }
    }
