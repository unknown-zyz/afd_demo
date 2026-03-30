"""
Detailed timing tracker for DBO pipeline analysis.

Records per-micro-batch timing for each stage:
- Attention compute
- FFN compute  
- Send (async)
- Recv wait

Outputs JSON timeline data for visualization.
"""

import json
import time
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any
from enum import Enum


class EventType(Enum):
    ATTN_COMPUTE = "attn_compute"
    FFN_COMPUTE = "ffn_compute"
    MOE_ROUTER = "moe_router"
    MOE_EXPERTS = "moe_experts"
    MOE_SHARED_OR_DENSE = "moe_shared_or_dense"
    SEND_START = "send_start"
    SEND_WAIT = "send_wait"
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
        elif event.event_type == EventType.SEND_WAIT.value:
            self.total_send_wait_ms += event.duration_ms
    
    @property
    def compute_ratio(self) -> float:
        """DBO efficiency: compute_time / e2e_time (higher is better)."""
        if self.total_time_ms == 0:
            return 0.0
        return self.total_compute_ms / self.total_time_ms
    
    def to_dict(self) -> Dict[str, Any]:
        return {
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
    Context manager for tracking timing events.
    
    Usage:
        tracker = TimingTracker("attention", num_layers=28, num_mb=2)
        
        for layer_idx in range(num_layers):
            for mb_idx in range(num_mb):
                with tracker.track(EventType.ATTN_COMPUTE, layer_idx, mb_idx):
                    # do attention compute
                    pass
        
        timing = tracker.finish()
        print(timing.summary())
        timing.save("results/timing.json")
    """
    
    def __init__(self, node: str, num_layers: int, num_micro_batches: int):
        self.timing = PipelineTiming(
            node=node,
            num_layers=num_layers,
            num_micro_batches=num_micro_batches,
        )
        self.start_time = time.perf_counter()
        self._current_event: Optional[Dict] = None
    
    def track(self, event_type: EventType, layer_idx: int, mb_idx: int):
        """Return a context manager to track an event."""
        return _TimingContext(self, event_type, layer_idx, mb_idx)
    
    def record_event(self, event_type: EventType, layer_idx: int, mb_idx: int,
                     start_time: float, end_time: float):
        """Directly record a timing event."""
        event = TimingEvent(
            event_type=event_type.value,
            layer_idx=layer_idx,
            mb_idx=mb_idx,
            start_time=start_time - self.start_time,  # Relative to pipeline start
            end_time=end_time - self.start_time,
            node=self.timing.node,
        )
        self.timing.add_event(event)
    
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
