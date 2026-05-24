"""
FFNQueue: Thread-safe queue for batching FFN computation requests.

Batches requests based on size and time thresholds for efficient
processing in the Coordinator-based MoE architecture.
"""

from collections import deque
from dataclasses import dataclass
from typing import List, Optional
import time
import threading
import torch


@dataclass
class _QueueItem:
    """Internal representation of a queued request."""
    request_id: str
    hidden: torch.Tensor
    weights: torch.Tensor
    src_rank: int
    src_slot: int
    arrival_time: float


@dataclass
class FFNBatch:
    """Batch of FFN computation requests."""
    request_ids: List[str]
    hidden: torch.Tensor          # [B, H]
    weights: torch.Tensor         # [B]
    src_rank: List[int]
    src_slot: List[int]


class FFNQueue:
    """
    Thread-safe queue for FFN computation requests.
    
    Returns batches when either:
    - Queue size reaches max_batch, OR
    - Oldest item has waited >= max_wait_ms
    """
    
    def __init__(self, max_batch: int, max_wait_ms: float = 5.0):
        """
        Args:
            max_batch: Maximum batch size
            max_wait_ms: Maximum wait time in milliseconds
        """
        self.max_batch = max_batch
        self.max_wait_ms = max_wait_ms / 1000.0  # Convert to seconds
        self._queue = deque()
        self._lock = threading.Lock()
    
    def push(self, request_id: str, hidden: torch.Tensor, weights: torch.Tensor,
             src_rank: int, src_slot: int) -> None:
        """
        Add a request to the queue.
        
        Args:
            request_id: Unique identifier for the request
            hidden: Hidden states tensor
            weights: Routing weights tensor
            src_rank: Source rank identifier
            src_slot: Source slot identifier
        """
        item = _QueueItem(
            request_id=request_id,
            hidden=hidden,
            weights=weights,
            src_rank=src_rank,
            src_slot=src_slot,
            arrival_time=time.monotonic()
        )
        
        with self._lock:
            self._queue.append(item)
    
    def pop_batch(self) -> Optional[FFNBatch]:
        """
        Attempt to pop a batch from the queue.
        
        Returns a batch if:
        - Queue size >= max_batch, OR
        - Oldest item waited >= max_wait_ms
        
        Returns:
            FFNBatch if conditions met, None otherwise
        """
        with self._lock:
            if len(self._queue) == 0:
                return None
            
            current_time = time.monotonic()
            oldest_wait = current_time - self._queue[0].arrival_time
            
            # Check if we should return a batch
            should_batch = (
                len(self._queue) >= self.max_batch or
                oldest_wait >= self.max_wait_ms
            )
            
            if not should_batch:
                return None
            
            # Determine batch size
            batch_size = min(len(self._queue), self.max_batch)
            
            # Pop items
            items = []
            for _ in range(batch_size):
                items.append(self._queue.popleft())
            
            # Build batch
            request_ids = [item.request_id for item in items]
            hidden_list = [item.hidden for item in items]
            weights_list = [item.weights for item in items]
            src_ranks = [item.src_rank for item in items]
            src_slots = [item.src_slot for item in items]
            
            # Stack tensors
            # Assumes all hidden tensors have the same shape
            hidden_batch = torch.stack(hidden_list, dim=0)
            weights_batch = torch.stack(weights_list, dim=0)
            
            return FFNBatch(
                request_ids=request_ids,
                hidden=hidden_batch,
                weights=weights_batch,
                src_rank=src_ranks,
                src_slot=src_slots
            )
    
    def __len__(self) -> int:
        """Return current queue size."""
        with self._lock:
            return len(self._queue)
