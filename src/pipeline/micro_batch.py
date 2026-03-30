"""
Micro-batch management for pipeline execution.

Handles splitting batches into micro-batches and managing their state.
"""

import logging
from typing import Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum

import torch

logger = logging.getLogger(__name__)


class MicroBatchState(Enum):
    """State of a micro-batch in the pipeline."""
    WAITING = 0      # Waiting to be processed
    IN_ATTENTION = 1  # Being processed by attention
    IN_TRANSIT_AF = 2 # Sent from attention to FFN
    IN_FFN = 3        # Being processed by FFN
    IN_TRANSIT_FA = 4 # Sent from FFN back to attention
    COMPLETED = 5     # Processing completed


@dataclass
class MicroBatch:
    """
    A micro-batch with its associated state and data.
    
    In the ping-pong pipeline, each micro-batch alternates between
    attention and FFN computation.
    """
    id: int
    
    # Input data
    input_ids: torch.Tensor
    attention_mask: Optional[torch.Tensor] = None
    position_ids: Optional[torch.Tensor] = None
    
    # Current state
    hidden_states: Optional[torch.Tensor] = None
    
    # Pipeline state
    state: MicroBatchState = MicroBatchState.WAITING
    current_layer: int = 0
    
    # Position embeddings (computed once, reused)
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    
    # Timing info (for profiling)
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    
    def advance_layer(self) -> None:
        """Move to the next layer."""
        self.current_layer += 1
    
    def reset_for_layer(self, layer_idx: int) -> None:
        """Reset state for a specific layer."""
        self.current_layer = layer_idx
        self.state = MicroBatchState.WAITING
    
    @property
    def batch_size(self) -> int:
        return self.input_ids.shape[0]
    
    @property
    def seq_len(self) -> int:
        return self.input_ids.shape[1]


class MicroBatchManager:
    """
    Manages micro-batches for pipeline execution.
    
    Handles:
    - Splitting input batch into micro-batches
    - Tracking micro-batch states
    - Merging results back
    """
    
    def __init__(
        self,
        num_micro_batches: int = 2,
        device: torch.device = torch.device("cuda"),
    ):
        """
        Initialize micro-batch manager.
        
        Args:
            num_micro_batches: Number of micro-batches (2 for ping-pong)
            device: Device for tensors
        """
        self.num_micro_batches = num_micro_batches
        self.device = device
        self.micro_batches: List[MicroBatch] = []
    
    def split_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> List[MicroBatch]:
        """
        Split a batch into micro-batches.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
        
        Returns:
            List of MicroBatch objects
        """
        batch_size = input_ids.shape[0]
        
        if batch_size < self.num_micro_batches:
            logger.warning(
                f"Batch size ({batch_size}) < num_micro_batches ({self.num_micro_batches}). "
                f"Using batch_size micro-batches."
            )
            actual_num_micro_batches = batch_size
        else:
            actual_num_micro_batches = self.num_micro_batches
        
        # Calculate micro-batch sizes
        base_size = batch_size // actual_num_micro_batches
        remainder = batch_size % actual_num_micro_batches
        
        self.micro_batches = []
        start_idx = 0
        
        for i in range(actual_num_micro_batches):
            # Distribute remainder across first few micro-batches
            mb_size = base_size + (1 if i < remainder else 0)
            end_idx = start_idx + mb_size
            
            mb_input_ids = input_ids[start_idx:end_idx].to(self.device)
            mb_attention_mask = None
            if attention_mask is not None:
                mb_attention_mask = attention_mask[start_idx:end_idx].to(self.device)
            
            micro_batch = MicroBatch(
                id=i,
                input_ids=mb_input_ids,
                attention_mask=mb_attention_mask,
            )
            self.micro_batches.append(micro_batch)
            
            start_idx = end_idx
            
        logger.info(
            f"Split batch into {len(self.micro_batches)} micro-batches: "
            f"sizes={[mb.batch_size for mb in self.micro_batches]}"
        )
        
        return self.micro_batches
    
    def merge_results(self, results: List[torch.Tensor]) -> torch.Tensor:
        """
        Merge micro-batch results back into a single batch.
        
        Args:
            results: List of result tensors from each micro-batch
        
        Returns:
            Merged tensor [batch_size, ...]
        """
        return torch.cat(results, dim=0)
    
    def get_micro_batch(self, idx: int) -> MicroBatch:
        """Get a specific micro-batch."""
        return self.micro_batches[idx]
    
    def all_completed(self) -> bool:
        """Check if all micro-batches have completed."""
        return all(mb.state == MicroBatchState.COMPLETED for mb in self.micro_batches)
    
    def reset(self) -> None:
        """Reset all micro-batches for a new run."""
        for mb in self.micro_batches:
            mb.state = MicroBatchState.WAITING
            mb.current_layer = 0
            mb.hidden_states = None


def create_position_ids(
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> torch.Tensor:
    """Create position IDs for a batch."""
    return torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)


def create_causal_mask(
    batch_size: int,
    seq_len: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Create a causal attention mask.
    
    Returns:
        Causal mask [batch_size, 1, seq_len, seq_len]
    """
    # Create causal mask (lower triangular)
    mask = torch.tril(
        torch.ones(seq_len, seq_len, device=device, dtype=dtype)
    )
    # Expand for batch and heads
    mask = mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1)
    return mask
