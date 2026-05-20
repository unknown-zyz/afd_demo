"""
ContinuousBatcher: Token-level micro-batching for attention outputs.

Groups tokens by destination FFN rank and chunks them into micro-batches
for efficient processing in the Coordinator-based MoE architecture.
"""

from dataclasses import dataclass
from typing import List
import torch


@dataclass
class MicroBatch:
    """Represents a micro-batch of tokens destined for the same primary FFN rank."""
    indices: torch.LongTensor      # [m] original token indices
    hidden: torch.Tensor           # [m, H]
    topk_indices: torch.Tensor     # [m, K]
    topk_weights: torch.Tensor     # [m, K]
    chunk_id: int
    primary_rank: int              # dominant dest rank in this MB


class ContinuousBatcher:
    """
    Splits attention outputs into micro-batches based on expert routing.
    
    Tokens are grouped by their primary destination rank (mode of expert assignments)
    and chunked to respect max_tokens_per_chunk limits.
    """
    
    def __init__(self, num_ep_ranks: int, num_experts: int, max_tokens_per_chunk: int = 4096):
        """
        Args:
            num_ep_ranks: Number of expert parallel ranks
            num_experts: Total number of experts
            max_tokens_per_chunk: Maximum tokens per micro-batch
        """
        self.num_ep_ranks = num_ep_ranks
        self.num_experts = num_experts
        self.max_tokens_per_chunk = max_tokens_per_chunk
    
    def split(self,
              hidden_states: torch.Tensor,        # [N, H]
              topk_indices: torch.Tensor,         # [N, K]
              topk_weights: torch.Tensor,         # [N, K]
              expert_to_rank: torch.Tensor        # [num_experts] long
             ) -> List[MicroBatch]:
        """
        Split tokens into micro-batches based on destination rank.
        
        Args:
            hidden_states: Token hidden states [N, H]
            topk_indices: Top-K expert indices per token [N, K]
            topk_weights: Top-K routing weights per token [N, K]
            expert_to_rank: Mapping from expert ID to rank [num_experts]
        
        Returns:
            List of MicroBatch objects
        """
        N, H = hidden_states.shape
        K = topk_indices.shape[1]
        
        # Step 1: Compute primary rank for each token
        # dest_ranks[n, k] = rank for token n's k-th expert
        dest_ranks = expert_to_rank[topk_indices]  # [N, K]
        
        # Compute mode (most frequent rank) for each token
        # For ties, choose smallest rank for determinism
        primary_ranks = torch.zeros(N, dtype=torch.long, device=dest_ranks.device)
        
        for n in range(N):
            ranks_for_token = dest_ranks[n]  # [K]
            # Count occurrences of each rank
            counts = torch.bincount(ranks_for_token, minlength=self.num_ep_ranks)
            # Find max count
            max_count = counts.max()
            # Among ranks with max count, choose smallest
            primary_ranks[n] = torch.where(counts == max_count)[0][0]
        
        # Step 2: Sort tokens by primary rank
        perm = torch.argsort(primary_ranks)
        sorted_ranks = primary_ranks[perm]
        
        # Step 3: Group by rank and chunk
        micro_batches = []
        chunk_id = 0
        
        i = 0
        while i < N:
            current_rank = sorted_ranks[i].item()
            # Find end of current rank group
            j = i
            while j < N and sorted_ranks[j].item() == current_rank:
                j += 1
            
            # Chunk this rank group
            rank_start = i
            while rank_start < j:
                chunk_end = min(rank_start + self.max_tokens_per_chunk, j)
                chunk_perm = perm[rank_start:chunk_end]
                
                mb = MicroBatch(
                    indices=chunk_perm,
                    hidden=hidden_states[chunk_perm],
                    topk_indices=topk_indices[chunk_perm],
                    topk_weights=topk_weights[chunk_perm],
                    chunk_id=chunk_id,
                    primary_rank=current_rank
                )
                micro_batches.append(mb)
                chunk_id += 1
                rank_start = chunk_end
            
            i = j
        
        return micro_batches
    
    def merge(self,
              outputs_per_mb: List[torch.Tensor],
              micro_batches: List[MicroBatch],
              total_tokens: int) -> torch.Tensor:
        """
        Merge micro-batch outputs back to original token order.
        
        Args:
            outputs_per_mb: List of output tensors, one per micro-batch
            micro_batches: List of MicroBatch objects (same order as outputs)
            total_tokens: Total number of tokens (N)
        
        Returns:
            Merged output tensor [N, H]
        """
        if len(outputs_per_mb) == 0:
            raise ValueError("No outputs to merge")
        
        # Get hidden dimension from first output
        H = outputs_per_mb[0].shape[-1]
        device = outputs_per_mb[0].device
        dtype = outputs_per_mb[0].dtype
        
        # Initialize output
        output = torch.zeros(total_tokens, H, dtype=dtype, device=device)
        
        # Track which indices are covered (for defensive checking)
        covered = torch.zeros(total_tokens, dtype=torch.bool, device=device)
        
        # Scatter outputs to original positions
        for mb, out in zip(micro_batches, outputs_per_mb):
            if len(mb.indices) != out.shape[0]:
                raise ValueError(
                    f"Mismatch: MicroBatch has {len(mb.indices)} indices "
                    f"but output has {out.shape[0]} tokens"
                )
            
            # Check no overlap
            if covered[mb.indices].any():
                raise ValueError("Overlapping indices detected in micro-batches")
            
            output[mb.indices] = out
            covered[mb.indices] = True
        
        # Verify all indices covered
        if not covered.all():
            raise ValueError(
                f"Not all tokens covered: {(~covered).sum().item()} tokens missing"
            )
        
        return output
