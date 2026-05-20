"""
Fallback MoE Communicator using torch.distributed all-to-all operations.

This module provides a torch.distributed-based implementation of expert parallel
communication for Mixture of Experts models, serving as a fallback when specialized
communication libraries (e.g., DeepEP) are not available.
"""

import logging
from typing import Dict, List, Optional

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


class FallbackMoECommunicator:
    """
    Fallback MoE communicator using torch.distributed all-to-all.
    
    This class handles expert-parallel communication for MoE layers using
    standard PyTorch distributed primitives. It dispatches tokens to expert ranks
    and combines results back, maintaining compatibility with the Coordinator API.
    """
    
    def __init__(
        self,
        ep_group: dist.ProcessGroup,
        hidden_size: int,
        num_experts: int,
        max_tokens_per_rank: int,
        device: torch.device,
    ):
        """
        Initialize the fallback MoE communicator.
        
        Args:
            ep_group: Expert parallel process group
            hidden_size: Hidden dimension size
            num_experts: Total number of experts
            max_tokens_per_rank: Maximum tokens per rank (for capacity planning)
            device: Device for tensor operations
        """
        self.ep_group = ep_group
        self.world_size = dist.get_world_size(ep_group)
        self.rank = dist.get_rank(ep_group)
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.max_tokens_per_rank = max_tokens_per_rank
        self.device = device
        
        # Routing table state
        self.expert_to_rank: Optional[torch.Tensor] = None
        self.version: Optional[int] = None
        self.mode: str = "normal"
        
        logger.debug(
            f"FallbackMoECommunicator initialized: rank={self.rank}, "
            f"world_size={self.world_size}, hidden_size={hidden_size}, "
            f"num_experts={num_experts}"
        )
    
    def update_routing_table(self, table: Dict) -> None:
        """
        Update the expert-to-rank routing table.
        
        Args:
            table: Dictionary containing:
                - version: int, routing table version
                - expert_to_rank: List[int], mapping from expert ID to rank
                - mode: str, operation mode (e.g., "normal", "recovery")
        """
        self.version = table["version"]
        expert_to_rank_list = table["expert_to_rank"]
        
        if len(expert_to_rank_list) != self.num_experts:
            raise ValueError(
                f"expert_to_rank length {len(expert_to_rank_list)} "
                f"does not match num_experts {self.num_experts}"
            )
        
        self.expert_to_rank = torch.tensor(
            expert_to_rank_list, dtype=torch.long, device=self.device
        )
        self.mode = table.get("mode", "normal")
        
        logger.debug(
            f"Routing table updated: version={self.version}, mode={self.mode}"
        )
    
    def set_mode(self, mode: str) -> None:
        """
        Set the operation mode.
        
        Args:
            mode: Operation mode string (no-op for fallback implementation)
        """
        self.mode = mode
    
    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> Dict:
        """
        Dispatch tokens to expert ranks via all-to-all communication.
        
        Args:
            hidden_states: Input token embeddings [N, H]
            topk_indices: Expert indices for each token [N, K]
            topk_weights: Expert weights for each token [N, K]
        
        Returns:
            Dictionary containing:
                - recv_hidden: Received token embeddings [M, H]
                - recv_weights: Received token weights [M]
                - send_counts: List of token counts sent to each rank
                - recv_counts: List of token counts received from each rank
                - send_perm: Permutation indices for reconstruction
                - original_shape: Original shape tuple (N, K, H)
                - topk_weights: Original topk_weights for combine phase
        """
        if self.expert_to_rank is None:
            raise RuntimeError(
                "update_routing_table must be called before dispatch"
            )
        
        N, H = hidden_states.shape
        K = topk_indices.shape[1]
        
        # Handle K=0 edge case (no experts selected)
        if K == 0:
            return {
                "recv_hidden": torch.zeros(0, H, device=self.device),
                "recv_weights": torch.zeros(0, device=self.device),
                "send_counts": [0] * self.world_size,
                "recv_counts": [0] * self.world_size,
                "send_perm": torch.zeros(0, dtype=torch.long, device=self.device),
                "original_shape": (N, K, H),
                "topk_weights": topk_weights,
            }
        
        # Flatten (N, K) pairs and look up destination ranks
        # For each token n and expert choice k, determine dest_rank
        topk_indices_flat = topk_indices.reshape(-1)  # [N*K]
        topk_weights_flat = topk_weights.reshape(-1)  # [N*K]
        
        # Look up destination rank for each (token, expert) pair
        dest_ranks = self.expert_to_rank[topk_indices_flat]  # [N*K]
        
        # Bucket tokens by destination rank
        send_counts = [0] * self.world_size
        for r in range(self.world_size):
            send_counts[r] = (dest_ranks == r).sum().item()
        
        # Create send buffer and permutation
        NK = N * K
        send_perm = torch.argsort(dest_ranks)  # Sort by destination rank
        
        # Gather hidden states for each (n, k) pair
        token_indices = torch.arange(N, device=self.device).unsqueeze(1).expand(N, K).reshape(-1)
        hidden_states_expanded = hidden_states[token_indices]  # [N*K, H]
        
        # Apply permutation to send in rank-order
        send_hidden = hidden_states_expanded[send_perm]  # [N*K, H]
        send_weights = topk_weights_flat[send_perm]  # [N*K]
        
        # Step 1: Exchange counts
        send_counts_tensor = torch.tensor(
            send_counts, dtype=torch.long, device=self.device
        )
        recv_counts_tensor = torch.zeros(
            self.world_size, dtype=torch.long, device=self.device
        )
        
        dist.all_to_all_single(
            recv_counts_tensor,
            send_counts_tensor,
            group=self.ep_group,
        )
        
        recv_counts = recv_counts_tensor.tolist()
        total_recv = sum(recv_counts)
        
        # Step 2: All-to-all hidden states
        recv_hidden = torch.empty(
            total_recv, H, dtype=hidden_states.dtype, device=self.device
        )
        
        dist.all_to_all_single(
            recv_hidden,
            send_hidden,
            output_split_sizes=recv_counts,
            input_split_sizes=send_counts,
            group=self.ep_group,
        )
        
        # Step 3: All-to-all weights
        recv_weights = torch.empty(
            total_recv, dtype=topk_weights.dtype, device=self.device
        )
        
        dist.all_to_all_single(
            recv_weights,
            send_weights,
            output_split_sizes=recv_counts,
            input_split_sizes=send_counts,
            group=self.ep_group,
        )
        
        logger.debug(
            f"Dispatch complete: sent {NK} tokens, received {total_recv} tokens"
        )
        
        return {
            "recv_hidden": recv_hidden,
            "recv_weights": recv_weights,
            "send_counts": send_counts,
            "recv_counts": recv_counts,
            "send_perm": send_perm,
            "original_shape": (N, K, H),
            "topk_weights": topk_weights,
        }
    
    def combine(
        self,
        ffn_outputs: torch.Tensor,
        dispatch_handle: Dict,
    ) -> torch.Tensor:
        """
        Combine expert outputs back to original token order.
        
        Args:
            ffn_outputs: Expert outputs [M, H]
            dispatch_handle: Handle returned from dispatch()
        
        Returns:
            Combined token embeddings [N, H]
        """
        N, K, H = dispatch_handle["original_shape"]
        send_counts = dispatch_handle["send_counts"]
        recv_counts = dispatch_handle["recv_counts"]
        send_perm = dispatch_handle["send_perm"]
        topk_weights = dispatch_handle["topk_weights"]
        
        # Handle K=0 edge case
        if K == 0:
            return torch.zeros(N, H, device=self.device, dtype=ffn_outputs.dtype)
        
        NK = N * K
        
        # Step 1: Reverse all-to-all to get back [N*K, H] in send_perm order
        gathered_hidden = torch.empty(
            NK, H, dtype=ffn_outputs.dtype, device=self.device
        )
        
        dist.all_to_all_single(
            gathered_hidden,
            ffn_outputs,
            output_split_sizes=send_counts,
            input_split_sizes=recv_counts,
            group=self.ep_group,
        )
        
        # Step 2: Apply inverse permutation to restore (n, k) order
        inverse_perm = torch.argsort(send_perm)
        hidden_restored = gathered_hidden[inverse_perm]  # [N*K, H]
        
        # Step 3: Reshape to [N, K, H]
        hidden_reshaped = hidden_restored.reshape(N, K, H)
        
        # Step 4: Weight and sum over K dimension
        topk_weights_expanded = topk_weights.unsqueeze(-1)  # [N, K, 1]
        weighted = hidden_reshaped * topk_weights_expanded  # [N, K, H]
        output = weighted.sum(dim=1)  # [N, H]
        
        logger.debug(f"Combine complete: output shape {output.shape}")
        
        return output
    
    @property
    def deepep_available(self) -> bool:
        """Check if DeepEP-Ascend is available."""
        return False
