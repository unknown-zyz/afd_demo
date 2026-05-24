"""
Factory for building MoE communicators.

This module provides a factory function to build the appropriate MoE communicator
implementation. FallbackMoECommunicator is the production default because the
DeepEP-Ascend cross-host RDMA path is still experimental on the current 910C
stack.
"""

import logging
from typing import Protocol, runtime_checkable

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


@runtime_checkable
class CommunicatorProtocol(Protocol):
    """
    Protocol defining the interface for MoE communicators.

    Both MoECommunicator and FallbackMoECommunicator implement this interface.
    """

    def update_routing_table(self, table: dict) -> None:
        """Update the expert-to-rank routing table."""
        ...

    def set_mode(self, mode: str) -> None:
        """Set the operation mode."""
        ...

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> dict:
        """Dispatch tokens to expert ranks."""
        ...

    def combine(
        self,
        ffn_outputs: torch.Tensor,
        dispatch_handle: dict,
    ) -> torch.Tensor:
        """Combine expert outputs back to original token order."""
        ...

    @property
    def deepep_available(self) -> bool:
        """Check if DeepEP-Ascend is available."""
        ...


def build_communicator(
    *,
    prefer_deepep: bool = False,
    ep_group: dist.ProcessGroup,
    hidden_size: int,
    num_experts: int,
    max_tokens_per_rank: int,
    device: torch.device,
    mode: str = "normal",
    nvl_bytes: int = 256 * 1024 * 1024,
) -> CommunicatorProtocol:
    """
    Build a MoE communicator.

    Returns FallbackMoECommunicator by default. DeepEP-backed MoECommunicator is
    used only when prefer_deepep=True and deep_ep is importable.

    Args:
        prefer_deepep: Whether to opt in to experimental DeepEP-Ascend
        ep_group: Expert parallel process group
        hidden_size: Hidden dimension size
        num_experts: Total number of experts
        max_tokens_per_rank: Maximum tokens per rank
        device: Device for tensor operations
        mode: Operation mode ("normal" or "low_latency")
        nvl_bytes: HCCS buffer size for DeepEP (default: 256MB)

    Returns:
        A communicator implementing CommunicatorProtocol
    """
    communicator_type = "unknown"
    communicator = None

    if prefer_deepep:
        try:
            from .moe_communicator import MoECommunicator

            communicator = MoECommunicator(
                ep_group=ep_group,
                hidden_size=hidden_size,
                num_experts=num_experts,
                max_tokens_per_rank=max_tokens_per_rank,
                device=device,
                mode=mode,
                nvl_bytes=nvl_bytes,
            )
            communicator_type = "DeepEP-Ascend MoECommunicator"
        except (ImportError, RuntimeError) as e:
            logger.info(f"DeepEP-Ascend not available ({e}), falling back to torch.distributed")
    else:
        logger.info("Using fallback communicator by default; pass --use-deepep to opt in to DeepEP")

    if communicator is None:
        from .fallback_a2a import FallbackMoECommunicator

        communicator = FallbackMoECommunicator(
            ep_group=ep_group,
            hidden_size=hidden_size,
            num_experts=num_experts,
            max_tokens_per_rank=max_tokens_per_rank,
            device=device,
        )
        communicator_type = "FallbackMoECommunicator (torch.distributed)"

    logger.info(f"Built communicator: {communicator_type}")

    return communicator
