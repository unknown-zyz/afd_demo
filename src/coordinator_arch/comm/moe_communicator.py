"""
MoE Communicator wrapping DeepEP-Ascend Buffer.

This module provides a wrapper around DeepEP-Ascend's distributed communication
for Mixture of Experts models on Ascend 910C NPUs. It exposes a stable API
compatible with FallbackMoECommunicator.
"""

import logging
from typing import Dict, Optional

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

# Lazy import deep_ep to avoid hard dependency
_DEEP_EP_AVAILABLE = False
_deep_ep = None

try:
    import deep_ep as _deep_ep
    _DEEP_EP_AVAILABLE = True
    logger.info("DeepEP-Ascend library available")
except ImportError:
    logger.debug("DeepEP-Ascend library not available, use FallbackMoECommunicator")


class MoECommunicator:
    """
    Wraps DeepEP-Ascend Buffer. Provides a stable API matching FallbackMoECommunicator.

    DeepEP-Ascend on 910C supports two modes:
      - normal: high throughput, larger buffer; used for Prefill or batch >= 64.
      - low_latency: lower latency, smaller transactions; used for Decode (batch < 64).
    DeepEP is experimental in this repository. FallbackMoECommunicator is the
    default production path; this class is used only when explicitly requested.
    """

    def __init__(
        self,
        ep_group: torch.distributed.ProcessGroup,
        hidden_size: int,
        num_experts: int,
        max_tokens_per_rank: int,
        device: torch.device,
        mode: str = "normal",
        nvl_bytes: int = 256 * 1024 * 1024,
    ) -> None:
        """
        Initialize the DeepEP-Ascend MoE communicator.

        Args:
            ep_group: Expert parallel process group
            hidden_size: Hidden dimension size
            num_experts: Total number of experts
            max_tokens_per_rank: Maximum tokens per rank (for capacity planning)
            device: Device for tensor operations
            mode: Operation mode - "normal" (high throughput) or "low_latency"
            nvl_bytes: HCCS communication buffer size (default: 256MB)

        Raises:
            RuntimeError: If deep_ep library is not available
        """
        if not _DEEP_EP_AVAILABLE:
            raise RuntimeError(
                "DeepEP-Ascend library not available. Please install deep_ep "
                "according to doc/deepep_ascend_install_report.md or use "
                "FallbackMoECommunicator instead."
            )

        self.ep_group = ep_group
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.max_tokens_per_rank = max_tokens_per_rank
        self.device = device
        self.nvl_bytes = nvl_bytes

        # State
        self.mode = mode
        self._buffer: Optional[object] = None  # deep_ep.Buffer instance
        self.expert_to_rank: Optional[torch.Tensor] = None
        self.version: Optional[int] = None

        self.world_size = self._get_world_size()

        logger.info(
            "MoECommunicator initialized (experimental): mode=%s, "
            "hidden_size=%s, num_experts=%s, world_size=%s, nvl_bytes=%s",
            mode, hidden_size, num_experts, self.world_size, nvl_bytes,
        )

    def _get_world_size(self) -> int:
        """Return EP group world size, with a safe fallback for mocked tests."""
        try:
            world_size = dist.get_world_size(self.ep_group)
            return world_size if isinstance(world_size, int) else 1
        except Exception:
            return 1

    @staticmethod
    def _unwrap_tensor(value):
        """DeepEP may return tensors directly or nested in (tensor, scale) tuples."""
        if isinstance(value, (tuple, list)):
            return value[0]
        return value

    def _create_buffer(self) -> None:
        """Create or recreate the DeepEP Buffer with current mode."""
        if self._buffer is not None:
            # Free old buffer if it exists
            del self._buffer
            self._buffer = None

        if self.mode == "low_latency":
            rdma_bytes = _deep_ep.Buffer.get_low_latency_rdma_size_hint(
                self.max_tokens_per_rank,
                self.hidden_size,
                self.world_size,
                self.num_experts,
            )
            num_qps_per_rank = max(1, self.num_experts // max(1, self.world_size))
            logger.debug(
                "Creating DeepEP low_latency Buffer: rdma_bytes=%s, qps_per_rank=%s",
                rdma_bytes, num_qps_per_rank,
            )
            self._buffer = _deep_ep.Buffer(
                group=self.ep_group,
                num_nvl_bytes=0,
                num_rdma_bytes=rdma_bytes,
                low_latency_mode=True,
                num_qps_per_rank=num_qps_per_rank,
            )
        else:
            logger.debug(
                "Creating DeepEP normal Buffer: nvl_bytes=%s",
                self.nvl_bytes,
            )
            self._buffer = _deep_ep.Buffer(
                group=self.ep_group,
                num_nvl_bytes=self.nvl_bytes,
                low_latency_mode=False,
            )

        logger.info(f"DeepEP Buffer created with mode={self.mode}")

    def update_routing_table(self, table: Dict) -> None:
        """
        Update the expert-to-rank routing table.

        Args:
            table: Dictionary containing:
                - version: int, routing table version
                - expert_to_rank: List[int], mapping from expert ID to rank
                - mode: str, operation mode hint (e.g., "normal", "low_latency")
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

        # Check if mode hint differs from current mode
        mode_hint = table.get("mode", "normal")
        if mode_hint != self.mode:
            logger.info(
                f"Routing table suggests mode change: {self.mode} -> {mode_hint}"
            )
            self.set_mode(mode_hint)

        logger.debug(
            f"Routing table updated: version={self.version}, mode={self.mode}"
        )

    def set_mode(self, mode: str) -> None:
        """
        Set the operation mode.

        If the new mode differs from the current mode, the internal Buffer
        is freed and reallocated with the new low_latency setting.

        Args:
            mode: Operation mode - "normal" or "low_latency"
        """
        if mode not in ("normal", "low_latency"):
            raise ValueError(f"Invalid mode: {mode}. Must be 'normal' or 'low_latency'")

        if mode == self.mode:
            logger.debug(f"Mode already set to {mode}, no change needed")
            return

        logger.info(f"Changing mode: {self.mode} -> {mode}")
        self.mode = mode

        # Recreate buffer with new mode
        if self._buffer is not None:
            self._create_buffer()

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> Dict:
        """
        Dispatch tokens to expert ranks via DeepEP-Ascend.

        Args:
            hidden_states: Input token embeddings [N, H]
            topk_indices: Expert indices for each token [N, K]
            topk_weights: Expert weights for each token [N, K]

        Returns:
            Dictionary handle containing:
                - recv_hidden: Dispatched token embeddings [M, H]
                - expert_token_nums: Number of tokens for each local expert
                - topk_indices: Cached topk_indices for combine phase
                - topk_weights: Cached topk_weights for combine phase
                - _mode: Current mode (for debugging)
        """
        # Lazy initialization of buffer
        if self._buffer is None:
            self._create_buffer()

        if self.mode == "low_latency":
            ret = self._buffer.low_latency_dispatch(
                hidden_states,
                topk_indices,
                self.max_tokens_per_rank,
                self.num_experts,
                use_fp8=False,
            )
            recv_hidden = self._unwrap_tensor(ret[0])
            expert_token_nums = ret[1]
            deepep_handle = ret[2]
            return {
                "recv_hidden": recv_hidden,
                "expert_token_nums": expert_token_nums,
                "deepep_handle": deepep_handle,
                "topk_indices": topk_indices,
                "topk_weights": topk_weights,
                "_mode": self.mode,
                "_comm_impl": "deepep",
            }

        layout = self._buffer.get_dispatch_layout(topk_indices, self.num_experts)
        (
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            num_tokens_per_expert,
            is_token_in_rank,
            _layout_event,
        ) = layout

        dispatch_kwargs = {
            "num_tokens_per_rank": num_tokens_per_rank,
            "num_tokens_per_rdma_rank": num_tokens_per_rdma_rank,
            "is_token_in_rank": is_token_in_rank,
            "num_tokens_per_expert": num_tokens_per_expert,
            "topk_idx": topk_indices,
            "topk_weights": topk_weights,
        }
        if hasattr(_deep_ep.Buffer, "get_dispatch_config"):
            dispatch_kwargs["config"] = _deep_ep.Buffer.get_dispatch_config(self.world_size)

        (
            recv_hidden,
            recv_topk_indices,
            recv_topk_weights,
            num_recv_per_expert_list,
            deepep_handle,
            _dispatch_event,
        ) = self._buffer.dispatch(hidden_states, **dispatch_kwargs)

        recv_hidden = self._unwrap_tensor(recv_hidden)

        logger.debug(
            "DeepEP dispatch complete: input_shape=%s, output_shape=%s, mode=%s",
            getattr(hidden_states, "shape", None),
            getattr(recv_hidden, "shape", None),
            self.mode,
        )

        return {
            "recv_hidden": recv_hidden,
            "expert_token_nums": num_recv_per_expert_list,
            "deepep_handle": deepep_handle,
            "topk_indices": topk_indices,
            "topk_weights": topk_weights,
            "recv_topk_indices": recv_topk_indices,
            "recv_topk_weights": recv_topk_weights,
            "_mode": self.mode,
            "_comm_impl": "deepep",
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
        if self._buffer is None:
            raise RuntimeError("Buffer not initialized. Call dispatch() first.")

        if "deepep_handle" not in dispatch_handle:
            raise KeyError("dispatch_handle missing DeepEP handle")

        if dispatch_handle.get("_mode") == "low_latency":
            combined, _event, _hook = self._buffer.low_latency_combine(
                ffn_outputs,
                dispatch_handle["topk_indices"],
                dispatch_handle["topk_weights"],
                dispatch_handle["deepep_handle"],
            )
            return combined

        combine_kwargs = {
            "topk_weights": dispatch_handle.get(
                "recv_topk_weights",
                dispatch_handle["topk_weights"],
            ),
        }
        if hasattr(_deep_ep.Buffer, "get_combine_config"):
            combine_kwargs["config"] = _deep_ep.Buffer.get_combine_config(self.world_size)

        combined_ret = self._buffer.combine(
            ffn_outputs,
            dispatch_handle["deepep_handle"],
            **combine_kwargs,
        )
        combined = self._unwrap_tensor(combined_ret)

        logger.debug(
            f"Combine complete: input shape {ffn_outputs.shape}, "
            f"output shape {combined.shape}"
        )

        return combined

    @property
    def deepep_available(self) -> bool:
        """Check if DeepEP-Ascend is available."""
        return True
