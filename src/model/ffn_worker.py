"""
FFN Worker - Handles the feed-forward network computation in AFD.

This worker runs on the FFN node and processes:
- Post-attention RMSNorm
- MLP/FFN computation
- Residual connections
"""

import logging
import math
import time
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import PreTrainedModel

logger = logging.getLogger(__name__)


@dataclass
class FFNStageTiming:
    """Fine-grained FFN stage timing in seconds."""
    router_s: float = 0.0
    experts_s: float = 0.0
    shared_or_dense_s: float = 0.0


class FFNLayer(nn.Module):
    """
    A single FFN layer extracted from Qwen model.
    
    This contains:
    - Post-attention RMSNorm
    - MLP (Gate + Up + Down projections with SiLU)
    - Residual add
    """
    
    def __init__(
        self,
        post_attention_layernorm: nn.Module,
        mlp: nn.Module,
        hidden_size: int,
        layer_idx: int,
        layer_device: torch.device,
    ):
        super().__init__()
        self.post_attention_layernorm = post_attention_layernorm
        self.mlp = mlp
        self.hidden_size = hidden_size
        self.layer_idx = layer_idx
        self.layer_device = layer_device
        self.is_sparse_moe = hasattr(self.mlp, "gate") and hasattr(self.mlp, "experts")
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        output_device: Optional[torch.device] = None,
        return_timing: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, FFNStageTiming]]:
        """
        Forward pass for FFN layer.
        
        Args:
            hidden_states: Pre-combined input (attn_output + residual, added on attention node).
                          If residual is provided separately, adds them first (legacy path).
            residual: Optional separate residual tensor (legacy compatibility).
        
        Returns:
            hidden_states: Output after FFN and residual connections
        """
        if output_device is None:
            output_device = self.layer_device

        if hidden_states.device != self.layer_device:
            hidden_states = hidden_states.to(self.layer_device, non_blocking=True)

        stage_timing = FFNStageTiming()

        # If residual is provided separately (legacy), add it first
        if residual is not None:
            if residual.device != self.layer_device:
                residual = residual.to(self.layer_device, non_blocking=True)
            hidden_states = residual + hidden_states

        # Store for second residual
        residual = hidden_states
        
        # FFN
        hidden_states = self.post_attention_layernorm(hidden_states)
        if self.is_sparse_moe:
            batch_size, seq_len, hidden_dim = hidden_states.shape
            hidden_states_2d = hidden_states.reshape(-1, hidden_dim)

            router_start = time.perf_counter()
            _, routing_weights, selected_experts = self.mlp.gate(hidden_states_2d)
            router_end = time.perf_counter()
            stage_timing.router_s = router_end - router_start

            experts_start = time.perf_counter()
            hidden_states = self.mlp.experts(hidden_states_2d, selected_experts, routing_weights)
            hidden_states = hidden_states.reshape(batch_size, seq_len, hidden_dim)
            experts_end = time.perf_counter()
            stage_timing.experts_s = experts_end - experts_start
        else:
            dense_start = time.perf_counter()
            hidden_states = self.mlp(hidden_states)
            dense_end = time.perf_counter()
            stage_timing.shared_or_dense_s = dense_end - dense_start
        
        # Second residual connection (FFN)
        hidden_states = residual + hidden_states

        if hidden_states.device != output_device:
            hidden_states = hidden_states.to(output_device, non_blocking=True)
        
        if return_timing:
            return hidden_states, stage_timing
        return hidden_states


class FFNWorker(nn.Module):
    """
    Worker that handles all FFN computations across layers.
    
    Responsibilities:
    - Post-attention layer norm
    - MLP computation
    - Residual connections
    """
    
    def __init__(
        self,
        model: PreTrainedModel,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
    ):
        """
        Initialize FFN worker from a pretrained model.
        
        Args:
            model: The full pretrained model to extract FFN from
            device: Device to place the weights on
            dtype: Data type for computation
        """
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.config = model.config
        self.hidden_size = model.config.hidden_size
        self.num_layers = model.config.num_hidden_layers
        self.role_devices = self._resolve_role_devices(device)
        
        # Extract and move components
        logger.info("Extracting FFN components from model...")
        
        # Extract FFN layers
        self.ffn_layers = nn.ModuleList()
        layers_per_device = max(1, math.ceil(self.num_layers / len(self.role_devices)))
        for idx, layer in enumerate(model.model.layers):
            layer_device_idx = min(idx // layers_per_device, len(self.role_devices) - 1)
            layer_device = self.role_devices[layer_device_idx]
            ffn_layer = FFNLayer(
                post_attention_layernorm=layer.post_attention_layernorm.to(device=layer_device, dtype=dtype),
                mlp=layer.mlp.to(device=layer_device, dtype=dtype),
                hidden_size=self.hidden_size,
                layer_idx=idx,
                layer_device=layer_device,
            )
            self.ffn_layers.append(ffn_layer)
        self.supports_moe_timing = any(layer.is_sparse_moe for layer in self.ffn_layers)

        logger.info(
            "FFNWorker initialized: layers=%d, devices=%s, moe_timing=%s",
            self.num_layers,
            [str(d) for d in self.role_devices],
            self.supports_moe_timing,
        )

    def _resolve_role_devices(self, primary_device: torch.device) -> list[torch.device]:
        """Resolve all visible CUDA devices for role-internal layer sharding."""
        if primary_device.type != "cuda" or not torch.cuda.is_available():
            return [primary_device]
        count = torch.cuda.device_count()
        if count <= 1:
            return [primary_device]
        return [torch.device(f"cuda:{idx}") for idx in range(count)]
    
    def forward_ffn_layer(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        attn_output: Optional[torch.Tensor] = None,
        residual: Optional[torch.Tensor] = None,
        return_timing: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, FFNStageTiming]]:
        """
        Run FFN for a single layer.
        
        Args:
            layer_idx: Index of the layer
            hidden_states: Pre-combined input (attn_output + residual)
            attn_output: Deprecated, use hidden_states instead
            residual: Deprecated, use hidden_states instead
        
        Returns:
            hidden_states: Output to send back to attention node
        """
        return self.ffn_layers[layer_idx](
            hidden_states,
            output_device=self.device,
            return_timing=return_timing,
        )
    
    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
    ) -> "FFNWorker":
        """
        Create FFNWorker from a pretrained model name.
        
        Note: This loads the full model then extracts FFN.
        For production, weights should be loaded directly.
        """
        from transformers import AutoModelForCausalLM
        
        logger.info(f"Loading model {model_name} for FFN extraction...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="cpu",  # Load to CPU first
            trust_remote_code=True,
        )
        
        worker = cls(model, device, dtype)
        
        # Free the original model
        del model
        torch.cuda.empty_cache()
        
        return worker
