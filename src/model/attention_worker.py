"""
Attention Worker - Handles the attention computation in AFD.

This worker runs on the attention node and processes:
- Input embedding (for first layer)
- RMSNorm (input)
- Self-Attention with RoPE
- Residual connection
- LM Head (for last layer)
"""

import logging
import inspect
import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple, List, Any, Union, Callable

import torch
import torch.nn as nn
from transformers import PreTrainedModel

from ..distributed import get_distributed_context

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AttentionOptimizationConfig:
    """Runtime controls for AttentionWorker optimization experiments."""

    attn_kernel: str = "hf"
    precopy_layer_inputs: bool = False
    attn_tp_size: int = 1
    fused_rmsnorm: bool = False
    fused_rope: bool = False
    stream_overlap: bool = False

    @classmethod
    def from_env(cls) -> "AttentionOptimizationConfig":
        return cls(
            attn_kernel=os.environ.get("AFD_ATTN_KERNEL", "hf"),
            precopy_layer_inputs=os.environ.get("AFD_ATTN_PRECOPY_LAYER_INPUTS", "0") == "1",
            attn_tp_size=int(os.environ.get("AFD_ATTN_TP_SIZE", "1")),
            fused_rmsnorm=os.environ.get("AFD_ATTN_FUSED_RMSNORM", "0") == "1",
            fused_rope=os.environ.get("AFD_ATTN_FUSED_ROPE", "0") == "1",
            stream_overlap=os.environ.get("AFD_ATTN_STREAM_OVERLAP", "0") == "1",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "attn_kernel": self.attn_kernel,
            "attn_precopy_layer_inputs": self.precopy_layer_inputs,
            "attn_tp_size": self.attn_tp_size,
            "attn_fused_rmsnorm": self.fused_rmsnorm,
            "attn_fused_rope": self.fused_rope,
            "attn_stream_overlap": self.stream_overlap,
        }


@dataclass(frozen=True)
class AttentionLayerInputs:
    attention_mask: Optional[torch.Tensor] = None
    position_ids: Optional[torch.Tensor] = None
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None


class AttentionLayerInputCache:
    """Pre-position layer-invariant tensors on each Attention layer device."""

    def __init__(
        self,
        inputs_by_device: dict[str, AttentionLayerInputs],
    ) -> None:
        self.inputs_by_device = inputs_by_device

    def get(self, layer_device: torch.device) -> AttentionLayerInputs:
        key = str(layer_device)
        try:
            return self.inputs_by_device[key]
        except KeyError as exc:
            raise RuntimeError(f"No cached attention inputs for layer device {key}") from exc


class AttentionLayer(nn.Module):
    """
    A single attention layer extracted from Qwen model.
    
    This contains:
    - Input RMSNorm
    - Self-Attention
    - Residual add
    """
    
    def __init__(
        self,
        input_layernorm: nn.Module,
        self_attn: nn.Module,
        hidden_size: int,
        layer_idx: int,
        layer_device: torch.device,
        output_device: torch.device,
        optimization_config: Optional[AttentionOptimizationConfig] = None,
    ):
        super().__init__()
        self.input_layernorm = input_layernorm
        self.self_attn = self_attn
        self.hidden_size = hidden_size
        self.layer_idx = layer_idx
        self.layer_device = layer_device
        self.output_device = output_device
        self.optimization_config = optimization_config or AttentionOptimizationConfig()
        self._forward_params = set(inspect.signature(self.self_attn.forward).parameters.keys())
        self._uses_position_embeddings = "position_embeddings" in self._forward_params
        self._uses_past_key_values = "past_key_values" in self._forward_params
        self._uses_past_key_value = "past_key_value" in self._forward_params
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Any] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        inputs_prepositioned: bool = False,
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, Optional[Any]],
    ]:
        """
        Forward pass for attention layer.
        
        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size]
            attention_mask: Attention mask
            position_ids: Position IDs for RoPE (legacy, used to compute position_embeddings)
            past_key_value: Cached KV state from previous tokens
            use_cache: Whether to return present_key_value
            position_embeddings: Pre-computed (cos, sin) for RoPE (new API)
        
        Returns:
            - attn_output: Output of attention (before residual add)
            - residual: The original hidden_states for residual connection
            - present_key_value: Updated cache (if use_cache=True)
        """
        if hidden_states.device != self.layer_device:
            hidden_states = hidden_states.to(self.layer_device, non_blocking=True)
        if inputs_prepositioned:
            if attention_mask is not None and attention_mask.device != self.layer_device:
                raise RuntimeError(
                    f"Layer {self.layer_idx} received cached attention_mask on "
                    f"{attention_mask.device}, expected {self.layer_device}"
                )
            if position_ids is not None and position_ids.device != self.layer_device:
                raise RuntimeError(
                    f"Layer {self.layer_idx} received cached position_ids on "
                    f"{position_ids.device}, expected {self.layer_device}"
                )
            if position_embeddings is not None:
                cos, sin = position_embeddings
                if cos.device != self.layer_device or sin.device != self.layer_device:
                    raise RuntimeError(
                        f"Layer {self.layer_idx} received cached position_embeddings on "
                        f"{cos.device}/{sin.device}, expected {self.layer_device}"
                    )
        else:
            if attention_mask is not None and attention_mask.device != self.layer_device:
                attention_mask = attention_mask.to(self.layer_device, non_blocking=True)
            if position_ids is not None and position_ids.device != self.layer_device:
                position_ids = position_ids.to(self.layer_device, non_blocking=True)
            if position_embeddings is not None:
                cos, sin = position_embeddings
                if cos.device != self.layer_device or sin.device != self.layer_device:
                    position_embeddings = (
                        cos.to(self.layer_device, non_blocking=True),
                        sin.to(self.layer_device, non_blocking=True),
                    )

        residual = hidden_states
        hidden_states = self._apply_input_layernorm(hidden_states)

        if self.optimization_config.attn_kernel == "npu-official":
            attn_output, present_key_value = self._forward_npu_official_attention(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            if attn_output.device != self.output_device:
                attn_output = attn_output.to(self.output_device, non_blocking=True)
            if residual.device != self.output_device:
                residual = residual.to(self.output_device, non_blocking=True)
            if use_cache:
                return attn_output, residual, present_key_value
            return attn_output, residual

        attn_kwargs = {
            "hidden_states": hidden_states,
            "attention_mask": attention_mask,
        }

        # transformers >= 5.x (Qwen2Attention): position_embeddings is mandatory.
        if self._uses_position_embeddings:
            if position_embeddings is None:
                raise ValueError(
                    f"Layer {self.layer_idx} requires position_embeddings, but received None."
                )
            attn_kwargs["position_embeddings"] = position_embeddings
            if self._uses_past_key_values:
                attn_kwargs["past_key_values"] = past_key_value
        else:
            # transformers 4.x fallback path.
            attn_kwargs["position_ids"] = position_ids
            if self._uses_past_key_value:
                attn_kwargs["past_key_value"] = past_key_value

        # Extra kwargs only exist in legacy API, so guard by signature.
        if "use_cache" in self._forward_params:
            attn_kwargs["use_cache"] = use_cache
        if "output_attentions" in self._forward_params:
            attn_kwargs["output_attentions"] = output_attentions

        outputs = self.self_attn(**attn_kwargs)

        if isinstance(outputs, torch.Tensor):
            attn_output = outputs
            present_key_value = past_key_value if use_cache else None
        else:
            if len(outputs) >= 3:
                attn_output = outputs[0]
                present_key_value = outputs[2]
            elif len(outputs) == 2:
                attn_output = outputs[0]
                # New API updates Cache in-place; return the cache object for compatibility.
                present_key_value = past_key_value if use_cache else None
            elif len(outputs) == 1:
                attn_output = outputs[0]
                present_key_value = past_key_value if use_cache else None
            else:
                raise RuntimeError(
                    f"Unexpected attention output length from layer {self.layer_idx}: {len(outputs)}"
                )

        if attn_output.device != self.output_device:
            attn_output = attn_output.to(self.output_device, non_blocking=True)
        if residual.device != self.output_device:
            residual = residual.to(self.output_device, non_blocking=True)

        if use_cache:
            return attn_output, residual, present_key_value
        return attn_output, residual

    def _apply_input_layernorm(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if not self.optimization_config.fused_rmsnorm:
            return self.input_layernorm(hidden_states)
        if self.layer_device.type != "npu":
            raise RuntimeError("--attn-fused-rmsnorm requires an NPU layer device.")
        weight = getattr(self.input_layernorm, "weight", None)
        if weight is None:
            raise RuntimeError("Fused RMSNorm requires an input_layernorm.weight tensor.")
        eps = getattr(
            self.input_layernorm,
            "variance_epsilon",
            getattr(self.input_layernorm, "eps", 1e-6),
        )
        try:
            import torch_npu  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError("--attn-fused-rmsnorm requires torch_npu.") from exc
        rms_norm = getattr(torch_npu, "npu_fused_rms_norm", None)
        if rms_norm is None:
            rms_norm = getattr(torch_npu, "npu_rms_norm", None)
        if rms_norm is None:
            raise RuntimeError("torch_npu does not provide npu_fused_rms_norm or npu_rms_norm.")
        return rms_norm(hidden_states, weight, float(eps))[0]

    def _apply_npu_rotary_mul(
        self,
        states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        if self.layer_device.type != "npu":
            raise RuntimeError("--attn-fused-rope requires an NPU layer device.")
        try:
            import torch_npu  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError("--attn-fused-rope requires torch_npu.") from exc
        rotary_mul = getattr(torch_npu, "npu_rotary_mul", None)
        if rotary_mul is None:
            raise RuntimeError("torch_npu does not provide npu_rotary_mul.")

        cos = self._reshape_rope_trig(cos, states)
        sin = self._reshape_rope_trig(sin, states)
        return rotary_mul(states, cos, sin, "half")

    def _reshape_rope_trig(self, trig: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        if trig.device != states.device:
            trig = trig.to(states.device, non_blocking=True)
        if trig.dtype != states.dtype:
            trig = trig.to(states.dtype)
        if trig.dim() == 2:
            trig = trig.unsqueeze(0).unsqueeze(0)
        elif trig.dim() == 3:
            trig = trig.unsqueeze(1)
        elif trig.dim() != 4:
            raise RuntimeError(f"Unsupported RoPE trig rank for npu_rotary_mul: {trig.dim()}")
        return trig.contiguous()

    def _forward_npu_official_attention(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]],
        past_key_value: Optional[Any],
        use_cache: bool,
        output_attentions: bool,
    ) -> Tuple[torch.Tensor, Optional[Any]]:
        if output_attentions:
            raise NotImplementedError("npu-official attention does not return attention weights.")
        if position_embeddings is None:
            raise ValueError("npu-official attention requires precomputed position_embeddings.")
        if self.layer_device.type != "npu":
            raise RuntimeError("npu-official attention requires an NPU layer device.")

        flash_prefill, flash_decode = self._load_npu_attention_ops()
        apply_rotary_pos_emb = self._load_apply_rotary_pos_emb()
        batch_size, seq_len, _ = hidden_states.shape
        head_dim = int(getattr(self.self_attn, "head_dim"))
        num_heads = self._num_attention_heads(head_dim)
        num_key_value_heads = self._num_key_value_heads(head_dim)
        scaling = float(getattr(self.self_attn, "scaling", 1.0 / math.sqrt(head_dim)))
        hidden_shape = (batch_size, seq_len, -1, head_dim)

        query_states = self.self_attn.q_proj(hidden_states).view(hidden_shape)
        key_states = self.self_attn.k_proj(hidden_states).view(hidden_shape)
        value_states = self.self_attn.v_proj(hidden_states).view(hidden_shape)
        if hasattr(self.self_attn, "q_norm"):
            query_states = self.self_attn.q_norm(query_states)
        if hasattr(self.self_attn, "k_norm"):
            key_states = self.self_attn.k_norm(key_states)

        query_states = query_states.transpose(1, 2).contiguous()
        key_states = key_states.transpose(1, 2).contiguous()
        value_states = value_states.transpose(1, 2).contiguous()
        cos, sin = position_embeddings
        if self.optimization_config.fused_rope:
            query_states = self._apply_npu_rotary_mul(query_states, cos, sin)
            key_states = self._apply_npu_rotary_mul(key_states, cos, sin)
        else:
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        query_states = query_states.contiguous()
        key_states = key_states.contiguous()

        if past_key_value is not None:
            key_states, value_states = past_key_value.update(
                key_states,
                value_states,
                self.layer_idx,
            )
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        npu_mask = self._to_npu_attention_mask(attention_mask)
        if seq_len == 1 and past_key_value is not None:
            attn_output = flash_decode(
                query_states,
                key_states,
                value_states,
                atten_mask=npu_mask,
                num_heads=num_heads,
                input_layout="BNSD",
                scale_value=scaling,
                num_key_value_heads=num_key_value_heads,
            )
        else:
            attn_output = flash_prefill(
                query_states,
                key_states,
                value_states,
                atten_mask=npu_mask,
                num_heads=num_heads,
                input_layout="BNSD",
                scale_value=scaling,
                num_key_value_heads=num_key_value_heads,
            )

        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, -1).contiguous()
        attn_output = self.self_attn.o_proj(attn_output)
        return attn_output, past_key_value if use_cache else None

    def _load_npu_attention_ops(self) -> Tuple[Callable[..., torch.Tensor], Callable[..., torch.Tensor]]:
        try:
            import torch_npu  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError("npu-official attention requires torch_npu.") from exc
        prefill = getattr(torch_npu, "npu_flash_attention", None)
        if prefill is None:
            prefill = getattr(torch_npu, "npu_prompt_flash_attention", None)
        decode = getattr(torch_npu, "npu_incre_flash_attention", None)
        if prefill is None or decode is None:
            raise RuntimeError(
                "npu-official attention requires torch_npu npu_prompt_flash_attention/"
                "npu_flash_attention and npu_incre_flash_attention."
            )
        return prefill, decode

    def _load_apply_rotary_pos_emb(self) -> Callable[..., Tuple[torch.Tensor, torch.Tensor]]:
        try:
            from transformers.models.qwen3_moe.modeling_qwen3_moe import apply_rotary_pos_emb
        except ImportError:
            from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
        return apply_rotary_pos_emb

    def _num_attention_heads(self, head_dim: int) -> int:
        value = getattr(self.self_attn, "num_heads", None)
        if value is None:
            value = getattr(getattr(self.self_attn, "config", None), "num_attention_heads", None)
        if value is None:
            value = self.self_attn.q_proj.out_features // head_dim
        return int(value)

    def _num_key_value_heads(self, head_dim: int) -> int:
        value = getattr(self.self_attn, "num_key_value_heads", None)
        if value is None:
            value = getattr(getattr(self.self_attn, "config", None), "num_key_value_heads", None)
        if value is None:
            value = self.self_attn.k_proj.out_features // head_dim
        return int(value)

    def _to_npu_attention_mask(self, attention_mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if attention_mask is None:
            return None
        if attention_mask.dtype == torch.bool:
            mask = attention_mask
        else:
            mask = attention_mask < 0
        if not bool(mask.any()):
            return None
        return mask.contiguous()


class AttentionWorker(nn.Module):
    """
    Worker that handles all attention computations across layers.
    
    Responsibilities:
    - Embedding layer (input processing)
    - Attention computation for all layers
    - LM head for output projection
    """
    
    def __init__(
        self,
        model: PreTrainedModel,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
        optimization_config: Optional[AttentionOptimizationConfig] = None,
    ):
        """
        Initialize attention worker from a pretrained model.
        
        Args:
            model: The full pretrained model to extract attention from
            device: Device to place the weights on
            dtype: Data type for computation
        """
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.config = model.config
        self.hidden_size = model.config.hidden_size
        self.num_layers = model.config.num_hidden_layers
        self.ctx = get_distributed_context()
        self.optimization_config = optimization_config or AttentionOptimizationConfig.from_env()
        if self.optimization_config.attn_kernel not in {"hf", "npu-official", "flash-attn-npu"}:
            raise ValueError(f"Unsupported attention kernel: {self.optimization_config.attn_kernel}")
        if self.optimization_config.attn_kernel == "flash-attn-npu":
            raise NotImplementedError(
                f"Attention kernel {self.optimization_config.attn_kernel!r} is not wired yet. "
                "Use --attn-kernel hf or --attn-kernel npu-official."
            )
        if self.optimization_config.attn_tp_size < 1:
            raise ValueError("--attn-tp-size must be >= 1")
        self.role_devices = self._resolve_role_devices(device)
        
        # Extract and move components
        logger.info("Extracting attention components from model...")
        
        # Embedding layer
        self.embed_tokens = model.model.embed_tokens.to(device=device, dtype=dtype)
        
        # Rotary embedding - in newer transformers, it's at model.model level
        if hasattr(model.model, 'rotary_emb'):
            self.rotary_emb = model.model.rotary_emb.to(device=device)
        elif hasattr(model.model.layers[0].self_attn, 'rotary_emb'):
            self.rotary_emb = model.model.layers[0].self_attn.rotary_emb.to(device=device)
        else:
            # Fallback: rotary_fn is a function, we'll handle it differently
            self.rotary_emb = None
            logger.warning("rotary_emb not found as module, will use rotary_fn")
        
        # Extract attention layers
        self.attention_layers = nn.ModuleList()
        # Reserve GPU memory on device 0 for NCCL/communicator overhead
        if len(self.role_devices) >= 2:
            layers_on_dev0 = max(1, self.num_layers // len(self.role_devices) - 3)
        else:
            layers_on_dev0 = self.num_layers
        for idx, layer in enumerate(model.model.layers):
            if len(self.role_devices) >= 2:
                layer_device_idx = 0 if idx < layers_on_dev0 else min(1, len(self.role_devices) - 1)
            else:
                layer_device_idx = 0
            layer_device = self.role_devices[layer_device_idx]
            attn_layer = AttentionLayer(
                input_layernorm=layer.input_layernorm.to(device=layer_device, dtype=dtype),
                self_attn=layer.self_attn.to(device=layer_device, dtype=dtype),
                hidden_size=self.hidden_size,
                layer_idx=idx,
                layer_device=layer_device,
                output_device=device,
                optimization_config=self.optimization_config,
            )
            self.attention_layers.append(attn_layer)
        
        # Final norm and LM head
        self.norm = model.model.norm.to(device=device, dtype=dtype)
        self.lm_head = model.lm_head.to(device=device, dtype=dtype)
        
        logger.info(
            "AttentionWorker initialized: layers=%d, devices=%s, optimizations=%s",
            self.num_layers,
            [str(d) for d in self.role_devices],
            self.optimization_config.to_dict(),
        )

    def _resolve_role_devices(self, primary_device: torch.device) -> List[torch.device]:
        """Resolve all visible accelerator devices for role-internal layer sharding."""
        from ..utils import device as devmod
        if primary_device.type not in ("cuda", "npu") or not devmod.is_available():
            return [primary_device]
        if self.ctx.ffn_ep_enabled:
            return [primary_device]
        count = devmod.device_count()
        if count <= 1:
            return [primary_device]
        return [torch.device(f"{primary_device.type}:{idx}") for idx in range(count)]
    
    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Embed input tokens."""
        return self.embed_tokens(input_ids)
    
    def get_position_embeddings(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Get rotary position embeddings."""
        if self.rotary_emb is None:
            # No rotary_emb module available, will be handled in attention layer
            return None
        # rotary_emb may have different signatures in different versions
        try:
            # New API: (hidden_states, position_ids)
            return self.rotary_emb(hidden_states, position_ids)
        except TypeError:
            # Old API: (hidden_states, seq_len=N)
            seq_len = position_ids.shape[-1]
            return self.rotary_emb(hidden_states, seq_len=seq_len)

    def prepare_layer_input_cache(
        self,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Optional[AttentionLayerInputCache]:
        """Copy layer-invariant attention inputs once per device when enabled."""
        if not self.optimization_config.precopy_layer_inputs:
            return None

        unique_devices = {str(layer.layer_device): layer.layer_device for layer in self.attention_layers}
        inputs_by_device: dict[str, AttentionLayerInputs] = {}
        for key, layer_device in unique_devices.items():
            cached_mask = (
                attention_mask.to(layer_device, non_blocking=True)
                if attention_mask is not None and attention_mask.device != layer_device
                else attention_mask
            )
            cached_position_ids = (
                position_ids.to(layer_device, non_blocking=True)
                if position_ids is not None and position_ids.device != layer_device
                else position_ids
            )
            cached_position_embeddings = None
            if position_embeddings is not None:
                cos, sin = position_embeddings
                cached_position_embeddings = (
                    cos.to(layer_device, non_blocking=True) if cos.device != layer_device else cos,
                    sin.to(layer_device, non_blocking=True) if sin.device != layer_device else sin,
                )
            inputs_by_device[key] = AttentionLayerInputs(
                attention_mask=cached_mask,
                position_ids=cached_position_ids,
                position_embeddings=cached_position_embeddings,
            )
        return AttentionLayerInputCache(inputs_by_device)
    
    def forward_attention_layer(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        past_key_value: Optional[Any] = None,
        layer_input_cache: Optional[AttentionLayerInputCache] = None,
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, Optional[Any]],
    ]:
        """
        Run attention for a single layer.
        
        Args:
            layer_idx: Index of the layer
            hidden_states: Input tensor [batch, seq_len, hidden_size]
            attention_mask: Attention mask
            position_ids: Position IDs for RoPE
            use_cache: Whether to return KV for caching
            past_key_value: Cached KV state from previous tokens
        
        Returns:
            - attn_output: Attention output (to be sent to FFN)
            - residual: Residual tensor (to be sent to FFN)
            - present_key_value: Updated cache if use_cache=True
        """
        cached_inputs = None
        if layer_input_cache is not None:
            cached_inputs = layer_input_cache.get(self.attention_layers[layer_idx].layer_device)
            attention_mask = cached_inputs.attention_mask
            position_ids = cached_inputs.position_ids
            position_embeddings = cached_inputs.position_embeddings

        return self.attention_layers[layer_idx](
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            use_cache=use_cache,
            past_key_value=past_key_value,
            inputs_prepositioned=cached_inputs is not None,
        )
    
    def forward_lm_head(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply final norm and LM head."""
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits
    
    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
        optimization_config: Optional[AttentionOptimizationConfig] = None,
    ) -> "AttentionWorker":
        """
        Create AttentionWorker from a pretrained model name.
        
        Note: This loads the full model then extracts attention.
        For production, weights should be loaded directly.
        """
        from transformers import AutoModelForCausalLM
        
        logger.info(f"Loading model {model_name} for attention extraction...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="cpu",  # Load to CPU first
            trust_remote_code=True,
        )
        
        worker = cls(model, device, dtype, optimization_config=optimization_config)
        
        # Free the original model
        del model
        torch.cuda.empty_cache()
        
        return worker
