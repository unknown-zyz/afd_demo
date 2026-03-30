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
from typing import Optional, Tuple, List, Any, Union

import torch
import torch.nn as nn
from transformers import PreTrainedModel

logger = logging.getLogger(__name__)


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
    ):
        super().__init__()
        self.input_layernorm = input_layernorm
        self.self_attn = self_attn
        self.hidden_size = hidden_size
        self.layer_idx = layer_idx
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
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

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

        if use_cache:
            return attn_output, residual, present_key_value
        return attn_output, residual


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
        for idx, layer in enumerate(model.model.layers):
            attn_layer = AttentionLayer(
                input_layernorm=layer.input_layernorm.to(device=device, dtype=dtype),
                self_attn=layer.self_attn.to(device=device, dtype=dtype),
                hidden_size=self.hidden_size,
                layer_idx=idx,
            )
            self.attention_layers.append(attn_layer)
        
        # Final norm and LM head
        self.norm = model.model.norm.to(device=device, dtype=dtype)
        self.lm_head = model.lm_head.to(device=device, dtype=dtype)
        
        logger.info(f"AttentionWorker initialized with {self.num_layers} layers")
    
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
    
    def forward_attention_layer(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        past_key_value: Optional[Any] = None,
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
        return self.attention_layers[layer_idx](
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            use_cache=use_cache,
            past_key_value=past_key_value,
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
        
        worker = cls(model, device, dtype)
        
        # Free the original model
        del model
        torch.cuda.empty_cache()
        
        return worker
