"""
Disaggregated Qwen Model - Orchestrates attention and FFN workers.

This module provides the main interface for running disaggregated inference,
coordinating between attention and FFN workers across nodes.
"""

import logging
from typing import Optional, Tuple, List, Dict, Any

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig
from transformers.cache_utils import DynamicCache

from .attention_worker import AttentionWorker
from .ffn_worker import FFNWorker
from ..distributed import get_distributed_context
from ..distributed.communicator import AFDCommunicator
from ..utils.sampling import sample_next_token, StoppingCriteria

logger = logging.getLogger(__name__)


class DisaggregatedQwenModel(nn.Module):
    """
    Disaggregated Qwen model that splits attention and FFN across nodes.
    
    This class manages the distributed execution of the model:
    - On attention node: runs AttentionWorker
    - On FFN node: runs FFNWorker
    
    Communication between nodes is handled by AFDCommunicator.
    """
    
    def __init__(
        self,
        config: Any,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
    ):
        """
        Initialize disaggregated model.
        
        Args:
            config: Model configuration
            device: Device for this node
            dtype: Data type for computation
        """
        super().__init__()
        self.config = config
        self.device = device
        self.dtype = dtype
        self.ctx = get_distributed_context()
        
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_hidden_layers
        
        # Workers (only one will be active per node)
        self.attention_worker: Optional[AttentionWorker] = None
        self.ffn_worker: Optional[FFNWorker] = None
        
        # KV Cache (attention node only) - using HF DynamicCache
        self.kv_cache: Optional[DynamicCache] = None
        
        # Communicator
        self.communicator: Optional[AFDCommunicator] = None
    
    def load_weights(self, model_name: str) -> None:
        """
        Load model weights for this node's role.
        
        Args:
            model_name: HuggingFace model name or path
        """
        logger.info(f"Loading weights for role: {self.ctx.role}")
        
        # Load full model first (will be optimized in production)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=self.dtype,
            device_map="cpu",
            trust_remote_code=True,
        )
        
        if self.ctx.is_attention_node:
            self.attention_worker = AttentionWorker(model, self.device, self.dtype)
        else:
            self.ffn_worker = FFNWorker(model, self.device, self.dtype)
        
        # Free the full model
        del model
        torch.cuda.empty_cache()
        
        logger.info(f"Weights loaded for {self.ctx.role} node")
    
    def setup_communicator(
        self,
        max_seq_len: int = 2048,
        max_batch_size: int = 4,
        num_buffers: int = 2,
    ) -> None:
        """
        Set up the communicator for inter-node communication.
        
        Args:
            max_seq_len: Maximum sequence length
            max_batch_size: Maximum batch size
            num_buffers: Number of buffers for pipelining
        """
        self.communicator = AFDCommunicator(
            hidden_size=self.hidden_size,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            num_buffers=num_buffers,
            dtype=self.dtype,
        )
        logger.info("Communicator set up")
    
    def forward_layer_sync(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Forward pass for a single layer (synchronous communication).
        
        This is a simple implementation without pipelining,
        useful for correctness validation.
        
        Args:
            layer_idx: Layer index
            hidden_states: Input hidden states
            attention_mask: Attention mask
            position_ids: Position IDs
            position_embeddings: Precomputed position embeddings
        
        Returns:
            Output hidden states
        """
        assert self.communicator is not None, "Communicator not set up"
        
        tag_base = layer_idx * 100
        
        if self.ctx.is_attention_node:
            # Compute attention
            assert self.attention_worker is not None
            logger.debug(f"Layer {layer_idx}: Attention node computing attention")
            attn_output, residual = self.attention_worker.forward_attention_layer(
                layer_idx=layer_idx,
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
            )
            
            # Send attn_output and residual to FFN node
            # Pack them together for efficiency
            batch_size, seq_len, _ = attn_output.shape
            packed = torch.cat([attn_output, residual], dim=-1)  # [B, S, 2*H]
            
            logger.debug(f"Layer {layer_idx}: Attention node sending to FFN, shape={packed.shape}, tag={tag_base}")
            self.communicator.send_sync(packed, tag=tag_base)
            logger.debug(f"Layer {layer_idx}: Attention node sent, now receiving from FFN")
            
            # Receive output from FFN node
            output = self.communicator.recv_sync(
                shape=(batch_size, seq_len, self.hidden_size),
                tag=tag_base + 1
            )
            logger.debug(f"Layer {layer_idx}: Attention node received from FFN")
            
            return output.clone()
            
        else:
            # FFN node
            assert self.ffn_worker is not None
            
            # Receive attn_output and residual from attention node
            batch_size, seq_len, _ = hidden_states.shape
            logger.debug(f"Layer {layer_idx}: FFN node receiving from Attention, tag={tag_base}")
            packed = self.communicator.recv_sync(
                shape=(batch_size, seq_len, self.hidden_size * 2),
                tag=tag_base
            )
            logger.debug(f"Layer {layer_idx}: FFN node received, shape={packed.shape}")
            
            # Unpack
            attn_output = packed[..., :self.hidden_size].clone()
            residual = packed[..., self.hidden_size:].clone()
            
            # Compute FFN
            logger.debug(f"Layer {layer_idx}: FFN node computing FFN")
            output = self.ffn_worker.forward_ffn_layer(
                layer_idx=layer_idx,
                attn_output=attn_output,
                residual=residual,
            )
            
            # Send output back to attention node
            logger.debug(f"Layer {layer_idx}: FFN node sending to Attention, tag={tag_base + 1}")
            self.communicator.send_sync(output, tag=tag_base + 1)
            logger.debug(f"Layer {layer_idx}: FFN node sent")
            
            return output
    
    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Full forward pass (synchronous, for validation).
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
        
        Returns:
            Logits [batch_size, seq_len, vocab_size] (attention node only)
        """
        batch_size, seq_len = input_ids.shape
        
        # Initial embedding and position setup (attention node)
        if self.ctx.is_attention_node:
            assert self.attention_worker is not None
            
            # Embed input
            hidden_states = self.attention_worker.embed(input_ids)
            
            # Create position IDs
            position_ids = torch.arange(
                seq_len, device=self.device
            ).unsqueeze(0).expand(batch_size, -1)
            
            # Get position embeddings
            position_embeddings = self.attention_worker.get_position_embeddings(
                hidden_states, position_ids
            )
            
            # Create causal mask
            if attention_mask is None:
                attention_mask = torch.ones(
                    batch_size, seq_len, device=self.device
                )
        else:
            # FFN node - create dummy tensors
            hidden_states = torch.zeros(
                batch_size, seq_len, self.hidden_size,
                device=self.device, dtype=self.dtype
            )
            position_ids = None
            position_embeddings = None
        
        # Run through all layers
        for layer_idx in range(self.num_layers):
            hidden_states = self.forward_layer_sync(
                layer_idx=layer_idx,
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids if self.ctx.is_attention_node else None,
                position_embeddings=position_embeddings if self.ctx.is_attention_node else None,
            )
        
        # Final logits (attention node only)
        if self.ctx.is_attention_node:
            logits = self.attention_worker.forward_lm_head(hidden_states)
            return logits
        else:
            return hidden_states  # Return last hidden state for FFN node
    
    def setup_kv_cache(self, max_batch_size: int = None, max_seq_len: int = None) -> None:
        """
        Initialize KV cache for generation (attention node only).
        
        Uses HuggingFace DynamicCache for compatibility.
        """
        if self.ctx.is_attention_node:
            self.kv_cache = DynamicCache()
            logger.info("KV Cache initialized (DynamicCache)")
    
    def reset_kv_cache(self) -> None:
        """Reset KV cache for new generation."""
        if self.ctx.is_attention_node:
            self.kv_cache = DynamicCache()
    
    def forward_layer_with_cache(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """
        Forward pass for a single layer with KV cache support.
        
        For prefill: processes full sequence, populates cache
        For decode: processes single token, uses/updates cache
        """
        assert self.communicator is not None, "Communicator not set up"
        
        tag_base = layer_idx * 100
        
        if self.ctx.is_attention_node:
            assert self.attention_worker is not None
            
            # Compute attention with cache
            attn_output, residual, present_kv = self.attention_worker.forward_attention_layer(
                layer_idx=layer_idx,
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                use_cache=use_cache,
                past_key_value=self.kv_cache,  # Pass full cache, layer extracts its own
            )
            
            # Update cache with new KV
            if use_cache and present_kv is not None:
                # DynamicCache is updated in-place by the attention layer
                pass
            
            # Send to FFN
            batch_size, seq_len, _ = attn_output.shape
            packed = torch.cat([attn_output, residual], dim=-1)
            self.communicator.send_sync(packed, tag=tag_base)
            
            # Receive from FFN
            output = self.communicator.recv_sync(
                shape=(batch_size, seq_len, self.hidden_size),
                tag=tag_base + 1
            )
            
            return output.clone()
            
        else:
            # FFN node - unchanged
            assert self.ffn_worker is not None
            
            batch_size, seq_len, _ = hidden_states.shape
            packed = self.communicator.recv_sync(
                shape=(batch_size, seq_len, self.hidden_size * 2),
                tag=tag_base
            )
            
            attn_output = packed[..., :self.hidden_size].clone()
            residual = packed[..., self.hidden_size:].clone()
            
            output = self.ffn_worker.forward_ffn_layer(
                layer_idx=layer_idx,
                attn_output=attn_output,
                residual=residual,
            )
            
            self.communicator.send_sync(output, tag=tag_base + 1)
            
            return output
    
    @torch.no_grad()
    def forward_prefill(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Prefill phase: process entire prompt, populate KV cache.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask
        
        Returns:
            Logits [batch_size, seq_len, vocab_size] (attention node only)
        """
        batch_size, seq_len = input_ids.shape
        
        if self.ctx.is_attention_node:
            assert self.attention_worker is not None
            
            # Reset KV cache
            self.reset_kv_cache()
            
            # Embed input
            hidden_states = self.attention_worker.embed(input_ids)
            
            # Position IDs for full sequence
            position_ids = torch.arange(
                seq_len, device=self.device
            ).unsqueeze(0).expand(batch_size, -1)
            position_embeddings = self.attention_worker.get_position_embeddings(
                hidden_states, position_ids
            )
            
            # Create causal mask
            if attention_mask is None:
                attention_mask = torch.ones(batch_size, seq_len, device=self.device)
            
            # Convert to 4D causal mask
            causal_mask = self._make_causal_mask(batch_size, seq_len, seq_len)
        else:
            hidden_states = torch.zeros(
                batch_size, seq_len, self.hidden_size,
                device=self.device, dtype=self.dtype
            )
            position_ids = None
            position_embeddings = None
            causal_mask = None
        
        # Run through layers
        for layer_idx in range(self.num_layers):
            hidden_states = self.forward_layer_with_cache(
                layer_idx=layer_idx,
                hidden_states=hidden_states,
                attention_mask=causal_mask if self.ctx.is_attention_node else None,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                use_cache=True,
            )
        
        # Return logits
        if self.ctx.is_attention_node:
            logits = self.attention_worker.forward_lm_head(hidden_states)
            return logits
        else:
            return hidden_states
    
    @torch.no_grad()
    def forward_decode(
        self,
        input_ids: torch.Tensor,  # [batch_size, 1]
    ) -> torch.Tensor:
        """
        Decode phase: process single token using KV cache.
        
        Args:
            input_ids: Next token ID [batch_size, 1]
        
        Returns:
            Logits [batch_size, 1, vocab_size] (attention node only)
        """
        batch_size = input_ids.shape[0]
        
        if self.ctx.is_attention_node:
            assert self.attention_worker is not None
            assert self.kv_cache is not None, "KV cache not initialized"
            
            # Embed single token
            hidden_states = self.attention_worker.embed(input_ids)
            
            # Position ID = current cache length (DynamicCache tracks seen_tokens)
            cur_pos = self.kv_cache.get_seq_length()
            position_ids = torch.full(
                (batch_size, 1), cur_pos,
                device=self.device, dtype=torch.long
            )
            position_embeddings = self.attention_worker.get_position_embeddings(
                hidden_states, position_ids
            )
            
            # Attention mask: attend to all previous + current
            total_len = cur_pos + 1
            attention_mask = self._make_causal_mask(batch_size, 1, total_len)
        else:
            hidden_states = torch.zeros(
                batch_size, 1, self.hidden_size,
                device=self.device, dtype=self.dtype
            )
            position_ids = None
            position_embeddings = None
            attention_mask = None
        
        # Run through layers
        for layer_idx in range(self.num_layers):
            hidden_states = self.forward_layer_with_cache(
                layer_idx=layer_idx,
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                use_cache=True,
            )
        
        # Return logits
        if self.ctx.is_attention_node:
            logits = self.attention_worker.forward_lm_head(hidden_states)
            return logits
        else:
            return hidden_states
    
    def _make_causal_mask(
        self,
        batch_size: int,
        query_len: int,
        key_len: int,
    ) -> torch.Tensor:
        """
        Create causal attention mask.
        
        For prefill: [B, 1, S, S] lower triangular
        For decode: [B, 1, 1, total_len] all ones (attend to all previous)
        """
        if query_len == key_len:
            # Prefill: causal mask
            mask = torch.tril(
                torch.ones(query_len, key_len, device=self.device, dtype=self.dtype)
            )
        else:
            # Decode: attend to all previous tokens
            mask = torch.ones(query_len, key_len, device=self.device, dtype=self.dtype)
        
        # Expand to [B, 1, Q, K]
        mask = mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1)
        
        # Convert to additive mask (0 for attend, -inf for mask)
        mask = (1.0 - mask) * torch.finfo(self.dtype).min
        
        return mask
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        do_sample: bool = True,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate text autoregressively.
        
        Args:
            input_ids: Input prompt [batch_size, seq_len]
            max_new_tokens: Maximum new tokens to generate
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Top-p (nucleus) filtering
            do_sample: If False, use greedy decoding
            eos_token_id: Token ID to stop at
            pad_token_id: Padding token ID
        
        Returns:
            Generated token IDs [batch_size, seq_len + num_generated]
        """
        if not self.ctx.is_attention_node:
            # FFN node: just participate in forward passes
            return self._generate_ffn_node(input_ids, max_new_tokens)
        
        # Initialize KV cache if needed
        batch_size, prompt_len = input_ids.shape
        max_seq_len = prompt_len + max_new_tokens
        
        if self.kv_cache is None:
            self.setup_kv_cache(batch_size, max_seq_len)
        
        # Stopping criteria
        stopping = StoppingCriteria(
            max_length=max_seq_len,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
        )
        
        # Prefill phase
        logits = self.forward_prefill(input_ids)
        
        # Sample first token
        next_token_logits = logits[:, -1, :]
        next_token = sample_next_token(
            next_token_logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
        )
        
        # Build output
        generated_ids = input_ids.clone()
        generated_ids = torch.cat([generated_ids, next_token], dim=1)
        
        # Decode loop
        for step in range(max_new_tokens - 1):
            # Forward decode
            logits = self.forward_decode(next_token)
            
            # Sample next token
            next_token_logits = logits[:, -1, :]
            next_token = sample_next_token(
                next_token_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=do_sample,
            )
            
            # Append to output
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            
            # Check stopping (but don't break - need to sync with FFN node)
            if eos_token_id is not None and (next_token == eos_token_id).all():
                # Still need to complete remaining steps for FFN sync
                # In future: add proper sync mechanism
                pass
        
        return generated_ids
    
    def _generate_ffn_node(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
    ) -> torch.Tensor:
        """
        FFN node participation in generation.
        
        Simply participates in prefill and decode forward passes.
        The attention node generates first token after prefill, then loops
        max_new_tokens - 1 times, so we need to match that.
        """
        batch_size, prompt_len = input_ids.shape
        
        # Prefill (attention node samples first token after this)
        self.forward_prefill(input_ids)
        
        # Decode loop: max_new_tokens - 1 iterations
        # (first token is sampled after prefill on attention node)
        for step in range(max_new_tokens - 1):
            # Create dummy token input
            dummy_token = torch.zeros(
                batch_size, 1, dtype=torch.long, device=self.device
            )
            self.forward_decode(dummy_token)
        
        # FFN node doesn't return meaningful output
        return input_ids
    
    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
        max_seq_len: int = 2048,
        max_batch_size: int = 4,
    ) -> "DisaggregatedQwenModel":
        """
        Create a disaggregated model from pretrained weights.
        
        Args:
            model_name: HuggingFace model name or path
            device: Device for this node
            dtype: Data type
            max_seq_len: Maximum sequence length for communicator
            max_batch_size: Maximum batch size for communicator
        
        Returns:
            Initialized DisaggregatedQwenModel
        """
        # Load config
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        
        # Create model
        model = cls(config, device, dtype)
        
        # Load weights
        model.load_weights(model_name)
        
        # Setup communicator
        model.setup_communicator(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
        )
        
        return model
