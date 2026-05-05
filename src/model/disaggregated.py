"""
Disaggregated Qwen Model - Orchestrates attention and FFN workers.

This module provides the main interface for running disaggregated inference,
coordinating between attention and FFN workers across nodes.
"""

import logging
import time
from typing import Optional, Tuple, List, Dict, Any

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig
from transformers.cache_utils import DynamicCache

from .attention_worker import AttentionWorker
from .ffn_worker import FFNWorker
from ..distributed import get_distributed_context
from ..distributed.communicator import AFDCommunicator
from ..utils import device as devmod
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
        self.model_type = str(getattr(config, "model_type", "unknown"))
        self.is_moe = ("moe" in self.model_type.lower()) or (getattr(config, "num_experts", 0) > 0)
        self.has_router = bool(getattr(config, "num_experts", 0) > 0)
        self.supports_moe_timing = False
        
        # Workers (only one will be active per node)
        self.attention_worker: Optional[AttentionWorker] = None
        self.ffn_worker: Optional[FFNWorker] = None
        
        # KV Cache (attention node only) - using HF DynamicCache
        self.kv_cache: Optional[DynamicCache] = None
        
        # Communicator
        self.communicator: Optional[AFDCommunicator] = None
        self._last_generation_metrics: Dict[str, Any] = {}
    
    def load_weights(self, model_name: str) -> None:
        """
        Load model weights for this node's role.
        
        Args:
            model_name: HuggingFace model name or path
        """
        import gc
        logger.info(f"Loading weights for role: {self.ctx.role}")
        
        # Load full model first with memory optimization
        # For large MoE models, we need low_cpu_mem_usage to avoid OOM
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=self.dtype,
            device_map="cpu",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        
        self.has_router = any(hasattr(layer.mlp, "gate") for layer in model.model.layers)
        self.supports_moe_timing = self.has_router

        if self.ctx.is_attention_node:
            self.attention_worker = AttentionWorker(model, self.device, self.dtype)
        else:
            self.ffn_worker = FFNWorker(model, self.device, self.dtype)
            self.supports_moe_timing = self.ffn_worker.supports_moe_timing
        
        # Free the full model and cleanup
        del model
        gc.collect()
        torch.cuda.empty_cache()
        
        logger.info(
            "Weights loaded for %s node | model_type=%s, moe=%s, router=%s, moe_timing=%s",
            self.ctx.role,
            self.model_type,
            self.is_moe,
            self.has_router,
            self.supports_moe_timing,
        )
    
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
            
            # Send pre-combined hidden states to FFN node (1×H instead of 2×H)
            batch_size, seq_len, _ = attn_output.shape
            packed = (attn_output + residual).contiguous()
            
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
            
            # Receive pre-combined hidden states from attention node
            batch_size, seq_len, _ = hidden_states.shape
            logger.debug(f"Layer {layer_idx}: FFN node receiving from Attention, tag={tag_base}")
            packed = self.communicator.recv_sync(
                shape=(batch_size, seq_len, self.hidden_size),
                tag=tag_base
            )
            logger.debug(f"Layer {layer_idx}: FFN node received, shape={packed.shape}")
            
            # Compute FFN (input is pre-combined: attn_output + residual)
            logger.debug(f"Layer {layer_idx}: FFN node computing FFN")
            output = self.ffn_worker.forward_ffn_layer(
                layer_idx=layer_idx,
                hidden_states=packed,
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
        
        # Final logits (attention node only).
        # Prefill only needs the last position's logits to sample the next token,
        # so slice before lm_head to avoid the (B, S, vocab) allocation that
        # OOMs at large (batch, seq) on NPU.
        if self.ctx.is_attention_node:
            last_hidden = hidden_states[:, -1:, :]
            logits = self.attention_worker.forward_lm_head(last_hidden)
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
            
            # Send pre-combined hidden states to FFN
            batch_size, seq_len, _ = attn_output.shape
            packed = (attn_output + residual).contiguous()
            self.communicator.send_sync(packed, tag=tag_base)
            
            # Receive from FFN
            output = self.communicator.recv_sync(
                shape=(batch_size, seq_len, self.hidden_size),
                tag=tag_base + 1
            )
            
            return output.clone()
            
        else:
            # FFN node
            assert self.ffn_worker is not None

            if self.ctx.is_ffn_expert_only and self.ctx.ffn_ep_enabled:
                output = self.ffn_worker.forward_ffn_layer(
                    layer_idx=layer_idx,
                    hidden_states=hidden_states,
                )
                return output[0] if isinstance(output, tuple) else output
            
            batch_size, seq_len, _ = hidden_states.shape
            packed = self.communicator.recv_sync(
                shape=(batch_size, seq_len, self.hidden_size),
                tag=tag_base
            )
            
            output = self.ffn_worker.forward_ffn_layer(
                layer_idx=layer_idx,
                hidden_states=packed,
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
        
        # Return logits (prefill: last-token only, see forward_prefill comment).
        if self.ctx.is_attention_node:
            last_hidden = hidden_states[:, -1:, :]
            logits = self.attention_worker.forward_lm_head(last_hidden)
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
        use_decode_dbo: bool = True,
        num_decode_micro_batches: int = 2,
        enable_timing: bool = False,
        timing_mode: str = "cuda_events",
        comm_timing_mode: str = "enqueue",
        decode_use_crosslayer: bool = False,
        af_comm_mode: str = "direct-hccl",
        controller_host: str = "127.0.0.1",
        controller_port: int = 40100,
    ) -> torch.Tensor:
        """
        Generate text autoregressively with optional Decode DBO.
        
        Args:
            input_ids: Input prompt [batch_size, seq_len]
            max_new_tokens: Maximum new tokens to generate
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Top-p (nucleus) filtering
            do_sample: If False, use greedy decoding
            eos_token_id: Token ID to stop at
            pad_token_id: Padding token ID
            use_decode_dbo: Whether to use DBO for decode phase
            num_decode_micro_batches: Number of micro-batches for decode DBO
            enable_timing: Whether to collect per-layer timing data
            timing_mode: "cuda_events" (zero-overhead) or "sync" (legacy)
            comm_timing_mode: "enqueue" for isend return overhead or
                "completion" for effective Work completion latency
            decode_use_crosslayer: Enable cross-layer micro-batch pipelining in decode DBO
            af_comm_mode: "direct-hccl" for current device-direct A/F communication,
                or "controller-cpu" for centralized CPU relay baseline.
            controller_host: CPU controller host for controller-cpu mode.
            controller_port: CPU controller port for controller-cpu mode.
        
        Returns:
            Generated token IDs [batch_size, seq_len + num_generated]
        """
        if not self.ctx.is_attention_node:
            # FFN node: just participate in forward passes
            return self._generate_ffn_node(
                input_ids, max_new_tokens, use_decode_dbo,
                num_decode_micro_batches, enable_timing, timing_mode,
                comm_timing_mode,
                decode_use_crosslayer,
                af_comm_mode,
                controller_host,
                controller_port,
            )
        
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
        
        devmod.synchronize()
        prefill_start = time.perf_counter()
        # Prefill phase (uses existing DBO scheduler)
        logits = self.forward_prefill(input_ids)
        devmod.synchronize()
        prefill_ms = (time.perf_counter() - prefill_start) * 1000
        
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
        
        # Initialize Decode DBO scheduler if enabled
        decode_scheduler = None
        if use_decode_dbo and batch_size >= num_decode_micro_batches:
            from ..pipeline import DecodeDBOScheduler
            ctx = get_distributed_context()
            decode_scheduler = DecodeDBOScheduler(
                model=self,
                num_micro_batches=num_decode_micro_batches,
                enable_timing=enable_timing,
                timing_mode=timing_mode,
                comm_timing_mode=comm_timing_mode,
                use_crosslayer=decode_use_crosslayer,
                af_comm_mode=af_comm_mode,
                controller_host=controller_host,
                controller_port=controller_port,
            )
            logger.info(f"Using Decode DBO with {num_decode_micro_batches} micro-batches")

        # Decode loop
        decode_steps = max(max_new_tokens - 1, 0)
        devmod.synchronize()
        self.ctx.barrier()
        decode_start = time.perf_counter()
        for step in range(max_new_tokens - 1):
            # Current position = KV cache length (accurate regardless of path)
            cur_pos = self.kv_cache.get_seq_length()
            position_ids = torch.full(
                (batch_size, 1), cur_pos,
                device=self.device, dtype=torch.long
            )
            
            if decode_scheduler is not None:
                # Use Decode DBO
                logits = decode_scheduler.forward_decode_dbo(
                    input_ids=next_token,
                    position_ids=position_ids,
                    kv_cache=self.kv_cache,
                )
            else:
                # Standard decode (no DBO)
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
        devmod.synchronize()
        self.ctx.barrier()
        decode_loop_ms = (time.perf_counter() - decode_start) * 1000
        decode_tpot_ms = decode_loop_ms / decode_steps if decode_steps > 0 else None
        self._last_generation_metrics = {
            "prefill_ms": prefill_ms,
            "decode_loop_ms": decode_loop_ms,
            "decode_steps": decode_steps,
            "decode_tpot_ms": decode_tpot_ms,
        }

        # Log Decode DBO stats
        if decode_scheduler is not None:
            logger.info(f"Decode DBO stats: {decode_scheduler.get_stats()}")
            # Store timing data for external access
            self._last_decode_timing = decode_scheduler.get_timing_data()
            if self._last_decode_timing is not None:
                self._last_decode_timing.prefill_ms = prefill_ms
                self._last_decode_timing.decode_loop_ms = decode_loop_ms
                self._last_decode_timing.decode_steps = decode_steps
                self._last_decode_timing.decode_tpot_ms = decode_tpot_ms
        
        return generated_ids
    
    def _generate_ffn_node(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        use_decode_dbo: bool = True,
        num_decode_micro_batches: int = 2,
        enable_timing: bool = False,
        timing_mode: str = "cuda_events",
        comm_timing_mode: str = "enqueue",
        decode_use_crosslayer: bool = False,
        af_comm_mode: str = "direct-hccl",
        controller_host: str = "127.0.0.1",
        controller_port: int = 40100,
    ) -> torch.Tensor:
        """
        FFN node participation in generation.
        
        Simply participates in prefill and decode forward passes.
        The attention node generates first token after prefill, then loops
        max_new_tokens - 1 times, so we need to match that.
        """
        batch_size, prompt_len = input_ids.shape
        
        devmod.synchronize()
        prefill_start = time.perf_counter()
        # Prefill (attention node samples first token after this)
        self.forward_prefill(input_ids)
        devmod.synchronize()
        prefill_ms = (time.perf_counter() - prefill_start) * 1000
        
        # Initialize Decode DBO scheduler if enabled
        decode_scheduler = None
        if use_decode_dbo and batch_size >= num_decode_micro_batches:
            from ..pipeline import DecodeDBOScheduler
            ctx = get_distributed_context()
            decode_scheduler = DecodeDBOScheduler(
                model=self,
                num_micro_batches=num_decode_micro_batches,
                enable_timing=enable_timing,
                timing_mode=timing_mode,
                comm_timing_mode=comm_timing_mode,
                use_crosslayer=decode_use_crosslayer,
                af_comm_mode=af_comm_mode,
                controller_host=controller_host,
                controller_port=controller_port,
            )
        
        # Decode loop: max_new_tokens - 1 iterations
        # (first token is sampled after prefill on attention node)
        decode_steps = max(max_new_tokens - 1, 0)
        devmod.synchronize()
        self.ctx.barrier()
        decode_start = time.perf_counter()
        for step in range(max_new_tokens - 1):
            # Create dummy token input
            dummy_token = torch.zeros(
                batch_size, 1, dtype=torch.long, device=self.device
            )
            dummy_position = torch.zeros(
                batch_size, 1, dtype=torch.long, device=self.device
            )
            
            if decode_scheduler is not None:
                # Use Decode DBO
                decode_scheduler.forward_decode_dbo(
                    input_ids=dummy_token,
                    position_ids=dummy_position,
                    kv_cache=None,  # FFN doesn't have kv_cache
                )
            else:
                self.forward_decode(dummy_token)
        devmod.synchronize()
        self.ctx.barrier()
        decode_loop_ms = (time.perf_counter() - decode_start) * 1000
        decode_tpot_ms = decode_loop_ms / decode_steps if decode_steps > 0 else None
        self._last_generation_metrics = {
            "prefill_ms": prefill_ms,
            "decode_loop_ms": decode_loop_ms,
            "decode_steps": decode_steps,
            "decode_tpot_ms": decode_tpot_ms,
        }

        # Log Decode DBO stats for FFN node
        if decode_scheduler is not None:
            logger.info(f"[FFN] Decode DBO stats: {decode_scheduler.get_stats()}")
            self._last_decode_timing = decode_scheduler.get_timing_data()
            if self._last_decode_timing is not None:
                self._last_decode_timing.prefill_ms = prefill_ms
                self._last_decode_timing.decode_loop_ms = decode_loop_ms
                self._last_decode_timing.decode_steps = decode_steps
                self._last_decode_timing.decode_tpot_ms = decode_tpot_ms
        
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
