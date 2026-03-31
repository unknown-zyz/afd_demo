"""
Decode DBO Scheduler - DBO pipeline for decode phase.

Key insight: During decode, we process multiple requests' current tokens as a batch.
This batch can be split into micro-batches for DBO overlap, similar to prefill.

Differences from prefill DBO:
- Each token position may be different (different KV cache lengths per request)
- We need to handle variable sequence lengths in KV cache
- Single token per request, but batched across requests
"""

import logging
import time
from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass

import torch
import torch.distributed as dist

from ..distributed import get_distributed_context
from ..utils.timing import TimingTracker, PipelineTiming, EventType

logger = logging.getLogger(__name__)


@dataclass
class DecodeDBOStats:
    """Statistics from Decode DBO execution."""
    total_time: float = 0.0
    compute_time: float = 0.0
    comm_time: float = 0.0
    num_tokens: int = 0
    num_micro_batches: int = 0
    
    # Detailed per-phase timing (summed across all layers and micro-batches)
    attn_compute_time: float = 0.0  # Attention computation
    ffn_compute_time: float = 0.0   # FFN computation
    a2f_comm_time: float = 0.0      # Attention → FFN communication
    f2a_comm_time: float = 0.0      # FFN → Attention communication
    
    @property
    def tokens_per_second(self) -> float:
        if self.total_time == 0:
            return 0.0
        return self.num_tokens / self.total_time
    
    def __str__(self) -> str:
        lines = [
            f"Decode DBO: {self.total_time*1000:.2f}ms | "
            f"计算: {self.compute_time*1000:.2f}ms | "
            f"通信: {self.comm_time*1000:.2f}ms | "
            f"{self.tokens_per_second:.1f} tok/s"
        ]
        # Add detailed breakdown if available
        if self.attn_compute_time > 0 or self.ffn_compute_time > 0:
            lines.append(
                f"  详细: Attn={self.attn_compute_time*1000:.2f}ms, "
                f"FFN={self.ffn_compute_time*1000:.2f}ms, "
                f"A2F={self.a2f_comm_time*1000:.2f}ms, "
                f"F2A={self.f2a_comm_time*1000:.2f}ms"
            )
        return "\n".join(lines)


class DecodeDBOScheduler:
    """
    Decode phase scheduler with DBO (Dual Batch Overlap).
    
    For decode, we batch multiple requests' current tokens together:
    - Request 0: generating token at position 10
    - Request 1: generating token at position 15
    - Request 2: generating token at position 12
    - Request 3: generating token at position 8
    
    These 4 tokens form a batch [4, 1], which we split into 2 micro-batches
    for DBO overlap.
    
    Pipeline pattern (2 micro-batches, 2 requests each):
    
    Attention Node:
      Layer 0: [Attn_MB0][Attn_MB1]
                 ↓isend   ↓isend
      Layer 1: [recv][Attn_MB0][Attn_MB1]
               ...
    
    FFN Node:
      Layer 0: [irecv][recv_wait][FFN_MB0][FFN_MB1]
                                   ↓isend   ↓isend
      Layer 1: [irecv][recv_wait][FFN_MB0][FFN_MB1]
               ...
    """
    
    def __init__(
        self,
        model,  # DisaggregatedQwenModel
        num_micro_batches: int = 2,
        enable_timing: bool = False,
    ):
        """
        Initialize decode DBO scheduler.
        
        Args:
            model: DisaggregatedQwenModel instance
            num_micro_batches: Number of micro-batches (2 for standard DBO)
            enable_timing: Whether to record detailed timing
        """
        self.model = model
        self.ctx = get_distributed_context()
        self.num_micro_batches = num_micro_batches
        self.enable_timing = enable_timing
        
        # Pending async operations
        self._pending_sends: List[dist.Work] = []
        
        # Stats
        self.stats = DecodeDBOStats()
        
        logger.debug(f"DecodeDBOScheduler initialized: num_mb={num_micro_batches}")
    
    def _get_tag(self, layer_idx: int, mb_idx: int, direction: str) -> int:
        """Get unique tag for send/recv matching."""
        dir_code = 0 if direction == "attn_to_ffn" else 1
        # Use different tag space from prefill (add 10000 offset)
        return 10000 + layer_idx * 1000 + mb_idx * 10 + dir_code
    
    def _split_batch(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Split decode batch into micro-batches.
        
        Args:
            input_ids: [batch_size, 1] - current tokens
            position_ids: [batch_size, 1] - position of each token
        
        Returns:
            List of (input_ids_mb, position_ids_mb) tuples
        """
        batch_size = input_ids.shape[0]
        
        if batch_size < self.num_micro_batches:
            # Can't split further, return as single batch
            return [(input_ids, position_ids)]
        
        # Split evenly
        mb_size = batch_size // self.num_micro_batches
        remainder = batch_size % self.num_micro_batches
        
        micro_batches = []
        start = 0
        for i in range(self.num_micro_batches):
            size = mb_size + (1 if i < remainder else 0)
            end = start + size
            micro_batches.append((
                input_ids[start:end],
                position_ids[start:end],
            ))
            start = end
        
        return micro_batches
    
    @torch.no_grad()
    def forward_decode_dbo(
        self,
        input_ids: torch.Tensor,  # [batch_size, 1]
        position_ids: torch.Tensor,  # [batch_size, 1]
        kv_cache,  # DynamicCache
    ) -> torch.Tensor:
        """
        Decode forward pass with DBO.
        
        Args:
            input_ids: Current tokens [batch_size, 1]
            position_ids: Position IDs [batch_size, 1]
            kv_cache: KV cache (DynamicCache)
        
        Returns:
            Logits [batch_size, 1, vocab_size] (attention node only)
        """
        start_time = time.perf_counter()
        batch_size = input_ids.shape[0]
        
        # Reset stats
        self.stats = DecodeDBOStats(
            num_tokens=batch_size,
            num_micro_batches=self.num_micro_batches,
        )
        
        # Split into micro-batches
        micro_batches = self._split_batch(input_ids, position_ids)
        actual_num_mb = len(micro_batches)
        
        # Sync metadata
        metadata = torch.tensor([batch_size, actual_num_mb], dtype=torch.long, device=self.model.device)
        dist.broadcast(metadata, src=0)
        
        if not self.ctx.is_attention_node:
            batch_size = metadata[0].item()
            actual_num_mb = metadata[1].item()
            # Recreate micro-batches with correct sizes
            mb_size = batch_size // actual_num_mb
            remainder = batch_size % actual_num_mb
            micro_batches = []
            for i in range(actual_num_mb):
                size = mb_size + (1 if i < remainder else 0)
                micro_batches.append((
                    torch.zeros(size, 1, dtype=torch.long, device=self.model.device),
                    torch.zeros(size, 1, dtype=torch.long, device=self.model.device),
                ))
        
        # Run pipeline
        if self.ctx.is_attention_node:
            result = self._run_attention_decode(micro_batches, kv_cache)
        else:
            self._run_ffn_decode(micro_batches)
            result = None
        
        # Record stats
        self.stats.total_time = time.perf_counter() - start_time
        
        return result
    
    def _run_attention_decode(
        self,
        micro_batches: List[Tuple[torch.Tensor, torch.Tensor]],
        kv_cache,
    ) -> torch.Tensor:
        """
        Attention node decode with DBO overlap.
        """
        assert self.model.attention_worker is not None
        
        num_mb = len(micro_batches)
        num_layers = self.model.num_layers
        
        # Initialize hidden states for each micro-batch
        mb_hidden_states = []
        mb_attention_masks = []
        mb_position_embeddings = []
        
        for mb_idx, (ids_mb, pos_mb) in enumerate(micro_batches):
            # Embed tokens
            hidden = self.model.attention_worker.embed(ids_mb)
            mb_hidden_states.append(hidden)
            
            # Get current cache length for this micro-batch
            cur_pos = kv_cache.get_seq_length() if mb_idx == 0 else kv_cache.get_seq_length()
            total_len = cur_pos + 1
            
            # Create attention mask
            mask = self.model._make_causal_mask(ids_mb.shape[0], 1, total_len)
            mb_attention_masks.append(mask)
            
            # Position embeddings
            pos_emb = self.model.attention_worker.get_position_embeddings(hidden, pos_mb)
            mb_position_embeddings.append(pos_emb)
        
        # Process layers with DBO
        for layer_idx in range(num_layers):
            send_handles = []
            packed_list = []
            
            # Compute attention for all micro-batches, send immediately after each
            for mb_idx in range(num_mb):
                compute_start = time.perf_counter()
                
                attn_output, residual = self.model.attention_worker.forward_attention_layer(
                    layer_idx=layer_idx,
                    hidden_states=mb_hidden_states[mb_idx],
                    attention_mask=mb_attention_masks[mb_idx],
                    position_ids=micro_batches[mb_idx][1],
                    position_embeddings=mb_position_embeddings[mb_idx],
                )
                packed = torch.cat([attn_output, residual], dim=-1)
                packed_list.append(packed)
                
                compute_end = time.perf_counter()
                compute_time = compute_end - compute_start
                self.stats.compute_time += compute_time
                self.stats.attn_compute_time += compute_time
                
                # Async send - overlaps with next MB's compute
                send_start = time.perf_counter()
                tag = self._get_tag(layer_idx, mb_idx, "attn_to_ffn")
                handle = dist.isend(packed.contiguous(), dst=self.ctx.peer_rank, tag=tag)
                send_handles.append((handle, send_start))
            
            # Wait for all sends and record A2F time
            for handle, send_start in send_handles:
                handle.wait()
                self.stats.a2f_comm_time += time.perf_counter() - send_start
            
            # Receive FFN results (F2A communication)
            recv_tensors = []
            for mb_idx in range(num_mb):
                recv_start = time.perf_counter()
                tag = self._get_tag(layer_idx, mb_idx, "ffn_to_attn")
                recv_tensor = torch.empty(
                    micro_batches[mb_idx][0].shape[0], 1, self.model.hidden_size,
                    dtype=self.model.dtype, device=self.model.device
                )
                dist.recv(recv_tensor, src=self.ctx.peer_rank, tag=tag)
                recv_tensors.append(recv_tensor)
                
                recv_time = time.perf_counter() - recv_start
                self.stats.comm_time += recv_time
                self.stats.f2a_comm_time += recv_time
            
            # Update hidden states
            for mb_idx in range(num_mb):
                mb_hidden_states[mb_idx] = recv_tensors[mb_idx].clone()
        
        # Generate logits
        results = []
        for mb_idx in range(num_mb):
            logits = self.model.attention_worker.forward_lm_head(mb_hidden_states[mb_idx])
            results.append(logits)
        
        # Merge results
        return torch.cat(results, dim=0)
    
    def _run_ffn_decode(
        self,
        micro_batches: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        """
        FFN node decode with DBO overlap.
        """
        assert self.model.ffn_worker is not None
        
        num_mb = len(micro_batches)
        num_layers = self.model.num_layers
        
        for layer_idx in range(num_layers):
            # Post all receives first (async)
            recv_handles = []
            recv_tensors = []
            
            for mb_idx in range(num_mb):
                batch_size = micro_batches[mb_idx][0].shape[0]
                tag = self._get_tag(layer_idx, mb_idx, "attn_to_ffn")
                recv_tensor = torch.empty(
                    batch_size, 1, self.model.hidden_size * 2,
                    dtype=self.model.dtype, device=self.model.device
                )
                handle = dist.irecv(recv_tensor, src=self.ctx.peer_rank, tag=tag)
                recv_handles.append(handle)
                recv_tensors.append(recv_tensor)
            
            # Process each MB: wait, compute, send
            send_handles = []
            
            for mb_idx in range(num_mb):
                # Wait for data (A2F receive)
                recv_start = time.perf_counter()
                recv_handles[mb_idx].wait()
                packed = recv_tensors[mb_idx]
                recv_time = time.perf_counter() - recv_start
                self.stats.comm_time += recv_time
                self.stats.a2f_comm_time += recv_time
                
                # Unpack
                attn_output = packed[..., :self.model.hidden_size].clone()
                residual = packed[..., self.model.hidden_size:].clone()
                
                # Compute FFN
                compute_start = time.perf_counter()
                output = self.model.ffn_worker.forward_ffn_layer(
                    layer_idx=layer_idx,
                    attn_output=attn_output,
                    residual=residual,
                )
                if isinstance(output, tuple):
                    output = output[0]
                output = output.contiguous()
                
                compute_time = time.perf_counter() - compute_start
                self.stats.compute_time += compute_time
                self.stats.ffn_compute_time += compute_time
                
                # Async send (F2A) - overlaps with next MB's processing
                send_start = time.perf_counter()
                tag = self._get_tag(layer_idx, mb_idx, "ffn_to_attn")
                handle = dist.isend(output, dst=self.ctx.peer_rank, tag=tag)
                send_handles.append((handle, send_start))
            
            # Wait for all sends and record F2A time
            for handle, send_start in send_handles:
                handle.wait()
                self.stats.f2a_comm_time += time.perf_counter() - send_start
    
    def get_stats(self) -> DecodeDBOStats:
        """Get statistics from last run."""
        return self.stats
