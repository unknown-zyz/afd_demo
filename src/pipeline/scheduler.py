"""
Pipeline Scheduler - Orchestrates the 2-micro-batch ping-pong pipeline.

Implements the core logic for overlapping computation and communication
in a disaggregated attention-FFN setup.
"""

import logging
from typing import Optional

import torch

from .micro_batch import MicroBatch, MicroBatchManager
from ..distributed import get_distributed_context
from ..utils.timing import TimingTracker, PipelineTiming, EventType

logger = logging.getLogger(__name__)


class SimplePipelineScheduler:
    """
    Simplified scheduler for 2-micro-batch pipeline without full async overlap.
    
    This is easier to understand and debug, useful for initial validation.
    Implements basic ping-pong with explicit synchronization points.
    """
    
    def __init__(
        self,
        model,  # DisaggregatedQwenModel
        num_micro_batches: int = 2,
        enable_timing: bool = False,
        timing_mode: str = "cuda_events",
    ):
        """
        Initialize simple scheduler.
        
        Args:
            model: DisaggregatedQwenModel instance
            num_micro_batches: Number of micro-batches
            enable_timing: Whether to record detailed per-layer timing events
            timing_mode: "cuda_events" (stream-level sync) or "sync" (device-level sync)
        """
        self.model = model
        self.ctx = get_distributed_context()
        self.num_micro_batches = num_micro_batches
        self.enable_timing = enable_timing
        self.timing_mode = timing_mode
        self._timing_data: Optional[PipelineTiming] = None
        self.mb_manager = MicroBatchManager(
            num_micro_batches=num_micro_batches,
            device=model.device,
        )
    
    def get_timing_data(self) -> Optional[PipelineTiming]:
        """Get detailed timing data (if timing was enabled)."""
        return self._timing_data

    def _forward_layer_timed(
        self,
        layer_idx: int,
        mb: MicroBatch,
        mb_idx: int,
        tracker: TimingTracker,
    ) -> torch.Tensor:
        """
        Process one layer for one micro-batch with per-event timing.
        
        Replicates the logic of model.forward_layer_sync() but records
        attn_compute, send_transfer, recv_wait, ffn_compute events.
        """
        assert self.model.communicator is not None
        tag_base = layer_idx * 100

        if self.ctx.is_attention_node:
            assert self.model.attention_worker is not None

            # 1. Attention compute
            tracker.mark_start(EventType.ATTN_COMPUTE, layer_idx, mb_idx)
            attn_output, residual = self.model.attention_worker.forward_attention_layer(
                layer_idx=layer_idx,
                hidden_states=mb.hidden_states,
                attention_mask=mb.attention_mask,
                position_ids=mb.position_ids if hasattr(mb, 'position_ids') else None,
                position_embeddings=mb.position_embeddings if hasattr(mb, 'position_embeddings') else None,
            )
            tracker.mark_end(EventType.ATTN_COMPUTE, layer_idx, mb_idx)

            # 2. A→F send (synchronous, measures actual transfer time)
            batch_size, seq_len, _ = attn_output.shape
            packed = (attn_output + residual).contiguous()
            tracker.mark_start(EventType.SEND_TRANSFER, layer_idx, mb_idx)
            self.model.communicator.send_sync(packed, tag=tag_base)
            tracker.mark_end(EventType.SEND_TRANSFER, layer_idx, mb_idx)

            # 3. F→A recv (wait for FFN result)
            tracker.mark_start(EventType.RECV_WAIT, layer_idx, mb_idx)
            output = self.model.communicator.recv_sync(
                shape=(batch_size, seq_len, self.model.hidden_size),
                tag=tag_base + 1,
            )
            tracker.mark_end(EventType.RECV_WAIT, layer_idx, mb_idx)

            return output.clone()
        else:
            assert self.model.ffn_worker is not None
            batch_size, seq_len, _ = mb.hidden_states.shape

            # 1. A→F recv (wait for attention result)
            tracker.mark_start(EventType.RECV_WAIT, layer_idx, mb_idx)
            packed = self.model.communicator.recv_sync(
                shape=(batch_size, seq_len, self.model.hidden_size),
                tag=tag_base,
            )
            tracker.mark_end(EventType.RECV_WAIT, layer_idx, mb_idx)

            # 2. FFN compute
            tracker.mark_start(EventType.FFN_COMPUTE, layer_idx, mb_idx)
            output = self.model.ffn_worker.forward_ffn_layer(
                layer_idx=layer_idx,
                hidden_states=packed,
            )
            tracker.mark_end(EventType.FFN_COMPUTE, layer_idx, mb_idx)

            # 3. F→A send (synchronous)
            tracker.mark_start(EventType.SEND_TRANSFER, layer_idx, mb_idx)
            self.model.communicator.send_sync(output, tag=tag_base + 1)
            tracker.mark_end(EventType.SEND_TRANSFER, layer_idx, mb_idx)

            return output

    @torch.no_grad()
    def run(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """
        Run inference with basic micro-batching.
        
        This processes micro-batches in a simple interleaved fashion.
        Not fully optimized but demonstrates the concept.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask
        
        Returns:
            Logits tensor (attention node) or None (FFN node)
        """
        # Split into micro-batches
        micro_batches = self.mb_manager.split_batch(input_ids, attention_mask)
        
        # Synchronize batch metadata between attention and FFN nodes
        # Attention node broadcasts the actual sequence lengths for each micro-batch
        import torch.distributed as dist
        
        for mb_idx, mb in enumerate(micro_batches):
            metadata = torch.tensor([mb.batch_size, mb.seq_len], dtype=torch.long, device=self.model.device)
            dist.broadcast(metadata, src=0)  # Broadcast from attention node (rank 0)
            actual_batch_size, actual_seq_len = metadata[0].item(), metadata[1].item()
            
            # FFN node might have wrong seq_len from placeholder input_ids
            # Update with actual values
            if not self.ctx.is_attention_node:
                mb._actual_seq_len = actual_seq_len
                mb._actual_batch_size = actual_batch_size
        
        # Initialize embeddings (attention node)
        if self.ctx.is_attention_node and self.model.attention_worker:
            for mb in micro_batches:
                mb.hidden_states = self.model.attention_worker.embed(mb.input_ids)
                
                # Create position IDs
                mb.position_ids = torch.arange(
                    mb.seq_len, device=self.model.device
                ).unsqueeze(0).expand(mb.batch_size, -1)
                
                # Create proper causal attention mask [batch, 1, seq, seq]
                if mb.attention_mask is not None:
                    # Convert [batch, seq] to [batch, 1, 1, seq] then expand to causal
                    seq_len = mb.seq_len
                    causal_mask = torch.tril(
                        torch.ones(seq_len, seq_len, device=self.model.device, dtype=self.model.dtype)
                    ).unsqueeze(0).unsqueeze(0)  # [1, 1, seq, seq]
                    causal_mask = causal_mask.expand(mb.batch_size, 1, -1, -1)
                    # Apply padding mask
                    padding_mask = mb.attention_mask.to(dtype=self.model.dtype).unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, seq]
                    combined_mask = causal_mask * padding_mask
                    # Convert to additive mask (0 -> -inf for masked positions, 1 -> 0)
                    mb.attention_mask = (1.0 - combined_mask) * torch.finfo(self.model.dtype).min
                else:
                    mb.attention_mask = None
                
                # Get position embeddings (not used in older transformers)
                mb.position_embeddings = self.model.attention_worker.get_position_embeddings(
                    mb.hidden_states,
                    mb.position_ids,
                )
        else:
            # FFN node - create placeholder hidden states with ACTUAL sizes from attention node
            for mb in micro_batches:
                actual_batch_size = getattr(mb, '_actual_batch_size', mb.batch_size)
                actual_seq_len = getattr(mb, '_actual_seq_len', mb.seq_len)
                mb.hidden_states = torch.zeros(
                    actual_batch_size, actual_seq_len, self.model.hidden_size,
                    device=self.model.device, dtype=self.model.dtype
                )
        
        # Initialize timing tracker if enabled
        tracker = None
        if self.enable_timing:
            node_name = "attention" if self.ctx.is_attention_node else "ffn"
            tracker = TimingTracker(
                node=node_name,
                num_layers=self.model.num_layers,
                num_micro_batches=self.num_micro_batches,
                mode=self.timing_mode,
            )

        # Process each layer
        for layer_idx in range(self.model.num_layers):
            # Process micro-batches in ping-pong order
            for mb_idx, mb in enumerate(micro_batches):
                if tracker is not None:
                    mb.hidden_states = self._forward_layer_timed(
                        layer_idx, mb, mb_idx, tracker,
                    )
                else:
                    mb.hidden_states = self.model.forward_layer_sync(
                        layer_idx=layer_idx,
                        hidden_states=mb.hidden_states,
                        attention_mask=mb.attention_mask,
                        position_ids=mb.position_ids if self.ctx.is_attention_node else None,
                        position_embeddings=mb.position_embeddings if self.ctx.is_attention_node else None,
                    )

        # Finalize timing
        if tracker is not None:
            self._timing_data = tracker.finish()
        
        # Generate logits (attention node)
        if self.ctx.is_attention_node and self.model.attention_worker:
            results = []
            for mb in micro_batches:
                logits = self.model.attention_worker.forward_lm_head(mb.hidden_states)
                results.append(logits)
            return self.mb_manager.merge_results(results)
        
        return None
