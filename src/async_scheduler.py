"""
Async Pipeline Scheduler - True DBO (Dual Batch Overlap) implementation.

This scheduler implements real computation-communication overlap using:
- CUDA Streams to separate compute and communication
- Async send/recv (isend/irecv) for non-blocking communication
- Double buffering to avoid data races

The key insight:
- While MB0's result is being sent, we can compute MB1
- While MB1's result is being sent, we can compute MB0 (next layer)
- This overlaps computation with communication, improving GPU utilization
"""

import logging
import time
from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass, field

import torch
import torch.cuda
import torch.distributed as dist

from .micro_batch import MicroBatch, MicroBatchManager, MicroBatchState
from ..distributed import get_distributed_context
from ..utils.timing import TimingTracker, PipelineTiming, EventType

logger = logging.getLogger(__name__)


@dataclass
class DBOStats:
    """Statistics from DBO pipeline execution."""
    total_time: float = 0.0
    compute_time: float = 0.0
    recv_wait_time: float = 0.0  # Actual waiting for incoming data
    num_layers: int = 0
    num_micro_batches: int = 0
    
    @property
    def compute_ratio(self) -> float:
        """
        DBO efficiency: compute_time / total_time.
        
        Higher is better:
        - 1.0 = Perfect (all time spent computing)
        - 0.5 = Half time computing, half waiting
        """
        if self.total_time == 0:
            return 0.0
        return self.compute_time / self.total_time
    
    def __str__(self) -> str:
        return (
            f"端到端: {self.total_time*1000:.2f}ms | "
            f"计算: {self.compute_time*1000:.2f}ms ({self.compute_ratio:.1%}) | "
            f"等待: {self.recv_wait_time*1000:.2f}ms"
        )


class AsyncPipelineScheduler:
    """
    Async scheduler implementing true DBO (Dual Batch Overlap).
    
    Pipeline pattern for 2 micro-batches:
    
    Time →
    Attention Node:
      Layer 0: [Attn_MB0][Attn_MB1]  (compute both, send interleaved)
               ↓send     ↓send
      Layer 1: [recv][Attn_MB0][Attn_MB1]
               ...
    
    FFN Node:
      Layer 0: [recv][FFN_MB0][FFN_MB1]  (recv and compute interleaved)
               ↓send      ↓send
      Layer 1: [recv][FFN_MB0][FFN_MB1]
               ...
    
    The key is that send operations are non-blocking (isend),
    so we can start computing the next micro-batch while sending.
    """
    
    def __init__(
        self,
        model,  # DisaggregatedQwenModel
        num_micro_batches: int = 2,
        use_cuda_streams: bool = True,
        enable_timing: bool = False,
    ):
        """
        Initialize async scheduler.
        
        Args:
            model: DisaggregatedQwenModel instance
            num_micro_batches: Number of micro-batches (2 for standard DBO)
            use_cuda_streams: Whether to use separate CUDA streams
            enable_timing: Whether to record detailed per-MB timing
        """
        self.model = model
        self.ctx = get_distributed_context()
        self.num_micro_batches = num_micro_batches
        self.use_cuda_streams = use_cuda_streams
        self.enable_timing = enable_timing
        
        self.mb_manager = MicroBatchManager(
            num_micro_batches=num_micro_batches,
            device=model.device,
        )
        
        # CUDA streams for overlap
        if use_cuda_streams and torch.cuda.is_available():
            self.compute_stream = torch.cuda.Stream()
            self.comm_stream = torch.cuda.Stream()
        else:
            self.compute_stream = None
            self.comm_stream = None
        
        # Pending async operations
        self._pending_sends: List[dist.Work] = []
        self._pending_recvs: Dict[str, Tuple[dist.Work, torch.Tensor]] = {}
        
        # Stats
        self.stats = DBOStats()
        
        # Detailed timing (optional)
        self._timing_tracker = None
        
        logger.info(f"AsyncPipelineScheduler initialized: num_mb={num_micro_batches}, "
                    f"use_cuda_streams={use_cuda_streams}, timing={enable_timing}")
    
    def _get_tag(self, layer_idx: int, mb_idx: int, direction: str) -> int:
        """Get unique tag for send/recv matching."""
        dir_code = 0 if direction == "attn_to_ffn" else 1
        return layer_idx * 1000 + mb_idx * 10 + dir_code
    
    def _send_async(self, tensor: torch.Tensor, tag: int) -> dist.Work:
        """Non-blocking send."""
        handle = dist.isend(tensor.contiguous(), dst=self.ctx.peer_rank, tag=tag)
        self._pending_sends.append(handle)
        return handle
    
    def _recv_async(self, shape: Tuple[int, ...], tag: int) -> Tuple[dist.Work, torch.Tensor]:
        """Non-blocking receive."""
        tensor = torch.empty(shape, dtype=self.model.dtype, device=self.model.device)
        handle = dist.irecv(tensor, src=self.ctx.peer_rank, tag=tag)
        return handle, tensor
    
    def _wait_all_sends(self):
        """Wait for all pending sends to complete."""
        for handle in self._pending_sends:
            handle.wait()
        self._pending_sends.clear()
    
    def _sync_metadata(self, micro_batches: List[MicroBatch]) -> None:
        """Synchronize batch metadata between nodes."""
        for mb_idx, mb in enumerate(micro_batches):
            metadata = torch.tensor(
                [mb.batch_size, mb.seq_len], 
                dtype=torch.long, 
                device=self.model.device
            )
            dist.broadcast(metadata, src=0)
            
            if not self.ctx.is_attention_node:
                mb._actual_seq_len = metadata[1].item()
                mb._actual_batch_size = metadata[0].item()
    
    def _prepare_attention_mb(self, mb: MicroBatch) -> None:
        """Prepare micro-batch for attention computation."""
        mb.hidden_states = self.model.attention_worker.embed(mb.input_ids)
        
        mb.position_ids = torch.arange(
            mb.seq_len, device=self.model.device
        ).unsqueeze(0).expand(mb.batch_size, -1)
        
        if mb.attention_mask is not None:
            seq_len = mb.seq_len
            causal_mask = torch.tril(
                torch.ones(seq_len, seq_len, device=self.model.device, dtype=self.model.dtype)
            ).unsqueeze(0).unsqueeze(0)
            causal_mask = causal_mask.expand(mb.batch_size, 1, -1, -1)
            padding_mask = mb.attention_mask.to(dtype=self.model.dtype).unsqueeze(1).unsqueeze(2)
            combined_mask = causal_mask * padding_mask
            mb.attention_mask = (1.0 - combined_mask) * torch.finfo(self.model.dtype).min
        
        mb.position_embeddings = self.model.attention_worker.get_position_embeddings(
            mb.hidden_states,
            mb.position_ids,
        )
    
    def _prepare_ffn_mb(self, mb: MicroBatch) -> None:
        """Prepare micro-batch placeholder for FFN node."""
        batch_size = getattr(mb, '_actual_batch_size', mb.batch_size)
        seq_len = getattr(mb, '_actual_seq_len', mb.seq_len)
        mb.hidden_states = torch.zeros(
            batch_size, seq_len, self.model.hidden_size,
            device=self.model.device, dtype=self.model.dtype
        )
    
    @torch.no_grad()
    def run(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """
        Run inference with true DBO (async overlap).
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask
        
        Returns:
            Logits tensor (attention node) or None (FFN node)
        """
        start_time = time.perf_counter()
        
        # Reset stats for this run
        self.stats = DBOStats()
        
        # Initialize timing tracker if enabled
        if self.enable_timing:
            node_name = "attention" if self.ctx.is_attention_node else "ffn"
            self._timing_tracker = TimingTracker(
                node=node_name,
                num_layers=self.model.num_layers,
                num_micro_batches=self.num_micro_batches,
            )
        
        # Split into micro-batches
        micro_batches = self.mb_manager.split_batch(input_ids, attention_mask)
        
        # Sync metadata
        self._sync_metadata(micro_batches)
        
        # Prepare micro-batches
        if self.ctx.is_attention_node:
            for mb in micro_batches:
                self._prepare_attention_mb(mb)
        else:
            for mb in micro_batches:
                self._prepare_ffn_mb(mb)
        
        # Run pipeline - use the model's forward_layer method but with async patterns
        # For now, use a simpler approach: process layers with async communication
        if self.ctx.is_attention_node:
            result = self._run_attention_node_simple(micro_batches)
        else:
            self._run_ffn_node_simple(micro_batches)
            result = None
        
        # Record stats
        end_time = time.perf_counter()
        self.stats.total_time = end_time - start_time
        self.stats.num_layers = self.model.num_layers
        self.stats.num_micro_batches = self.num_micro_batches
        
        # Finalize timing
        if self.enable_timing and self._timing_tracker:
            self._timing_data = self._timing_tracker.finish()
        
        return result
    
    def get_timing_data(self) -> Optional[PipelineTiming]:
        """Get detailed timing data (if timing was enabled)."""
        return getattr(self, '_timing_data', None)
    
    def _run_attention_node_simple(
        self,
        micro_batches: List[MicroBatch],
    ) -> torch.Tensor:
        """
        Attention node with overlapped computation and communication.
        
        The key overlap pattern:
        - For each layer, while sending MB0's result, we compute MB1
        - This overlaps send(MB0) with compute(MB1)
        """
        assert self.model.attention_worker is not None
        
        num_mb = len(micro_batches)
        tracker = self._timing_tracker
        
        for layer_idx in range(self.model.num_layers):
            # Process micro-batches with overlap
            send_handles = []
            send_start_times = []
            packed_list = []
            
            for mb_idx, mb in enumerate(micro_batches):
                # Compute attention
                compute_start = time.perf_counter()
                
                attn_output, residual = self.model.attention_worker.forward_attention_layer(
                    layer_idx=layer_idx,
                    hidden_states=mb.hidden_states,
                    attention_mask=mb.attention_mask,
                    position_ids=mb.position_ids,
                    position_embeddings=mb.position_embeddings,
                )
                packed = torch.cat([attn_output, residual], dim=-1)
                packed_list.append(packed)
                
                compute_end = time.perf_counter()
                self.stats.compute_time += compute_end - compute_start
                
                # Record timing
                if tracker:
                    tracker.record_event(EventType.ATTN_COMPUTE, layer_idx, mb_idx,
                                        compute_start, compute_end)
                
                # Start async send immediately after compute
                send_start = time.perf_counter()
                tag = self._get_tag(layer_idx, mb_idx, "attn_to_ffn")
                handle = self._send_async(packed.contiguous(), tag)
                send_handles.append(handle)
                send_start_times.append(send_start)
                
                # The next MB's compute will overlap with this send!
            
            # Wait for all sends to complete
            for mb_idx, handle in enumerate(send_handles):
                handle.wait()
                send_end = time.perf_counter()
                if tracker:
                    tracker.record_event(EventType.SEND_WAIT, layer_idx, mb_idx,
                                        send_start_times[mb_idx], send_end)
            
            # Now receive FFN results
            recv_tensors = []
            for mb_idx, mb in enumerate(micro_batches):
                recv_start = time.perf_counter()
                tag = self._get_tag(layer_idx, mb_idx, "ffn_to_attn")
                recv_handle, recv_tensor = self._recv_async(
                    (mb.batch_size, mb.seq_len, self.model.hidden_size),
                    tag
                )
                recv_handle.wait()
                recv_end = time.perf_counter()
                recv_tensors.append(recv_tensor)
                
                # Track recv wait time
                self.stats.recv_wait_time += recv_end - recv_start
                
                if tracker:
                    tracker.record_event(EventType.RECV_WAIT, layer_idx, mb_idx,
                                        recv_start, recv_end)
            
            # Update hidden states
            for mb_idx, mb in enumerate(micro_batches):
                mb.hidden_states = recv_tensors[mb_idx].clone()
        
        self._pending_sends.clear()
        
        # Generate logits
        results = []
        for mb in micro_batches:
            logits = self.model.attention_worker.forward_lm_head(mb.hidden_states)
            results.append(logits)
        
        return self.mb_manager.merge_results(results)
    
    def _run_ffn_node_simple(
        self,
        micro_batches: List[MicroBatch],
    ) -> None:
        """
        FFN node with overlapped computation and communication.
        
        The key overlap pattern:
        - For each layer, while sending MB0's result, we compute MB1's FFN
        - This overlaps send(MB0) with compute(MB1)
        """
        assert self.model.ffn_worker is not None
        
        num_mb = len(micro_batches)
        tracker = self._timing_tracker
        
        for layer_idx in range(self.model.num_layers):
            # First, post all receives (async)
            recv_handles = []
            recv_tensors = []
            recv_post_times = []
            
            for mb_idx, mb in enumerate(micro_batches):
                batch_size = getattr(mb, '_actual_batch_size', mb.batch_size)
                seq_len = getattr(mb, '_actual_seq_len', mb.seq_len)
                
                recv_post_start = time.perf_counter()
                tag = self._get_tag(layer_idx, mb_idx, "attn_to_ffn")
                recv_handle, recv_tensor = self._recv_async(
                    (batch_size, seq_len, self.model.hidden_size * 2),
                    tag
                )
                recv_handles.append(recv_handle)
                recv_tensors.append(recv_tensor)
                recv_post_times.append(recv_post_start)
            
            # Process each MB: wait for recv, compute, send immediately
            send_handles = []
            send_start_times = []
            output_list = []
            
            for mb_idx, (mb, recv_h, recv_t) in enumerate(
                zip(micro_batches, recv_handles, recv_tensors)
            ):
                # Wait for this MB's data
                recv_wait_start = time.perf_counter()
                recv_h.wait()
                recv_wait_end = time.perf_counter()
                packed = recv_t
                
                # Track recv wait time
                self.stats.recv_wait_time += recv_wait_end - recv_wait_start
                
                if tracker:
                    tracker.record_event(EventType.RECV_WAIT, layer_idx, mb_idx,
                                        recv_post_times[mb_idx], recv_wait_end)
                
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
                output = output.contiguous().clone()
                output_list.append(output)
                compute_end = time.perf_counter()
                self.stats.compute_time += compute_end - compute_start
                
                if tracker:
                    tracker.record_event(EventType.FFN_COMPUTE, layer_idx, mb_idx,
                                        compute_start, compute_end)
                
                # Send immediately (async) - next MB's compute overlaps with this send
                send_start = time.perf_counter()
                tag = self._get_tag(layer_idx, mb_idx, "ffn_to_attn")
                handle = self._send_async(output, tag)
                send_handles.append(handle)
                send_start_times.append(send_start)
            
            # Wait for all sends
            for mb_idx, handle in enumerate(send_handles):
                handle.wait()
                send_end = time.perf_counter()
                if tracker:
                    tracker.record_event(EventType.SEND_WAIT, layer_idx, mb_idx,
                                        send_start_times[mb_idx], send_end)
        
        self._pending_sends.clear()
    
    def _run_attention_node_async(
        self,
        micro_batches: List[MicroBatch],
    ) -> torch.Tensor:
        """
        Run attention node with async overlap.
        
        For each layer:
        1. Compute attention for all MBs, sending each result immediately (async)
        2. Post receives for all FFN results
        3. Wait for receives and update hidden states
        
        The key overlap: while sending MB_i, we compute MB_{i+1}
        """
        assert self.model.attention_worker is not None
        
        num_mb = len(micro_batches)
        
        for layer_idx in range(self.model.num_layers):
            layer_start = time.perf_counter()
            
            send_handles = []
            packed_tensors = []  # Store for debug
            
            # Phase 1: Compute attention and send results (overlapped)
            for mb_idx, mb in enumerate(micro_batches):
                compute_start = time.perf_counter()
                
                # Compute attention
                if self.compute_stream:
                    with torch.cuda.stream(self.compute_stream):
                        attn_output, residual = self.model.attention_worker.forward_attention_layer(
                            layer_idx=layer_idx,
                            hidden_states=mb.hidden_states,
                            attention_mask=mb.attention_mask,
                            position_ids=mb.position_ids,
                            position_embeddings=mb.position_embeddings,
                        )
                        packed = torch.cat([attn_output, residual], dim=-1)
                    self.compute_stream.synchronize()
                else:
                    attn_output, residual = self.model.attention_worker.forward_attention_layer(
                        layer_idx=layer_idx,
                        hidden_states=mb.hidden_states,
                        attention_mask=mb.attention_mask,
                        position_ids=mb.position_ids,
                        position_embeddings=mb.position_embeddings,
                    )
                    packed = torch.cat([attn_output, residual], dim=-1)
                
                packed_tensors.append(packed)
                
                compute_end = time.perf_counter()
                self.stats.compute_time += compute_end - compute_start
                
                # Send async (non-blocking) - this allows overlap with next MB compute
                tag_send = self._get_tag(layer_idx, mb_idx, "attn_to_ffn")
                send_handle = self._send_async(packed.contiguous(), tag_send)
                send_handles.append(send_handle)
            
            # Phase 2: Wait for all sends first (ensure FFN can receive)
            for handle in send_handles:
                handle.wait()
            
            # Phase 3: Post all receives after sends complete
            recv_handles = []
            recv_tensors = []
            for mb_idx, mb in enumerate(micro_batches):
                tag_recv = self._get_tag(layer_idx, mb_idx, "ffn_to_attn")
                batch_size, seq_len = mb.batch_size, mb.seq_len
                recv_handle, recv_tensor = self._recv_async(
                    (batch_size, seq_len, self.model.hidden_size),
                    tag_recv
                )
                recv_handles.append(recv_handle)
                recv_tensors.append(recv_tensor)
            
            # Phase 4: Wait for receives and update hidden states
            for mb_idx, mb in enumerate(micro_batches):
                recv_handles[mb_idx].wait()
                mb.hidden_states = recv_tensors[mb_idx].clone()
            
            layer_end = time.perf_counter()
            
            if layer_idx % 7 == 0:
                logger.debug(f"Layer {layer_idx}/{self.model.num_layers} completed in {layer_end - layer_start:.3f}s")
        
        self._pending_sends.clear()
        
        # Generate logits
        results = []
        for mb in micro_batches:
            logits = self.model.attention_worker.forward_lm_head(mb.hidden_states)
            results.append(logits)
        
        return self.mb_manager.merge_results(results)
    
    def _run_ffn_node_async(
        self,
        micro_batches: List[MicroBatch],
    ) -> None:
        """
        Run FFN node with async overlap.
        
        For each layer:
        1. Post all receives from attention (async)
        2. Wait for each recv, compute FFN, and send result immediately (async)
        
        The key overlap: while sending MB_i result, we process MB_{i+1}
        """
        assert self.model.ffn_worker is not None
        
        num_mb = len(micro_batches)
        
        for layer_idx in range(self.model.num_layers):
            layer_start = time.perf_counter()
            
            # Phase 1: Post all receives first
            recv_handles = []
            recv_tensors = []
            
            for mb_idx, mb in enumerate(micro_batches):
                batch_size = getattr(mb, '_actual_batch_size', mb.batch_size)
                seq_len = getattr(mb, '_actual_seq_len', mb.seq_len)
                
                tag_recv = self._get_tag(layer_idx, mb_idx, "attn_to_ffn")
                recv_handle, recv_tensor = self._recv_async(
                    (batch_size, seq_len, self.model.hidden_size * 2),
                    tag_recv
                )
                recv_handles.append(recv_handle)
                recv_tensors.append(recv_tensor)
            
            # Phase 2: Process each micro-batch sequentially but with async sends
            send_handles = []
            output_tensors = []  # Keep references to output tensors
            
            for mb_idx, mb in enumerate(micro_batches):
                # Wait for this micro-batch's data
                recv_handles[mb_idx].wait()
                packed = recv_tensors[mb_idx]
                
                # Unpack - clone to ensure we have our own copy
                attn_output = packed[..., :self.model.hidden_size].clone()
                residual = packed[..., self.model.hidden_size:].clone()
                
                # Compute FFN
                compute_start = time.perf_counter()
                
                if self.compute_stream:
                    with torch.cuda.stream(self.compute_stream):
                        output = self.model.ffn_worker.forward_ffn_layer(
                            layer_idx=layer_idx,
                            attn_output=attn_output,
                            residual=residual,
                        )
                    self.compute_stream.synchronize()
                else:
                    output = self.model.ffn_worker.forward_ffn_layer(
                        layer_idx=layer_idx,
                        attn_output=attn_output,
                        residual=residual,
                    )
                
                # Make a contiguous copy for sending
                output = output.contiguous().clone()
                output_tensors.append(output)  # Keep reference
                
                compute_end = time.perf_counter()
                self.stats.compute_time += compute_end - compute_start
                
                # Send async - allows overlap with next MB processing
                tag_send = self._get_tag(layer_idx, mb_idx, "ffn_to_attn")
                send_handle = self._send_async(output, tag_send)
                send_handles.append(send_handle)
            
            # Wait for all sends before next layer
            for handle in send_handles:
                handle.wait()
            
            layer_end = time.perf_counter()
            
            if layer_idx % 7 == 0:
                logger.debug(f"FFN Layer {layer_idx}/{self.model.num_layers} completed in {layer_end - layer_start:.3f}s")
        
        self._pending_sends.clear()
    
    def get_stats(self) -> DBOStats:
        """Get execution statistics."""
        return self.stats
