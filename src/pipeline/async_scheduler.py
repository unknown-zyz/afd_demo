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
import threading
import queue
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


class SendTransferMonitor:
    """
    Monitor for measuring actual send transfer time using background polling.
    
    When isend() is called, NCCL starts the transfer asynchronously.
    This monitor polls handle.is_completed() to detect when transfer actually finishes,
    independent of when wait() is called.
    
    Usage:
        monitor = SendTransferMonitor(poll_interval=0.0001)  # 100μs
        monitor.start_monitoring(handle, start_time, layer_idx, mb_idx, direction)
        # ... do other work ...
        handle.wait()
        results = monitor.collect_results()  # [(layer, mb, direction, start, end), ...]
    """
    
    def __init__(self, poll_interval: float = 0.0001):
        """
        Args:
            poll_interval: Polling interval in seconds (default 100μs)
        """
        self.poll_interval = poll_interval
        self._results: queue.Queue = queue.Queue()
        self._threads: List[threading.Thread] = []
    
    def start_monitoring(
        self,
        handle: dist.Work,
        start_time: float,
        layer_idx: int,
        mb_idx: int,
        direction: str = "a2f",
    ) -> None:
        """
        Start background thread to poll for send completion.
        
        Args:
            handle: The dist.Work handle from isend()
            start_time: Time when isend() was called
            layer_idx: Layer index for recording
            mb_idx: Micro-batch index for recording
            direction: "a2f" (attention→ffn) or "f2a" (ffn→attention)
        """
        thread = threading.Thread(
            target=self._poll_completion,
            args=(handle, start_time, layer_idx, mb_idx, direction),
            daemon=True,
        )
        thread.start()
        self._threads.append(thread)
    
    def _poll_completion(
        self,
        handle: dist.Work,
        start_time: float,
        layer_idx: int,
        mb_idx: int,
        direction: str,
    ) -> None:
        """Background polling thread."""
        while not handle.is_completed():
            time.sleep(self.poll_interval)
        end_time = time.perf_counter()
        self._results.put((layer_idx, mb_idx, direction, start_time, end_time))
    
    def collect_results(self) -> List[Tuple[int, int, str, float, float]]:
        """
        Collect all completed monitoring results.
        
        Returns:
            List of (layer_idx, mb_idx, direction, start_time, end_time) tuples
        """
        # Wait for all threads to finish (with timeout)
        for thread in self._threads:
            thread.join(timeout=1.0)
        
        results = []
        while not self._results.empty():
            try:
                results.append(self._results.get_nowait())
            except queue.Empty:
                break
        
        # Clear threads list for reuse
        self._threads.clear()
        return results
    
    def clear(self) -> None:
        """Clear all pending results and threads."""
        self._threads.clear()
        while not self._results.empty():
            try:
                self._results.get_nowait()
            except queue.Empty:
                break


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
        timing_mode: str = "cuda_events",
    ):
        """
        Initialize async scheduler.
        
        Args:
            model: DisaggregatedQwenModel instance
            num_micro_batches: Number of micro-batches (2 for standard DBO)
            use_cuda_streams: Whether to use separate CUDA streams
            enable_timing: Whether to record detailed per-MB timing
            timing_mode: "cuda_events" (zero-overhead) or "sync" (legacy)
        """
        self.model = model
        self.ctx = get_distributed_context()
        self.num_micro_batches = num_micro_batches
        self.use_cuda_streams = use_cuda_streams
        self.enable_timing = enable_timing
        self.timing_mode = timing_mode
        
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
        
        # Send transfer monitor (only used when timing is enabled)
        self._send_monitor: Optional[SendTransferMonitor] = None
        
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
                mode=self.timing_mode,
            )
            # Initialize send transfer monitor for real transfer time measurement
            self._send_monitor = SendTransferMonitor(poll_interval=0.0001)
        else:
            self._send_monitor = None
        
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
        Attention node with cross-layer pipelined computation and communication.
        
        跨层 MB 流水线优化：recv 和下一层 compute 交错进行，每个 MB 独立推进。
        MB0 的 Layer L+1 attention 不再等待 MB1 的 Layer L recv 完成。
        
        流水线结构：
          Layer 0:   compute(MB0) → send(MB0) → compute(MB1) → send(MB1)
          Layer 1~N-1: post recv(all MBs, prev_layer) →
                       per-MB: wait recv → update → compute → send
          Last recv: recv(last_layer) → update hidden_states
        """
        assert self.model.attention_worker is not None
        
        num_mb = len(micro_batches)
        num_layers = self.model.num_layers
        tracker = self._timing_tracker
        monitor = self._send_monitor
        
        # === Layer 0: compute ALL MBs first, then send ===
        # Deferred-send: compute all MBs before issuing any isend.
        # This gives FFN time to finish initialization and post its irecv,
        # preventing NCCL flow-control blocking on the first A2F sends.
        # (Without this, MB0's send blocks 15-24ms waiting for FFN to drain.)
        prev_send_handles = []
        layer0_outputs = []
        
        for mb_idx, mb in enumerate(micro_batches):
            # Stream sync for accurate timing (only sync compute stream, not NCCL)
            if tracker:
                torch.cuda.current_stream().synchronize()
            compute_start = time.perf_counter()
            
            attn_output, residual = self.model.attention_worker.forward_attention_layer(
                layer_idx=0,
                hidden_states=mb.hidden_states,
                attention_mask=mb.attention_mask,
                position_ids=mb.position_ids,
                position_embeddings=mb.position_embeddings,
            )
            # Pre-add residual on attention side to halve A2F data (1×H instead of 2×H)
            packed = (attn_output + residual).contiguous()
            
            if tracker:
                torch.cuda.current_stream().synchronize()
            compute_end = time.perf_counter()
            self.stats.compute_time += compute_end - compute_start
            
            if tracker:
                tracker.record_event(EventType.ATTN_COMPUTE, 0, mb_idx,
                                    compute_start, compute_end)
            
            layer0_outputs.append(packed)
        
        # Now send all Layer 0 outputs (FFN has had time to post irecv)
        for mb_idx, packed in enumerate(layer0_outputs):
            send_start = time.perf_counter()
            tag = self._get_tag(0, mb_idx, "attn_to_ffn")
            handle = self._send_async(packed, tag)
            prev_send_handles.append(handle)
            
            if monitor:
                monitor.start_monitoring(handle, send_start, 0, mb_idx, "a2f")
        
        # === Layers 1 ~ num_layers-1: 交错 recv(prev_layer) + compute(curr_layer) ===
        for layer_idx in range(1, num_layers):
            prev_layer = layer_idx - 1
            
            # Post all recvs for previous layer (non-blocking，尽早占位)
            recv_handles = []
            recv_start_times = []
            for mb_idx, mb in enumerate(micro_batches):
                recv_start = time.perf_counter()
                tag = self._get_tag(prev_layer, mb_idx, "ffn_to_attn")
                recv_handle, recv_tensor = self._recv_async(
                    (mb.batch_size, mb.seq_len, self.model.hidden_size),
                    tag
                )
                recv_handles.append((recv_handle, recv_tensor))
                recv_start_times.append(recv_start)
            
            # Wait all sends from previous layer
            for handle in prev_send_handles:
                handle.wait()
            
            # Collect and record real transfer times from monitor
            if monitor and tracker:
                for result in monitor.collect_results():
                    l_idx, m_idx, direction, start, end = result
                    tracker.record_event(EventType.SEND_TRANSFER, l_idx, m_idx, start, end)
            
            # 逐 MB 处理：recv 完一个就立即 compute 下一层，不等其他 MB
            send_handles = []
            
            for mb_idx, mb in enumerate(micro_batches):
                # Wait for this MB's recv from previous layer
                recv_handle, recv_tensor = recv_handles[mb_idx]
                recv_handle.wait()
                recv_end = time.perf_counter()
                
                # Track recv wait time
                self.stats.recv_wait_time += recv_end - recv_start_times[mb_idx]
                
                if tracker:
                    tracker.record_event(EventType.RECV_WAIT, prev_layer, mb_idx,
                                        recv_start_times[mb_idx], recv_end)
                
                # Update hidden states from FFN result
                mb.hidden_states = recv_tensor.clone()
                
                # Immediately compute this MB's current layer attention
                if tracker:
                    torch.cuda.current_stream().synchronize()
                compute_start = time.perf_counter()
                
                attn_output, residual = self.model.attention_worker.forward_attention_layer(
                    layer_idx=layer_idx,
                    hidden_states=mb.hidden_states,
                    attention_mask=mb.attention_mask,
                    position_ids=mb.position_ids,
                    position_embeddings=mb.position_embeddings,
                )
                packed = (attn_output + residual).contiguous()
                
                if tracker:
                    torch.cuda.current_stream().synchronize()
                compute_end = time.perf_counter()
                self.stats.compute_time += compute_end - compute_start
                
                if tracker:
                    tracker.record_event(EventType.ATTN_COMPUTE, layer_idx, mb_idx,
                                        compute_start, compute_end)
                
                # Send async — next MB's recv.wait will overlap with this send
                send_start = time.perf_counter()
                tag = self._get_tag(layer_idx, mb_idx, "attn_to_ffn")
                handle = self._send_async(packed, tag)
                send_handles.append(handle)
                
                if monitor:
                    monitor.start_monitoring(handle, send_start, layer_idx, mb_idx, "a2f")
            
            prev_send_handles = send_handles
        
        # === 最后一层的 recv: wait sends + recv + update ===
        last_layer = num_layers - 1
        
        for handle in prev_send_handles:
            handle.wait()
        
        if monitor and tracker:
            for result in monitor.collect_results():
                l_idx, m_idx, direction, start, end = result
                tracker.record_event(EventType.SEND_TRANSFER, l_idx, m_idx, start, end)
        
        for mb_idx, mb in enumerate(micro_batches):
            recv_start = time.perf_counter()
            tag = self._get_tag(last_layer, mb_idx, "ffn_to_attn")
            recv_handle, recv_tensor = self._recv_async(
                (mb.batch_size, mb.seq_len, self.model.hidden_size),
                tag
            )
            recv_handle.wait()
            recv_end = time.perf_counter()
            
            self.stats.recv_wait_time += recv_end - recv_start
            
            if tracker:
                tracker.record_event(EventType.RECV_WAIT, last_layer, mb_idx,
                                    recv_start, recv_end)
            
            mb.hidden_states = recv_tensor.clone()
        
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
        monitor = self._send_monitor
        
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
                    (batch_size, seq_len, self.model.hidden_size),
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
                hidden_states_in = recv_t
                
                # Track recv wait time
                self.stats.recv_wait_time += recv_wait_end - recv_wait_start
                
                if tracker:
                    tracker.record_event(EventType.RECV_WAIT, layer_idx, mb_idx,
                                        recv_wait_start, recv_wait_end)
                
                # Stream sync for accurate timing (only sync compute stream, not NCCL)
                if tracker:
                    torch.cuda.current_stream().synchronize()
                
                # Compute FFN (input is pre-combined: attn_output + residual)
                compute_start = time.perf_counter()
                ffn_result = self.model.ffn_worker.forward_ffn_layer(
                    layer_idx=layer_idx,
                    hidden_states=hidden_states_in,
                    return_timing=bool(tracker and self.model.supports_moe_timing),
                )
                if isinstance(ffn_result, tuple):
                    output, stage_timing = ffn_result
                else:
                    output, stage_timing = ffn_result, None
                output = output.contiguous().clone()
                output_list.append(output)
                if tracker:
                    torch.cuda.current_stream().synchronize()
                compute_end = time.perf_counter()
                self.stats.compute_time += compute_end - compute_start
                
                if tracker:
                    tracker.record_event(EventType.FFN_COMPUTE, layer_idx, mb_idx,
                                        compute_start, compute_end)
                
                # MoE sub-stage timing (uses CPU timestamps from ffn_worker)
                if tracker and stage_timing is not None:
                    if stage_timing.router_s > 0:
                        tracker.record_event(
                            EventType.MOE_ROUTER,
                            layer_idx,
                            mb_idx,
                            compute_start,
                            compute_start + stage_timing.router_s,
                        )
                    if stage_timing.experts_s > 0:
                        experts_start = compute_start + stage_timing.router_s
                        tracker.record_event(
                            EventType.MOE_EXPERTS,
                            layer_idx,
                            mb_idx,
                            experts_start,
                            experts_start + stage_timing.experts_s,
                        )
                    if stage_timing.shared_or_dense_s > 0:
                        shared_start = compute_start + stage_timing.router_s + stage_timing.experts_s
                        tracker.record_event(
                            EventType.MOE_SHARED_OR_DENSE,
                            layer_idx,
                            mb_idx,
                            shared_start,
                            shared_start + stage_timing.shared_or_dense_s,
                        )
                
                # Send immediately (async) - next MB's compute overlaps with this send
                send_start = time.perf_counter()
                tag = self._get_tag(layer_idx, mb_idx, "ffn_to_attn")
                handle = self._send_async(output, tag)
                send_handles.append(handle)
                send_start_times.append(send_start)
                
                # Start polling for actual transfer completion (if timing enabled)
                if monitor:
                    monitor.start_monitoring(handle, send_start, layer_idx, mb_idx, "f2a")
            
            # Wait for all sends
            for mb_idx, handle in enumerate(send_handles):
                handle.wait()
            
            # Collect and record real transfer times from monitor
            if monitor and tracker:
                for result in monitor.collect_results():
                    l_idx, m_idx, direction, start, end = result
                    tracker.record_event(EventType.SEND_TRANSFER, l_idx, m_idx, start, end)
        
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
                        packed = (attn_output + residual).contiguous()
                    self.compute_stream.synchronize()
                else:
                    attn_output, residual = self.model.attention_worker.forward_attention_layer(
                        layer_idx=layer_idx,
                        hidden_states=mb.hidden_states,
                        attention_mask=mb.attention_mask,
                        position_ids=mb.position_ids,
                        position_embeddings=mb.position_embeddings,
                    )
                    packed = (attn_output + residual).contiguous()
                
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
                    (batch_size, seq_len, self.model.hidden_size),
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
                hidden_states_in = recv_tensors[mb_idx]
                
                # Compute FFN (input is pre-combined: attn_output + residual)
                compute_start = time.perf_counter()
                
                if self.compute_stream:
                    with torch.cuda.stream(self.compute_stream):
                        ffn_result = self.model.ffn_worker.forward_ffn_layer(
                            layer_idx=layer_idx,
                            hidden_states=hidden_states_in,
                            return_timing=False,
                        )
                    self.compute_stream.synchronize()
                else:
                    ffn_result = self.model.ffn_worker.forward_ffn_layer(
                        layer_idx=layer_idx,
                        hidden_states=hidden_states_in,
                        return_timing=False,
                    )
                output = ffn_result[0] if isinstance(ffn_result, tuple) else ffn_result
                
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
