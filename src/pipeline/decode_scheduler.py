"""
Decode DBO Scheduler - Pipeline for decode phase with micro-batch overlap.

Implements the DBO (Dual Batch Overlap) principle on BOTH attention and FFN sides:
- Attention side: per-MB attention → isend(MB_i) || attention(MB_{i+1})
- FFN side: compute(MB0) → isend(MB0) || compute(MB1) → isend(MB1)

KV cache (DynamicCache) stores tensors as [batch, heads, seq, dim], so the batch
dimension is naturally sliceable. Each micro-batch runs attention with its own
batch slice of the KV cache, then the updated slices are merged back.
"""

import logging
import time
from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass

import torch
import torch.distributed as dist

from ..distributed import get_distributed_context
from ..utils.timing import TimingTracker, PipelineTiming, EventType
from .async_scheduler import SendTransferMonitor

logger = logging.getLogger(__name__)


@dataclass
class DecodeDBOStats:
    """Statistics from Decode DBO execution."""
    total_time: float = 0.0
    compute_time: float = 0.0
    comm_time: float = 0.0
    num_tokens: int = 0
    num_layers: int = 0

    attn_compute_time: float = 0.0
    ffn_compute_time: float = 0.0
    a2f_comm_time: float = 0.0
    f2a_comm_time: float = 0.0

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
    Decode phase scheduler with micro-batch DBO overlap.

    DBO pattern per layer:
    - Attention node: for each MB: attention(MB_i) → isend(MB_i)
      Key overlap: isend(MB_i) || attention(MB_{i+1})
      KV cache is batch-sliced so each MB only touches its own batch range.
    - FFN node: irecv all MBs → for each: wait recv → compute → isend result
      Key overlap: isend(MB_i result) || compute(MB_{i+1})
    - Attention node: recv all MB results → merge → next layer
    """

    def __init__(
        self,
        model,  # DisaggregatedQwenModel
        num_micro_batches: int = 2,
        enable_timing: bool = False,
        timing_mode: str = "cuda_events",
    ):
        self.model = model
        self.ctx = get_distributed_context()
        self.num_micro_batches = num_micro_batches
        self.enable_timing = enable_timing
        self.timing_mode = timing_mode
        self.stats = DecodeDBOStats()
        self._timing_data: Optional[PipelineTiming] = None
        # Track timing on step 1 (skip step 0 warmup)
        self._timing_step = 1
        self._current_step = 0
        # Send monitor for accurate async send timing
        self._send_monitor: Optional[SendTransferMonitor] = None
        logger.debug(
            f"DecodeDBOScheduler initialized: num_mb={num_micro_batches}"
        )

    def _get_tag(self, layer_idx: int, mb_idx: int, direction: str) -> int:
        """Unique tag per (layer, micro-batch, direction)."""
        dir_offset = 0 if direction == "a2f" else 1
        return 10000 + layer_idx * (self.num_micro_batches * 2) + mb_idx * 2 + dir_offset

    def _compute_mb_sizes(self, batch_size: int) -> List[int]:
        """Compute micro-batch sizes (handles uneven splits)."""
        num_mb = min(self.num_micro_batches, batch_size)
        base = batch_size // num_mb
        remainder = batch_size % num_mb
        return [base + (1 if i < remainder else 0) for i in range(num_mb)]

    @torch.no_grad()
    def forward_decode_dbo(
        self,
        input_ids: torch.Tensor,   # [batch_size, 1]
        position_ids: torch.Tensor, # [batch_size, 1]
        kv_cache,                   # DynamicCache (attention node) or None (FFN node)
    ) -> Optional[torch.Tensor]:
        """
        Decode forward pass with micro-batch DBO overlap.

        Returns:
            Logits [batch_size, 1, vocab_size] on attention node, None on FFN node.
        """
        start_time = time.perf_counter()
        batch_size = input_ids.shape[0]

        self.stats = DecodeDBOStats(num_tokens=batch_size)

        should_track = self.enable_timing and self._current_step == self._timing_step
        tracker = None
        if should_track:
            actual_num_mb = min(self.num_micro_batches, batch_size)
            tracker = TimingTracker(
                node=self.ctx.role,
                num_layers=self.model.num_layers,
                num_micro_batches=actual_num_mb,
                mode=self.timing_mode,
            )
            self._send_monitor = SendTransferMonitor(poll_interval=0.0001)
        else:
            self._send_monitor = None

        if self.ctx.is_attention_node:
            result = self._run_attention_decode(input_ids, position_ids, kv_cache, tracker)
        else:
            self._run_ffn_decode(batch_size, tracker)
            result = None

        if should_track and tracker is not None:
            self._timing_data = tracker.finish()

        self.stats.total_time = time.perf_counter() - start_time
        self.stats.num_layers = self.model.num_layers
        self._current_step += 1
        return result

    def _run_attention_decode(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        kv_cache,
        tracker: Optional[TimingTracker] = None,
    ) -> torch.Tensor:
        """
        Attention node: per-MB attention with batch-sliced KV cache.

        For each layer:
        1. Snapshot the current layer's KV cache tensors
        2. For each micro-batch:
           a. Set cache to this MB's batch slice
           b. Run attention (updates the sliced cache in-place)
           c. isend result immediately (overlaps with next MB's compute)
        3. Merge all MB cache updates back into the full KV cache
        4. Recv FFN results for all MBs
        """
        assert self.model.attention_worker is not None

        batch_size = input_ids.shape[0]
        num_layers = self.model.num_layers
        mb_sizes = self._compute_mb_sizes(batch_size)
        num_mb = len(mb_sizes)

        # Precompute batch offsets for each MB
        mb_offsets = []
        offset = 0
        for s in mb_sizes:
            mb_offsets.append(offset)
            offset += s

        hidden_states = self.model.attention_worker.embed(input_ids)
        position_embeddings = self.model.attention_worker.get_position_embeddings(
            hidden_states, position_ids
        )
        cur_pos = kv_cache.get_seq_length()
        total_len = cur_pos + 1
        attention_mask = self.model._make_causal_mask(batch_size, 1, total_len)

        for layer_idx in range(num_layers):
            cache_layer = kv_cache.layers[layer_idx]
            orig_keys = cache_layer.keys
            orig_values = cache_layer.values

            mb_updated_keys = []
            mb_updated_values = []
            send_handles = []

            for mb_idx in range(num_mb):
                start = mb_offsets[mb_idx]
                end = start + mb_sizes[mb_idx]

                if tracker:
                    tracker.mark_start(EventType.ATTN_COMPUTE, layer_idx, mb_idx)
                compute_start = time.perf_counter()

                # Temporarily set cache to this MB's batch slice
                cache_layer.keys = orig_keys[start:end]
                cache_layer.values = orig_values[start:end]

                # Slice inputs for this micro-batch
                mb_hidden = hidden_states[start:end]
                mb_mask = attention_mask[start:end]
                mb_pos_ids = position_ids[start:end]
                mb_pos_emb = None
                if position_embeddings is not None:
                    cos, sin = position_embeddings
                    mb_pos_emb = (cos[start:end], sin[start:end])

                attn_output, residual, _ = self.model.attention_worker.forward_attention_layer(
                    layer_idx=layer_idx,
                    hidden_states=mb_hidden,
                    attention_mask=mb_mask,
                    position_ids=mb_pos_ids,
                    position_embeddings=mb_pos_emb,
                    use_cache=True,
                    past_key_value=kv_cache,
                )

                # Collect updated KV (attention appended 1 new token for this MB's batch slice)
                mb_updated_keys.append(cache_layer.keys)
                mb_updated_values.append(cache_layer.values)

                packed = (attn_output + residual).contiguous()

                if tracker:
                    tracker.mark_end(EventType.ATTN_COMPUTE, layer_idx, mb_idx)
                compute_end = time.perf_counter()
                compute_time = compute_end - compute_start
                self.stats.compute_time += compute_time
                self.stats.attn_compute_time += compute_time

                # Async send immediately — next MB's attention overlaps with this send
                send_start = time.perf_counter()
                tag = self._get_tag(layer_idx, mb_idx, "a2f")
                handle = dist.isend(packed, dst=self.ctx.peer_rank, tag=tag)
                send_handles.append(handle)
                if self._send_monitor:
                    self._send_monitor.start_monitoring(
                        handle, send_start, layer_idx, mb_idx, "a2f")

            # Reconstruct full KV cache for this layer from all MB updates
            cache_layer.keys = torch.cat(mb_updated_keys, dim=0)
            cache_layer.values = torch.cat(mb_updated_values, dim=0)

            # Wait for all sends
            for handle in send_handles:
                handle.wait()

            # Receive micro-batch results from FFN
            recv_chunks = []
            for mb_idx, mb_size in enumerate(mb_sizes):
                if tracker:
                    tracker.mark_start(EventType.RECV_WAIT, layer_idx, mb_idx)
                recv_start = time.perf_counter()
                tag = self._get_tag(layer_idx, mb_idx, "f2a")
                recv_tensor = torch.empty(
                    mb_size, 1, self.model.hidden_size,
                    dtype=self.model.dtype, device=self.model.device,
                )
                dist.recv(recv_tensor, src=self.ctx.peer_rank, tag=tag)
                recv_end = time.perf_counter()
                recv_chunks.append(recv_tensor)
                if tracker:
                    tracker.mark_end(EventType.RECV_WAIT, layer_idx, mb_idx)
                self.stats.f2a_comm_time += recv_end - recv_start

            hidden_states = torch.cat(recv_chunks, dim=0)

        # Collect send monitor results for accurate timing
        if self._send_monitor and tracker:
            for layer_idx, mb_idx, direction, start, end in self._send_monitor.collect_results():
                tracker.record_event(EventType.SEND_TRANSFER, layer_idx, mb_idx, start, end)
                self.stats.a2f_comm_time += end - start

        self.stats.comm_time = self.stats.a2f_comm_time + self.stats.f2a_comm_time
        return self.model.attention_worker.forward_lm_head(hidden_states)

    def _post_irecvs(self, layer_idx: int, mb_sizes: List[int]) -> Tuple[List, List]:
        """Post irecvs for all MBs of a given layer. Returns (handles, tensors)."""
        handles = []
        tensors = []
        for mb_idx, mb_size in enumerate(mb_sizes):
            tag = self._get_tag(layer_idx, mb_idx, "a2f")
            recv_tensor = torch.empty(
                mb_size, 1, self.model.hidden_size,
                dtype=self.model.dtype, device=self.model.device,
            )
            handle = dist.irecv(recv_tensor, src=self.ctx.peer_rank, tag=tag)
            handles.append(handle)
            tensors.append(recv_tensor)
        return handles, tensors

    def _run_ffn_decode(self, batch_size: int, tracker: Optional[TimingTracker] = None) -> None:
        """
        FFN node: micro-batch DBO overlap with pre-posted irecvs.

        Key optimization: irecvs for the NEXT layer are posted before the current
        layer starts processing, so data arrives into buffers while we compute.
        This eliminates the ~1.9ms/layer MB0 recv_wait bubble.

        Flow:
        1. Post L0 irecvs upfront
        2. For each layer:
           a. Post L+1 irecvs (if not last layer) — data arrives during compute
           b. For each MB: wait recv → compute → isend
        3. Wait all sends at the end
        """
        assert self.model.ffn_worker is not None

        num_layers = self.model.num_layers
        mb_sizes = self._compute_mb_sizes(batch_size)
        num_mb = len(mb_sizes)

        # Pre-post L0 irecvs before any processing
        curr_recv_handles, curr_recv_tensors = self._post_irecvs(0, mb_sizes)

        all_send_handles = []

        for layer_idx in range(num_layers):
            # Post NEXT layer's irecvs early (double-buffer)
            # Data arrives into these buffers while we process current layer
            next_recv_handles, next_recv_tensors = None, None
            if layer_idx + 1 < num_layers:
                next_recv_handles, next_recv_tensors = self._post_irecvs(
                    layer_idx + 1, mb_sizes
                )

            # Process each MB: wait recv → compute → isend
            for mb_idx in range(num_mb):
                # Wait for this MB's data from attention
                if tracker:
                    tracker.mark_start(EventType.RECV_WAIT, layer_idx, mb_idx)
                recv_start = time.perf_counter()
                curr_recv_handles[mb_idx].wait()
                recv_end = time.perf_counter()
                if tracker:
                    tracker.mark_end(EventType.RECV_WAIT, layer_idx, mb_idx)
                self.stats.a2f_comm_time += recv_end - recv_start

                # Compute FFN
                if tracker:
                    tracker.mark_start(EventType.FFN_COMPUTE, layer_idx, mb_idx)
                compute_start = time.perf_counter()
                output = self.model.ffn_worker.forward_ffn_layer(
                    layer_idx=layer_idx,
                    hidden_states=curr_recv_tensors[mb_idx],
                )
                if isinstance(output, tuple):
                    output = output[0]
                output = output.contiguous()
                if tracker:
                    tracker.mark_end(EventType.FFN_COMPUTE, layer_idx, mb_idx)
                compute_end = time.perf_counter()
                compute_time = compute_end - compute_start
                self.stats.compute_time += compute_time
                self.stats.ffn_compute_time += compute_time

                # Send result back immediately (async)
                tag = self._get_tag(layer_idx, mb_idx, "f2a")
                send_start = time.perf_counter()
                handle = dist.isend(output, dst=self.ctx.peer_rank, tag=tag)
                all_send_handles.append(handle)
                if self._send_monitor:
                    self._send_monitor.start_monitoring(
                        handle, send_start, layer_idx, mb_idx, "f2a")

            # Swap to next layer's pre-posted irecvs
            if next_recv_handles is not None:
                curr_recv_handles = next_recv_handles
                curr_recv_tensors = next_recv_tensors

        # Wait for all sends to complete at the end
        for handle in all_send_handles:
            handle.wait()

        # Collect send monitor results for accurate timing
        if self._send_monitor and tracker:
            for layer_idx, mb_idx, direction, start, end in self._send_monitor.collect_results():
                tracker.record_event(EventType.SEND_TRANSFER, layer_idx, mb_idx, start, end)
                self.stats.f2a_comm_time += end - start

        self.stats.comm_time = self.stats.a2f_comm_time + self.stats.f2a_comm_time

    def get_stats(self) -> DecodeDBOStats:
        """Get statistics from last run."""
        return self.stats

    def get_timing_data(self) -> Optional[PipelineTiming]:
        """Get per-layer timing data (tracked for one representative decode step)."""
        return self._timing_data
