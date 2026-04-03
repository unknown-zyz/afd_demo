"""
Decode DBO Scheduler - Pipeline for decode phase with micro-batch overlap.

Implements the same DBO (Dual Batch Overlap) principle as the prefill scheduler:
- FFN side: compute(MB0) → isend(MB0) || compute(MB1) → isend(MB1)
- The isend of one micro-batch overlaps with the compute of the next

Key difference from prefill: attention computation is full-batch because
KV cache is shared across all batch entries and cannot be split. Only the
FFN communication and computation are micro-batched.
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
    - Attention node: full-batch attention → split output → isend micro-batches to FFN
    - FFN node: irecv all MBs → for each: wait recv → compute → isend result
      Key overlap: isend(MB_i result) || compute(MB_{i+1})
    - Attention node: recv all MB results → merge

    Attention is always full-batch because the KV cache is shared across
    all batch entries. The micro-batch overlap happens on the FFN side.
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
        Attention node: full-batch attention → split → isend micro-batches → recv results.
        """
        assert self.model.attention_worker is not None

        batch_size = input_ids.shape[0]
        num_layers = self.model.num_layers
        mb_sizes = self._compute_mb_sizes(batch_size)
        num_mb = len(mb_sizes)

        hidden_states = self.model.attention_worker.embed(input_ids)
        position_embeddings = self.model.attention_worker.get_position_embeddings(
            hidden_states, position_ids
        )
        cur_pos = kv_cache.get_seq_length()
        total_len = cur_pos + 1
        attention_mask = self.model._make_causal_mask(batch_size, 1, total_len)

        for layer_idx in range(num_layers):
            # Full-batch attention (KV cache requires full batch)
            if tracker:
                torch.cuda.current_stream().synchronize()
            compute_start = time.perf_counter()
            attn_output, residual, _ = self.model.attention_worker.forward_attention_layer(
                layer_idx=layer_idx,
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                use_cache=True,
                past_key_value=kv_cache,
            )
            packed = (attn_output + residual).contiguous()
            if tracker:
                torch.cuda.current_stream().synchronize()
            compute_end = time.perf_counter()
            compute_time = compute_end - compute_start
            self.stats.compute_time += compute_time
            self.stats.attn_compute_time += compute_time
            if tracker:
                tracker.record_event(EventType.ATTN_COMPUTE, layer_idx, 0,
                                    compute_start, compute_end)

            # Split output into micro-batches and async send
            chunks = packed.split(mb_sizes, dim=0)
            send_handles = []
            for mb_idx, chunk in enumerate(chunks):
                send_start = time.perf_counter()
                tag = self._get_tag(layer_idx, mb_idx, "a2f")
                handle = dist.isend(chunk.contiguous(), dst=self.ctx.peer_rank, tag=tag)
                send_handles.append((handle, send_start, mb_idx))

            # Wait for all sends
            for handle, send_start, mb_idx in send_handles:
                handle.wait()
                send_end = time.perf_counter()
                if tracker:
                    tracker.record_event(EventType.SEND_TRANSFER, layer_idx, mb_idx,
                                        send_start, send_end)
                self.stats.a2f_comm_time += send_end - send_start

            # Receive micro-batch results from FFN
            recv_chunks = []
            for mb_idx, mb_size in enumerate(mb_sizes):
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
                    tracker.record_event(EventType.RECV_WAIT, layer_idx, mb_idx,
                                        recv_start, recv_end)
                self.stats.f2a_comm_time += recv_end - recv_start

            hidden_states = torch.cat(recv_chunks, dim=0)

        self.stats.comm_time = self.stats.a2f_comm_time + self.stats.f2a_comm_time
        return self.model.attention_worker.forward_lm_head(hidden_states)

    def _run_ffn_decode(self, batch_size: int, tracker: Optional[TimingTracker] = None) -> None:
        """
        FFN node: micro-batch DBO overlap.
        
        For each layer:
        1. Post all irecvs upfront
        2. For each MB: wait recv → compute FFN → isend result
           Key: isend(MB_i) overlaps with compute(MB_{i+1})
        3. Wait for all sends
        """
        assert self.model.ffn_worker is not None

        num_layers = self.model.num_layers
        mb_sizes = self._compute_mb_sizes(batch_size)
        num_mb = len(mb_sizes)

        for layer_idx in range(num_layers):
            # Post all irecvs upfront (non-blocking)
            recv_handles = []
            recv_tensors = []
            for mb_idx, mb_size in enumerate(mb_sizes):
                tag = self._get_tag(layer_idx, mb_idx, "a2f")
                recv_tensor = torch.empty(
                    mb_size, 1, self.model.hidden_size,
                    dtype=self.model.dtype, device=self.model.device,
                )
                handle = dist.irecv(recv_tensor, src=self.ctx.peer_rank, tag=tag)
                recv_handles.append(handle)
                recv_tensors.append(recv_tensor)

            # Process each MB: wait recv → compute → isend
            send_handles = []
            for mb_idx in range(num_mb):
                # Wait for this MB's data from attention
                recv_start = time.perf_counter()
                recv_handles[mb_idx].wait()
                recv_end = time.perf_counter()
                if tracker:
                    tracker.record_event(EventType.RECV_WAIT, layer_idx, mb_idx,
                                        recv_start, recv_end)
                self.stats.a2f_comm_time += recv_end - recv_start

                # Compute FFN
                if tracker:
                    torch.cuda.current_stream().synchronize()
                compute_start = time.perf_counter()
                output = self.model.ffn_worker.forward_ffn_layer(
                    layer_idx=layer_idx,
                    hidden_states=recv_tensors[mb_idx],
                )
                if isinstance(output, tuple):
                    output = output[0]
                output = output.contiguous()
                if tracker:
                    torch.cuda.current_stream().synchronize()
                compute_end = time.perf_counter()
                compute_time = compute_end - compute_start
                self.stats.compute_time += compute_time
                self.stats.ffn_compute_time += compute_time
                if tracker:
                    tracker.record_event(EventType.FFN_COMPUTE, layer_idx, mb_idx,
                                        compute_start, compute_end)

                # Send result back immediately (async)
                # Next MB's compute will overlap with this send!
                tag = self._get_tag(layer_idx, mb_idx, "f2a")
                send_start = time.perf_counter()
                handle = dist.isend(output, dst=self.ctx.peer_rank, tag=tag)
                send_handles.append((handle, send_start, mb_idx))

            # Wait for all sends to complete
            for handle, send_start, mb_idx in send_handles:
                handle.wait()
                send_end = time.perf_counter()
                if tracker:
                    tracker.record_event(EventType.SEND_TRANSFER, layer_idx, mb_idx,
                                        send_start, send_end)
                self.stats.f2a_comm_time += send_end - send_start

        self.stats.comm_time = self.stats.a2f_comm_time + self.stats.f2a_comm_time

    def get_stats(self) -> DecodeDBOStats:
        """Get statistics from last run."""
        return self.stats

    def get_timing_data(self) -> Optional[PipelineTiming]:
        """Get per-layer timing data (tracked for one representative decode step)."""
        return self._timing_data
