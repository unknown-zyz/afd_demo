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
    Decode phase scheduler with cross-layer micro-batch pipelining.

    MB0 can proceed to layer L+1 as soon as its F2A result arrives,
    without waiting for MB1's layer L to complete.

    Attention node:
      Layer 0 (warmup): compute+send all MBs, post irecv for F2A.
      Layers 1+: per-MB wait(F2A[mb]) → compute → isend, then post
                 next layer irecv.

    FFN node:
      Pre-post irecv for layer 0.  Per layer: wait(A2F[mb]) → compute
      → isend, then post next layer irecv BEFORE waiting current sends.

    Key overlaps:
    - Cross-layer: MB0 layer L+1 starts while MB1 layer L is still in flight.
    - Intra-layer: compute(MB_{i+1}) overlaps with isend(MB_i) transfer.
    - FFN: next-layer irecv posted before current sends complete.
    """

    def __init__(
        self,
        model,  # DisaggregatedQwenModel
        num_micro_batches: int = 2,
        enable_timing: bool = False,
        timing_mode: str = "cuda_events",
        comm_timing_mode: str = "enqueue",
        use_crosslayer: bool = False,
    ):
        self.model = model
        self.ctx = get_distributed_context()
        self.num_micro_batches = num_micro_batches
        self.enable_timing = enable_timing
        self.timing_mode = timing_mode
        self.comm_timing_mode = comm_timing_mode
        self.stats = DecodeDBOStats()
        self._timing_data: Optional[PipelineTiming] = None
        # Track timing on 0-based step 1: skip step 0 warmup/cold-start effects.
        self._timing_step = 1
        self._current_step = 0
        # Send timing is controlled by comm_timing_mode: enqueue overhead by
        # default, or effective Work completion latency for communication
        # overlap profiling.
        # Cross-layer pipelining switch:
        #   - False (default): layer-synchronous — drain current-layer sends
        #     before posting next-layer irecvs. Cleaner baseline.
        #   - True: post next-layer irecvs before draining current-layer sends,
        #     enabling cross-layer micro-batch pipeline.
        self.use_crosslayer = use_crosslayer
        # Eagerly init directional groups (collective: both nodes must reach here)
        _ = self.ctx.a2f_group
        logger.debug(
            f"DecodeDBOScheduler initialized: num_mb={num_micro_batches}, "
            f"use_crosslayer={use_crosslayer}, comm_timing={comm_timing_mode}"
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

    def _use_ep_overlap(self, num_mb: int) -> bool:
        return (
            self.ctx.ffn_ep_enabled
            and self.ctx.config is not None
            and self.ctx.config.ffn_ep_backend == "broadcast_reduce_overlap"
            and num_mb >= 2
        )

    def _record_ffn_stage_timing(
        self,
        tracker: TimingTracker,
        layer_idx: int,
        mb_idx: int,
        compute_start: float,
        stage_timing: Any,
    ) -> None:
        """Record fine-grained MoE/EP stages within the FFN compute span."""
        def _record_if_absolute(
            event_type: EventType,
            start_attr: str,
            end_attr: str,
        ) -> bool:
            start = getattr(stage_timing, start_attr, 0.0)
            end = getattr(stage_timing, end_attr, 0.0)
            if start > 0 and end > start:
                tracker.record_event(event_type, layer_idx, mb_idx, start, end)
                return True
            return False

        has_absolute_stages = any(
            getattr(stage_timing, attr, 0.0) > 0
            for attr in (
                "router_start_s",
                "experts_start_s",
                "shared_or_dense_start_s",
                "ep_dispatch_start_s",
                "ep_local_experts_start_s",
                "ep_reduce_start_s",
            )
        )
        if has_absolute_stages:
            _record_if_absolute(EventType.MOE_ROUTER, "router_start_s", "router_end_s")
            _record_if_absolute(EventType.MOE_EXPERTS, "experts_start_s", "experts_end_s")
            _record_if_absolute(
                EventType.MOE_SHARED_OR_DENSE,
                "shared_or_dense_start_s",
                "shared_or_dense_end_s",
            )
            _record_if_absolute(
                EventType.EP_DISPATCH,
                "ep_dispatch_start_s",
                "ep_dispatch_wait_end_s",
            )
            _record_if_absolute(
                EventType.EP_DISPATCH_WAIT,
                "ep_dispatch_wait_start_s",
                "ep_dispatch_wait_end_s",
            )
            _record_if_absolute(
                EventType.EP_LOCAL_EXPERTS,
                "ep_local_experts_start_s",
                "ep_local_experts_end_s",
            )
            reduce_enqueue_done = getattr(stage_timing, "ep_reduce_enqueue_done_s", 0.0)
            reduce_wait_start = getattr(stage_timing, "ep_reduce_wait_start_s", 0.0)
            if reduce_enqueue_done > 0 and reduce_wait_start > reduce_enqueue_done:
                tracker.record_event(
                    EventType.EP_OVERLAP_HIDDEN,
                    layer_idx,
                    mb_idx,
                    reduce_enqueue_done,
                    reduce_wait_start,
                )
            _record_if_absolute(
                EventType.EP_REDUCE,
                "ep_reduce_start_s",
                "ep_reduce_wait_end_s",
            )
            _record_if_absolute(
                EventType.EP_REDUCE_WAIT,
                "ep_reduce_wait_start_s",
                "ep_reduce_wait_end_s",
            )
            return

        cursor = compute_start
        router_s = getattr(stage_timing, "router_s", 0.0)
        if router_s > 0:
            tracker.record_event(
                EventType.MOE_ROUTER,
                layer_idx,
                mb_idx,
                cursor,
                cursor + router_s,
            )
            cursor += router_s

        ep_dispatch_s = getattr(stage_timing, "ep_dispatch_s", 0.0)
        if ep_dispatch_s > 0:
            tracker.record_event(
                EventType.EP_DISPATCH,
                layer_idx,
                mb_idx,
                cursor,
                cursor + ep_dispatch_s,
            )
            cursor += ep_dispatch_s

        ep_dispatch_wait_s = getattr(stage_timing, "ep_dispatch_wait_s", 0.0)
        if ep_dispatch_wait_s > 0:
            tracker.record_event(
                EventType.EP_DISPATCH_WAIT,
                layer_idx,
                mb_idx,
                max(compute_start, cursor - ep_dispatch_wait_s),
                cursor,
            )

        ep_local_experts_s = getattr(stage_timing, "ep_local_experts_s", 0.0)
        if ep_local_experts_s > 0:
            tracker.record_event(
                EventType.EP_LOCAL_EXPERTS,
                layer_idx,
                mb_idx,
                cursor,
                cursor + ep_local_experts_s,
            )
            cursor += ep_local_experts_s
        else:
            experts_s = getattr(stage_timing, "experts_s", 0.0)
            if experts_s > 0:
                tracker.record_event(
                    EventType.MOE_EXPERTS,
                    layer_idx,
                    mb_idx,
                    cursor,
                    cursor + experts_s,
                )
                cursor += experts_s

        ep_reduce_s = getattr(stage_timing, "ep_reduce_s", 0.0)
        if ep_reduce_s > 0:
            tracker.record_event(
                EventType.EP_REDUCE,
                layer_idx,
                mb_idx,
                cursor,
                cursor + ep_reduce_s,
            )
            cursor += ep_reduce_s

        ep_reduce_wait_s = getattr(stage_timing, "ep_reduce_wait_s", 0.0)
        if ep_reduce_wait_s > 0:
            tracker.record_event(
                EventType.EP_REDUCE_WAIT,
                layer_idx,
                mb_idx,
                max(compute_start, cursor - ep_reduce_wait_s),
                cursor,
            )

        ep_overlap_hidden_s = getattr(stage_timing, "ep_overlap_hidden_s", 0.0)
        if ep_overlap_hidden_s > 0:
            tracker.record_event(
                EventType.EP_OVERLAP_HIDDEN,
                layer_idx,
                mb_idx,
                max(compute_start, cursor - ep_reduce_wait_s - ep_overlap_hidden_s),
                max(compute_start, cursor - ep_reduce_wait_s),
            )

        shared_or_dense_s = getattr(stage_timing, "shared_or_dense_s", 0.0)
        if shared_or_dense_s > 0:
            tracker.record_event(
                EventType.MOE_SHARED_OR_DENSE,
                layer_idx,
                mb_idx,
                cursor,
                cursor + shared_or_dense_s,
            )

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
            # Synchronize both nodes so their TimingTracker baselines align
            dist.barrier()
            actual_num_mb = min(self.num_micro_batches, batch_size)
            tracker = TimingTracker(
                node=self.ctx.role,
                num_layers=self.model.num_layers,
                num_micro_batches=actual_num_mb,
                mode=self.timing_mode,
                comm_timing_mode=self.comm_timing_mode,
            )

        if self.ctx.is_attention_node:
            result = self._run_attention_decode(input_ids, position_ids, kv_cache, tracker)
        elif self.ctx.is_ffn_expert_only and self.ctx.ffn_ep_enabled:
            num_mb = min(self.num_micro_batches, batch_size)
            if self._use_ep_overlap(num_mb):
                self._run_ffn_ep_overlap_decode(batch_size, tracker, is_coordinator=False)
            else:
                self._run_ffn_ep_expert_decode(batch_size, tracker)
            result = None
        else:
            self._run_ffn_decode(batch_size, tracker)
            result = None

        if should_track and tracker is not None:
            self._timing_data = tracker.finish()
            self._timing_data.timed_decode_step = self._timing_step
            self._timing_data.timed_decode_step_base = "0-based"
            self._timing_data.timed_decode_step_note = (
                "step 0 skipped to avoid warmup/cold-start timing"
            )

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
        Attention node: cross-layer micro-batch pipeline with directional groups.

        Uses two separate NCCL process groups to break FIFO constraint:
          - a2f_group: ATT isend → FFN irecv (A2F direction)
          - f2a_group: FFN isend → ATT irecv (F2A direction)

        This allows posting F2A irecv per-MB immediately after each A2F isend,
        without blocking subsequent A2F sends (they're on different communicators).

        Layer 0 (warmup):
          compute+send each MB, post F2A irecv per-MB right after send.

        Layers 1+:
          Per MB: wait(F2A[L-1, mb]) → compute → isend(A2F[L, mb]) →
                  irecv(F2A[L, mb]). F2A irecv is posted ~(num_mb-1)*mb_time
                  earlier than the old "post all after layer" approach.
        """
        assert self.model.attention_worker is not None

        batch_size = input_ids.shape[0]
        num_layers = self.model.num_layers
        mb_sizes = self._compute_mb_sizes(batch_size)
        num_mb = len(mb_sizes)
        peer = self.ctx.peer_rank
        a2f_group = self.ctx.a2f_group
        f2a_group = self.ctx.f2a_group

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

        # ── Layer 0: warmup — compute and send all MBs ──
        # FIFO invariant: both a2f_group and f2a_group see operations in
        # layer-major/MB-major order: [L0MB0, L0MB1, L1MB0, L1MB1, ...].
        # ATT side enqueues: isend(a2f), irecv(f2a) per-MB in this order.
        # FFN side enqueues: irecv(a2f), isend(f2a) per-MB in this order.
        layer_idx = 0
        cache_layer = kv_cache.layers[layer_idx]
        orig_keys = cache_layer.keys
        orig_values = cache_layer.values
        mb_updated_keys = []
        mb_updated_values = []
        prev_send_handle = None
        # Per-MB F2A irecv handles (posted right after each send)
        f2a_recv_handles: List[Optional[dist.Work]] = [None] * num_mb
        f2a_recv_tensors: List[Optional[torch.Tensor]] = [None] * num_mb

        for mb_idx in range(num_mb):
            start = mb_offsets[mb_idx]
            end = start + mb_sizes[mb_idx]

            if tracker:
                tracker.mark_start(EventType.ATTN_COMPUTE, layer_idx, mb_idx)
            compute_start = time.perf_counter()

            cache_layer.keys = orig_keys[start:end]
            cache_layer.values = orig_values[start:end]

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

            mb_updated_keys.append(cache_layer.keys)
            mb_updated_values.append(cache_layer.values)

            packed = (attn_output + residual).contiguous()

            if tracker:
                tracker.mark_end(EventType.ATTN_COMPUTE, layer_idx, mb_idx)
            compute_end = time.perf_counter()
            compute_time = compute_end - compute_start
            self.stats.compute_time += compute_time
            self.stats.attn_compute_time += compute_time

            # Wait prev MB's send before issuing the next (overlap compute with send)
            if prev_send_handle is not None:
                prev_send_handle.wait()
                if tracker:
                    tracker.observe_send_completion(prev_send_handle)

            send_start = time.perf_counter()
            tag = self._get_tag(layer_idx, mb_idx, "a2f")
            prev_send_handle = dist.isend(packed, dst=peer, tag=tag, group=a2f_group)
            if tracker:
                tracker.record_send(prev_send_handle, layer_idx, mb_idx, send_start, packed)
            if self.use_crosslayer:
                # Post F2A irecv per-MB on f2a_group (doesn't block a2f_group sends)
                f2a_tag = self._get_tag(0, mb_idx, "f2a")
                recv_tensor = torch.empty(
                    mb_sizes[mb_idx], 1, self.model.hidden_size,
                    dtype=self.model.dtype, device=self.model.device,
                )
                f2a_recv_handles[mb_idx] = dist.irecv(
                    recv_tensor, src=peer, tag=f2a_tag, group=f2a_group,
                )
                f2a_recv_tensors[mb_idx] = recv_tensor

        if prev_send_handle is not None:
            prev_send_handle.wait()
            if tracker:
                tracker.observe_send_completion(prev_send_handle)

        if not self.use_crosslayer:
            # Post all F2A irecvs for layer 0 AFTER sends drain
            for mb_idx in range(num_mb):
                f2a_tag = self._get_tag(0, mb_idx, "f2a")
                recv_tensor = torch.empty(
                    mb_sizes[mb_idx], 1, self.model.hidden_size,
                    dtype=self.model.dtype, device=self.model.device,
                )
                f2a_recv_handles[mb_idx] = dist.irecv(
                    recv_tensor, src=peer, tag=f2a_tag, group=f2a_group,
                )
                f2a_recv_tensors[mb_idx] = recv_tensor

        cache_layer.keys = torch.cat(mb_updated_keys, dim=0)
        cache_layer.values = torch.cat(mb_updated_values, dim=0)

        # ── Layers 1+: cross-layer pipeline with per-MB irecv posting ──
        for layer_idx in range(1, num_layers):
            cache_layer = kv_cache.layers[layer_idx]
            orig_keys = cache_layer.keys
            orig_values = cache_layer.values
            mb_updated_keys = []
            mb_updated_values = []
            prev_send_handle = None
            next_f2a_handles: List[Optional[dist.Work]] = [None] * num_mb
            next_f2a_tensors: List[Optional[torch.Tensor]] = [None] * num_mb

            for mb_idx in range(num_mb):
                start = mb_offsets[mb_idx]
                end = start + mb_sizes[mb_idx]

                # Wait for THIS MB's F2A from previous layer
                if tracker:
                    tracker.mark_start(EventType.RECV_WAIT, layer_idx - 1, mb_idx)
                recv_start = time.perf_counter()
                f2a_recv_handles[mb_idx].wait()
                recv_end = time.perf_counter()
                if tracker:
                    tracker.mark_end(EventType.RECV_WAIT, layer_idx - 1, mb_idx)
                self.stats.f2a_comm_time += recv_end - recv_start

                if tracker:
                    tracker.mark_start(EventType.ATTN_COMPUTE, layer_idx, mb_idx)
                compute_start = time.perf_counter()

                cache_layer.keys = orig_keys[start:end]
                cache_layer.values = orig_values[start:end]

                # Hidden states come from F2A recv of previous layer
                mb_hidden = f2a_recv_tensors[mb_idx]
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

                mb_updated_keys.append(cache_layer.keys)
                mb_updated_values.append(cache_layer.values)

                packed = (attn_output + residual).contiguous()

                if tracker:
                    tracker.mark_end(EventType.ATTN_COMPUTE, layer_idx, mb_idx)
                compute_end = time.perf_counter()
                compute_time = compute_end - compute_start
                self.stats.compute_time += compute_time
                self.stats.attn_compute_time += compute_time

                if prev_send_handle is not None:
                    prev_send_handle.wait()
                    if tracker:
                        tracker.observe_send_completion(prev_send_handle)

                send_start = time.perf_counter()
                tag = self._get_tag(layer_idx, mb_idx, "a2f")
                prev_send_handle = dist.isend(packed, dst=peer, tag=tag, group=a2f_group)
                if tracker:
                    tracker.record_send(prev_send_handle, layer_idx, mb_idx, send_start, packed)
                if self.use_crosslayer:
                    # Post F2A irecv for THIS layer per-MB on f2a_group
                    f2a_tag = self._get_tag(layer_idx, mb_idx, "f2a")
                    recv_tensor = torch.empty(
                        mb_sizes[mb_idx], 1, self.model.hidden_size,
                        dtype=self.model.dtype, device=self.model.device,
                    )
                    next_f2a_handles[mb_idx] = dist.irecv(
                        recv_tensor, src=peer, tag=f2a_tag, group=f2a_group,
                    )
                    next_f2a_tensors[mb_idx] = recv_tensor

            if prev_send_handle is not None:
                prev_send_handle.wait()
                if tracker:
                    tracker.observe_send_completion(prev_send_handle)

            if not self.use_crosslayer:
                # Post all F2A irecvs for this layer AFTER sends drain
                for mb_idx in range(num_mb):
                    f2a_tag = self._get_tag(layer_idx, mb_idx, "f2a")
                    recv_tensor = torch.empty(
                        mb_sizes[mb_idx], 1, self.model.hidden_size,
                        dtype=self.model.dtype, device=self.model.device,
                    )
                    next_f2a_handles[mb_idx] = dist.irecv(
                        recv_tensor, src=peer, tag=f2a_tag, group=f2a_group,
                    )
                    next_f2a_tensors[mb_idx] = recv_tensor

            cache_layer.keys = torch.cat(mb_updated_keys, dim=0)
            cache_layer.values = torch.cat(mb_updated_values, dim=0)

            # Swap handles for next layer iteration
            f2a_recv_handles = next_f2a_handles
            f2a_recv_tensors = next_f2a_tensors

        # ── Wait final layer F2A ──
        for mb_idx in range(num_mb):
            if tracker:
                tracker.mark_start(EventType.RECV_WAIT, num_layers - 1, mb_idx)
            recv_start = time.perf_counter()
            f2a_recv_handles[mb_idx].wait()
            recv_end = time.perf_counter()
            if tracker:
                tracker.mark_end(EventType.RECV_WAIT, num_layers - 1, mb_idx)
            self.stats.f2a_comm_time += recv_end - recv_start

        hidden_states = torch.cat(f2a_recv_tensors, dim=0)

        self.stats.comm_time = self.stats.a2f_comm_time + self.stats.f2a_comm_time
        return self.model.attention_worker.forward_lm_head(hidden_states)

    def _run_ffn_decode(self, batch_size: int, tracker: Optional[TimingTracker] = None) -> None:
        """
        FFN node: cross-layer micro-batch pipeline with directional groups.

        Uses a2f_group for irecv (receiving from ATT) and f2a_group for isend
        (sending back to ATT). Posts next-layer A2F irecv per-MB right after
        each MB's compute+isend, allowing the ATT node to start sending
        next-layer data before the current layer finishes.
        """
        assert self.model.ffn_worker is not None

        num_layers = self.model.num_layers
        mb_sizes = self._compute_mb_sizes(batch_size)
        num_mb = len(mb_sizes)
        if self._use_ep_overlap(num_mb):
            self._run_ffn_ep_overlap_decode(batch_size, tracker, is_coordinator=True)
            return
        peer = self.ctx.peer_rank
        a2f_group = self.ctx.a2f_group
        f2a_group = self.ctx.f2a_group

        # Pre-post irecv for layer 0 on a2f_group
        a2f_recv_handles: List[Optional[dist.Work]] = [None] * num_mb
        a2f_recv_tensors: List[Optional[torch.Tensor]] = [None] * num_mb
        for mb_idx, mb_size in enumerate(mb_sizes):
            tag = self._get_tag(0, mb_idx, "a2f")
            recv_tensor = torch.empty(
                mb_size, 1, self.model.hidden_size,
                dtype=self.model.dtype, device=self.model.device,
            )
            a2f_recv_handles[mb_idx] = dist.irecv(
                recv_tensor, src=peer, tag=tag, group=a2f_group,
            )
            a2f_recv_tensors[mb_idx] = recv_tensor

        for layer_idx in range(num_layers):
            # Process each MB: wait recv → compute → isend → post next-layer irecv
            send_handles = []
            next_a2f_handles: List[Optional[dist.Work]] = [None] * num_mb
            next_a2f_tensors: List[Optional[torch.Tensor]] = [None] * num_mb

            for mb_idx in range(num_mb):
                if tracker:
                    tracker.mark_start(EventType.RECV_WAIT, layer_idx, mb_idx)
                recv_start = time.perf_counter()
                a2f_recv_handles[mb_idx].wait()
                recv_end = time.perf_counter()
                if tracker:
                    tracker.mark_end(EventType.RECV_WAIT, layer_idx, mb_idx)
                self.stats.a2f_comm_time += recv_end - recv_start

                if tracker:
                    tracker.mark_start(EventType.FFN_COMPUTE, layer_idx, mb_idx)
                compute_start = time.perf_counter()
                ffn_result = self.model.ffn_worker.forward_ffn_layer(
                    layer_idx=layer_idx,
                    hidden_states=a2f_recv_tensors[mb_idx],
                    return_timing=bool(tracker and self.model.supports_moe_timing),
                )
                if isinstance(ffn_result, tuple):
                    output, stage_timing = ffn_result
                else:
                    output, stage_timing = ffn_result, None
                output = output.contiguous()
                if tracker:
                    tracker.mark_end(EventType.FFN_COMPUTE, layer_idx, mb_idx)
                    if stage_timing is not None:
                        self._record_ffn_stage_timing(
                            tracker,
                            layer_idx,
                            mb_idx,
                            compute_start,
                            stage_timing,
                        )
                compute_end = time.perf_counter()
                compute_time = compute_end - compute_start
                self.stats.compute_time += compute_time
                self.stats.ffn_compute_time += compute_time

                tag = self._get_tag(layer_idx, mb_idx, "f2a")
                send_start = time.perf_counter()
                handle = dist.isend(output, dst=peer, tag=tag, group=f2a_group)
                if tracker:
                    tracker.record_send(handle, layer_idx, mb_idx, send_start, output)
                send_handles.append(handle)

            # Post next-layer A2F irecvs.
            #   use_crosslayer=True: post BEFORE draining sends → next-layer
            #     irecvs can be matched while this-layer F2A sends complete,
            #     enabling cross-layer micro-batch pipeline.
            #   use_crosslayer=False: post AFTER draining sends → layer-synchronous
            #     baseline. Incurs ~5ms/layer irecv match latency but gives a
            #     clean non-pipelined reference.
            def _post_next_layer_irecvs():
                if layer_idx + 1 < num_layers:
                    for mb_idx in range(num_mb):
                        next_tag = self._get_tag(layer_idx + 1, mb_idx, "a2f")
                        recv_tensor = torch.empty(
                            mb_sizes[mb_idx], 1, self.model.hidden_size,
                            dtype=self.model.dtype, device=self.model.device,
                        )
                        next_a2f_handles[mb_idx] = dist.irecv(
                            recv_tensor, src=peer, tag=next_tag, group=a2f_group,
                        )
                        next_a2f_tensors[mb_idx] = recv_tensor

            if self.use_crosslayer:
                _post_next_layer_irecvs()

            # Wait all sends complete
            for handle in send_handles:
                handle.wait()
                if tracker:
                    tracker.observe_send_completion(handle)

            if not self.use_crosslayer:
                _post_next_layer_irecvs()

            if layer_idx + 1 < num_layers:
                a2f_recv_handles = next_a2f_handles
                a2f_recv_tensors = next_a2f_tensors

        self.stats.comm_time = self.stats.a2f_comm_time + self.stats.f2a_comm_time

    def _record_ep_item_timing(
        self,
        tracker: Optional[TimingTracker],
        layer_idx: int,
        mb_idx: int,
        item,
    ) -> None:
        stage_timing = item.timing
        compute_time = stage_timing.router_s + stage_timing.ep_local_experts_s
        self.stats.compute_time += compute_time
        self.stats.ffn_compute_time += compute_time
        if tracker is None:
            return

        event_start = getattr(item, "ffn_event_start_s", time.perf_counter())
        tracker.record_event(
            EventType.FFN_COMPUTE,
            layer_idx,
            mb_idx,
            event_start,
            event_start + max(compute_time, 0.0),
        )
        self._record_ffn_stage_timing(
            tracker,
            layer_idx,
            mb_idx,
            event_start,
            stage_timing,
        )

    def _run_ffn_ep_overlap_decode(
        self,
        batch_size: int,
        tracker: Optional[TimingTracker] = None,
        *,
        is_coordinator: bool,
    ) -> None:
        """FFN EP decode path with in-layer MB overlap for dispatch/reduce."""
        assert self.model.ffn_worker is not None

        num_layers = self.model.num_layers
        mb_sizes = self._compute_mb_sizes(batch_size)
        num_mb = len(mb_sizes)

        if is_coordinator:
            peer = self.ctx.peer_rank
            a2f_group = self.ctx.a2f_group
            f2a_group = self.ctx.f2a_group
            a2f_recv_handles: List[Optional[dist.Work]] = [None] * num_mb
            a2f_recv_tensors: List[Optional[torch.Tensor]] = [None] * num_mb
            for mb_idx, mb_size in enumerate(mb_sizes):
                tag = self._get_tag(0, mb_idx, "a2f")
                recv_tensor = torch.empty(
                    mb_size, 1, self.model.hidden_size,
                    dtype=self.model.dtype, device=self.model.device,
                )
                a2f_recv_handles[mb_idx] = dist.irecv(
                    recv_tensor, src=peer, tag=tag, group=a2f_group,
                )
                a2f_recv_tensors[mb_idx] = recv_tensor
        else:
            peer = None
            a2f_group = None
            f2a_group = None
            a2f_recv_handles = []
            a2f_recv_tensors = []

        for layer_idx in range(num_layers):
            layer = self.model.ffn_worker.ffn_layers[layer_idx]
            items = []
            send_handles: List[dist.Work] = []
            next_a2f_handles: List[Optional[dist.Work]] = [None] * num_mb
            next_a2f_tensors: List[Optional[torch.Tensor]] = [None] * num_mb

            for mb_idx, mb_size in enumerate(mb_sizes):
                if is_coordinator:
                    if tracker:
                        tracker.mark_start(EventType.RECV_WAIT, layer_idx, mb_idx)
                    recv_start = time.perf_counter()
                    assert a2f_recv_handles[mb_idx] is not None
                    a2f_recv_handles[mb_idx].wait()
                    recv_end = time.perf_counter()
                    if tracker:
                        tracker.mark_end(EventType.RECV_WAIT, layer_idx, mb_idx)
                    self.stats.a2f_comm_time += recv_end - recv_start
                    hidden_states = a2f_recv_tensors[mb_idx]
                else:
                    hidden_states = torch.empty(
                        mb_size, 1, self.model.hidden_size,
                        dtype=self.model.dtype, device=self.model.device,
                    )

                item_start = time.perf_counter()
                item = layer.create_work_item(
                    hidden_states=hidden_states,
                    output_device=self.model.device,
                )
                setattr(item, "ffn_event_start_s", item_start)
                layer.dispatch_async(item)
                items.append(item)

            previous: Optional[Tuple[int, Any]] = None

            def _finish_and_send(done_mb_idx: int, done_item) -> None:
                layer.finish_reduce(done_item)
                output = layer.finish_output(done_item)
                self._record_ep_item_timing(tracker, layer_idx, done_mb_idx, done_item)
                if is_coordinator:
                    output = output.contiguous()
                    tag = self._get_tag(layer_idx, done_mb_idx, "f2a")
                    send_start = time.perf_counter()
                    handle = dist.isend(output, dst=peer, tag=tag, group=f2a_group)
                    if tracker:
                        tracker.record_send(handle, layer_idx, done_mb_idx, send_start, output)
                    send_handles.append(handle)

            for mb_idx, item in enumerate(items):
                layer.finish_dispatch(item)
                layer.compute_local(item)
                layer.reduce_async(item)
                if previous is not None:
                    prev_mb_idx, prev_item = previous
                    _finish_and_send(prev_mb_idx, prev_item)
                previous = (mb_idx, item)

            if previous is not None:
                prev_mb_idx, prev_item = previous
                _finish_and_send(prev_mb_idx, prev_item)

            if is_coordinator:
                def _post_next_layer_irecvs():
                    if layer_idx + 1 < num_layers:
                        for mb_idx, mb_size in enumerate(mb_sizes):
                            next_tag = self._get_tag(layer_idx + 1, mb_idx, "a2f")
                            recv_tensor = torch.empty(
                                mb_size, 1, self.model.hidden_size,
                                dtype=self.model.dtype, device=self.model.device,
                            )
                            next_a2f_handles[mb_idx] = dist.irecv(
                                recv_tensor, src=peer, tag=next_tag, group=a2f_group,
                            )
                            next_a2f_tensors[mb_idx] = recv_tensor

                if self.use_crosslayer:
                    _post_next_layer_irecvs()

                for handle in send_handles:
                    handle.wait()
                    if tracker:
                        tracker.observe_send_completion(handle)

                if not self.use_crosslayer:
                    _post_next_layer_irecvs()

                if layer_idx + 1 < num_layers:
                    a2f_recv_handles = next_a2f_handles
                    a2f_recv_tensors = next_a2f_tensors

        self.stats.comm_time = self.stats.a2f_comm_time + self.stats.f2a_comm_time

    def _run_ffn_ep_expert_decode(self, batch_size: int, tracker: Optional[TimingTracker] = None) -> None:
        """FFN EP non-coordinator decode loop: only participate in EP collectives."""
        assert self.model.ffn_worker is not None

        num_layers = self.model.num_layers
        mb_sizes = self._compute_mb_sizes(batch_size)
        num_mb = len(mb_sizes)

        for layer_idx in range(num_layers):
            for mb_idx in range(num_mb):
                hidden_states = torch.empty(
                    mb_sizes[mb_idx],
                    1,
                    self.model.hidden_size,
                    dtype=self.model.dtype,
                    device=self.model.device,
                )
                if tracker:
                    tracker.mark_start(EventType.FFN_COMPUTE, layer_idx, mb_idx)
                compute_start = time.perf_counter()
                ffn_result = self.model.ffn_worker.forward_ffn_layer(
                    layer_idx=layer_idx,
                    hidden_states=hidden_states,
                    return_timing=bool(tracker and self.model.supports_moe_timing),
                )
                if isinstance(ffn_result, tuple):
                    _, stage_timing = ffn_result
                else:
                    stage_timing = None
                if tracker:
                    tracker.mark_end(EventType.FFN_COMPUTE, layer_idx, mb_idx)
                    if stage_timing is not None:
                        self._record_ffn_stage_timing(
                            tracker,
                            layer_idx,
                            mb_idx,
                            compute_start,
                            stage_timing,
                        )
                compute_time = time.perf_counter() - compute_start
                self.stats.compute_time += compute_time
                self.stats.ffn_compute_time += compute_time

    def get_stats(self) -> DecodeDBOStats:
        """Get statistics from last run."""
        return self.stats

    def get_timing_data(self) -> Optional[PipelineTiming]:
        """Get per-layer timing data tracked on 0-based decode step 1."""
        return self._timing_data
