"""
Decode DBO Scheduler - Pipeline for decode phase.

For decode (seq=1 per token), micro-batching provides negligible overlap
because the compute-to-communication ratio is very low. The cuBLAS kernel
overhead of multiple small FFN calls actually dominates any overlap saving.

This scheduler uses full-batch communication (single send/recv per layer)
to avoid kernel launch overhead while maintaining the scheduler interface
for future cross-layer pipelining.
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
    Decode phase scheduler.

    Uses full-batch attention (with global KV cache) and full-batch FFN
    communication. Micro-batching is avoided because for decode (seq=1):
    - cuBLAS kernel overhead for 2×FFN(b/2) ≈ 2× FFN(b) (kernel selection issue)
    - NCCL latency per extra op ≈ 150-300μs, exceeding any overlap benefit
    - The correct KV cache is maintained by computing attention for the full batch
    """

    def __init__(
        self,
        model,  # DisaggregatedQwenModel
        num_micro_batches: int = 2,
        enable_timing: bool = False,
    ):
        self.model = model
        self.ctx = get_distributed_context()
        self.enable_timing = enable_timing
        self.stats = DecodeDBOStats()
        logger.debug("DecodeDBOScheduler initialized (full-batch mode)")

    def _get_tag(self, layer_idx: int) -> int:
        """Get unique tags for A2F and F2A communication."""
        return 10000 + layer_idx * 10

    @torch.no_grad()
    def forward_decode_dbo(
        self,
        input_ids: torch.Tensor,   # [batch_size, 1]
        position_ids: torch.Tensor, # [batch_size, 1]
        kv_cache,                   # DynamicCache (attention node) or None (FFN node)
    ) -> Optional[torch.Tensor]:
        """
        Decode forward pass.

        Returns:
            Logits [batch_size, 1, vocab_size] on attention node, None on FFN node.
        """
        start_time = time.perf_counter()
        batch_size = input_ids.shape[0]

        self.stats = DecodeDBOStats(num_tokens=batch_size)

        if self.ctx.is_attention_node:
            result = self._run_attention_decode(input_ids, position_ids, kv_cache)
        else:
            self._run_ffn_decode(batch_size)
            result = None

        self.stats.total_time = time.perf_counter() - start_time
        self.stats.num_layers = self.model.num_layers
        return result

    def _run_attention_decode(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        kv_cache,
    ) -> torch.Tensor:
        """
        Attention node: full-batch attention + full-batch communication.
        """
        assert self.model.attention_worker is not None

        batch_size = input_ids.shape[0]
        num_layers = self.model.num_layers

        hidden_states = self.model.attention_worker.embed(input_ids)
        position_embeddings = self.model.attention_worker.get_position_embeddings(
            hidden_states, position_ids
        )
        cur_pos = kv_cache.get_seq_length()
        total_len = cur_pos + 1
        attention_mask = self.model._make_causal_mask(batch_size, 1, total_len)

        for layer_idx in range(num_layers):
            # Full-batch attention (KV cache updated in-place)
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
            compute_end = time.perf_counter()
            compute_time = compute_end - compute_start
            self.stats.compute_time += compute_time
            self.stats.attn_compute_time += compute_time

            # Send full batch to FFN
            tag_a2f = self._get_tag(layer_idx)
            send_start = time.perf_counter()
            dist.send(packed, dst=self.ctx.peer_rank, tag=tag_a2f)
            send_end = time.perf_counter()
            self.stats.a2f_comm_time += send_end - send_start

            # Receive full batch from FFN
            tag_f2a = self._get_tag(layer_idx) + 1
            recv_start = time.perf_counter()
            hidden_states = torch.empty(
                batch_size, 1, self.model.hidden_size,
                dtype=self.model.dtype, device=self.model.device,
            )
            dist.recv(hidden_states, src=self.ctx.peer_rank, tag=tag_f2a)
            recv_end = time.perf_counter()
            self.stats.f2a_comm_time += recv_end - recv_start
            self.stats.comm_time += (send_end - send_start) + (recv_end - recv_start)

        return self.model.attention_worker.forward_lm_head(hidden_states)

    def _run_ffn_decode(self, batch_size: int) -> None:
        """
        FFN node: receive full batch → compute FFN → send back.
        """
        assert self.model.ffn_worker is not None

        num_layers = self.model.num_layers

        for layer_idx in range(num_layers):
            # Receive full batch from attention
            tag_a2f = self._get_tag(layer_idx)
            recv_start = time.perf_counter()
            hidden_states = torch.empty(
                batch_size, 1, self.model.hidden_size,
                dtype=self.model.dtype, device=self.model.device,
            )
            dist.recv(hidden_states, src=self.ctx.peer_rank, tag=tag_a2f)
            recv_end = time.perf_counter()
            self.stats.a2f_comm_time += recv_end - recv_start

            # Full-batch FFN (single kernel launch)
            compute_start = time.perf_counter()
            output = self.model.ffn_worker.forward_ffn_layer(
                layer_idx=layer_idx,
                hidden_states=hidden_states,
            )
            if isinstance(output, tuple):
                output = output[0]
            output = output.contiguous()
            compute_end = time.perf_counter()
            compute_time = compute_end - compute_start
            self.stats.compute_time += compute_time
            self.stats.ffn_compute_time += compute_time

            # Send full batch back to attention
            tag_f2a = self._get_tag(layer_idx) + 1
            send_start = time.perf_counter()
            dist.send(output, dst=self.ctx.peer_rank, tag=tag_f2a)
            send_end = time.perf_counter()
            self.stats.f2a_comm_time += send_end - send_start
            self.stats.comm_time += (recv_end - recv_start) + (send_end - send_start)

    def get_stats(self) -> DecodeDBOStats:
        """Get statistics from last run."""
        return self.stats
