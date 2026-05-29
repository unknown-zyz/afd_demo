"""Expert-parallel MoE helpers for NPU FFN prototypes."""

from __future__ import annotations

import time
import os
from dataclasses import dataclass, field
from typing import Optional, Sequence

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ExpertShardPlan:
    """Static owner mapping for routed experts."""

    num_experts: int
    ep_size: int
    ep_rank: int
    policy: str = "round_robin"
    expert_to_rank: Optional[Sequence[int]] = None

    def __post_init__(self) -> None:
        if self.num_experts <= 0:
            raise ValueError("num_experts must be positive")
        if self.ep_size <= 0:
            raise ValueError("ep_size must be positive")
        if not 0 <= self.ep_rank < self.ep_size:
            raise ValueError(f"ep_rank must be in [0, {self.ep_size}), got {self.ep_rank}")
        if self.policy not in {"round_robin", "contiguous", "explicit"}:
            raise ValueError(f"Unsupported expert shard policy: {self.policy}")
        if self.policy == "explicit":
            if self.expert_to_rank is None:
                raise ValueError("expert_to_rank is required for explicit expert shard policy")
            mapping = tuple(int(rank) for rank in self.expert_to_rank)
            if len(mapping) != self.num_experts:
                raise ValueError(
                    f"expert_to_rank len {len(mapping)} != num_experts {self.num_experts}"
                )
            invalid = [rank for rank in mapping if rank < 0 or rank >= self.ep_size]
            if invalid:
                raise ValueError(
                    f"expert_to_rank values must be in [0, {self.ep_size}); got {invalid[:5]}"
                )
            object.__setattr__(self, "expert_to_rank", mapping)
        elif self.expert_to_rank is not None:
            raise ValueError("expert_to_rank is only valid with explicit expert shard policy")

    @property
    def local_expert_ids(self) -> list[int]:
        if self.policy == "explicit":
            assert self.expert_to_rank is not None
            return [
                idx for idx, owner_rank in enumerate(self.expert_to_rank)
                if owner_rank == self.ep_rank
            ]
        if self.policy == "round_robin":
            return [idx for idx in range(self.num_experts) if idx % self.ep_size == self.ep_rank]
        base = self.num_experts // self.ep_size
        remainder = self.num_experts % self.ep_size
        start = self.ep_rank * base + min(self.ep_rank, remainder)
        count = base + (1 if self.ep_rank < remainder else 0)
        return list(range(start, start + count))

    @staticmethod
    def all_assignments(
        num_experts: int,
        ep_size: int,
        policy: str = "round_robin",
        expert_to_rank: Optional[Sequence[int]] = None,
    ) -> list[list[int]]:
        return [
            ExpertShardPlan(num_experts, ep_size, ep_rank, policy, expert_to_rank).local_expert_ids
            for ep_rank in range(ep_size)
        ]

    @property
    def expert_to_rank_map(self) -> list[int]:
        """Return global expert owner ranks for this shard policy."""
        if self.policy == "explicit":
            assert self.expert_to_rank is not None
            return list(self.expert_to_rank)
        mapping = [0] * self.num_experts
        for rank, expert_ids in enumerate(
            ExpertShardPlan.all_assignments(
                self.num_experts,
                self.ep_size,
                self.policy,
                self.expert_to_rank,
            )
        ):
            for expert_id in expert_ids:
                mapping[expert_id] = rank
        return mapping


@dataclass
class EPStageTiming:
    """Fine-grained EP timing in seconds."""

    router_s: float = 0.0
    experts_s: float = 0.0
    shared_or_dense_s: float = 0.0
    ep_dispatch_s: float = 0.0
    ep_local_experts_s: float = 0.0
    ep_reduce_s: float = 0.0
    ep_dispatch_enqueue_s: float = 0.0
    ep_reduce_enqueue_s: float = 0.0
    ep_dispatch_wait_s: float = 0.0
    ep_reduce_wait_s: float = 0.0
    ep_overlap_hidden_s: float = 0.0
    ep_dispatch_bytes: int = 0
    ep_reduce_bytes: int = 0
    ep_active_experts: int = 0
    ep_local_assignments: int = 0
    router_start_s: float = 0.0
    router_end_s: float = 0.0
    ep_dispatch_start_s: float = 0.0
    ep_dispatch_enqueue_done_s: float = 0.0
    ep_dispatch_wait_start_s: float = 0.0
    ep_dispatch_wait_end_s: float = 0.0
    ep_local_experts_start_s: float = 0.0
    ep_local_experts_end_s: float = 0.0
    ep_reduce_start_s: float = 0.0
    ep_reduce_enqueue_done_s: float = 0.0
    ep_reduce_wait_start_s: float = 0.0
    ep_reduce_wait_end_s: float = 0.0
    ep_finish_output_start_s: float = 0.0
    ep_finish_output_end_s: float = 0.0


@dataclass
class EPWorkItem:
    """State for one EP micro-batch while overlap collectives are in flight."""

    hidden_states: torch.Tensor
    output_device: torch.device
    batch_size: int
    seq_len: int
    hidden_dim: int
    hidden_2d: torch.Tensor
    selected_experts: torch.Tensor
    routing_weights: torch.Tensor
    timing: EPStageTiming = field(default_factory=EPStageTiming)
    residual_out: Optional[torch.Tensor] = None
    partial: Optional[torch.Tensor] = None
    dispatch_handles: list[dist.Work] = field(default_factory=list)
    reduce_handle: Optional[dist.Work] = None
    dispatch_start_s: float = 0.0
    dispatch_enqueue_done_s: float = 0.0
    reduce_start_s: float = 0.0
    reduce_enqueue_done_s: float = 0.0
    fused_buf: Optional[torch.Tensor] = None
    fused_layout: Optional[tuple] = None
    all2all_dispatch_handle: Optional[dict] = None
    all2all_combine_handle: Optional[dict] = None
    all2all_ffn_outputs: Optional[torch.Tensor] = None


class ShardedExperts(nn.Module):
    """Compute the weighted MoE contribution for a rank-local expert shard."""

    def __init__(
        self,
        experts,
        plan: ExpertShardPlan,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.plan = plan
        self.device = device
        self.dtype = dtype
        self.local_expert_ids = plan.local_expert_ids
        self.local_expert_id_set = set(self.local_expert_ids)
        self.act_fn = experts.act_fn
        self.num_total_experts = plan.num_experts

        if not self.local_expert_ids:
            self.gate_up_stack = None
            self.down_stack = None
        else:
            gate_up_list = [
                experts.gate_up_proj[eid].detach().to(device=device, dtype=dtype).contiguous()
                for eid in self.local_expert_ids
            ]
            down_list = [
                experts.down_proj[eid].detach().to(device=device, dtype=dtype).contiguous()
                for eid in self.local_expert_ids
            ]
            self.gate_up_stack = nn.Parameter(
                torch.stack(gate_up_list, dim=0).contiguous(), requires_grad=False
            )
            self.down_stack = nn.Parameter(
                torch.stack(down_list, dim=0).contiguous(), requires_grad=False
            )

        local_mask = torch.zeros(self.num_total_experts, dtype=torch.bool, device=device)
        local_index = torch.full((self.num_total_experts,), -1, dtype=torch.long, device=device)
        for eid in self.local_expert_ids:
            local_mask[eid] = True
            local_index[eid] = self.local_expert_ids.index(eid)
        self.register_buffer("_local_mask_lut", local_mask, persistent=False)
        self.register_buffer("_local_index_lut", local_index, persistent=False)
        self._local_idx_of_py = {int(eid): i for i, eid in enumerate(self.local_expert_ids)}

    def forward_local(
        self,
        hidden_2d: torch.Tensor,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, int, int]:
        """Return this shard's dense partial output plus activity counters.

        Active-only grouped path: build a flat (assignment, hidden) tensor for
        all tokens routed to this shard, sort by expert id, and run one GEMM
        pair per active local expert using stacked weights. Avoids per-step
        host syncs from `.cpu().tolist()` and replaces N small `F.linear` calls
        per expert with a single contiguous slice per active expert.
        """

        partial = torch.zeros_like(hidden_2d)
        if not self.local_expert_ids or self.gate_up_stack is None:
            return partial, 0, 0

        top_k = selected_experts.shape[-1]
        flat_experts = selected_experts.reshape(-1)
        flat_weights = routing_weights.reshape(-1)

        # On-device mask for "is this assignment routed to a local expert?"
        local_hits = self._local_mask_lut[flat_experts]
        active_pos = local_hits.nonzero(as_tuple=False).squeeze(-1)
        if active_pos.numel() == 0:
            return partial, 0, 0

        expert_per_assign = flat_experts[active_pos]
        token_per_assign = active_pos // top_k
        weight_per_assign = flat_weights[active_pos]

        # Sort assignments by expert id so each expert owns a contiguous slice.
        sort_idx = expert_per_assign.argsort()
        ex_sorted = expert_per_assign[sort_idx]
        tok_sorted = token_per_assign[sort_idx]
        w_sorted = weight_per_assign[sort_idx]

        x = hidden_2d.index_select(0, tok_sorted)

        uniq, counts = torch.unique_consecutive(ex_sorted, return_counts=True)
        # Single host sync per layer-MB for driving the small loop (<= num local experts).
        uniq_list = uniq.tolist()
        counts_list = counts.tolist()

        out = torch.empty_like(x)
        offset = 0
        for eid, cnt in zip(uniq_list, counts_list):
            local_idx = self._local_idx_of_py[int(eid)]
            seg = x.narrow(0, offset, cnt)
            gu = F.linear(seg, self.gate_up_stack[local_idx])
            gate, up = gu.chunk(2, dim=-1)
            hidden = self.act_fn(gate) * up
            out_seg = F.linear(hidden, self.down_stack[local_idx])
            out.narrow(0, offset, cnt).copy_(out_seg)
            offset += cnt

        weighted = out * w_sorted.unsqueeze(-1).to(out.dtype)
        partial.index_add_(0, tok_sorted, weighted)
        return partial, len(uniq_list), int(active_pos.numel())

    def forward_dispatched(
        self,
        recv_hidden: torch.Tensor,
        recv_experts: torch.Tensor,
    ) -> tuple[torch.Tensor, int, int]:
        """Run local experts for assignment-level all-to-all dispatched tokens.

        `recv_hidden[i]` is one token copy routed to global expert
        `recv_experts[i]`. Outputs are intentionally unweighted; routing weights
        are applied once during the all-to-all combine phase.
        """

        if recv_hidden.numel() == 0:
            return torch.empty_like(recv_hidden), 0, 0
        if self.gate_up_stack is None or not self.local_expert_ids:
            raise RuntimeError(
                f"Rank {self.plan.ep_rank} received {recv_hidden.shape[0]} assignments "
                "but owns no experts"
            )

        recv_experts = recv_experts.to(torch.long).contiguous()
        local_indices = self._local_index_lut[recv_experts]
        invalid = local_indices < 0
        if bool(invalid.any().item()):
            bad = recv_experts[invalid][:8].detach().cpu().tolist()
            raise RuntimeError(
                f"Rank {self.plan.ep_rank} received non-local expert IDs {bad}; "
                f"local experts={self.local_expert_ids}"
            )

        sort_idx = recv_experts.argsort()
        ex_sorted = recv_experts[sort_idx]
        local_sorted = local_indices[sort_idx]
        x_sorted = recv_hidden.index_select(0, sort_idx)

        uniq, counts = torch.unique_consecutive(ex_sorted, return_counts=True)
        uniq_list = uniq.tolist()
        counts_list = counts.tolist()

        out_sorted = torch.empty_like(x_sorted)
        offset = 0
        for _eid, cnt in zip(uniq_list, counts_list):
            local_idx = int(local_sorted[offset].item())
            seg = x_sorted.narrow(0, offset, cnt)
            gu = F.linear(seg, self.gate_up_stack[local_idx])
            gate, up = gu.chunk(2, dim=-1)
            hidden = self.act_fn(gate) * up
            out_seg = F.linear(hidden, self.down_stack[local_idx])
            out_sorted.narrow(0, offset, cnt).copy_(out_seg)
            offset += cnt

        inverse = torch.empty_like(sort_idx)
        inverse.scatter_(0, sort_idx, torch.arange(sort_idx.numel(), device=sort_idx.device))
        out = out_sorted.index_select(0, inverse)
        return out, len(uniq_list), int(recv_hidden.shape[0])


def sync_if_needed(device: torch.device) -> None:
    if os.environ.get("AFD_EP_SYNC_TIMING", "0") != "1":
        return
    if device.type == "npu" and hasattr(torch, "npu"):
        torch.npu.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


class EPFFNLayer(nn.Module):
    """One FFN layer backed by FFN-rank expert parallelism."""

    def __init__(
        self,
        *,
        post_attention_layernorm: Optional[nn.Module],
        gate: Optional[nn.Module],
        sharded_experts: ShardedExperts,
        hidden_size: int,
        top_k: int,
        layer_idx: int,
        layer_device: torch.device,
        ctx,
    ) -> None:
        super().__init__()
        self.post_attention_layernorm = post_attention_layernorm
        self.gate = gate
        self.sharded_experts = sharded_experts
        self.hidden_size = hidden_size
        self.top_k = top_k
        self.layer_idx = layer_idx
        self.layer_device = layer_device
        self.ctx = ctx
        self.is_sparse_moe = True
        self._all2all_comm = None
        if self.ctx.config is not None and self.ctx.config.ffn_ep_backend == "all_to_all_single":
            from ..coordinator_arch.comm.fallback_a2a import FallbackMoECommunicator

            self._all2all_comm = FallbackMoECommunicator(
                ep_group=self.ctx.ffn_ep_group,
                hidden_size=hidden_size,
                num_experts=sharded_experts.plan.num_experts,
                max_tokens_per_rank=int(os.environ.get("AFD_EP_MAX_TOKENS_PER_RANK", "1048576")),
                device=layer_device,
                dispatch_weights=False,
                dispatch_experts=False,
                metadata_src_rank=self.ctx.ffn_coordinator_rank,
            )
            self._all2all_comm.update_routing_table(
                {
                    "version": 1,
                    "expert_to_rank": sharded_experts.plan.expert_to_rank_map,
                    "mode": "normal",
                }
            )

    @property
    def is_coordinator(self) -> bool:
        return self.ctx.is_ffn_coordinator

    @property
    def use_all_to_all(self) -> bool:
        return self._all2all_comm is not None

    def _empty_all2all_inputs(
        self,
        hidden_dim: int,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.empty(0, hidden_dim, device=self.layer_device, dtype=dtype)
        experts = torch.empty(0, self.top_k, device=self.layer_device, dtype=torch.long)
        weights = torch.empty(0, self.top_k, device=self.layer_device, dtype=dtype)
        return hidden, experts, weights

    def _broadcast_inputs(
        self,
        hidden_2d: torch.Tensor,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> None:
        group = self.ctx.ffn_ep_dispatch_group
        src = self.ctx.ffn_coordinator_rank
        h_bytes = hidden_2d.numel() * hidden_2d.element_size()
        s_bytes = selected_experts.numel() * selected_experts.element_size()
        r_bytes = routing_weights.numel() * routing_weights.element_size()
        fused_buf = torch.empty(h_bytes + s_bytes + r_bytes, dtype=torch.uint8, device=self.layer_device)
        if self.is_coordinator:
            fused_buf[:h_bytes].copy_(hidden_2d.contiguous().view(torch.uint8).reshape(-1))
            fused_buf[h_bytes:h_bytes + s_bytes].copy_(selected_experts.contiguous().view(torch.uint8).reshape(-1))
            fused_buf[h_bytes + s_bytes:].copy_(routing_weights.contiguous().view(torch.uint8).reshape(-1))
        dist.broadcast(fused_buf, src=src, group=group)
        if not self.is_coordinator:
            hidden_2d.view(torch.uint8).reshape(-1).copy_(fused_buf[:h_bytes])
            selected_experts.view(torch.uint8).reshape(-1).copy_(fused_buf[h_bytes:h_bytes + s_bytes])
            routing_weights.view(torch.uint8).reshape(-1).copy_(fused_buf[h_bytes + s_bytes:])

    def create_work_item(
        self,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        output_device: Optional[torch.device] = None,
    ) -> EPWorkItem:
        """Prepare router inputs for one EP micro-batch without running collectives."""
        if output_device is None:
            output_device = self.layer_device
        if hidden_states.device != self.layer_device:
            hidden_states = hidden_states.to(self.layer_device, non_blocking=True)

        batch_size, seq_len, hidden_dim = hidden_states.shape
        tokens = batch_size * seq_len
        timing = EPStageTiming()

        if self.is_coordinator:
            if self.post_attention_layernorm is None or self.gate is None:
                raise RuntimeError("FFN EP coordinator requires layernorm and router gate")
            if residual is not None:
                if residual.device != self.layer_device:
                    residual = residual.to(self.layer_device, non_blocking=True)
                hidden_states = residual + hidden_states
            residual_out = hidden_states
            normed = self.post_attention_layernorm(hidden_states)
            hidden_2d = normed.reshape(tokens, hidden_dim).contiguous()
            sync_if_needed(self.layer_device)
            router_start = time.perf_counter()
            _, routing_weights, selected_experts = self.gate(hidden_2d)
            sync_if_needed(self.layer_device)
            router_end = time.perf_counter()
            timing.router_s = router_end - router_start
            timing.router_start_s = router_start
            timing.router_end_s = router_end
            selected_experts = selected_experts.to(torch.int64).contiguous()
            routing_weights = routing_weights.contiguous()
        else:
            residual_out = None
            if self.use_all_to_all:
                hidden_2d, selected_experts, routing_weights = self._empty_all2all_inputs(
                    hidden_dim,
                    hidden_states.dtype,
                )
            else:
                hidden_2d = torch.empty(tokens, hidden_dim, device=self.layer_device, dtype=hidden_states.dtype)
                selected_experts = torch.empty(tokens, self.top_k, device=self.layer_device, dtype=torch.int64)
                routing_weights = torch.empty(tokens, self.top_k, device=self.layer_device, dtype=hidden_states.dtype)

        return EPWorkItem(
            hidden_states=hidden_states,
            output_device=output_device,
            batch_size=batch_size,
            seq_len=seq_len,
            hidden_dim=hidden_dim,
            hidden_2d=hidden_2d,
            selected_experts=selected_experts,
            routing_weights=routing_weights,
            residual_out=residual_out,
            timing=timing,
        )

    def dispatch_async(self, item: EPWorkItem) -> None:
        """Enqueue coordinator-to-expert EP broadcast for one micro-batch (fused)."""
        if self.use_all_to_all:
            item.dispatch_start_s = time.perf_counter()
            assert self._all2all_comm is not None
            item.all2all_dispatch_handle = self._all2all_comm.dispatch_async(
                item.hidden_2d,
                item.selected_experts,
                item.routing_weights,
            )
            item.dispatch_enqueue_done_s = time.perf_counter()
            item.timing.ep_dispatch_enqueue_s = item.dispatch_enqueue_done_s - item.dispatch_start_s
            item.timing.ep_dispatch_start_s = item.dispatch_start_s
            item.timing.ep_dispatch_enqueue_done_s = item.dispatch_enqueue_done_s
            item.timing.ep_dispatch_bytes = (
                item.hidden_2d.numel() * item.hidden_2d.element_size()
                + item.selected_experts.numel() * item.selected_experts.element_size()
                + item.routing_weights.numel() * item.routing_weights.element_size()
            )
            return

        group = self.ctx.ffn_ep_dispatch_group
        src = self.ctx.ffn_coordinator_rank

        h = item.hidden_2d
        s = item.selected_experts
        r = item.routing_weights
        h_bytes = h.numel() * h.element_size()
        s_bytes = s.numel() * s.element_size()
        r_bytes = r.numel() * r.element_size()
        total_bytes = h_bytes + s_bytes + r_bytes

        item.dispatch_start_s = time.perf_counter()
        fused_buf = torch.empty(total_bytes, dtype=torch.uint8, device=self.layer_device)
        if self.is_coordinator:
            fused_buf[:h_bytes].copy_(h.contiguous().view(torch.uint8).reshape(-1))
            fused_buf[h_bytes:h_bytes + s_bytes].copy_(s.contiguous().view(torch.uint8).reshape(-1))
            fused_buf[h_bytes + s_bytes:].copy_(r.contiguous().view(torch.uint8).reshape(-1))
        else:
            item.hidden_2d = fused_buf[:h_bytes].view(h.dtype).reshape(h.shape)
            item.selected_experts = fused_buf[h_bytes:h_bytes + s_bytes].view(s.dtype).reshape(s.shape)
            item.routing_weights = fused_buf[h_bytes + s_bytes:].view(r.dtype).reshape(r.shape)
        item.fused_buf = fused_buf
        item.fused_layout = (h_bytes, s_bytes, r_bytes)
        item.dispatch_handles = [
            dist.broadcast(fused_buf, src=src, group=group, async_op=True),
        ]
        item.dispatch_enqueue_done_s = time.perf_counter()
        item.timing.ep_dispatch_enqueue_s = item.dispatch_enqueue_done_s - item.dispatch_start_s
        item.timing.ep_dispatch_start_s = item.dispatch_start_s
        item.timing.ep_dispatch_enqueue_done_s = item.dispatch_enqueue_done_s
        item.timing.ep_dispatch_bytes = total_bytes

    def finish_dispatch(self, item: EPWorkItem) -> None:
        """Wait until dispatch inputs are ready for local expert compute."""
        wait_start = time.perf_counter()
        if self.use_all_to_all:
            assert self._all2all_comm is not None
            if item.all2all_dispatch_handle is None:
                raise RuntimeError("all_to_all finish_dispatch called before dispatch_async")
            item.all2all_dispatch_handle = self._all2all_comm.wait_dispatch(
                item.all2all_dispatch_handle
            )
        else:
            for handle in item.dispatch_handles:
                handle.wait()
        wait_end = time.perf_counter()
        item.timing.ep_dispatch_wait_s = wait_end - wait_start
        item.timing.ep_dispatch_s = wait_end - item.dispatch_start_s
        item.timing.ep_dispatch_wait_start_s = wait_start
        item.timing.ep_dispatch_wait_end_s = wait_end

    def compute_local(self, item: EPWorkItem) -> None:
        """Run this rank's local expert shard for a dispatched micro-batch."""
        local_start = time.perf_counter()
        if self.use_all_to_all:
            if item.all2all_dispatch_handle is None:
                raise RuntimeError("all_to_all compute_local called before finish_dispatch")
            partial, active, assignments = self.sharded_experts.forward_dispatched(
                item.all2all_dispatch_handle["recv_hidden"],
                item.all2all_dispatch_handle["recv_experts"],
            )
        else:
            partial, active, assignments = self.sharded_experts.forward_local(
                item.hidden_2d,
                item.selected_experts,
                item.routing_weights,
            )
        sync_if_needed(self.layer_device)
        local_end = time.perf_counter()
        item.timing.ep_local_experts_s = local_end - local_start
        item.timing.experts_s = item.timing.ep_local_experts_s
        item.timing.ep_active_experts = active
        item.timing.ep_local_assignments = assignments
        item.timing.ep_local_experts_start_s = local_start
        item.timing.ep_local_experts_end_s = local_end
        item.partial = partial

    def reduce_async(self, item: EPWorkItem) -> None:
        """Enqueue partial-output reduce for one micro-batch."""
        if item.partial is None:
            raise RuntimeError("EP reduce_async called before compute_local")
        if self.use_all_to_all:
            if item.all2all_dispatch_handle is None:
                raise RuntimeError("all_to_all reduce_async called before dispatch")
            item.reduce_start_s = time.perf_counter()
            assert self._all2all_comm is not None
            item.all2all_combine_handle = self._all2all_comm.combine_async(
                item.partial,
                item.all2all_dispatch_handle,
            )
            item.reduce_enqueue_done_s = time.perf_counter()
            item.timing.ep_reduce_enqueue_s = item.reduce_enqueue_done_s - item.reduce_start_s
            item.timing.ep_reduce_start_s = item.reduce_start_s
            item.timing.ep_reduce_enqueue_done_s = item.reduce_enqueue_done_s
            item.timing.ep_reduce_bytes = item.partial.numel() * item.partial.element_size()
            return
        item.reduce_start_s = time.perf_counter()
        item.reduce_handle = dist.reduce(
            item.partial,
            dst=self.ctx.ffn_coordinator_rank,
            op=dist.ReduceOp.SUM,
            group=self.ctx.ffn_ep_reduce_group,
            async_op=True,
        )
        item.reduce_enqueue_done_s = time.perf_counter()
        item.timing.ep_reduce_enqueue_s = item.reduce_enqueue_done_s - item.reduce_start_s
        item.timing.ep_reduce_start_s = item.reduce_start_s
        item.timing.ep_reduce_enqueue_done_s = item.reduce_enqueue_done_s
        item.timing.ep_reduce_bytes = item.partial.numel() * item.partial.element_size()

    def finish_reduce(self, item: EPWorkItem) -> None:
        """Wait for partial-output reduce, tracking how much delay was hidden."""
        wait_start = time.perf_counter()
        item.timing.ep_overlap_hidden_s = max(0.0, wait_start - item.reduce_enqueue_done_s)
        if self.use_all_to_all:
            if item.all2all_combine_handle is None:
                raise RuntimeError("all_to_all finish_reduce called before reduce_async")
            assert self._all2all_comm is not None
            item.all2all_ffn_outputs = self._all2all_comm.wait_combine(
                item.all2all_combine_handle
            )
        else:
            if item.reduce_handle is None:
                raise RuntimeError("EP finish_reduce called before reduce_async")
            item.reduce_handle.wait()
        sync_if_needed(self.layer_device)
        wait_end = time.perf_counter()
        item.timing.ep_reduce_wait_s = wait_end - wait_start
        item.timing.ep_reduce_s = wait_end - item.reduce_start_s
        item.timing.ep_reduce_wait_start_s = wait_start
        item.timing.ep_reduce_wait_end_s = wait_end

    def finish_output(self, item: EPWorkItem):
        """Return the coordinator output after reduce; expert ranks keep a dummy tensor."""
        output_start = time.perf_counter()
        if not self.is_coordinator:
            output_end = time.perf_counter()
            item.timing.ep_finish_output_start_s = output_start
            item.timing.ep_finish_output_end_s = output_end
            return item.hidden_states
        if self.use_all_to_all:
            if item.all2all_ffn_outputs is None or item.residual_out is None:
                raise RuntimeError("all_to_all coordinator output requires combine output and residual")
            output = item.all2all_ffn_outputs.reshape(item.batch_size, item.seq_len, item.hidden_dim)
            output = item.residual_out + output
            if output.device != item.output_device:
                output = output.to(item.output_device, non_blocking=True)
            sync_if_needed(self.layer_device)
            output_end = time.perf_counter()
            item.timing.ep_finish_output_start_s = output_start
            item.timing.ep_finish_output_end_s = output_end
            return output
        if item.partial is None or item.residual_out is None:
            raise RuntimeError("EP coordinator output requires reduced partial and residual")
        output = item.partial.reshape(item.batch_size, item.seq_len, item.hidden_dim)
        output = item.residual_out + output
        if output.device != item.output_device:
            output = output.to(item.output_device, non_blocking=True)
        sync_if_needed(self.layer_device)
        output_end = time.perf_counter()
        item.timing.ep_finish_output_start_s = output_start
        item.timing.ep_finish_output_end_s = output_end
        return output

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        output_device: Optional[torch.device] = None,
        return_timing: bool = False,
    ):
        if output_device is None:
            output_device = self.layer_device
        if hidden_states.device != self.layer_device:
            hidden_states = hidden_states.to(self.layer_device, non_blocking=True)

        timing = EPStageTiming()
        batch_size, seq_len, hidden_dim = hidden_states.shape
        tokens = batch_size * seq_len

        if self.is_coordinator:
            if self.post_attention_layernorm is None or self.gate is None:
                raise RuntimeError("FFN EP coordinator requires layernorm and router gate")
            if residual is not None:
                if residual.device != self.layer_device:
                    residual = residual.to(self.layer_device, non_blocking=True)
                hidden_states = residual + hidden_states
            residual_out = hidden_states
            normed = self.post_attention_layernorm(hidden_states)
            hidden_2d = normed.reshape(tokens, hidden_dim).contiguous()
            sync_if_needed(self.layer_device)
            router_start = time.perf_counter()
            _, routing_weights, selected_experts = self.gate(hidden_2d)
            sync_if_needed(self.layer_device)
            router_end = time.perf_counter()
            timing.router_s = router_end - router_start
            timing.router_start_s = router_start
            timing.router_end_s = router_end
            selected_experts = selected_experts.to(torch.int64).contiguous()
            routing_weights = routing_weights.contiguous()
        else:
            residual_out = None
            if self.use_all_to_all:
                hidden_2d, selected_experts, routing_weights = self._empty_all2all_inputs(
                    hidden_dim,
                    hidden_states.dtype,
                )
            else:
                hidden_2d = torch.empty(tokens, hidden_dim, device=self.layer_device, dtype=hidden_states.dtype)
                selected_experts = torch.empty(tokens, self.top_k, device=self.layer_device, dtype=torch.int64)
                routing_weights = torch.empty(tokens, self.top_k, device=self.layer_device, dtype=hidden_states.dtype)

        if self.use_all_to_all:
            item = EPWorkItem(
                hidden_states=hidden_states,
                output_device=output_device,
                batch_size=batch_size,
                seq_len=seq_len,
                hidden_dim=hidden_dim,
                hidden_2d=hidden_2d,
                selected_experts=selected_experts,
                routing_weights=routing_weights,
                residual_out=residual_out,
                timing=timing,
            )
            self.dispatch_async(item)
            self.finish_dispatch(item)
            self.compute_local(item)
            self.reduce_async(item)
            self.finish_reduce(item)
            output = self.finish_output(item)
            if return_timing:
                return output, item.timing
            return output

        sync_if_needed(self.layer_device)
        dispatch_start = time.perf_counter()
        self._broadcast_inputs(hidden_2d, selected_experts, routing_weights)
        sync_if_needed(self.layer_device)
        dispatch_end = time.perf_counter()
        dispatch_h_bytes = hidden_2d.numel() * hidden_2d.element_size()
        dispatch_s_bytes = selected_experts.numel() * selected_experts.element_size()
        dispatch_r_bytes = routing_weights.numel() * routing_weights.element_size()
        timing.ep_dispatch_s = dispatch_end - dispatch_start
        timing.ep_dispatch_start_s = dispatch_start
        timing.ep_dispatch_enqueue_done_s = dispatch_end
        timing.ep_dispatch_wait_start_s = dispatch_start
        timing.ep_dispatch_wait_end_s = dispatch_end
        timing.ep_dispatch_bytes = dispatch_h_bytes + dispatch_s_bytes + dispatch_r_bytes

        local_start = time.perf_counter()
        partial, active, assignments = self.sharded_experts.forward_local(
            hidden_2d,
            selected_experts,
            routing_weights,
        )
        sync_if_needed(self.layer_device)
        local_end = time.perf_counter()
        timing.ep_local_experts_s = local_end - local_start
        timing.experts_s = timing.ep_local_experts_s
        timing.ep_active_experts = active
        timing.ep_local_assignments = assignments
        timing.ep_local_experts_start_s = local_start
        timing.ep_local_experts_end_s = local_end

        reduce_start = time.perf_counter()
        dist.reduce(
            partial,
            dst=self.ctx.ffn_coordinator_rank,
            op=dist.ReduceOp.SUM,
            group=self.ctx.ffn_ep_reduce_group,
        )
        sync_if_needed(self.layer_device)
        reduce_end = time.perf_counter()
        timing.ep_reduce_s = reduce_end - reduce_start
        timing.ep_reduce_start_s = reduce_start
        timing.ep_reduce_enqueue_done_s = reduce_end
        timing.ep_reduce_wait_start_s = reduce_start
        timing.ep_reduce_wait_end_s = reduce_end
        timing.ep_reduce_bytes = partial.numel() * partial.element_size()

        if not self.is_coordinator:
            if return_timing:
                return hidden_states, timing
            return hidden_states

        output_start = time.perf_counter()
        output = partial.reshape(batch_size, seq_len, hidden_dim)
        output = residual_out + output
        if output.device != output_device:
            output = output.to(output_device, non_blocking=True)
        sync_if_needed(self.layer_device)
        output_end = time.perf_counter()
        timing.ep_finish_output_start_s = output_start
        timing.ep_finish_output_end_s = output_end
        if return_timing:
            return output, timing
        return output
