"""Expert-parallel MoE helpers for NPU FFN prototypes."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable, Optional

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

    def __post_init__(self) -> None:
        if self.num_experts <= 0:
            raise ValueError("num_experts must be positive")
        if self.ep_size <= 0:
            raise ValueError("ep_size must be positive")
        if not 0 <= self.ep_rank < self.ep_size:
            raise ValueError(f"ep_rank must be in [0, {self.ep_size}), got {self.ep_rank}")
        if self.policy not in {"round_robin", "contiguous"}:
            raise ValueError(f"Unsupported expert shard policy: {self.policy}")

    @property
    def local_expert_ids(self) -> list[int]:
        if self.policy == "round_robin":
            return [idx for idx in range(self.num_experts) if idx % self.ep_size == self.ep_rank]
        base = self.num_experts // self.ep_size
        remainder = self.num_experts % self.ep_size
        start = self.ep_rank * base + min(self.ep_rank, remainder)
        count = base + (1 if self.ep_rank < remainder else 0)
        return list(range(start, start + count))

    @staticmethod
    def all_assignments(num_experts: int, ep_size: int, policy: str = "round_robin") -> list[list[int]]:
        return [
            ExpertShardPlan(num_experts, ep_size, ep_rank, policy).local_expert_ids
            for ep_rank in range(ep_size)
        ]


@dataclass
class EPStageTiming:
    """Fine-grained EP timing in seconds."""

    router_s: float = 0.0
    experts_s: float = 0.0
    shared_or_dense_s: float = 0.0
    ep_dispatch_s: float = 0.0
    ep_local_experts_s: float = 0.0
    ep_reduce_s: float = 0.0
    ep_dispatch_wait_s: float = 0.0
    ep_reduce_wait_s: float = 0.0
    ep_overlap_hidden_s: float = 0.0
    ep_active_experts: int = 0
    ep_local_assignments: int = 0


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

        gate_up = {}
        down = {}
        for expert_id in self.local_expert_ids:
            gate_up[str(expert_id)] = nn.Parameter(
                experts.gate_up_proj[expert_id].detach().to(device=device, dtype=dtype).contiguous(),
                requires_grad=False,
            )
            down[str(expert_id)] = nn.Parameter(
                experts.down_proj[expert_id].detach().to(device=device, dtype=dtype).contiguous(),
                requires_grad=False,
            )
        self.gate_up_proj = nn.ParameterDict(gate_up)
        self.down_proj = nn.ParameterDict(down)

    def forward_local(
        self,
        hidden_2d: torch.Tensor,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, int, int]:
        """Return this shard's dense partial output plus activity counters."""

        partial = torch.zeros_like(hidden_2d)
        active = 0
        assignments = 0
        for expert_id in self.local_expert_ids:
            token_idx, topk_idx = torch.where(selected_experts == expert_id)
            if token_idx.numel() == 0:
                continue
            active += 1
            assignments += int(token_idx.numel())
            x = hidden_2d[token_idx].contiguous()
            gate_up = F.linear(x, self.gate_up_proj[str(expert_id)])
            gate, up = gate_up.chunk(2, dim=-1)
            hidden = self.act_fn(gate) * up
            out = F.linear(hidden, self.down_proj[str(expert_id)])
            weights = routing_weights[token_idx, topk_idx, None].to(out.dtype)
            partial.index_add_(0, token_idx, out * weights)
        return partial, active, assignments


def sync_if_needed(device: torch.device) -> None:
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

    @property
    def is_coordinator(self) -> bool:
        return self.ctx.is_ffn_coordinator

    def _broadcast_inputs(
        self,
        hidden_2d: torch.Tensor,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> None:
        group = self.ctx.ffn_ep_dispatch_group
        src = self.ctx.ffn_coordinator_rank
        dist.broadcast(hidden_2d, src=src, group=group)
        dist.broadcast(selected_experts, src=src, group=group)
        dist.broadcast(routing_weights, src=src, group=group)

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
            timing.router_s = time.perf_counter() - router_start
            selected_experts = selected_experts.to(torch.int64).contiguous()
            routing_weights = routing_weights.contiguous()
        else:
            residual_out = None
            hidden_2d = torch.empty(tokens, hidden_dim, device=self.layer_device, dtype=hidden_states.dtype)
            selected_experts = torch.empty(tokens, self.top_k, device=self.layer_device, dtype=torch.int64)
            routing_weights = torch.empty(tokens, self.top_k, device=self.layer_device, dtype=hidden_states.dtype)

        sync_if_needed(self.layer_device)
        dispatch_start = time.perf_counter()
        self._broadcast_inputs(hidden_2d, selected_experts, routing_weights)
        sync_if_needed(self.layer_device)
        timing.ep_dispatch_s = time.perf_counter() - dispatch_start

        local_start = time.perf_counter()
        partial, active, assignments = self.sharded_experts.forward_local(
            hidden_2d,
            selected_experts,
            routing_weights,
        )
        sync_if_needed(self.layer_device)
        timing.ep_local_experts_s = time.perf_counter() - local_start
        timing.experts_s = timing.ep_local_experts_s
        timing.ep_active_experts = active
        timing.ep_local_assignments = assignments

        reduce_start = time.perf_counter()
        dist.reduce(
            partial,
            dst=self.ctx.ffn_coordinator_rank,
            op=dist.ReduceOp.SUM,
            group=self.ctx.ffn_ep_reduce_group,
        )
        sync_if_needed(self.layer_device)
        timing.ep_reduce_s = time.perf_counter() - reduce_start

        if not self.is_coordinator:
            if return_timing:
                return hidden_states, timing
            return hidden_states

        output = partial.reshape(batch_size, seq_len, hidden_dim)
        output = residual_out + output
        if output.device != output_device:
            output = output.to(output_device, non_blocking=True)
        if return_timing:
            return output, timing
        return output
