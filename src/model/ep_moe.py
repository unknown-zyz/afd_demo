"""Expert-parallel MoE helpers for NPU FFN prototypes."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "0").lower() in {"1", "true", "yes", "on"}


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
    assist_info_for_combine: Optional[torch.Tensor] = None
    expert_token_nums: Optional[torch.Tensor] = None
    ep_recv_counts: Optional[torch.Tensor] = None
    tp_recv_counts: Optional[torch.Tensor] = None
    expand_scales: Optional[torch.Tensor] = None
    combined: Optional[torch.Tensor] = None
    dispatch_handles: list[dist.Work] = field(default_factory=list)
    reduce_handle: Optional[dist.Work] = None
    dispatch_start_s: float = 0.0
    dispatch_enqueue_done_s: float = 0.0
    reduce_start_s: float = 0.0
    reduce_enqueue_done_s: float = 0.0
    fused_buf: Optional[torch.Tensor] = None
    fused_layout: Optional[tuple] = None


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
        self.act_fn = getattr(experts, "act_fn", None)
        if self.act_fn is None and len(experts) > 0:
            self.act_fn = getattr(experts[0], "act_fn", F.silu)
        self.num_total_experts = plan.num_experts

        if not self.local_expert_ids:
            self.gate_up_stack = None
            self.down_stack = None
        else:
            gate_up_list = [self._load_gate_up_weight(experts, eid, device, dtype) for eid in self.local_expert_ids]
            down_list = [self._load_down_weight(experts, eid, device, dtype) for eid in self.local_expert_ids]
            self.gate_up_stack = nn.Parameter(
                torch.stack(gate_up_list, dim=0).contiguous(), requires_grad=False
            )
            self.down_stack = nn.Parameter(
                torch.stack(down_list, dim=0).contiguous(), requires_grad=False
            )

        local_mask = torch.zeros(self.num_total_experts, dtype=torch.bool, device=device)
        for eid in self.local_expert_ids:
            local_mask[eid] = True
        self.register_buffer("_local_mask_lut", local_mask, persistent=False)
        self._local_idx_of_py = {int(eid): i for i, eid in enumerate(self.local_expert_ids)}

    @staticmethod
    def _load_gate_up_weight(experts, expert_id: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if hasattr(experts, "gate_up_proj"):
            return experts.gate_up_proj[expert_id].detach().to(device=device, dtype=dtype).contiguous()
        expert = experts[expert_id]
        if not hasattr(expert, "gate_proj") or not hasattr(expert, "up_proj"):
            raise RuntimeError(f"Expert {expert_id} does not expose gate_proj/up_proj weights")
        gate = expert.gate_proj.weight.detach()
        up = expert.up_proj.weight.detach()
        return torch.cat([gate, up], dim=0).to(device=device, dtype=dtype).contiguous()

    @staticmethod
    def _load_down_weight(experts, expert_id: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if hasattr(experts, "down_proj"):
            return experts.down_proj[expert_id].detach().to(device=device, dtype=dtype).contiguous()
        expert = experts[expert_id]
        if not hasattr(expert, "down_proj"):
            raise RuntimeError(f"Expert {expert_id} does not expose down_proj weights")
        return expert.down_proj.weight.detach().to(device=device, dtype=dtype).contiguous()

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

    def forward_packed(
        self,
        expand_x: torch.Tensor,
        expert_token_nums: torch.Tensor,
    ) -> tuple[torch.Tensor, int, int]:
        """Run local experts on dispatch_v2 packed contiguous expert segments."""
        if not self.local_expert_ids or self.gate_up_stack is None:
            return torch.zeros_like(expand_x), 0, 0

        counts = [int(v) for v in expert_token_nums.detach().cpu().tolist()]
        if len(counts) < len(self.local_expert_ids):
            raise RuntimeError(
                f"dispatch_v2 returned {len(counts)} expert counts for "
                f"{len(self.local_expert_ids)} local experts"
            )

        output = torch.zeros(expand_x.shape, dtype=expand_x.dtype, device=expand_x.device)
        offset = 0
        active = 0
        assignments = 0
        trace = _env_flag("AFD_NPU_MOE_V2_TRACE_EXPERTS")
        max_active_rows_env = os.environ.get("AFD_NPU_MOE_V2_MAX_ACTIVE_ROWS")
        max_active_rows = int(max_active_rows_env) if max_active_rows_env else 0
        for local_idx, count in enumerate(counts[:len(self.local_expert_ids)]):
            if count < 0:
                raise RuntimeError(f"dispatch_v2 expert_token_nums contains negative count {count}")
            if count == 0:
                continue
            if offset + count > expand_x.shape[0]:
                raise RuntimeError(
                    f"dispatch_v2 packed segment exceeds expand_x rows: "
                    f"offset={offset}, count={count}, rows={expand_x.shape[0]}"
                )
            seg = expand_x.narrow(0, offset, count)
            if trace:
                print(
                    f"[npu_moe_v2_trace] pid={os.getpid()} expert={self.local_expert_ids[local_idx]} "
                    f"packed_count={count} stage=packed_segment_start",
                    flush=True,
                )
            active_rows = seg.abs().sum(dim=-1).ne(0)
            row_idx = active_rows.nonzero(as_tuple=False).squeeze(-1)
            if row_idx.numel() == 0:
                offset += count
                continue
            if max_active_rows > 0 and row_idx.numel() > max_active_rows:
                row_idx = row_idx[:max_active_rows]
            seg_active = seg.index_select(0, row_idx)
            if trace:
                print(
                    f"[npu_moe_v2_trace] pid={os.getpid()} expert={self.local_expert_ids[local_idx]} "
                    f"active_rows={int(row_idx.numel())} stage=gate_up_start",
                    flush=True,
                )
            gu = F.linear(seg_active, self.gate_up_stack[local_idx])
            if trace:
                print(
                    f"[npu_moe_v2_trace] pid={os.getpid()} expert={self.local_expert_ids[local_idx]} "
                    "stage=gate_up_done",
                    flush=True,
                )
            gate, up = gu.chunk(2, dim=-1)
            hidden = self.act_fn(gate) * up
            if trace:
                print(
                    f"[npu_moe_v2_trace] pid={os.getpid()} expert={self.local_expert_ids[local_idx]} "
                    "stage=down_start",
                    flush=True,
                )
            out_seg = F.linear(hidden, self.down_stack[local_idx])
            if trace:
                print(
                    f"[npu_moe_v2_trace] pid={os.getpid()} expert={self.local_expert_ids[local_idx]} "
                    "stage=down_done",
                    flush=True,
                )
            output.narrow(0, offset, count).index_copy_(0, row_idx, out_seg)
            active += 1
            assignments += int(row_idx.numel())
            offset += count
        return output, active, assignments


def sync_if_needed(device: torch.device) -> None:
    if os.environ.get("AFD_EP_SYNC_TIMING", "0") != "1":
        return
    if device.type == "npu" and hasattr(torch, "npu"):
        torch.npu.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def _debug_sync(device: torch.device) -> None:
    if device.type == "npu" and hasattr(torch, "npu"):
        torch.npu.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def parse_gate_output(gate_output, top_k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (routing_weights, selected_experts) across HF Qwen gate variants."""
    values = gate_output if isinstance(gate_output, (tuple, list)) else (gate_output,)
    routing_weights = None
    selected_experts = None
    for value in values:
        if not isinstance(value, torch.Tensor) or value.ndim < 2 or value.shape[-1] != top_k:
            continue
        if value.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
            selected_experts = value
        elif value.dtype.is_floating_point:
            routing_weights = value
    if routing_weights is None or selected_experts is None:
        logits = next(
            (
                value
                for value in values
                if isinstance(value, torch.Tensor)
                and value.ndim == 2
                and value.shape[-1] > top_k
                and value.dtype.is_floating_point
            ),
            None,
        )
        if logits is not None:
            probs = F.softmax(logits.float(), dim=-1)
            routing_weights, selected_experts = torch.topk(probs, top_k, dim=-1)
            denom = routing_weights.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(routing_weights.dtype).eps)
            routing_weights = routing_weights / denom
            return routing_weights.to(dtype=logits.dtype), selected_experts
        shapes = [
            (tuple(value.shape), str(value.dtype))
            for value in values
            if isinstance(value, torch.Tensor)
        ]
        raise RuntimeError(f"Could not parse gate output for top_k={top_k}; tensor outputs={shapes}")
    return routing_weights, selected_experts


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
        self._npu_moe_hcomm: Optional[str] = None

        if self.use_npu_moe_v2:
            self._validate_npu_moe_v2()

    @property
    def is_coordinator(self) -> bool:
        return self.ctx.is_ffn_coordinator

    @property
    def use_npu_moe_v2(self) -> bool:
        return self.ctx.config is not None and self.ctx.config.ffn_ep_backend == "npu_moe_v2"

    def _validate_npu_moe_v2(self) -> None:
        if self.layer_device.type != "npu":
            raise RuntimeError("ffn_ep_backend=npu_moe_v2 requires an Ascend NPU device")
        if self.sharded_experts.plan.policy != "contiguous":
            raise RuntimeError(
                "ffn_ep_backend=npu_moe_v2 requires --ep-expert-policy contiguous; "
                "Ascend dispatch_v2 currently rejects expert_shard_type != 0."
            )
        try:
            import torch_npu  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("ffn_ep_backend=npu_moe_v2 requires torch_npu") from exc
        for op_name in ("npu_moe_distribute_dispatch_v2", "npu_moe_distribute_combine_v2"):
            packet = getattr(torch.ops.npu, op_name, None)
            if packet is None:
                raise RuntimeError(f"torch_npu is missing {op_name}; cannot enable npu_moe_v2")
            schema = str(packet.default._schema)
            if "comm_alg" not in schema:
                raise RuntimeError(
                    f"{op_name} schema does not expose comm_alg; "
                    "npu_moe_v2 requires torch_npu/CANN with comm_alg=fullmesh_v2 support"
                )

    def _get_npu_moe_hcomm(self) -> str:
        if self._npu_moe_hcomm is not None:
            return self._npu_moe_hcomm
        group = self.ctx.ffn_ep_group
        if group is None:
            raise RuntimeError("npu_moe_v2 requires an initialized FFN EP process group")
        backend = group._get_backend(torch.device("npu"))
        # `group` is the FFN EP subgroup (global ranks may be offset by the
        # attention rank), so the HCCL comm name must be queried by EP-local rank.
        self._npu_moe_hcomm = backend.get_hccl_comm_name(self.ctx.ffn_ep_rank)
        return self._npu_moe_hcomm

    def _npu_moe_common_kwargs(self, item: EPWorkItem) -> dict[str, Any]:
        global_bs = int(os.environ.get("AFD_NPU_MOE_V2_GLOBAL_BS", "0"))
        if global_bs <= 0:
            global_bs = item.hidden_2d.shape[0] * self.ctx.ffn_ep_size
        return {
            "group_ep": self._get_npu_moe_hcomm(),
            "ep_world_size": self.ctx.ffn_ep_size,
            "ep_rank_id": self.ctx.ffn_ep_rank,
            "moe_expert_num": self.sharded_experts.num_total_experts,
            "global_bs": global_bs,
            "comm_alg": "fullmesh_v2",
            "expert_shard_type": 0,
        }

    def _trace_npu_moe_v2(self, item: EPWorkItem, stage: str) -> None:
        if not _env_flag("AFD_NPU_MOE_V2_TRACE"):
            return
        trace_layer = os.environ.get("AFD_NPU_MOE_V2_TRACE_LAYER")
        if trace_layer not in (None, "", "all") and self.layer_idx != int(trace_layer):
            return
        print(
            f"[npu_moe_v2_trace] rank={self.ctx.rank} ep_rank={self.ctx.ffn_ep_rank} "
            f"layer={self.layer_idx} stage={stage} tokens={item.hidden_2d.shape[0]} "
            f"hidden={item.hidden_dim}",
            flush=True,
        )

    def _npu_moe_debug_layer_matches(self) -> bool:
        abort_layer = os.environ.get("AFD_NPU_MOE_V2_ABORT_LAYER")
        return abort_layer in (None, "", "all") or self.layer_idx == int(abort_layer)

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
            routing_weights, selected_experts = parse_gate_output(self.gate(hidden_2d), self.top_k)
            sync_if_needed(self.layer_device)
            router_end = time.perf_counter()
            timing.router_s = router_end - router_start
            timing.router_start_s = router_start
            timing.router_end_s = router_end
            selected_experts = selected_experts.to(torch.int64).contiguous()
            routing_weights = routing_weights.contiguous()
        else:
            residual_out = None
            if self.use_npu_moe_v2:
                hidden_2d = torch.zeros(tokens, hidden_dim, device=self.layer_device, dtype=hidden_states.dtype)
                selected_experts = torch.zeros(tokens, self.top_k, device=self.layer_device, dtype=torch.int64)
                routing_weights = torch.zeros(tokens, self.top_k, device=self.layer_device, dtype=hidden_states.dtype)
            else:
                hidden_2d = torch.empty(tokens, hidden_dim, device=self.layer_device, dtype=hidden_states.dtype)
                selected_experts = torch.empty(tokens, self.top_k, device=self.layer_device, dtype=torch.int64)
                routing_weights = torch.empty(tokens, self.top_k, device=self.layer_device, dtype=hidden_states.dtype)

        dump_dir = os.environ.get("AFD_NPU_MOE_V2_DUMP_DIR")
        dump_layer = int(os.environ.get("AFD_NPU_MOE_V2_DUMP_LAYER", "0"))
        if self.use_npu_moe_v2 and dump_dir and self.layer_idx == dump_layer:
            os.makedirs(dump_dir, exist_ok=True)
            if self.is_coordinator:
                torch.save(
                    {
                        "hidden_2d": hidden_2d.detach().cpu(),
                        "selected_experts": selected_experts.detach().cpu(),
                        "routing_weights": routing_weights.detach().cpu(),
                        "top_k": self.top_k,
                        "num_experts": self.sharded_experts.num_total_experts,
                        "ep_size": self.ctx.ffn_ep_size,
                        "hidden_dim": hidden_dim,
                        "tokens": tokens,
                    },
                    os.path.join(dump_dir, f"layer{self.layer_idx}_rank{self.ctx.rank}.pt"),
                )
            if os.environ.get("AFD_NPU_MOE_V2_DUMP_ABORT", "0") == "1":
                raise RuntimeError(f"Dumped npu_moe_v2 layer {self.layer_idx} inputs to {dump_dir}")

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
        if self.use_npu_moe_v2:
            self._dispatch_npu_moe_v2(item)
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
        if self.use_npu_moe_v2:
            item.timing.ep_dispatch_wait_start_s = item.dispatch_enqueue_done_s
            item.timing.ep_dispatch_wait_end_s = item.dispatch_enqueue_done_s
            return
        wait_start = time.perf_counter()
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
        if self.use_npu_moe_v2:
            if item.partial is None or item.expert_token_nums is None:
                raise RuntimeError("npu_moe_v2 compute_local called before dispatch_v2")
            self._trace_npu_moe_v2(item, "compute_local_start")
            if _env_flag("AFD_NPU_MOE_V2_ZERO_EXPERTS"):
                partial = torch.zeros_like(item.partial)
                active = 0
                assignments = 0
            else:
                partial, active, assignments = self.sharded_experts.forward_packed(
                    item.partial,
                    item.expert_token_nums,
                )
            if _env_flag("AFD_NPU_MOE_V2_FORCE_SYNC"):
                _debug_sync(self.layer_device)
                self._trace_npu_moe_v2(item, "compute_local_synced")
            self._trace_npu_moe_v2(item, "compute_local_done")
            if _env_flag("AFD_NPU_MOE_V2_ABORT_AFTER_COMPUTE") and self._npu_moe_debug_layer_matches():
                raise RuntimeError("AFD_NPU_MOE_V2_ABORT_AFTER_COMPUTE")
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
        if self.use_npu_moe_v2:
            self._combine_npu_moe_v2(item)
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
        if self.use_npu_moe_v2:
            item.timing.ep_overlap_hidden_s = 0.0
            item.timing.ep_reduce_wait_start_s = item.reduce_enqueue_done_s
            item.timing.ep_reduce_wait_end_s = item.reduce_enqueue_done_s
            return
        if item.reduce_handle is None:
            raise RuntimeError("EP finish_reduce called before reduce_async")
        wait_start = time.perf_counter()
        item.timing.ep_overlap_hidden_s = max(0.0, wait_start - item.reduce_enqueue_done_s)
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
        if item.partial is None or item.residual_out is None:
            raise RuntimeError("EP coordinator output requires reduced partial and residual")
        source = item.combined if self.use_npu_moe_v2 else item.partial
        if source is None:
            raise RuntimeError("EP coordinator output requires combined npu_moe_v2 output")
        output = source.reshape(item.batch_size, item.seq_len, item.hidden_dim)
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
        if self.use_npu_moe_v2:
            item = self.create_work_item(hidden_states, residual=residual, output_device=output_device)
            self.dispatch_async(item)
            self.finish_dispatch(item)
            self.compute_local(item)
            self.reduce_async(item)
            self.finish_reduce(item)
            output = self.finish_output(item)
            if return_timing:
                return output, item.timing
            return output
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
            routing_weights, selected_experts = parse_gate_output(self.gate(hidden_2d), self.top_k)
            sync_if_needed(self.layer_device)
            router_end = time.perf_counter()
            timing.router_s = router_end - router_start
            timing.router_start_s = router_start
            timing.router_end_s = router_end
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

    def _dispatch_npu_moe_v2(self, item: EPWorkItem) -> None:
        import torch_npu  # type: ignore[import-not-found]

        item.dispatch_start_s = time.perf_counter()
        self._trace_npu_moe_v2(item, "dispatch_start")
        expert_ids = item.selected_experts.to(torch.int32).contiguous()
        expert_scales = item.routing_weights.to(torch.float32).contiguous()
        kwargs = self._npu_moe_common_kwargs(item)
        output = torch_npu.npu_moe_distribute_dispatch_v2(
            x=item.hidden_2d.contiguous(),
            expert_ids=expert_ids,
            expert_scales=expert_scales,
            **kwargs,
        )
        self._trace_npu_moe_v2(item, "dispatch_returned")
        if _env_flag("AFD_NPU_MOE_V2_FORCE_SYNC"):
            _debug_sync(self.layer_device)
            self._trace_npu_moe_v2(item, "dispatch_synced")
        else:
            sync_if_needed(self.layer_device)
        dispatch_end = time.perf_counter()
        expand_x, _dynamic_scales, assist, expert_token_nums, ep_recv_counts, tp_recv_counts, expand_scales = output
        item.selected_experts = expert_ids
        item.routing_weights = expert_scales
        item.partial = expand_x
        item.assist_info_for_combine = assist
        item.expert_token_nums = expert_token_nums
        item.ep_recv_counts = ep_recv_counts
        item.tp_recv_counts = tp_recv_counts
        item.expand_scales = expand_scales
        item.dispatch_enqueue_done_s = dispatch_end
        item.timing.ep_dispatch_s = dispatch_end - item.dispatch_start_s
        item.timing.ep_dispatch_enqueue_s = item.timing.ep_dispatch_s
        item.timing.ep_dispatch_start_s = item.dispatch_start_s
        item.timing.ep_dispatch_enqueue_done_s = dispatch_end
        item.timing.ep_dispatch_wait_start_s = item.dispatch_start_s
        item.timing.ep_dispatch_wait_end_s = dispatch_end
        item.timing.ep_dispatch_bytes = item.hidden_2d.numel() * item.hidden_2d.element_size()
        if _env_flag("AFD_NPU_MOE_V2_ABORT_AFTER_DISPATCH") and self._npu_moe_debug_layer_matches():
            raise RuntimeError("AFD_NPU_MOE_V2_ABORT_AFTER_DISPATCH")

    def _combine_npu_moe_v2(self, item: EPWorkItem) -> None:
        import torch_npu  # type: ignore[import-not-found]

        required = (
            item.assist_info_for_combine,
            item.ep_recv_counts,
            item.tp_recv_counts,
            item.expand_scales,
        )
        if any(value is None for value in required):
            raise RuntimeError("npu_moe_v2 combine called before dispatch metadata is available")
        item.reduce_start_s = time.perf_counter()
        self._trace_npu_moe_v2(item, "combine_start")
        kwargs = self._npu_moe_common_kwargs(item)
        combine_input = torch.empty(item.partial.shape, dtype=item.partial.dtype, device=item.partial.device)
        combine_input.copy_(item.partial)
        combined = torch_npu.npu_moe_distribute_combine_v2(
            expand_x=combine_input,
            expert_ids=item.selected_experts.to(torch.int32).contiguous(),
            assist_info_for_combine=item.assist_info_for_combine,
            ep_send_counts=item.ep_recv_counts,
            expert_scales=item.routing_weights.to(torch.float32).contiguous(),
            tp_send_counts=item.tp_recv_counts,
            expand_scales=item.expand_scales,
            **kwargs,
        )
        self._trace_npu_moe_v2(item, "combine_returned")
        if _env_flag("AFD_NPU_MOE_V2_FORCE_SYNC"):
            _debug_sync(self.layer_device)
            self._trace_npu_moe_v2(item, "combine_synced")
        else:
            sync_if_needed(self.layer_device)
        reduce_end = time.perf_counter()
        item.combined = combined
        item.reduce_enqueue_done_s = reduce_end
        item.timing.ep_reduce_s = reduce_end - item.reduce_start_s
        item.timing.ep_reduce_enqueue_s = item.timing.ep_reduce_s
        item.timing.ep_reduce_start_s = item.reduce_start_s
        item.timing.ep_reduce_enqueue_done_s = reduce_end
        item.timing.ep_reduce_wait_start_s = item.reduce_start_s
        item.timing.ep_reduce_wait_end_s = reduce_end
        item.timing.ep_reduce_bytes = combine_input.numel() * combine_input.element_size()
        if _env_flag("AFD_NPU_MOE_V2_ABORT_AFTER_COMBINE") and self._npu_moe_debug_layer_matches():
            raise RuntimeError("AFD_NPU_MOE_V2_ABORT_AFTER_COMBINE")
