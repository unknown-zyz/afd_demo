#!/usr/bin/env python3
"""Check EPFFN packed compute with torch_npu MoE dispatch/combine v2."""

from __future__ import annotations

import argparse
import json
import os
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.nn.functional as F

from src.model.ep_moe import ExpertShardPlan, ShardedExperts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate npu_moe_v2 EPFFN semantics on NPU.")
    parser.add_argument("--tokens", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--intermediate", type=int, default=64)
    parser.add_argument("--experts", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument(
        "--routing-pattern",
        choices=["round_robin", "random", "hot"],
        default="round_robin",
    )
    parser.add_argument("--input-file", default="")
    parser.add_argument("--model-name", default="")
    parser.add_argument("--layer-idx", type=int, default=0)
    parser.add_argument("--simulate-af-p2p", action="store_true")
    parser.add_argument("--global-bs-multiplier", type=int, default=1)
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--comm-alg", default="fullmesh_v2")
    parser.add_argument("--use-transfer-to-npu", action="store_true")
    parser.add_argument(
        "--ffn-subgroup-offset",
        type=int,
        default=0,
        help="Reserve ranks [0, offset) as non-FFN ranks; remaining ranks form the EP group.",
    )
    parser.add_argument(
        "--extra-ep-groups",
        type=int,
        default=0,
        help="Create additional FFN EP groups after the primary group to mimic src.distributed.",
    )
    parser.add_argument("--output", default="")
    return parser.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16}[name]


def init_dist() -> tuple[int, int, int]:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.npu.set_device(local_rank)
    dist.init_process_group("hccl", init_method="env://")
    return rank, local_rank, world


def get_hcomm(group, ep_rank: int) -> str:
    backend = group._get_backend(torch.device("npu"))
    return backend.get_hccl_comm_name(ep_rank)


def make_experts(args: argparse.Namespace, plan: ExpertShardPlan, dtype: torch.dtype):
    if args.model_name:
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=dtype,
            device_map="cpu",
            trust_remote_code=True,
        )
        return model.model.layers[args.layer_idx].mlp.experts

    gate_up = []
    down = []
    for expert_id in range(args.experts):
        gen = torch.Generator(device="cpu")
        gen.manual_seed(10_000 + int(expert_id))
        gu = torch.randn(
            2 * args.intermediate,
            args.hidden,
            generator=gen,
            dtype=torch.float32,
        ).mul_(0.01)
        gen.manual_seed(20_000 + int(expert_id))
        dw = torch.randn(
            args.hidden,
            args.intermediate,
            generator=gen,
            dtype=torch.float32,
        ).mul_(0.01)
        gate_up.append(gu.to(device="npu", dtype=dtype))
        down.append(dw.to(device="npu", dtype=dtype))
    return SimpleNamespace(
        num_experts=args.experts,
        gate_up_proj=gate_up,
        down_proj=down,
        act_fn=F.silu,
    )


def make_source_inputs(args: argparse.Namespace, dtype: torch.dtype, rank: int):
    if rank == 0:
        if args.input_file:
            payload = torch.load(args.input_file, map_location="cpu")
            x = payload["hidden_2d"].to("npu", dtype=dtype).contiguous()
            expert_ids = payload["selected_experts"].to("npu", dtype=torch.int64).contiguous()
            weights = payload["routing_weights"].to("npu", dtype=dtype).contiguous()
        else:
            gen = torch.Generator(device="cpu")
            gen.manual_seed(1234)
            x = torch.randn(args.tokens, args.hidden, generator=gen, dtype=torch.float32).to("npu", dtype=dtype)
            if args.routing_pattern == "random":
                gen.manual_seed(4321)
                scores = torch.randn(args.tokens, args.experts, generator=gen, dtype=torch.float32)
                weights, expert_ids = torch.topk(F.softmax(scores, dim=-1), args.top_k, dim=-1)
                weights = weights / weights.sum(dim=-1, keepdim=True)
                expert_ids = expert_ids.to("npu", dtype=torch.int64).contiguous()
                weights = weights.to("npu", dtype=dtype).contiguous()
            elif args.routing_pattern == "hot":
                expert_ids = torch.arange(args.top_k, device="npu", dtype=torch.int64).repeat(args.tokens, 1)
                weights = torch.full((args.tokens, args.top_k), 1.0 / args.top_k, device="npu", dtype=dtype)
            else:
                expert_ids = (
                    torch.arange(args.tokens * args.top_k, device="npu", dtype=torch.int64)
                    .reshape(args.tokens, args.top_k)
                    .remainder(args.experts)
                    .contiguous()
                )
                weights = torch.full((args.tokens, args.top_k), 1.0 / args.top_k, device="npu", dtype=dtype)
    else:
        x = torch.zeros(args.tokens, args.hidden, device="npu", dtype=dtype)
        expert_ids = torch.zeros(args.tokens, args.top_k, device="npu", dtype=torch.int64)
        weights = torch.zeros(args.tokens, args.top_k, device="npu", dtype=dtype)
    return x, expert_ids, weights


def broadcast_reference_inputs(x: torch.Tensor, expert_ids: torch.Tensor, weights: torch.Tensor) -> None:
    dist.broadcast(x, src=0)
    dist.broadcast(expert_ids, src=0)
    dist.broadcast(weights, src=0)


def main() -> None:
    args = parse_args()
    if args.use_transfer_to_npu:
        from torch_npu.contrib import transfer_to_npu  # noqa: F401
    rank, _local_rank, world = init_dist()
    ffn_ranks = list(range(args.ffn_subgroup_offset, world))
    ep_world = len(ffn_ranks)
    if rank < args.ffn_subgroup_offset:
        group = dist.new_group(ranks=ffn_ranks, backend="hccl")
        for _ in range(args.extra_ep_groups):
            dist.new_group(ranks=ffn_ranks, backend="hccl")
        if args.simulate_af_p2p:
            send_tensor = torch.zeros(args.tokens, args.hidden, device="npu", dtype=dtype_from_name(args.dtype))
            dist.send(send_tensor, dst=args.ffn_subgroup_offset, tag=0)
            recv_tensor = torch.empty_like(send_tensor)
            dist.recv(recv_tensor, src=args.ffn_subgroup_offset, tag=1)
        dist.barrier()
        dist.destroy_process_group()
        return
    group = dist.new_group(ranks=ffn_ranks, backend="hccl")
    for _ in range(args.extra_ep_groups):
        dist.new_group(ranks=ffn_ranks, backend="hccl")
    ep_rank = rank - args.ffn_subgroup_offset
    if args.experts % ep_world != 0:
        raise ValueError("--experts must be divisible by WORLD_SIZE for contiguous EP sharding")
    dtype = dtype_from_name(args.dtype)
    if args.simulate_af_p2p and rank == args.ffn_subgroup_offset:
        recv_tensor = torch.empty(args.tokens, args.hidden, device="npu", dtype=dtype)
        dist.recv(recv_tensor, src=0, tag=0)
    plan = ExpertShardPlan(args.experts, ep_world, ep_rank, policy="contiguous")
    sharded = ShardedExperts(make_experts(args, plan, dtype), plan, device=torch.device("npu"), dtype=dtype)
    hcomm = get_hcomm(group, ep_rank)

    source_x, source_experts, source_weights = make_source_inputs(args, dtype, ep_rank)
    ref_x = source_x.clone()
    ref_experts = source_experts.clone()
    ref_weights = source_weights.clone()
    dist.broadcast(ref_x, src=args.ffn_subgroup_offset, group=group)
    dist.broadcast(ref_experts, src=args.ffn_subgroup_offset, group=group)
    dist.broadcast(ref_weights, src=args.ffn_subgroup_offset, group=group)
    ref_partial, ref_active, ref_assignments = sharded.forward_local(ref_x, ref_experts, ref_weights)
    dist.reduce(ref_partial, dst=args.ffn_subgroup_offset, op=dist.ReduceOp.SUM, group=group)
    torch.npu.synchronize()

    import torch_npu  # type: ignore[import-not-found]

    dispatch_out = torch_npu.npu_moe_distribute_dispatch_v2(
        x=source_x,
        expert_ids=source_experts.to(torch.int32),
        expert_scales=source_weights.to(torch.float32),
        group_ep=hcomm,
        ep_world_size=ep_world,
        ep_rank_id=ep_rank,
        moe_expert_num=args.experts,
        global_bs=args.tokens * args.global_bs_multiplier,
        expert_shard_type=0,
        comm_alg=args.comm_alg,
    )
    expand_x, _dynamic_scales, assist, expert_token_nums, ep_recv_counts, tp_recv_counts, expand_scales = dispatch_out
    packed_out, packed_active, packed_assignments = sharded.forward_packed(expand_x, expert_token_nums)
    combined = torch_npu.npu_moe_distribute_combine_v2(
        expand_x=packed_out,
        expert_ids=source_experts.to(torch.int32),
        assist_info_for_combine=assist,
        ep_send_counts=ep_recv_counts,
        expert_scales=source_weights.to(torch.float32),
        tp_send_counts=tp_recv_counts,
        expand_scales=expand_scales,
        group_ep=hcomm,
        ep_world_size=ep_world,
        ep_rank_id=ep_rank,
        moe_expert_num=args.experts,
        global_bs=args.tokens * args.global_bs_multiplier,
        expert_shard_type=0,
        comm_alg=args.comm_alg,
    )
    torch.npu.synchronize()
    if args.simulate_af_p2p and rank == args.ffn_subgroup_offset:
        dist.send(combined.to(dtype), dst=0, tag=1)

    diff = None
    if rank == args.ffn_subgroup_offset:
        diff = float((combined.float() - ref_partial.float()).abs().max().item())
    row = {
        "rank": rank,
        "ep_rank": ep_rank,
        "ep_world": ep_world,
        "tokens": args.tokens,
        "hidden": args.hidden,
        "experts": args.experts,
        "top_k": args.top_k,
        "comm_alg": args.comm_alg,
        "routing_pattern": args.routing_pattern,
        "local_experts": plan.local_expert_ids,
        "ref_active": ref_active,
        "ref_assignments": ref_assignments,
        "packed_active": packed_active,
        "packed_assignments": packed_assignments,
        "expert_token_nums": expert_token_nums.detach().cpu().tolist(),
        "rank0_max_abs_diff": diff,
    }
    gathered = [None for _ in range(ep_world)] if rank == args.ffn_subgroup_offset else None
    dist.gather_object(row, gathered, dst=args.ffn_subgroup_offset, group=group)
    if rank == args.ffn_subgroup_offset:
        payload = {"ok": diff is not None and diff < 1e-2, "max_abs_diff": diff, "rows": gathered}
        print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
        if not payload["ok"]:
            raise SystemExit(1)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
