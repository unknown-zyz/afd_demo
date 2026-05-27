#!/usr/bin/env python3
"""Probe Ascend torch_npu MoE distribute dispatch/combine APIs."""

from __future__ import annotations

import argparse
import inspect
import os
import time
from typing import Any

import torch
import torch.distributed as dist


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe torch_npu MoE distribute APIs on NPU.")
    parser.add_argument("--mode", choices=["schema", "smoke"], default="schema")
    parser.add_argument("--tokens", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--experts", type=int, default=16)
    parser.add_argument(
        "--global-bs",
        type=int,
        default=None,
        help="global_bs passed to MoE distribute ops. Defaults to tokens * ep_world_size.",
    )
    parser.add_argument("--use-v2", action="store_true", help="Use *_v2 APIs in smoke mode.")
    parser.add_argument(
        "--comm-alg",
        default="",
        help="Optional comm_alg for *_v2 APIs when the installed torch_npu schema exposes it.",
    )
    return parser.parse_args()


def log(message: str) -> None:
    print(message, flush=True)


def schema_probe() -> None:
    import torch_npu  # type: ignore[import-not-found]

    names = [
        "npu_moe_distribute_dispatch",
        "npu_moe_distribute_combine",
        "npu_moe_distribute_dispatch_v2",
        "npu_moe_distribute_combine_v2",
    ]
    log(f"torch={torch.__version__}")
    log(f"torch_npu={getattr(torch_npu, '__version__', None)}")
    for name in names:
        obj = getattr(torch_npu, name, None)
        log(f"=== {name} ===")
        log(f"available={obj is not None}")
        if obj is not None:
            try:
                log(f"signature={inspect.signature(obj)}")
            except Exception as exc:
                log(f"signature_error={type(exc).__name__}: {exc}")
        packet = getattr(torch.ops.npu, name, None)
        if packet is not None:
            try:
                log(f"schema={packet.default._schema}")
            except Exception as exc:
                log(f"schema_error={type(exc).__name__}: {exc}")


def get_hcomm_info(rank: int) -> str:
    backend = dist.group.WORLD._get_backend(torch.device("npu"))
    return backend.get_hccl_comm_name(rank)


def op_supports_kwarg(op_name: str, kwarg: str) -> bool:
    packet = getattr(torch.ops.npu, op_name, None)
    if packet is None:
        return False
    try:
        return kwarg in str(packet.default._schema)
    except Exception:
        return False


def maybe_add_comm_alg(kwargs: dict[str, Any], op_name: str, comm_alg: str) -> None:
    if not comm_alg:
        return
    if not op_supports_kwarg(op_name, "comm_alg"):
        raise RuntimeError(f"{op_name} schema does not expose comm_alg; cannot pass {comm_alg!r}")
    kwargs["comm_alg"] = comm_alg


def init_hccl() -> tuple[int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.npu.set_device(local_rank)
    log(f"local_rank={local_rank} before init_process_group")
    dist.init_process_group("hccl", init_method="env://")
    rank = dist.get_rank()
    world = dist.get_world_size()
    log(f"rank={rank} world={world} after init_process_group")
    return rank, world


def smoke(args: argparse.Namespace) -> None:
    import torch_npu  # type: ignore[import-not-found]

    rank, world = init_hccl()
    hcomm = get_hcomm_info(rank)
    log(f"rank={rank} hcomm_len={len(hcomm)}")
    global_bs = args.global_bs if args.global_bs is not None else args.tokens * world

    torch.manual_seed(1234 + rank)
    x = torch.randn(args.tokens, args.hidden, device="npu", dtype=torch.bfloat16)
    expert_ids = (
        torch.arange(args.tokens * args.top_k, device="npu", dtype=torch.int32)
        .view(args.tokens, args.top_k)
        .remainder(args.experts)
        .contiguous()
    )
    weights = torch.full(
        (args.tokens, args.top_k),
        1.0 / args.top_k,
        device="npu",
        dtype=torch.float32,
    )

    dispatch = torch_npu.npu_moe_distribute_dispatch_v2 if args.use_v2 else torch_npu.npu_moe_distribute_dispatch
    combine = torch_npu.npu_moe_distribute_combine_v2 if args.use_v2 else torch_npu.npu_moe_distribute_combine

    torch.npu.synchronize()
    start = time.perf_counter()
    log(f"rank={rank} before dispatch")
    dispatch_kwargs: dict[str, Any] = {
        "x": x,
        "expert_ids": expert_ids,
        "group_ep": hcomm,
        "ep_world_size": world,
        "ep_rank_id": rank,
        "moe_expert_num": args.experts,
        "expert_scales": weights,
        "global_bs": global_bs,
    }
    if args.use_v2:
        maybe_add_comm_alg(dispatch_kwargs, "npu_moe_distribute_dispatch_v2", args.comm_alg)
    output = dispatch(**dispatch_kwargs)
    torch.npu.synchronize()
    dispatch_ms = (time.perf_counter() - start) * 1000.0
    expand_x, _dynamic_scales, expand_idx, expert_token_nums, ep_recv_counts, tp_recv_counts, expand_scales = output
    log(
        "rank={} after dispatch dispatch_ms={:.3f} expand_x={} expert_token_nums={} ep_recv_counts={}".format(
            rank,
            dispatch_ms,
            tuple(expand_x.shape),
            expert_token_nums.cpu().tolist(),
            ep_recv_counts.cpu().tolist(),
        )
    )

    kwargs: dict[str, Any] = {
        "expand_x": expand_x,
        "expert_ids": expert_ids,
        "ep_send_counts": ep_recv_counts,
        "expert_scales": weights,
        "tp_send_counts": tp_recv_counts,
        "expand_scales": expand_scales,
        "group_ep": hcomm,
        "ep_world_size": world,
        "ep_rank_id": rank,
        "moe_expert_num": args.experts,
        "global_bs": global_bs,
    }
    if args.use_v2:
        kwargs["assist_info_for_combine"] = expand_idx
        maybe_add_comm_alg(kwargs, "npu_moe_distribute_combine_v2", args.comm_alg)
    else:
        kwargs["expand_idx"] = expand_idx

    start = time.perf_counter()
    combined = combine(**kwargs)
    torch.npu.synchronize()
    combine_ms = (time.perf_counter() - start) * 1000.0
    diff = (combined.float() - x.float()).abs().max().item()
    log(f"rank={rank} after combine combine_ms={combine_ms:.3f} combined={tuple(combined.shape)} diff={diff:.6g}")
    dist.barrier()
    dist.destroy_process_group()


def main() -> None:
    args = parse_args()
    if args.mode == "schema":
        schema_probe()
    else:
        smoke(args)


if __name__ == "__main__":
    main()
