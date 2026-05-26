#!/usr/bin/env python3
"""Benchmark FFN EP communication backends on Ascend NPU.

This script intentionally benchmarks communication/reorder only. It compares the
current EPFFNLayer-style full broadcast + dense reduce with torch_npu's official
MoE distribute dispatch/combine APIs when the runtime supports the requested
shape.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any, Callable

import torch
import torch.distributed as dist


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark MoE dispatch/combine communication on NPU.")
    parser.add_argument("--backend", choices=["broadcast_reduce", "npu_moe_distribute"], required=True)
    parser.add_argument("--tokens", default="4,16,64", help="Comma-separated local token counts.")
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--experts", type=int, default=128)
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--use-v2", action="store_true", help="Use *_v2 MoE distribute APIs.")
    parser.add_argument("--global-bs", type=int, default=None, help="global_bs for official API.")
    parser.add_argument("--output", required=True, help="JSON output path written by rank 0.")
    return parser.parse_args()


def parse_int_list(spec: str) -> list[int]:
    values = [int(part.strip()) for part in spec.split(",") if part.strip()]
    if not values or any(value < 1 for value in values):
        raise ValueError(f"Invalid positive integer list: {spec!r}")
    return values


def dtype_from_name(name: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16}[name]


def log(message: str) -> None:
    print(message, flush=True)


def sync() -> None:
    torch.npu.synchronize()


def init_dist() -> tuple[int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.npu.set_device(local_rank)
    dist.init_process_group("hccl", init_method="env://")
    return dist.get_rank(), dist.get_world_size()


def get_hcomm_info(rank: int) -> str:
    backend = dist.group.WORLD._get_backend(torch.device("npu"))
    return backend.get_hccl_comm_name(rank)


def bench(fn: Callable[[], None], warmup: int, iters: int, repeats: int) -> tuple[float, list[float]]:
    for _ in range(warmup):
        fn()
    sync()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        for _ in range(iters):
            fn()
        sync()
        samples.append((time.perf_counter() - start) * 1000.0 / iters)
    return float(statistics.median(samples)), samples


def make_routing(tokens: int, top_k: int, experts: int) -> tuple[torch.Tensor, torch.Tensor]:
    selected = (
        torch.arange(tokens * top_k, device="npu", dtype=torch.int64)
        .view(tokens, top_k)
        .remainder(experts)
        .contiguous()
    )
    weights = torch.full((tokens, top_k), 1.0 / top_k, device="npu", dtype=torch.bfloat16)
    return selected, weights


def broadcast_reduce_case(
    tokens: int,
    hidden: int,
    top_k: int,
    experts: int,
    dtype: torch.dtype,
    rank: int,
    world: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    del experts, world
    hidden_2d = torch.randn(tokens, hidden, device="npu", dtype=dtype)
    selected, weights = make_routing(tokens, top_k, args.experts)
    weights = weights.to(dtype=dtype)
    partial = torch.randn(tokens, hidden, device="npu", dtype=dtype)

    h_bytes = hidden_2d.numel() * hidden_2d.element_size()
    s_bytes = selected.numel() * selected.element_size()
    w_bytes = weights.numel() * weights.element_size()
    total_bytes = h_bytes + s_bytes + w_bytes
    fused_buf = torch.empty(total_bytes, dtype=torch.uint8, device="npu")

    def pack_if_src() -> None:
        if rank == 0:
            fused_buf[:h_bytes].copy_(hidden_2d.contiguous().view(torch.uint8).reshape(-1))
            fused_buf[h_bytes : h_bytes + s_bytes].copy_(selected.contiguous().view(torch.uint8).reshape(-1))
            fused_buf[h_bytes + s_bytes :].copy_(weights.contiguous().view(torch.uint8).reshape(-1))

    def dispatch_run() -> None:
        pack_if_src()
        handle = dist.broadcast(fused_buf, src=0, async_op=True)
        handle.wait()

    def reduce_run() -> None:
        handle = dist.reduce(partial, dst=0, op=dist.ReduceOp.SUM, async_op=True)
        handle.wait()

    dispatch_ms, dispatch_samples = bench(dispatch_run, args.warmup, args.iters, args.repeats)
    reduce_ms, reduce_samples = bench(reduce_run, args.warmup, args.iters, args.repeats)
    return {
        "backend": "broadcast_reduce",
        "rank": rank,
        "tokens": tokens,
        "hidden": hidden,
        "top_k": top_k,
        "dispatch_bytes": total_bytes,
        "reduce_bytes": partial.numel() * partial.element_size(),
        "dispatch_ms": dispatch_ms,
        "reduce_ms": reduce_ms,
        "total_ms": dispatch_ms + reduce_ms,
        "dispatch_samples_ms": dispatch_samples,
        "reduce_samples_ms": reduce_samples,
        "ok": True,
    }


def official_case(
    tokens: int,
    hidden: int,
    top_k: int,
    experts: int,
    dtype: torch.dtype,
    rank: int,
    world: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    import torch_npu  # type: ignore[import-not-found]

    hcomm = get_hcomm_info(rank)
    x = torch.randn(tokens, hidden, device="npu", dtype=dtype)
    selected, weights = make_routing(tokens, top_k, experts)
    expert_ids = selected.to(torch.int32)
    weights = weights.to(torch.float32)
    global_bs = args.global_bs if args.global_bs is not None else tokens * world
    dispatch = torch_npu.npu_moe_distribute_dispatch_v2 if args.use_v2 else torch_npu.npu_moe_distribute_dispatch
    combine = torch_npu.npu_moe_distribute_combine_v2 if args.use_v2 else torch_npu.npu_moe_distribute_combine

    state: dict[str, torch.Tensor] = {}

    def dispatch_once() -> None:
        output = dispatch(
            x=x,
            expert_ids=expert_ids,
            group_ep=hcomm,
            ep_world_size=world,
            ep_rank_id=rank,
            moe_expert_num=experts,
            expert_scales=weights,
            global_bs=global_bs,
        )
        state["expand_x"], _, state["expand_idx"], state["expert_token_nums"], state["ep_recv_counts"], state[
            "tp_recv_counts"
        ], state["expand_scales"] = output

    try:
        dispatch_ms, dispatch_samples = bench(dispatch_once, args.warmup, args.iters, args.repeats)
    except Exception as exc:
        return {
            "backend": "npu_moe_distribute_v2" if args.use_v2 else "npu_moe_distribute",
            "rank": rank,
            "tokens": tokens,
            "hidden": hidden,
            "top_k": top_k,
            "ok": False,
            "stage": "dispatch",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }

    def combine_once() -> None:
        kwargs: dict[str, Any] = {
            "expand_x": state["expand_x"],
            "expert_ids": expert_ids,
            "ep_send_counts": state["ep_recv_counts"],
            "expert_scales": weights,
            "tp_send_counts": state["tp_recv_counts"],
            "expand_scales": state["expand_scales"],
            "group_ep": hcomm,
            "ep_world_size": world,
            "ep_rank_id": rank,
            "moe_expert_num": experts,
            "global_bs": global_bs,
        }
        if args.use_v2:
            kwargs["assist_info_for_combine"] = state["expand_idx"]
        else:
            kwargs["expand_idx"] = state["expand_idx"]
        state["combined"] = combine(**kwargs)

    try:
        combine_ms, combine_samples = bench(combine_once, args.warmup, args.iters, args.repeats)
        diff = (state["combined"].float() - x.float()).abs().max().item()
    except Exception as exc:
        return {
            "backend": "npu_moe_distribute_v2" if args.use_v2 else "npu_moe_distribute",
            "rank": rank,
            "tokens": tokens,
            "hidden": hidden,
            "top_k": top_k,
            "ok": False,
            "stage": "combine",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }

    return {
        "backend": "npu_moe_distribute_v2" if args.use_v2 else "npu_moe_distribute",
        "rank": rank,
        "tokens": tokens,
        "hidden": hidden,
        "top_k": top_k,
        "global_bs": global_bs,
        "dispatch_ms": dispatch_ms,
        "combine_ms": combine_ms,
        "total_ms": dispatch_ms + combine_ms,
        "dispatch_samples_ms": dispatch_samples,
        "combine_samples_ms": combine_samples,
        "max_abs_diff": diff,
        "ok": True,
    }


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row.get("backend"), row.get("tokens"), row.get("hidden"), row.get("top_k"), row.get("ok"))
        grouped.setdefault(key, []).append(row)
    result = []
    for (backend, tokens, hidden, top_k, ok), group in grouped.items():
        out: dict[str, Any] = {
            "backend": backend,
            "tokens": tokens,
            "hidden": hidden,
            "top_k": top_k,
            "ok": ok,
            "rank_count": len(group),
        }
        if ok:
            for field in ("dispatch_ms", "reduce_ms", "combine_ms", "total_ms"):
                values = [float(row[field]) for row in group if row.get(field) is not None]
                if values:
                    out[f"{field}_median"] = float(statistics.median(values))
                    out[f"{field}_max"] = float(max(values))
        else:
            first = group[0]
            out["stage"] = first.get("stage")
            out["error_type"] = first.get("error_type")
            out["error"] = str(first.get("error", "")).splitlines()[0]
        result.append(out)
    return sorted(result, key=lambda row: (str(row["backend"]), int(row["hidden"]), int(row["tokens"])))


def main() -> None:
    args = parse_args()
    rank, world = init_dist()
    dtype = dtype_from_name(args.dtype)
    rows = []
    for tokens in parse_int_list(args.tokens):
        if args.backend == "broadcast_reduce":
            row = broadcast_reduce_case(tokens, args.hidden, args.top_k, args.experts, dtype, rank, world, args)
        else:
            row = official_case(tokens, args.hidden, args.top_k, args.experts, dtype, rank, world, args)
        rows.append(row)
        log(json.dumps(row, ensure_ascii=False))

    gathered: list[list[dict[str, Any]]] | None = [None for _ in range(world)] if rank == 0 else None
    dist.gather_object(rows, gathered, dst=0)
    if rank == 0:
        assert gathered is not None
        flat = [row for part in gathered for row in part]
        payload = {
            "backend": args.backend,
            "world_size": world,
            "dtype": args.dtype,
            "warmup": args.warmup,
            "iters": args.iters,
            "repeats": args.repeats,
            "rows": flat,
            "aggregate": aggregate(flat),
        }
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        log(f"wrote {output}")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
