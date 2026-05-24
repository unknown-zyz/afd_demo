"""
D3 dispatch+combine round-trip latency benchmark (DeepEP-Ascend cross-host).

Env vars:
  RANK, WORLD_SIZE, LOCAL_RANK
  MASTER_ADDR, MASTER_PORT
  HCCL_IF_BASE_PORT, HCCL_IF_IP

Usage (each rank):
  python3 scripts/cross_host_deepep_rt_bench.py --iters 50 --warmup 10 \
      --num-tokens 64 --hidden 4096 --topk 4 --num-experts 16
"""

import argparse
import os
import sys
import time

import torch
import torch.distributed as dist


def log(rank, msg):
    print(f"[rank{rank}] {msg}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num-tokens", type=int, default=64)
    p.add_argument("--hidden", type=int, default=4096)
    p.add_argument("--topk", type=int, default=4)
    p.add_argument("--num-experts", type=int, default=16)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--nvl-bytes", type=int, default=256 * 1024 * 1024)
    args = p.parse_args()

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    log(rank, f"start: world={world} local={local_rank} tokens={args.num_tokens} hidden={args.hidden} topk={args.topk} experts={args.num_experts}")

    import torch_npu  # noqa: F401
    torch.npu.set_device(local_rank)
    device = torch.device(f"npu:{local_rank}")

    dist.init_process_group(
        backend="hccl",
        init_method="env://",
        rank=rank,
        world_size=world,
    )
    log(rank, "hccl pg ok")

    import deep_ep

    buf = deep_ep.Buffer(
        group=dist.group.WORLD,
        num_nvl_bytes=args.nvl_bytes,
        low_latency_mode=False,
    )
    log(rank, "buffer (normal) created")

    cfg = deep_ep.Buffer.get_dispatch_config(world)
    combine_cfg = deep_ep.Buffer.get_combine_config(world)

    torch.manual_seed(42 + rank)
    x = torch.randn(args.num_tokens, args.hidden, dtype=torch.bfloat16, device=device)
    topk_idx = torch.stack([
        torch.randperm(args.num_experts, device=device)[: args.topk]
        for _ in range(args.num_tokens)
    ]).to(torch.int64)
    topk_w = torch.rand(args.num_tokens, args.topk, dtype=torch.float32, device=device)
    topk_w = topk_w / topk_w.sum(dim=-1, keepdim=True)

    def one_rt():
        (num_tokens_per_rank, num_tokens_per_rdma_rank, num_tokens_per_expert,
         is_token_in_rank, _evt) = buf.get_dispatch_layout(topk_idx, args.num_experts)
        recv_x, recv_topk_idx, recv_topk_w, num_recv_per_expert_list, handle, _evt = buf.dispatch(
            x,
            num_tokens_per_rank=num_tokens_per_rank,
            num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
            is_token_in_rank=is_token_in_rank,
            num_tokens_per_expert=num_tokens_per_expert,
            topk_idx=topk_idx,
            topk_weights=topk_w,
            config=cfg,
        )
        # echo as FFN output
        out = recv_x if isinstance(recv_x, torch.Tensor) else recv_x[0]
        combined, _w, _evt = buf.combine(out, handle, topk_weights=recv_topk_w, config=combine_cfg)
        return combined

    # warmup
    log(rank, "warmup...")
    for i in range(args.warmup):
        try:
            _ = one_rt()
        except Exception as e:
            log(rank, f"WARMUP FAIL @ iter {i}: {type(e).__name__}: {e}")
            sys.exit(2)
    torch.npu.synchronize()
    dist.barrier()
    log(rank, "warmup ok, starting timed iters")

    lat_us = []
    for _ in range(args.iters):
        torch.npu.synchronize()
        t0 = time.perf_counter()
        _ = one_rt()
        torch.npu.synchronize()
        t1 = time.perf_counter()
        lat_us.append((t1 - t0) * 1e6)

    lat_us.sort()
    mean = sum(lat_us) / len(lat_us)
    p50 = lat_us[len(lat_us) // 2]
    p99 = lat_us[max(0, int(len(lat_us) * 0.99) - 1)]
    log(rank, f"RT us: mean={mean:.1f} p50={p50:.1f} p99={p99:.1f} min={lat_us[0]:.1f} max={lat_us[-1]:.1f} n={len(lat_us)}")

    dist.barrier()
    dist.destroy_process_group()
    log(rank, "D3 RT BENCH DONE")


if __name__ == "__main__":
    main()
