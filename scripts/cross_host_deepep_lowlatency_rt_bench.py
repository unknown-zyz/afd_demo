"""
D3 dispatch+combine round-trip latency benchmark via DeepEP **low_latency** mode.

Same env contract as cross_host_deepep_rt_bench.py:
  RANK, WORLD_SIZE, LOCAL_RANK, MASTER_ADDR, MASTER_PORT
  HCCL_IF_BASE_PORT, HCCL_IF_IP
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
    p.add_argument("--topk", type=int, default=8)
    p.add_argument("--num-experts", type=int, default=128)
    p.add_argument("--num-max", type=int, default=128,
                   help="num_max_dispatch_tokens_per_rank")
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--use-fp8", action="store_true")
    args = p.parse_args()

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    log(rank, f"start: world={world} local={local_rank} tokens={args.num_tokens} hidden={args.hidden} topk={args.topk} experts={args.num_experts} num_max={args.num_max} fp8={args.use_fp8}")

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

    rdma_bytes = deep_ep.Buffer.get_low_latency_rdma_size_hint(
        args.num_max, args.hidden, world, args.num_experts
    )
    log(rank, f"rdma_size_hint={rdma_bytes} bytes")

    buf = deep_ep.Buffer(
        group=dist.group.WORLD,
        num_nvl_bytes=0,
        num_rdma_bytes=rdma_bytes,
        low_latency_mode=True,
        num_qps_per_rank=args.num_experts // world,
    )
    log(rank, "buffer (low_latency) created")

    torch.manual_seed(42 + rank)
    x = torch.randn(args.num_tokens, args.hidden, dtype=torch.bfloat16, device=device)
    topk_idx = torch.stack([
        torch.randperm(args.num_experts, device=device)[: args.topk]
        for _ in range(args.num_tokens)
    ]).to(torch.int64)
    topk_w = torch.rand(args.num_tokens, args.topk, dtype=torch.float32, device=device)
    topk_w = topk_w / topk_w.sum(dim=-1, keepdim=True)

    def one_rt():
        ret = buf.low_latency_dispatch(
            x, topk_idx,
            args.num_max, args.num_experts,
            use_fp8=args.use_fp8,
        )
        recv = ret[0]
        recv_x = recv if isinstance(recv, torch.Tensor) else recv[0]
        handle = ret[2]
        # echo as FFN output
        combined, _evt, _hook = buf.low_latency_combine(
            recv_x.to(torch.bfloat16) if recv_x.dtype != torch.bfloat16 else recv_x,
            topk_idx, topk_w, handle,
        )
        return combined

    log(rank, "warmup...")
    for i in range(args.warmup):
        try:
            _ = one_rt()
            torch.npu.synchronize()
        except Exception as e:
            log(rank, f"WARMUP FAIL @ iter {i}: {type(e).__name__}: {e}")
            dist.destroy_process_group()
            sys.exit(2)

    log(rank, "measure...")
    times_ms = []
    for i in range(args.iters):
        torch.npu.synchronize()
        t0 = time.perf_counter()
        _ = one_rt()
        torch.npu.synchronize()
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)

    import statistics
    times_ms.sort()
    n = len(times_ms)
    log(rank, f"iters={n} mean={statistics.mean(times_ms):.3f}ms p50={times_ms[n//2]:.3f}ms p99={times_ms[min(n-1,int(n*0.99))]:.3f}ms min={times_ms[0]:.3f}ms max={times_ms[-1]:.3f}ms")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
