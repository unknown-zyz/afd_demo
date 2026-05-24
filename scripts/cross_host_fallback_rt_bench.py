"""
D4 fallback all_to_all_single round-trip latency benchmark (HCCL cross-host).

Mimics MoE dispatch+combine semantics via two `all_to_all_single` calls
(scatter to expert ranks → echo → gather back). Used to validate the
FallbackMoECommunicator code path when DeepEP-Ascend is unavailable.

Env vars: RANK WORLD_SIZE LOCAL_RANK MASTER_ADDR MASTER_PORT HCCL_IF_BASE_PORT HCCL_IF_IP
"""

import argparse
import os
import time

import torch
import torch.distributed as dist


def log(rank, msg):
    print(f"[rank{rank}] {msg}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num-tokens", type=int, default=64)
    p.add_argument("--hidden", type=int, default=4096)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--warmup", type=int, default=10)
    args = p.parse_args()

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    log(rank, f"start: world={world} local={local_rank} tokens={args.num_tokens} hidden={args.hidden}")

    import torch_npu  # noqa: F401
    torch.npu.set_device(local_rank)
    device = torch.device(f"npu:{local_rank}")

    dist.init_process_group(backend="hccl", init_method="env://", rank=rank, world_size=world)
    log(rank, "hccl pg ok")

    assert args.num_tokens % world == 0, "tokens must divide world for equal split"
    per = args.num_tokens // world

    x = torch.randn(args.num_tokens, args.hidden, dtype=torch.bfloat16, device=device)

    def one_rt():
        # dispatch: all_to_all_single — each rank sends `per` tokens to each rank
        sent = torch.empty_like(x)
        dist.all_to_all_single(sent, x.contiguous())
        # echo (no FFN) — combine: send back
        recv = torch.empty_like(x)
        dist.all_to_all_single(recv, sent.contiguous())
        return recv

    log(rank, "warmup...")
    for _ in range(args.warmup):
        _ = one_rt()
    torch.npu.synchronize()
    dist.barrier()
    log(rank, "warmup ok, timed iters")

    lat_us = []
    for _ in range(args.iters):
        torch.npu.synchronize()
        t0 = time.perf_counter()
        _ = one_rt()
        torch.npu.synchronize()
        lat_us.append((time.perf_counter() - t0) * 1e6)

    lat_us.sort()
    mean = sum(lat_us) / len(lat_us)
    p50 = lat_us[len(lat_us) // 2]
    p99 = lat_us[max(0, int(len(lat_us) * 0.99) - 1)]
    log(rank, f"RT us: mean={mean:.1f} p50={p50:.1f} p99={p99:.1f} min={lat_us[0]:.1f} max={lat_us[-1]:.1f} n={len(lat_us)} payload={args.num_tokens*args.hidden*2/1024:.1f} KiB")

    dist.barrier()
    dist.destroy_process_group()
    log(rank, "D4 FALLBACK RT BENCH DONE")


if __name__ == "__main__":
    main()
