"""
Cross-host smoke that drives FallbackMoECommunicator.dispatch/combine
(the actual class used by AttentionWorker/FFNWorker), not just raw
all_to_all_single.

Env: RANK WORLD_SIZE LOCAL_RANK MASTER_ADDR MASTER_PORT HCCL_IF_BASE_PORT HCCL_IF_IP
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
    p.add_argument("--num-experts", type=int, default=16)
    p.add_argument("--topk", type=int, default=4)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--warmup", type=int, default=5)
    args = p.parse_args()

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    log(rank, f"start: world={world} local={local_rank} N={args.num_tokens} H={args.hidden} E={args.num_experts} K={args.topk}")

    import torch_npu  # noqa
    torch.npu.set_device(local_rank)
    device = torch.device(f"npu:{local_rank}")

    dist.init_process_group(backend="hccl", init_method="env://", rank=rank, world_size=world)
    log(rank, "hccl pg ok")

    # Make sure src is importable
    import sys, pathlib
    repo = pathlib.Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo))
    from src.coordinator_arch.comm.fallback_a2a import FallbackMoECommunicator

    # Use default WORLD group as ep_group; creating a new_group on HCCL requires
    # an additional bind port which collides with HCCL_IF_BASE_PORT on the same host.
    ep_group = dist.group.WORLD
    comm = FallbackMoECommunicator(
        ep_group=ep_group,
        hidden_size=args.hidden,
        num_experts=args.num_experts,
        max_tokens_per_rank=args.num_tokens,
        device=device,
    )
    # Uniform expert→rank routing table
    expert_to_rank = [(e * world) // args.num_experts for e in range(args.num_experts)]
    comm.update_routing_table({"version": 1, "expert_to_rank": expert_to_rank, "mode": "normal"})
    log(rank, f"comm ready; expert_to_rank head={expert_to_rank[:8]}")

    # Build deterministic top-k routing
    N, H, K = args.num_tokens, args.hidden, args.topk
    hidden_states = torch.randn(N, H, dtype=torch.bfloat16, device=device) + rank
    idx = (torch.arange(N * K, device=device) % args.num_experts).view(N, K).long()
    w = torch.full((N, K), 1.0 / K, device=device, dtype=hidden_states.dtype)

    def one_step():
        h = comm.dispatch(hidden_states, idx, w)
        ffn_out = h["recv_hidden"]  # identity FFN
        merged = comm.combine(ffn_out, h)
        return merged

    log(rank, "warmup...")
    for _ in range(args.warmup):
        _ = one_step()
    torch.npu.synchronize()
    dist.barrier()
    log(rank, "warmup ok, timed iters")

    lat = []
    for _ in range(args.iters):
        torch.npu.synchronize()
        t0 = time.perf_counter()
        out = one_step()
        torch.npu.synchronize()
        lat.append((time.perf_counter() - t0) * 1e6)

    # Light correctness: output shape must match hidden_states
    assert out.shape == hidden_states.shape, f"shape mismatch {out.shape} vs {hidden_states.shape}"

    lat.sort()
    mean = sum(lat) / len(lat)
    p50 = lat[len(lat) // 2]
    p99 = lat[max(0, int(len(lat) * 0.99) - 1)]
    log(rank, f"FallbackMoECommunicator RT us: mean={mean:.1f} p50={p50:.1f} p99={p99:.1f} min={lat[0]:.1f} max={lat[-1]:.1f} n={len(lat)}")
    log(rank, "FALLBACK COMM SMOKE PASS")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
