#!/usr/bin/env python3
"""Benchmark NCCL P2P send latency after idle gaps."""
import os
import sys
import time
import torch
import torch.distributed as dist

def main():
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 2))
    master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
    master_port = os.environ.get("MASTER_PORT", "29700")

    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port

    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    device = torch.device("cuda:0")
    peer = 1 - rank

    # Data tensors (same size as real pipeline: batch=2, seq=128, hidden=4096, bf16)
    data = torch.randn(2, 128, 4096, dtype=torch.bfloat16, device=device)
    recv_buf = torch.empty_like(data)

    # Warmup
    for _ in range(5):
        if rank == 0:
            dist.isend(data, peer).wait()
            dist.irecv(recv_buf, peer).wait()
        else:
            dist.irecv(recv_buf, peer).wait()
            dist.isend(data, peer).wait()

    dist.barrier()
    if rank == 0:
        print(f"Tensor size: {data.numel() * data.element_size() / 1024:.0f} KB")

    # Test: send after varying idle gaps
    gaps_ms = [0, 1, 5, 10, 15, 20, 25, 30, 40, 50]

    for gap in gaps_ms:
        latencies = []
        for trial in range(5):
            # Rank 0 sends, Rank 1 receives
            if rank == 0:
                # Do a "baseline" send first (proxy is warm)
                h = dist.isend(data, peer)
                h.wait()
                # Now sleep to let proxy go idle
                torch.cuda.synchronize()
                time.sleep(gap / 1000.0)
                # Measure send latency after the gap
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                h = dist.isend(data, peer)
                h.wait()
                t1 = time.perf_counter()
                latencies.append((t1 - t0) * 1000)
            else:
                # Rank 1 receives both sends
                dist.irecv(recv_buf, peer).wait()
                dist.irecv(recv_buf, peer).wait()

            dist.barrier()

        if rank == 0:
            avg = sum(latencies) / len(latencies)
            mn = min(latencies)
            mx = max(latencies)
            print(f"Gap={gap:3d}ms: avg={avg:6.2f}ms  min={mn:6.2f}ms  max={mx:6.2f}ms")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
