#!/usr/bin/env python3
"""
Measure communication latency between nodes.

Performs ping-pong test to measure P2P communication latency:
- Attention node sends a tensor to FFN node
- FFN node sends it back
- Repeat many times and calculate average latency
"""

import argparse
import time
import torch
import torch.distributed as dist

def ping_pong_test(num_iterations=100, tensor_size=1024):
    """
    Ping-pong latency test.
    
    Args:
        num_iterations: Number of round trips
        tensor_size: Size of tensor to send (in float32 elements)
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    if world_size != 2:
        raise ValueError(f"This test requires exactly 2 processes, got {world_size}")
    
    peer_rank = 1 - rank  # 0 <-> 1
    device = torch.device(f"cuda:{rank}")
    
    # Create test tensor
    tensor = torch.randn(tensor_size, device=device, dtype=torch.float32)
    
    # Warmup
    for _ in range(10):
        if rank == 0:
            dist.send(tensor, dst=peer_rank)
            dist.recv(tensor, src=peer_rank)
        else:
            dist.recv(tensor, src=peer_rank)
            dist.send(tensor, dst=peer_rank)
    
    # Measure
    latencies = []
    
    for i in range(num_iterations):
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        if rank == 0:
            # Ping-pong: send -> recv
            dist.send(tensor, dst=peer_rank)
            dist.recv(tensor, src=peer_rank)
        else:
            # Pong-ping: recv -> send
            dist.recv(tensor, src=peer_rank)
            dist.send(tensor, dst=peer_rank)
        
        torch.cuda.synchronize()
        end = time.perf_counter()
        
        if rank == 0:
            # Only rank 0 measures (full round trip)
            latency_ms = (end - start) * 1000
            latencies.append(latency_ms)
    
    if rank == 0:
        import numpy as np
        latencies = np.array(latencies)
        
        print(f"\n{'='*60}")
        print(f"Communication Latency Test Results")
        print(f"{'='*60}")
        print(f"Tensor size: {tensor_size} floats ({tensor_size*4} bytes)")
        print(f"Iterations: {num_iterations}")
        print(f"")
        print(f"Round-trip latency:")
        print(f"  Mean:   {latencies.mean():.3f} ms")
        print(f"  Median: {np.median(latencies):.3f} ms")
        print(f"  Std:    {latencies.std():.3f} ms")
        print(f"  Min:    {latencies.min():.3f} ms")
        print(f"  Max:    {latencies.max():.3f} ms")
        print(f"  P95:    {np.percentile(latencies, 95):.3f} ms")
        print(f"  P99:    {np.percentile(latencies, 99):.3f} ms")
        print(f"")
        print(f"One-way latency (estimated): {latencies.mean()/2:.3f} ms")
        print(f"{'='*60}\n")

def main():
    parser = argparse.ArgumentParser(description="Measure communication latency")
    parser.add_argument("--iterations", type=int, default=100, help="Number of iterations")
    parser.add_argument("--tensor-size", type=int, default=1024, help="Tensor size (floats)")
    parser.add_argument("--local-test", action="store_true", help="Local test mode")
    parser.add_argument("--master-addr", default="127.0.0.1", help="Master address")
    parser.add_argument("--master-port", default="29500", help="Master port")
    parser.add_argument("--rank", type=int, default=None, help="Rank (auto from env if not set)")
    parser.add_argument("--world-size", type=int, default=2, help="World size")
    
    args = parser.parse_args()
    
    # Initialize distributed
    if args.local_test:
        # torchrun sets these automatically
        pass
    else:
        import os
        if args.rank is not None:
            os.environ["RANK"] = str(args.rank)
        os.environ["WORLD_SIZE"] = str(args.world_size)
        os.environ["MASTER_ADDR"] = args.master_addr
        os.environ["MASTER_PORT"] = args.master_port
    
    dist.init_process_group(backend="nccl")
    
    try:
        ping_pong_test(args.iterations, args.tensor_size)
    finally:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
