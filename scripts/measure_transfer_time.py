#!/usr/bin/env python3
"""
测量单机环境下 GPU 间数据传输时间。

使用同步 send/recv 测量纯传输时间，不包含 DBO 重叠。
"""

import argparse
import time
import torch
import torch.distributed as dist


def measure_transfer_time(
    tensor_size: tuple,
    num_iterations: int = 10,
    warmup: int = 3,
    dtype: torch.dtype = torch.bfloat16,
) -> dict:
    """
    测量从 rank 0 发送到 rank 1 的传输时间。
    
    Args:
        tensor_size: 张量形状 (batch, seq_len, hidden_dim)
        num_iterations: 测量次数
        warmup: 预热次数
        dtype: 数据类型
    
    Returns:
        dict: 包含 mean, min, max, std 的统计数据
    """
    rank = dist.get_rank()
    device = torch.device(f"cuda:{rank}")
    
    # 创建测试张量
    if rank == 0:
        tensor = torch.randn(tensor_size, dtype=dtype, device=device)
    else:
        tensor = torch.empty(tensor_size, dtype=dtype, device=device)
    
    # 计算数据量
    num_elements = tensor.numel()
    bytes_per_element = tensor.element_size()
    total_bytes = num_elements * bytes_per_element
    total_mb = total_bytes / (1024 * 1024)
    
    times = []
    
    for i in range(warmup + num_iterations):
        dist.barrier()
        
        if rank == 0:
            torch.cuda.synchronize()
            start = time.perf_counter()
            dist.send(tensor, dst=1)
            torch.cuda.synchronize()
            end = time.perf_counter()
            
            if i >= warmup:
                times.append((end - start) * 1000)  # ms
        else:
            dist.recv(tensor, src=0)
    
    if rank == 0:
        import statistics
        result = {
            "shape": tensor_size,
            "dtype": str(dtype),
            "total_mb": total_mb,
            "num_iterations": num_iterations,
            "mean_ms": statistics.mean(times),
            "min_ms": min(times),
            "max_ms": max(times),
            "std_ms": statistics.stdev(times) if len(times) > 1 else 0,
            "bandwidth_gbps": (total_mb / statistics.mean(times)) * 1000 * 8 / 1024,
        }
        return result
    return None


def main():
    parser = argparse.ArgumentParser(description="测量 GPU 间传输时间")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=2048)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()
    
    # 初始化分布式环境
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    
    if rank == 0:
        print("="*60)
        print("GPU 间数据传输时间测量")
        print("="*60)
        print()
    
    # 测量不同配置
    configs = [
        # (batch, seq_len, hidden) - 模拟不同的 micro-batch 大小
        (1, 128, 2048),   # 小 batch
        (2, 128, 2048),   # 中 batch  
        (4, 128, 2048),   # 大 batch
        (args.batch_size, args.seq_len, args.hidden_dim),  # 自定义
    ]
    
    # 去重
    configs = list(dict.fromkeys(configs))
    
    results = []
    for config in configs:
        result = measure_transfer_time(
            tensor_size=config,
            num_iterations=args.iterations,
            warmup=args.warmup,
        )
        
        if rank == 0 and result:
            results.append(result)
            print(f"Shape: {result['shape']}")
            print(f"  Data size: {result['total_mb']:.2f} MB")
            print(f"  Transfer time: {result['mean_ms']:.3f} ± {result['std_ms']:.3f} ms")
            print(f"  Bandwidth: {result['bandwidth_gbps']:.2f} Gb/s")
            print()
    
    dist.destroy_process_group()
    
    if rank == 0:
        # 总结
        print("="*60)
        print("Summary")
        print("="*60)
        print()
        print(f"{'Shape':<20} {'Size (MB)':<12} {'Time (ms)':<15} {'BW (Gb/s)':<12}")
        print("-"*60)
        for r in results:
            print(f"{str(r['shape']):<20} {r['total_mb']:<12.2f} {r['mean_ms']:<15.3f} {r['bandwidth_gbps']:<12.2f}")


if __name__ == "__main__":
    main()
