"""NCCL P2P warmup utilities."""

import time
import logging
import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


def warmup_p2p(
    peer_rank: int,
    device: torch.device,
    num_rounds: int = 3,
    tensor_size: int = 1024,  # bytes
    dtype: torch.dtype = torch.float16,
    extra_groups: list = None,
) -> dict:
    """
    在初始化阶段预热 NCCL P2P 通道。

    对 peer_rank 执行 num_rounds 轮双向 isend/irecv，
    强制 NCCL proxy thread 启动并建立 P2P 通道。

    Args:
        extra_groups: Additional NCCL process groups to warm up (e.g., a2f/f2a groups).

    Returns:
        dict: {
            'cold_latency_ms': float,  # 首次通信延迟
            'warm_latency_ms': float,  # 预热后平均延迟
            'num_rounds': int,
        }
    """
    elem_size = torch.tensor([], dtype=dtype).element_size()
    num_elements = max(1, tensor_size // elem_size)

    latencies = []
    for i in range(num_rounds):
        send_tensor = torch.ones(num_elements, dtype=dtype, device=device)
        recv_tensor = torch.empty(num_elements, dtype=dtype, device=device)

        torch.cuda.synchronize()
        start = time.perf_counter()

        rank = dist.get_rank()
        if rank < peer_rank:
            handle_s = dist.isend(send_tensor, dst=peer_rank)
            handle_r = dist.irecv(recv_tensor, src=peer_rank)
        else:
            handle_r = dist.irecv(recv_tensor, src=peer_rank)
            handle_s = dist.isend(send_tensor, dst=peer_rank)

        handle_s.wait()
        handle_r.wait()
        torch.cuda.synchronize()

        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies.append(elapsed_ms)
        logger.info(f"[Warmup] round {i}: {elapsed_ms:.3f}ms")

    result = {
        'cold_latency_ms': latencies[0] if latencies else 0,
        'warm_latency_ms': (
            sum(latencies[1:]) / max(1, len(latencies) - 1)
            if len(latencies) > 1 else 0
        ),
        'num_rounds': num_rounds,
        'all_latencies_ms': latencies,
    }
    logger.info(
        f"[Warmup] cold={result['cold_latency_ms']:.3f}ms, "
        f"warm={result['warm_latency_ms']:.3f}ms"
    )

    # Warm up directional groups (a2f, f2a) to avoid cold-start on pipeline traffic
    if extra_groups:
        for gi, group in enumerate(extra_groups):
            for r in range(max(2, num_rounds)):
                s = torch.ones(num_elements, dtype=dtype, device=device)
                rv = torch.empty(num_elements, dtype=dtype, device=device)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                if rank < peer_rank:
                    hs = dist.isend(s, dst=peer_rank, group=group)
                    hr = dist.irecv(rv, src=peer_rank, group=group)
                else:
                    hr = dist.irecv(rv, src=peer_rank, group=group)
                    hs = dist.isend(s, dst=peer_rank, group=group)
                hs.wait()
                hr.wait()
                torch.cuda.synchronize()
                ms = (time.perf_counter() - t0) * 1000
                logger.info(f"[Warmup] group[{gi}] round {r}: {ms:.3f}ms")

    return result
