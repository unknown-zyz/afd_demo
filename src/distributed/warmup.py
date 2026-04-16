"""NCCL P2P 预热与保活模块。"""

import time
import threading
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


class P2PKeepalive:
    """
    保活线程：定期发送心跳防止 NCCL proxy thread 休眠。

    如果通信间隔超过 interval_s，自动发送小消息保持通道活跃。

    Thread-safety: Use pause()/resume() around inference to prevent
    heartbeat NCCL ops from conflicting with scheduler NCCL ops.
    pause() blocks until any in-flight heartbeat completes.
    """

    def __init__(
        self,
        peer_rank: int,
        device: torch.device,
        interval_s: float = 0.5,
        tensor_size: int = 64,  # bytes
        dtype: torch.dtype = torch.float16,
    ):
        self.peer_rank = peer_rank
        self.device = device
        self.interval_s = interval_s
        self.dtype = dtype

        elem_size = torch.tensor([], dtype=dtype).element_size()
        self.num_elements = max(1, tensor_size // elem_size)

        self._last_comm_time = time.monotonic()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._heartbeat_count = 0
        self._lock = threading.Lock()
        # Gate lock: heartbeat acquires during NCCL ops; pause() acquires to
        # block until in-flight heartbeat finishes, then holds to prevent new ones.
        self._gate = threading.Lock()
        self._paused = False

    def start(self):
        """启动保活线程。"""
        self._stop_event.clear()
        self._paused = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info(
            f"[Keepalive] started: peer={self.peer_rank}, interval={self.interval_s}s"
        )

    def stop(self):
        """停止保活线程。"""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5.0)
        logger.info(f"[Keepalive] stopped: sent {self._heartbeat_count} heartbeats")

    def pause(self):
        """Pause heartbeats. Blocks until any in-flight heartbeat completes."""
        self._gate.acquire()
        self._paused = True

    def resume(self):
        """Resume heartbeats after inference."""
        with self._lock:
            self._last_comm_time = time.monotonic()
        self._paused = False
        try:
            self._gate.release()
        except RuntimeError:
            pass

    def notify_comm(self):
        """通知保活线程：刚刚发生了一次正常通信。"""
        with self._lock:
            self._last_comm_time = time.monotonic()

    def _run(self):
        """后台线程：检查是否需要发送心跳。"""
        heartbeat_tag = 99999
        rank = dist.get_rank()

        while not self._stop_event.is_set():
            self._stop_event.wait(self.interval_s / 2)

            if self._paused or self._stop_event.is_set():
                continue

            with self._lock:
                idle_time = time.monotonic() - self._last_comm_time

            if idle_time >= self.interval_s and not self._stop_event.is_set():
                if not self._gate.acquire(timeout=0.01):
                    continue
                try:
                    if self._paused or self._stop_event.is_set():
                        continue

                    send_t = torch.zeros(
                        self.num_elements, dtype=self.dtype, device=self.device
                    )
                    recv_t = torch.empty(
                        self.num_elements, dtype=self.dtype, device=self.device
                    )

                    if rank < self.peer_rank:
                        hs = dist.isend(send_t, dst=self.peer_rank, tag=heartbeat_tag)
                        hr = dist.irecv(recv_t, src=self.peer_rank, tag=heartbeat_tag)
                    else:
                        hr = dist.irecv(recv_t, src=self.peer_rank, tag=heartbeat_tag)
                        hs = dist.isend(send_t, dst=self.peer_rank, tag=heartbeat_tag)

                    hs.wait()
                    hr.wait()

                    with self._lock:
                        self._last_comm_time = time.monotonic()
                        self._heartbeat_count += 1
                except Exception as e:
                    if not self._stop_event.is_set():
                        logger.warning(f"[Keepalive] heartbeat failed: {e}")
                finally:
                    self._gate.release()
