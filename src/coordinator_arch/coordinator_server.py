"""
Coordinator gRPC server (asyncio).

Implements the `Coordinator` service defined in `proto/coordinator.proto`:
- RegisterWorker: workers register on startup, get initial routing table version.
- GetRoutingTable: one-shot pull.
- SubscribeRoutingTable: server-side stream; pushes whenever version bumps.
- UpdateMetrics: workers push metrics; triggers rebalance computation.

Rebalance algorithm lives in `router.py` (LoadAwareRouter).

See doc/12-coordinator-arch.md §3, §4, §6 for design.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import grpc

from .proto import coordinator_pb2 as pb
from .proto import coordinator_pb2_grpc as pb_grpc
from .router import LoadAwareRouter

logger = logging.getLogger(__name__)


@dataclass
class _WorkerEntry:
    info: pb.WorkerInfo
    last_metrics: Optional[pb.WorkerMetrics] = None
    last_seen_us: int = field(default_factory=lambda: int(time.time() * 1e6))


class CoordinatorServicer(pb_grpc.CoordinatorServicer):
    """In-memory coordinator implementation."""

    def __init__(
        self,
        num_experts: int,
        attn_world: int,
        ffn_world: int,
        initial_mode: str = "low_latency",
        stale_threshold_s: float = 5.0,
        min_rebalance_interval_ms: float = 100.0,
        imbalance_threshold: float = 0.10,
    ):
        self.num_experts = num_experts
        self.attn_world = attn_world
        self.ffn_world = ffn_world
        self.stale_threshold_s = stale_threshold_s
        self.min_rebalance_interval_s = min_rebalance_interval_ms / 1000.0
        self.imbalance_threshold = imbalance_threshold

        # registry: (role, rank) -> entry
        self._workers: Dict[Tuple[str, int], _WorkerEntry] = {}
        self._workers_lock = asyncio.Lock()

        # routing state
        uniform = [
            (e * ffn_world) // num_experts for e in range(num_experts)
        ]
        self._table = pb.RoutingTable(
            version=1,
            expert_to_rank=uniform,
            mode=initial_mode,
            valid_from_us=int(time.time() * 1e6),
        )
        self._table_lock = asyncio.Lock()
        self._table_cv = asyncio.Condition()

        self._router = LoadAwareRouter(
            num_experts=num_experts,
            ffn_world=ffn_world,
            attn_world=attn_world,
            imbalance_threshold=imbalance_threshold,
        )
        self._last_rebalance_ts = 0.0

    # ---------- RPC handlers ----------
    async def RegisterWorker(self, request: pb.WorkerInfo, context) -> pb.RegisterAck:
        key = (request.role, request.rank)
        async with self._workers_lock:
            self._workers[key] = _WorkerEntry(info=request)
        logger.info(
            "RegisterWorker role=%s rank=%d host=%s dev=%d (total=%d)",
            request.role, request.rank, request.host, request.device_id, len(self._workers),
        )
        async with self._table_lock:
            version = self._table.version
        return pb.RegisterAck(
            initial_table_version=version,
            assigned_group="default",
            accepted=True,
            msg="ok",
        )

    async def GetRoutingTable(self, request: pb.Empty, context) -> pb.RoutingTable:
        async with self._table_lock:
            return pb.RoutingTable(
                version=self._table.version,
                expert_to_rank=list(self._table.expert_to_rank),
                mode=self._table.mode,
                valid_from_us=self._table.valid_from_us,
            )

    async def SubscribeRoutingTable(self, request: pb.Empty, context):
        last_version = -1
        try:
            while True:
                async with self._table_cv:
                    await self._table_cv.wait_for(
                        lambda: self._table.version != last_version
                    )
                    table = pb.RoutingTable(
                        version=self._table.version,
                        expert_to_rank=list(self._table.expert_to_rank),
                        mode=self._table.mode,
                        valid_from_us=self._table.valid_from_us,
                    )
                last_version = table.version
                yield table
        except asyncio.CancelledError:
            return

    async def UpdateMetrics(self, request: pb.WorkerMetrics, context) -> pb.Ack:
        key = (request.role, request.rank)
        async with self._workers_lock:
            entry = self._workers.get(key)
            if entry is None:
                return pb.Ack(ok=False, msg=f"unknown worker {key}")
            entry.last_metrics = request
            entry.last_seen_us = int(time.time() * 1e6)
        # Try rebalance (throttled, lock-free read).
        now = time.time()
        if now - self._last_rebalance_ts >= self.min_rebalance_interval_s:
            self._last_rebalance_ts = now
            asyncio.create_task(self._maybe_rebalance())
        return pb.Ack(ok=True, msg="")

    # ---------- helpers ----------
    async def _maybe_rebalance(self) -> None:
        async with self._workers_lock:
            ffn_metrics = [
                e.last_metrics
                for e in self._workers.values()
                if e.info.role == "ffn" and e.last_metrics is not None
            ]
        if len(ffn_metrics) < self.ffn_world:
            return  # not all workers reporting yet
        async with self._table_lock:
            prev_table = list(self._table.expert_to_rank)
        new_table = self._router.rebalance(ffn_metrics, prev_table)
        if new_table is None:
            return
        async with self._table_lock:
            self._table = pb.RoutingTable(
                version=self._table.version + 1,
                expert_to_rank=new_table,
                mode=self._table.mode,
                valid_from_us=int(time.time() * 1e6),
            )
            logger.info(
                "RoutingTable v=%d pushed (mode=%s)",
                self._table.version, self._table.mode,
            )
        async with self._table_cv:
            self._table_cv.notify_all()

    async def set_mode(self, mode: str) -> None:
        """Programmatically switch mode (called by admin endpoint or test)."""
        async with self._table_lock:
            self._table = pb.RoutingTable(
                version=self._table.version + 1,
                expert_to_rank=list(self._table.expert_to_rank),
                mode=mode,
                valid_from_us=int(time.time() * 1e6),
            )
        async with self._table_cv:
            self._table_cv.notify_all()

    async def sweep_stale_workers(self) -> None:
        """Background loop: remove workers with stale metrics."""
        while True:
            await asyncio.sleep(self.stale_threshold_s / 2)
            now_us = int(time.time() * 1e6)
            removed = []
            async with self._workers_lock:
                for key in list(self._workers.keys()):
                    e = self._workers[key]
                    if (now_us - e.last_seen_us) / 1e6 > self.stale_threshold_s:
                        del self._workers[key]
                        removed.append(key)
            if removed:
                logger.warning("Sweep removed stale workers: %s", removed)
                # Trigger rebalance with whoever's left.
                await self._maybe_rebalance()


async def serve(
    bind: str,
    num_experts: int,
    attn_world: int,
    ffn_world: int,
    mode: str = "low_latency",
) -> Tuple[grpc.aio.Server, CoordinatorServicer]:
    server = grpc.aio.server()
    svc = CoordinatorServicer(
        num_experts=num_experts,
        attn_world=attn_world,
        ffn_world=ffn_world,
        initial_mode=mode,
    )
    pb_grpc.add_CoordinatorServicer_to_server(svc, server)
    server.add_insecure_port(bind)
    await server.start()
    logger.info("Coordinator listening on %s", bind)
    asyncio.create_task(svc.sweep_stale_workers())
    return server, svc


def _parse_args():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--bind", default="0.0.0.0:50051")
    p.add_argument("--num-experts", type=int, required=True)
    p.add_argument("--attn-world", type=int, required=True)
    p.add_argument("--ffn-world", type=int, required=True)
    p.add_argument("--mode", choices=["normal", "low_latency"], default="low_latency")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


async def _main_async():
    args = _parse_args()
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    server, _svc = await serve(
        bind=args.bind,
        num_experts=args.num_experts,
        attn_world=args.attn_world,
        ffn_world=args.ffn_world,
        mode=args.mode,
    )
    await server.wait_for_termination()


def main():
    asyncio.run(_main_async())


if __name__ == "__main__":
    main()
