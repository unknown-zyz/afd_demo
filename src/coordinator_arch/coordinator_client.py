"""
Coordinator gRPC client for AttentionWorker and FFNWorker.

When `grpc` and the generated proto stubs are importable, uses real gRPC.
Otherwise falls back to a no-op stub (useful for unit tests on macOS without
the toolchain).
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

try:
    import grpc
    from .proto import coordinator_pb2 as _pb
    from .proto import coordinator_pb2_grpc as _pb_grpc
    _HAS_GRPC = True
except Exception as _exc:  # pragma: no cover
    _HAS_GRPC = False
    logger.info("gRPC stubs not available (%s); CoordinatorClient runs in stub mode", _exc)


class CoordinatorClient:
    """Real gRPC client when available; stub otherwise.

    All methods are safe to call when the coordinator is unreachable —
    they log and return sensible defaults so workers can keep running on
    cached routing tables.
    """

    def __init__(self, addr: str, connect_timeout_s: float = 2.0):
        self.addr = addr
        self._lock = threading.Lock()
        self._channel = None
        self._stub = None
        self._subscribe_thread: Optional[threading.Thread] = None
        self._subscribe_stop = threading.Event()

        if not _HAS_GRPC:
            logger.info("CoordinatorClient stub mode: %s", addr)
            return
        try:
            self._channel = grpc.insecure_channel(addr)
            self._stub = _pb_grpc.CoordinatorStub(self._channel)
            # Probe with a lightweight unary call; tolerate failure (offline mode).
            try:
                self._stub.GetRoutingTable(_pb.Empty(), timeout=connect_timeout_s)
                logger.info("CoordinatorClient connected: %s", addr)
            except Exception as probe_e:
                logger.warning(
                    "CoordinatorClient probe failed for %s: %s (will retry on first real call)",
                    addr, probe_e,
                )
        except Exception as e:
            logger.warning("CoordinatorClient failed to create channel %s: %s", addr, e)
            self._stub = None

    # ---------------- registration ----------------
    def register_worker(self, info: Dict[str, Any]) -> Dict[str, Any]:
        if self._stub is None:
            return {"success": False, "msg": "offline", "initial_table_version": 0}
        try:
            req = _pb.WorkerInfo(
                role=info.get("role", ""),
                rank=int(info.get("rank", 0)),
                host=info.get("host", ""),
                device_id=int(info.get("device_id", 0)),
                world_size=int(info.get("world_size", 0)),
                local_experts=list(info.get("local_experts", []) or []),
                deepep_endpoint=info.get("deepep_endpoint", ""),
            )
            ack = self._stub.RegisterWorker(req)
            return {
                "success": ack.accepted,
                "msg": ack.msg,
                "initial_table_version": ack.initial_table_version,
                "assigned_group": ack.assigned_group,
            }
        except Exception as e:
            logger.warning("register_worker RPC failed: %s", e)
            return {"success": False, "msg": str(e), "initial_table_version": 0}

    # ---------------- routing table ----------------
    def get_routing_table(self, timeout_s: Optional[float] = None) -> Optional[Dict[str, Any]]:
        if self._stub is None:
            return None
        try:
            r = self._stub.GetRoutingTable(_pb.Empty(), timeout=timeout_s)
            return {
                "version": r.version,
                "expert_to_rank": list(r.expert_to_rank),
                "mode": r.mode,
                "valid_from_us": r.valid_from_us,
            }
        except Exception as e:
            logger.warning("get_routing_table RPC failed: %s", e)
            return None

    def subscribe_routing_table(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Spawn background thread that calls callback(table_dict) on every push."""
        if self._stub is None:
            logger.info("subscribe_routing_table: no-op (offline)")
            return
        if self._subscribe_thread is not None and self._subscribe_thread.is_alive():
            logger.warning("subscribe_routing_table already running")
            return

        def _run():
            backoff = 0.5
            while not self._subscribe_stop.is_set():
                try:
                    for r in self._stub.SubscribeRoutingTable(_pb.Empty()):
                        if self._subscribe_stop.is_set():
                            return
                        callback({
                            "version": r.version,
                            "expert_to_rank": list(r.expert_to_rank),
                            "mode": r.mode,
                            "valid_from_us": r.valid_from_us,
                        })
                    backoff = 0.5
                except Exception as e:
                    logger.warning("subscribe stream error: %s; retry in %.1fs", e, backoff)
                    if self._subscribe_stop.wait(backoff):
                        return
                    backoff = min(backoff * 2, 5.0)

        t = threading.Thread(target=_run, name="coord-subscribe", daemon=True)
        t.start()
        self._subscribe_thread = t

    # ---------------- metrics ----------------
    def update_metrics(self, metrics: Dict[str, Any]) -> None:
        if self._stub is None:
            return
        try:
            req = _pb.WorkerMetrics(
                role=metrics.get("role", ""),
                rank=int(metrics.get("rank", 0)),
                queue_len_avg=float(metrics.get("queue_len_avg", 0.0)),
                dispatch_rate_tps=float(metrics.get("dispatch_rate_tps", 0.0)),
                cache_miss_rate=float(metrics.get("cache_miss_rate", 0.0)),
                timestamp_us=int(metrics.get("timestamp_us", time.time() * 1e6)),
                per_expert_load=list(metrics.get("per_expert_load", []) or []),
            )
            self._stub.UpdateMetrics(req)
        except Exception as e:
            logger.debug("update_metrics RPC failed: %s", e)

    # ---------------- lifecycle ----------------
    def close(self) -> None:
        self._subscribe_stop.set()
        if self._subscribe_thread is not None:
            self._subscribe_thread.join(timeout=2.0)
        if self._channel is not None:
            try:
                self._channel.close()
            except Exception:
                pass
            self._channel = None
        self._stub = None
        logger.info("CoordinatorClient closed: %s", self.addr)
