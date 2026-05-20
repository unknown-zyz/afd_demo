"""
Coordinator gRPC client stub for AttentionWorker and FFNWorker.

This is a placeholder implementation that logs calls. The real gRPC integration
will be implemented once the proto schema task (p1-proto-schema) lands.
"""

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class CoordinatorClient:
    """
    Stub gRPC client for communicating with the Coordinator server.
    
    This placeholder logs all method calls and returns mock data. It will be
    replaced with real gRPC implementation once proto schema is available.
    """
    
    def __init__(self, addr: str):
        """
        Initialize coordinator client stub.
        
        Args:
            addr: Coordinator address in format "host:port"
        """
        self.addr = addr
        self._connected = False
        logger.info(f"CoordinatorClient stub initialized with addr={addr}")
        
        # Try to parse addr for basic validation
        try:
            host, port = addr.split(":")
            port = int(port)
            self._connected = True
            logger.debug(f"Parsed coordinator address: host={host}, port={port}")
        except (ValueError, AttributeError) as e:
            logger.warning(f"Failed to parse coordinator address '{addr}': {e}")
            self._connected = False
    
    def register_worker(self, info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Register worker with the coordinator.
        
        Args:
            info: Worker information dictionary containing:
                - rank: int, global rank
                - role: str, "attn" or "ffn"
                - host: str, hostname
                - device_id: int, local device index
                - capabilities: dict, optional metadata
        
        Returns:
            Acknowledgment dictionary with:
                - success: bool
                - message: str
        """
        logger.info(
            f"[STUB] register_worker called: rank={info.get('rank')}, "
            f"role={info.get('role')}, addr={self.addr}"
        )
        
        if not self._connected:
            logger.warning("Coordinator not connected, registration skipped")
            return {"success": False, "message": "Coordinator unreachable"}
        
        return {
            "success": True,
            "message": f"Worker rank={info.get('rank')} registered (stub)"
        }
    
    def get_routing_table(self) -> Optional[Dict[str, Any]]:
        """
        Fetch the current routing table from coordinator.
        
        Returns:
            Routing table dictionary with:
                - version: int
                - expert_to_rank: List[int]
                - mode: str
            Returns None if coordinator is unreachable.
        """
        logger.debug(f"[STUB] get_routing_table called: addr={self.addr}")
        
        if not self._connected:
            logger.warning("Coordinator not connected, returning None")
            return None
        
        # Return stub routing table
        return {
            "version": 0,
            "expert_to_rank": [],  # Will be filled by caller
            "mode": "normal"
        }
    
    def update_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Send worker metrics to coordinator.
        
        Args:
            metrics: Metrics dictionary containing:
                - rank: int
                - timestamp: float
                - tokens_processed: int
                - queue_depth: int (for FFN workers)
                - latency_p50_ms: float
                - latency_p99_ms: float
                - custom: dict, optional additional metrics
        """
        logger.debug(
            f"[STUB] update_metrics called: rank={metrics.get('rank')}, "
            f"tokens={metrics.get('tokens_processed')}"
        )
        
        if not self._connected:
            logger.debug("Coordinator not connected, metrics not sent")
            return
        
        # Stub: just log
        logger.debug(f"Metrics would be sent to {self.addr}: {metrics}")
    
    def close(self) -> None:
        """Close gRPC connection."""
        logger.info(f"[STUB] Closing coordinator client: addr={self.addr}")
        self._connected = False
