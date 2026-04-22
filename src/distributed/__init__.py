"""
Distributed initialization utilities for AFD Demo.

Handles multi-node setup with attention and FFN workers on separate nodes.
"""

import os
import logging
from typing import Optional, Tuple
from dataclasses import dataclass

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


@dataclass
class DistributedConfig:
    """Configuration for distributed setup."""
    world_size: int = 2
    rank: int = 0
    local_rank: int = 0
    backend: str = "nccl"
    init_method: str = "env://"
    master_addr: Optional[str] = None
    master_port: Optional[str] = None
    
    # Role assignment
    attn_node_rank: int = 0
    ffn_node_rank: int = 1


class DistributedContext:
    """
    Manages distributed context for AFD.
    
    In a 2-node setup:
    - Node 0 (rank 0): Attention worker
    - Node 1 (rank 1): FFN worker
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self.config: Optional[DistributedConfig] = None
        self._initialized = False
        self._a2f_group = None   # ATT→FFN directional group
        self._f2a_group = None   # FFN→ATT directional group
        self._warmup_result: Optional[dict] = None
        self._keepalive = None
        
    def initialize(self, config: Optional[DistributedConfig] = None) -> None:
        """
        Initialize distributed environment.
        
        Args:
            config: Distributed configuration. If None, will be inferred from environment.
        """
        if self._initialized:
            logger.warning("Distributed context already initialized")
            return
            
        if config is None:
            config = self._config_from_env()
        self.config = config
        
        # Set environment variables (required for multi-machine setup)
        if config.master_addr:
            os.environ["MASTER_ADDR"] = config.master_addr
        if config.master_port:
            os.environ["MASTER_PORT"] = str(config.master_port)
        
        # Also set RANK and WORLD_SIZE for torch.distributed
        os.environ["RANK"] = str(config.rank)
        os.environ["WORLD_SIZE"] = str(config.world_size)
        
        logger.info(f"Initializing distributed: rank={config.rank}, "
                    f"master={config.master_addr}:{config.master_port}")
        
        # Set compute device before process group init.
        device_id = None
        from ..utils import device as devmod
        if devmod.is_available():
            devmod.set_device(config.local_rank)
            # device_id param to init_process_group is a torch.device; only
            # meaningful for nccl. HCCL on NPU uses torch.npu device implicitly.
            if config.backend == "nccl":
                device_id = torch.device(f"cuda:{config.local_rank}")
        
        # Initialize process group
        if not dist.is_initialized():
            dist.init_process_group(
                backend=config.backend,
                init_method=config.init_method,
                world_size=config.world_size,
                rank=config.rank,
                device_id=device_id,
            )
        
        # Create directional NCCL process groups lazily (only when DBO is used).
        # See a2f_group / f2a_group properties below.
        # Store comm_ranks for lazy creation.
        self._comm_ranks = [config.attn_node_rank, config.ffn_node_rank]
        
        self._initialized = True
        logger.info(
            f"Distributed initialized: rank={config.rank}, "
            f"world_size={config.world_size}, role={self.role}"
        )
    
    def _config_from_env(self) -> DistributedConfig:
        """Create config from environment variables."""
        return DistributedConfig(
            world_size=int(os.environ.get("WORLD_SIZE", 2)),
            rank=int(os.environ.get("RANK", 0)),
            local_rank=int(os.environ.get("LOCAL_RANK", 0)),
            backend=os.environ.get("BACKEND", "nccl"),
            init_method=os.environ.get("INIT_METHOD", "env://"),
            master_addr=os.environ.get("MASTER_ADDR"),
            master_port=os.environ.get("MASTER_PORT"),
        )
    
    @property
    def is_initialized(self) -> bool:
        return self._initialized
    
    @property
    def rank(self) -> int:
        assert self.config is not None
        return self.config.rank
    
    @property
    def world_size(self) -> int:
        assert self.config is not None
        return self.config.world_size
    
    @property
    def local_rank(self) -> int:
        assert self.config is not None
        return self.config.local_rank
    
    @property
    def device(self) -> torch.device:
        """Get the device for this rank (cuda / npu / cpu)."""
        from ..utils import device as devmod
        if devmod.DEVICE_TYPE in ("cuda", "npu"):
            return torch.device(f"{devmod.DEVICE_TYPE}:{self.local_rank}")
        return torch.device("cpu")
    
    @property
    def role(self) -> str:
        """Get the role of this node: 'attention' or 'ffn'."""
        assert self.config is not None
        if self.config.rank == self.config.attn_node_rank:
            return "attention"
        elif self.config.rank == self.config.ffn_node_rank:
            return "ffn"
        else:
            return "unknown"
    
    @property
    def is_attention_node(self) -> bool:
        return self.role == "attention"
    
    @property
    def is_ffn_node(self) -> bool:
        return self.role == "ffn"
    
    @property
    def peer_rank(self) -> int:
        """Get the rank of the peer node."""
        assert self.config is not None
        if self.is_attention_node:
            return self.config.ffn_node_rank
        else:
            return self.config.attn_node_rank
    
    @property
    def a2f_group(self):
        """Get the ATT→FFN directional NCCL group (lazy init)."""
        if self._a2f_group is None:
            self._init_directional_groups()
        return self._a2f_group
    
    @property
    def f2a_group(self):
        """Get the FFN→ATT directional NCCL group (lazy init)."""
        if self._f2a_group is None:
            self._init_directional_groups()
        return self._f2a_group
    
    def _init_directional_groups(self):
        """Create directional NCCL groups for cross-layer pipelining.
        
        Called lazily on first access to a2f_group or f2a_group.
        Directional groups use separate NCCL communicators to break FIFO:
          - a2f_group: ATT isend → FFN irecv (Attention-to-FFN direction)
          - f2a_group: FFN isend → ATT irecv (FFN-to-Attention direction)
        """
        logger.info("Creating directional NCCL groups (a2f, f2a)...")
        self._a2f_group = dist.new_group(ranks=self._comm_ranks)
        self._f2a_group = dist.new_group(ranks=self._comm_ranks)
        # Warm up the new groups to avoid cold-start latency
        from .warmup import warmup_p2p
        warmup_p2p(
            self.peer_rank, self.device, num_rounds=3,
            extra_groups=[self._a2f_group, self._f2a_group],
        )
        logger.info("Directional NCCL groups created and warmed up")
    
    def warmup(self, num_rounds=3, keepalive=False, keepalive_interval=0.5):
        """预热 P2P 通道并可选启动保活。"""
        from .warmup import warmup_p2p, P2PKeepalive

        # Only warm up directional groups if they've been created
        extra = []
        if self._a2f_group is not None and self._f2a_group is not None:
            extra = [self._a2f_group, self._f2a_group]

        result = warmup_p2p(
            self.peer_rank, self.device, num_rounds=num_rounds,
            extra_groups=extra,
        )
        self._warmup_result = result

        if keepalive:
            self._keepalive = P2PKeepalive(
                self.peer_rank, self.device, interval_s=keepalive_interval
            )
            self._keepalive.start()
        return result

    def barrier(self) -> None:
        """Synchronize all processes."""
        if dist.is_initialized():
            dist.barrier()
    
    def cleanup(self) -> None:
        """Cleanup distributed resources."""
        if self._keepalive is not None:
            self._keepalive.stop()
            self._keepalive = None
        if dist.is_initialized():
            backend = dist.get_backend()
            # Workaround: PyTorch 2.7 + NCCL 2.26 may abort on explicit destroy_process_group()
            # in some environments (ncclProxyDestroy refCount assertion). Let process exit naturally.
            if backend == "nccl":
                logger.warning(
                    "Skip destroy_process_group() for NCCL to avoid known teardown crash; "
                    "resources will be released on process exit."
                )
            else:
                dist.destroy_process_group()
            self._a2f_group = None
            self._f2a_group = None
        self._initialized = False
        logger.info("Distributed context cleaned up")


def get_distributed_context() -> DistributedContext:
    """Get the singleton distributed context."""
    return DistributedContext()


def init_distributed(
    rank: int,
    world_size: int,
    master_addr: str = "localhost",
    master_port: str = "29500",
    backend: str = "nccl",
) -> DistributedContext:
    """
    Convenience function to initialize distributed context.
    
    Args:
        rank: Global rank of this process
        world_size: Total number of processes
        master_addr: Master node address
        master_port: Master node port
        backend: Communication backend (nccl, gloo, etc.)
    
    Returns:
        Initialized DistributedContext
    """
    config = DistributedConfig(
        world_size=world_size,
        rank=rank,
        local_rank=rank,  # Simplified for 2-node case
        backend=backend,
        init_method=f"tcp://{master_addr}:{master_port}",
        master_addr=master_addr,
        master_port=master_port,
    )
    
    ctx = get_distributed_context()
    ctx.initialize(config)
    return ctx
