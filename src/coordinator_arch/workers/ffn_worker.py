"""
FFN Worker entrypoint for Coordinator-based MoE architecture.

This worker handles feed-forward network computation with dynamic routing
and expert parallelism via DeepEP-Ascend or fallback torch.distributed.
"""

import argparse
import logging
import sys
from typing import Dict, List, Optional

import torch
import torch.distributed as dist

from ..coordinator_client import CoordinatorClient
from ..comm.factory import build_communicator
from ..batching.ffn_queue import FFNQueue

logger = logging.getLogger(__name__)


def _get_dev_module():
    """Get device module (torch_npu or stub) for NPU operations."""
    try:
        import torch_npu
        return torch_npu
    except ImportError:
        logger.warning("torch_npu not available, using CPU stub")
        
        class StubModule:
            @staticmethod
            def npu_grouped_matmul(*args, **kwargs):
                raise NotImplementedError("torch_npu not available")
        
        return StubModule()


class FFNWorker:
    """
    FFN Worker for dynamic MoE architecture.
    
    Responsibilities:
    - Register with Coordinator
    - Maintain local expert weights (skeleton: identity FFN)
    - Receive dispatched tokens via communicator
    - Process FFN computation
    - Return results via combine() path
    """
    
    def __init__(self, args):
        """
        Initialize FFN Worker.
        
        Args:
            args: Parsed command-line arguments
        """
        self.args = args
        self.device = self._init_device()
        self.coord = CoordinatorClient(args.coord_addr)
        
        if not args.no_init_dist:
            self._init_distributed()
        
        self.ep_group = self._build_ep_group() if not args.no_init_dist else None
        
        if args.no_init_dist:
            self.comm = None
            logger.warning("Communicator not built (--no-init-dist)")
        else:
            self.comm = build_communicator(
                prefer_deepep=not args.use_fallback,
                ep_group=self.ep_group,
                hidden_size=args.hidden_size,
                num_experts=args.num_experts,
                max_tokens_per_rank=args.max_batch * 8,
                device=self.device,
                mode=args.mode,
            )
        
        self.routing_table = self._fetch_initial_routing_table()
        if self.comm is not None:
            self.comm.update_routing_table(self.routing_table)
        
        self.queue = FFNQueue(
            max_batch=args.max_batch,
            max_wait_ms=args.max_wait_ms,
        )
        
        # SKELETON: Identity FFN, no actual weights loaded
        self.local_expert_ids = self._compute_local_experts()
        
        logger.info(
            f"FFNWorker initialized: rank={args.rank}, world={args.world}, "
            f"local_experts={self.local_expert_ids}, device={self.device}"
        )
    
    def _init_device(self) -> torch.device:
        """Initialize device based on device_id argument."""
        if self.args.device_id == -1:
            device = torch.device("cpu")
            logger.info("Using CPU device")
        else:
            dev_module = _get_dev_module()
            if hasattr(dev_module, 'npu'):
                device = torch.device(f"npu:{self.args.device_id}")
                logger.info(f"Using NPU device: {device}")
            else:
                # Fallback to CUDA if available
                if torch.cuda.is_available():
                    device = torch.device(f"cuda:{self.args.device_id}")
                    logger.info(f"Using CUDA device: {device}")
                else:
                    device = torch.device("cpu")
                    logger.warning(
                        f"Device {self.args.device_id} requested but NPU/CUDA "
                        f"not available, using CPU"
                    )
        return device
    
    def _init_distributed(self):
        """Initialize torch.distributed backend."""
        if dist.is_initialized():
            logger.info("torch.distributed already initialized")
            return
        
        backend = "hccl" if self.device.type == "npu" else "nccl"
        
        logger.info(
            f"Initializing distributed: backend={backend}, "
            f"rank={self.args.rank}, world_size={self.args.world}"
        )
        
        dist.init_process_group(
            backend=backend,
            init_method=f"tcp://{self.args.master_addr}:{self.args.master_port}",
            world_size=self.args.world,
            rank=self.args.rank,
        )
        
        logger.info(f"Distributed initialized: rank={dist.get_rank()}")
    
    def _build_ep_group(self) -> dist.ProcessGroup:
        """Build expert parallel process group."""
        if not dist.is_initialized():
            raise RuntimeError("torch.distributed not initialized")
        
        # For FFN workers, the EP group is all FFN ranks
        ranks = list(range(self.args.world))
        ep_group = dist.new_group(ranks)
        
        logger.info(f"EP group created with ranks: {ranks}")
        return ep_group
    
    def _fetch_initial_routing_table(self) -> Dict:
        """
        Fetch initial routing table from coordinator.
        
        Returns uniform distribution as default.
        """
        # Try to get from coordinator (stub will return empty)
        table = self.coord.get_routing_table()
        
        # If empty, create uniform default
        if not table.get("expert_to_rank"):
            expert_to_rank = [
                i % self.args.world for i in range(self.args.num_experts)
            ]
            table = {
                "version": 0,
                "expert_to_rank": expert_to_rank,
                "mode": self.args.mode,
            }
            logger.info(
                f"Using uniform routing table: {self.args.num_experts} experts "
                f"distributed across {self.args.world} ranks"
            )
        
        return table
    
    def _compute_local_experts(self) -> List[int]:
        """
        Compute which experts this rank owns based on routing table.
        
        Returns:
            List of expert IDs assigned to this rank
        """
        expert_to_rank = self.routing_table["expert_to_rank"]
        local_experts = [
            expert_id
            for expert_id, rank in enumerate(expert_to_rank)
            if rank == self.args.rank
        ]
        
        logger.debug(
            f"Rank {self.args.rank} owns {len(local_experts)} experts: "
            f"{local_experts[:10]}{'...' if len(local_experts) > 10 else ''}"
        )
        
        return local_experts
    
    def register_with_coordinator(self):
        """Register this worker with the coordinator."""
        info = {
            "worker_type": "ffn",
            "rank": self.args.rank,
            "world_size": self.args.world,
            "device_info": str(self.device),
            "num_local_experts": len(self.local_expert_ids),
            "local_expert_ids": self.local_expert_ids[:10],  # First 10 for logging
        }
        
        response = self.coord.register_worker(info)
        logger.info(f"Registered with coordinator: {response}")
    
    def run_once(self, dispatch_handle: Dict) -> torch.Tensor:
        """
        Testing API: process one dispatch handle and return FFN outputs.
        
        For testing the skeleton without a full serve loop. Takes a handle
        from AttentionWorker's dispatch() and produces ffn_outputs to pass
        back to combine().
        
        Args:
            dispatch_handle: Dictionary from comm.dispatch() containing:
                - recv_hidden: Dispatched tokens [M, H]
                - topk_indices: Top-k expert indices
                - topk_weights: Top-k expert weights
        
        Returns:
            FFN outputs tensor [M, H]
        """
        recv_hidden = dispatch_handle["recv_hidden"]
        
        # SKELETON: Identity function (no actual FFN computation)
        ffn_outputs = self._forward_ffn(
            hidden=recv_hidden,
            topk_indices=dispatch_handle.get("topk_indices"),
        )
        
        return ffn_outputs
    
    def _forward_ffn(
        self, hidden: torch.Tensor, topk_indices: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        Execute FFN computation on local experts.
        
        SKELETON: Returns identity (input = output).
        Real implementation would use torch_npu.npu_grouped_matmul over
        local_expert_ids.
        
        Args:
            hidden: Input hidden states [M, H]
            topk_indices: Expert indices (unused in skeleton)
        
        Returns:
            FFN outputs [M, H]
        """
        # SKELETON: Identity operation
        # Real implementation:
        # 1. Group tokens by local expert ID
        # 2. Call npu_grouped_matmul with expert weights
        # 3. Apply activation and combine
        return hidden.clone()
    
    def serve_forever(self):
        """Main serving loop (skeleton: heartbeat-only).

        Periodically pushes a metrics heartbeat to the coordinator so the
        worker isn't reaped by the stale-sweep. Real batch processing will
        be wired in once the dispatch/combine path is enabled.
        """
        logger.info("FFNWorker.serve_forever() - heartbeat-only skeleton")
        import time
        while True:
            time.sleep(2.0)
            if self.coord is not None:
                self.coord.update_metrics({
                    "role": "ffn",
                    "rank": self.args.rank,
                    "queue_len_avg": float(getattr(self.queue, "avg_len", 0.0))
                    if hasattr(self, "queue") else 0.0,
                    "timestamp_us": int(time.time() * 1e6),
                })
    
    def shutdown(self):
        """Cleanup resources."""
        logger.info("FFNWorker shutting down")
        
        if hasattr(self, 'ep_group') and self.ep_group is not None:
            # Process group cleanup handled by torch.distributed
            pass
        
        if dist.is_initialized():
            dist.destroy_process_group()
            logger.info("Distributed process group destroyed")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="FFN Worker for Coordinator-based MoE"
    )
    
    # Coordinator
    parser.add_argument(
        "--coord-addr",
        type=str,
        required=True,
        help="Coordinator address (host:port)",
    )
    
    # Distributed
    parser.add_argument("--rank", type=int, required=True, help="Global FFN rank")
    parser.add_argument(
        "--world", type=int, required=True, help="FFN EP world size"
    )
    parser.add_argument(
        "--attn-world", type=int, required=True, help="ATTN DP world size"
    )
    parser.add_argument(
        "--master-addr", type=str, default="localhost", help="Master address"
    )
    parser.add_argument(
        "--master-port", type=int, default=29500, help="Master port"
    )
    
    # Model
    parser.add_argument(
        "--model-path",
        type=str,
        default="/models/Qwen3-30B-A3B",
        help="Path to model (e.g., /models/Qwen3-30B-A3B)",
    )
    parser.add_argument(
        "--hidden-size", type=int, default=2048, help="Hidden dimension size"
    )
    parser.add_argument(
        "--num-experts", type=int, default=128, help="Total number of experts"
    )
    parser.add_argument(
        "--topk", type=int, default=8, help="Top-K experts per token"
    )
    
    # Batching
    parser.add_argument(
        "--max-batch", type=int, default=64, help="Maximum batch size"
    )
    parser.add_argument(
        "--max-wait-ms", type=float, default=5.0, help="Maximum wait time in ms"
    )
    
    # Communication
    parser.add_argument(
        "--mode",
        type=str,
        choices=["normal", "low_latency"],
        default="normal",
        help="Communication mode",
    )
    parser.add_argument(
        "--use-fallback",
        action="store_true",
        help="Use fallback torch.distributed instead of DeepEP",
    )
    
    # Device
    parser.add_argument(
        "--device-id", type=int, default=0, help="Local device ID (-1 for CPU)"
    )
    
    # Testing
    parser.add_argument(
        "--no-init-dist",
        action="store_true",
        help="Skip distributed initialization (for testing)",
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
    )
    
    args = parse_args()
    
    logger.info(f"Starting FFN Worker with args: {args}")
    
    worker = FFNWorker(args)
    worker.register_with_coordinator()
    
    try:
        worker.serve_forever()
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    finally:
        worker.shutdown()


if __name__ == "__main__":
    main()
