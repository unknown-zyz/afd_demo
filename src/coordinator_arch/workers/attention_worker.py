"""
AttentionWorker entrypoint for Coordinator-based MoE architecture.

This module implements the attention worker process that:
1. Runs attention computation on input tokens
2. Routes tokens to FFN workers via MoE communicator
3. Combines FFN outputs back to original token order
4. Registers with Coordinator for dynamic load balancing
"""

import argparse
import logging
import socket
from typing import Dict, List, Optional, Any

import torch

logger = logging.getLogger(__name__)


def _get_dev_module():
    """
    Get torch_npu module if available, otherwise return stub.
    
    Returns a module with set_device() and synchronize() methods.
    """
    try:
        import torch_npu
        return torch_npu.npu
    except ImportError:
        logger.debug("torch_npu not available, using stub NPU module")
        
        class StubNPU:
            @staticmethod
            def set_device(device_id):
                pass
            
            @staticmethod
            def synchronize():
                pass
        
        return StubNPU()


class AttentionWorker:
    """
    Attention worker for Coordinator-based MoE architecture.
    
    Handles:
    - Attention computation (skeleton: identity for now)
    - Token routing and dispatch to FFN workers
    - Micro-batching via ContinuousBatcher
    - Coordinator registration and metrics reporting
    """
    
    def __init__(self, args: argparse.Namespace):
        """
        Initialize AttentionWorker.
        
        Args:
            args: Command-line arguments namespace
        """
        self.args = args
        self.device = self._init_device()
        self.coord = self._connect_coord()
        
        # Conditionally initialize distributed
        if not getattr(args, 'no_init_dist', False):
            self._init_distributed()
            self.ep_group = self._build_ep_group()
        else:
            logger.warning("Distributed init skipped (--no-init-dist)")
            self.ep_group = None
        
        # Initialize batcher and communicator
        from ..batching.continuous_batcher import ContinuousBatcher
        from ..comm.factory import build_communicator
        
        self.batcher = ContinuousBatcher(
            num_ep_ranks=args.ffn_world,
            num_experts=args.num_experts,
            max_tokens_per_chunk=args.max_tokens_per_chunk
        )
        
        # Build communicator (only if distributed is initialized)
        if self.ep_group is not None:
            prefer_deepep = bool(getattr(args, "use_deepep", False)) and not args.use_fallback
            self.comm = build_communicator(
                prefer_deepep=prefer_deepep,
                ep_group=self.ep_group,
                hidden_size=args.hidden_size,
                num_experts=args.num_experts,
                max_tokens_per_rank=args.max_tokens_per_chunk,
                device=self.device,
                mode=args.mode
            )
            logger.info("AttentionWorker communicator=%s", type(self.comm).__name__)
        else:
            self.comm = None
            logger.warning("Communicator not built (distributed not initialized)")
        
        # Initialize routing table
        self.routing_table = self._fetch_initial_routing_table()
        self._routing_table_tensor = None
        
        if self.comm is not None:
            self.comm.update_routing_table(self.routing_table)
        
        logger.info(
            f"AttentionWorker initialized: rank={args.rank}, "
            f"world={args.world}, ffn_world={args.ffn_world}, "
            f"device={self.device}, mode={args.mode}"
        )
    
    def _init_device(self) -> torch.device:
        """Initialize device and return torch.device."""
        dev_module = _get_dev_module()
        
        if self.args.device_id >= 0:
            dev_module.set_device(self.args.device_id)
            device_type = 'npu' if hasattr(dev_module, '__name__') and 'npu' in str(dev_module) else 'cuda'
            
            # Detect actual device type
            try:
                import torch_npu
                device_type = 'npu'
            except ImportError:
                if torch.cuda.is_available():
                    device_type = 'cuda'
                else:
                    device_type = 'cpu'
            
            device = torch.device(f'{device_type}:{self.args.device_id}')
            logger.info(f"Device initialized: {device}")
        else:
            device = torch.device('cpu')
            logger.info("Using CPU device")
        
        return device
    
    def _connect_coord(self):
        """Connect to coordinator via gRPC stub."""
        from ..coordinator_client import CoordinatorClient
        
        try:
            client = CoordinatorClient(self.args.coord_addr)
            logger.info(f"Connected to coordinator at {self.args.coord_addr}")
            return client
        except Exception as e:
            logger.warning(f"Failed to connect to coordinator: {e}")
            return None
    
    def _init_distributed(self):
        """Initialize torch.distributed process group."""
        import torch.distributed as dist
        
        # Determine backend
        try:
            import torch_npu
            backend = 'hccl'
            logger.info("Using HCCL backend for distributed")
        except ImportError:
            if torch.cuda.is_available():
                backend = 'nccl'
            else:
                backend = 'gloo'
            logger.info(f"Using {backend} backend for distributed")
        
        try:
            init_method = f"tcp://{self.args.master_addr}:{self.args.master_port}"
            
            dist.init_process_group(
                backend=backend,
                init_method=init_method,
                world_size=self.args.world + self.args.ffn_world,
                rank=self.args.rank
            )
            
            logger.info(
                f"Distributed initialized: rank={self.args.rank}, "
                f"world_size={self.args.world + self.args.ffn_world}, "
                f"backend={backend}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize distributed: {e}")
            raise
    
    def _build_ep_group(self):
        """Build expert parallel process group (all ATTN + FFN ranks)."""
        import torch.distributed as dist
        
        # EP group includes all ranks: ATTN (0..world-1) + FFN (world..world+ffn_world-1)
        all_ranks = list(range(self.args.world + self.args.ffn_world))
        ep_group = dist.new_group(ranks=all_ranks)
        
        logger.info(
            f"EP group created: ranks={all_ranks[:3]}...{all_ranks[-3:]} "
            f"(total {len(all_ranks)})"
        )
        
        return ep_group
    
    def _fetch_initial_routing_table(self) -> Dict[str, Any]:
        """
        Fetch initial routing table from coordinator.
        
        Falls back to uniform distribution if coordinator is unreachable.
        """
        if self.coord is not None:
            try:
                table = self.coord.get_routing_table()
                if table is not None and table.get("expert_to_rank"):
                    logger.info(f"Fetched routing table version {table['version']}")
                    return table
            except Exception as e:
                logger.warning(f"Failed to fetch routing table from coordinator: {e}")
        
        # Fallback: uniform expert-to-rank mapping
        logger.info("Using uniform fallback routing table")
        num_experts = self.args.num_experts
        ffn_world = self.args.ffn_world
        
        # Distribute experts uniformly across FFN ranks
        # FFN ranks start at offset = self.args.world
        expert_to_rank = [
            self.args.world + (e * ffn_world) // num_experts
            for e in range(num_experts)
        ]
        
        return {
            "version": 0,
            "expert_to_rank": expert_to_rank,
            "mode": self.args.mode
        }
    
    @property
    def routing_table_tensor(self) -> torch.Tensor:
        """Get cached routing table as LongTensor on device."""
        if self._routing_table_tensor is None:
            expert_to_rank = self.routing_table["expert_to_rank"]
            self._routing_table_tensor = torch.tensor(
                expert_to_rank,
                dtype=torch.long,
                device=self.device
            )
        return self._routing_table_tensor
    
    def register_with_coordinator(self):
        """Register this worker with the coordinator."""
        if self.coord is None:
            logger.warning("Coordinator not connected, skipping registration")
            return
        
        try:
            info = {
                "rank": self.args.rank,
                "role": "attn",
                "host": socket.gethostname(),
                "device_id": self.args.device_id,
                "capabilities": {
                    "hidden_size": self.args.hidden_size,
                    "max_tokens_per_chunk": self.args.max_tokens_per_chunk,
                }
            }
            
            response = self.coord.register_worker(info)
            
            if response.get("success"):
                logger.info(f"Successfully registered with coordinator: {response.get('message')}")
            else:
                logger.warning(f"Registration failed: {response.get('message')}")
        
        except Exception as e:
            logger.error(f"Exception during registration: {e}")
    
    def run_step(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Run one end-to-end attention->dispatch->combine step.
        
        Args:
            batch: Input tensor [B, S, H]
        
        Returns:
            Output tensor [B, S, H]
        """
        if self.comm is None:
            logger.warning("Communicator not available, returning identity")
            return batch
        
        # Step 1: Forward attention (skeleton: identity reshape)
        hidden = self._forward_attention(batch)  # [B*S, H]
        
        # Step 2: Route tokens to experts (skeleton: deterministic top-k)
        topk_idx, topk_w = self._route_topk(hidden)  # [N, K], [N, K]
        
        # Step 3: Get routing table
        expert_to_rank = self.routing_table_tensor  # [num_experts]
        
        # Step 4: Split into micro-batches
        mbs = self.batcher.split(hidden, topk_idx, topk_w, expert_to_rank)
        
        logger.debug(f"Split into {len(mbs)} micro-batches")
        
        # Step 5: Dispatch micro-batches
        handles = []
        for mb in mbs:
            h = self.comm.dispatch_async(mb.hidden, mb.topk_indices, mb.topk_weights)
            handles.append(h)
            # Pipeline opportunity: kick off next MB's attention here in real impl
        
        # Step 6: Await FFN and combine
        outputs = []
        for h, mb in zip(handles, mbs):
            # In skeleton: pretend FFN returned identity-shaped output
            ffn_out = self._await_ffn(h)
            combined_h = self.comm.combine_async(ffn_out, h)
            combined = self.comm.wait_combine(combined_h)
            outputs.append(combined)
        
        # Step 7: Merge micro-batches back to original order
        merged = self.batcher.merge(outputs, mbs, total_tokens=hidden.shape[0])
        
        # Reshape back to [B, S, H]
        return merged.view_as(batch)
    
    def _forward_attention(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Forward attention computation.
        
        SKELETON: identity reshape. Real implementation would call Qwen3 attention layers.
        
        Args:
            batch: Input [B, S, H]
        
        Returns:
            Hidden states [B*S, H]
        """
        B, S, H = batch.shape
        return batch.reshape(B * S, H).contiguous()
    
    def _route_topk(self, hidden: torch.Tensor) -> tuple:
        """
        Route tokens to top-k experts.
        
        SKELETON: deterministic top-k for testing.
        Real implementation would use learned router.
        
        Args:
            hidden: Hidden states [N, H]
        
        Returns:
            - topk_indices: Expert indices [N, K]
            - topk_weights: Expert weights [N, K]
        """
        N = hidden.shape[0]
        K = self.args.topk
        
        # Deterministic routing: token i -> experts [i%num_experts, (i+1)%num_experts, ...]
        idx = (torch.arange(N * K, device=hidden.device) % self.args.num_experts).view(N, K).long()
        w = torch.full((N, K), 1.0 / K, device=hidden.device, dtype=hidden.dtype)
        
        return idx, w
    
    def _await_ffn(self, handle: Dict) -> torch.Tensor:
        """
        Await FFN computation result.
        
        SKELETON: echo back recv_hidden as FFN output (identity FFN).
        Real implementation would receive actual FFN outputs.
        
        Args:
            handle: Dispatch handle from communicator
        
        Returns:
            FFN outputs
        """
        # Identity FFN: output = input
        handle = self.comm.wait_dispatch(handle)
        return handle["recv_hidden"]
    
    def shutdown(self):
        """Shutdown worker and cleanup resources."""
        logger.info("Shutting down AttentionWorker")
        
        # Close coordinator connection
        if self.coord is not None:
            try:
                self.coord.close()
            except Exception as e:
                logger.warning(f"Error closing coordinator connection: {e}")
        
        # Destroy process group
        try:
            import torch.distributed as dist
            if dist.is_initialized():
                dist.destroy_process_group()
                logger.info("Distributed process group destroyed")
        except Exception as e:
            logger.warning(f"Error destroying process group: {e}")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="AttentionWorker for Coordinator-based MoE"
    )
    
    # Coordinator and distributed config
    parser.add_argument("--coord-addr", type=str, required=True,
                        help="Coordinator address (host:port)")
    parser.add_argument("--rank", type=int, required=True,
                        help="Global rank within ATTN_DP_GROUP")
    parser.add_argument("--world", type=int, required=True,
                        help="ATTN DP world size")
    parser.add_argument("--ffn-world", type=int, required=True,
                        help="FFN EP world size")
    parser.add_argument("--master-addr", type=str, default="localhost",
                        help="Master address for distributed init")
    parser.add_argument("--master-port", type=int, default=29500,
                        help="Master port for distributed init")
    
    # Model config
    parser.add_argument("--model-path", type=str, default="/models/Qwen3-30B-A3B",
                        help="Path to model weights")
    parser.add_argument("--hidden-size", type=int, default=2048,
                        help="Hidden dimension size")
    parser.add_argument("--num-experts", type=int, default=128,
                        help="Total number of experts")
    parser.add_argument("--topk", type=int, default=8,
                        help="Top-K experts per token")
    
    # Runtime config
    parser.add_argument("--max-tokens-per-chunk", type=int, default=4096,
                        help="Maximum tokens per micro-batch")
    parser.add_argument("--mode", type=str, default="normal",
                        choices=["normal", "low_latency"],
                        help="Operation mode")
    parser.add_argument("--device-id", type=int, default=0,
                        help="Local NPU/GPU device index (-1 for CPU)")
    
    # Flags
    parser.add_argument("--use-deepep", action="store_true",
                        help="Opt in to experimental DeepEP communicator (default: fallback_a2a)")
    parser.add_argument("--use-fallback", action="store_true",
                        help="Deprecated no-op: fallback_a2a is the default")
    parser.add_argument("--no-init-dist", action="store_true",
                        help="Skip distributed init (unit test mode)")
    
    return parser.parse_args()


def main():
    """Main entrypoint for AttentionWorker."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
    )
    
    args = parse_args()
    
    logger.info("Starting AttentionWorker")
    logger.info(f"Arguments: {vars(args)}")
    
    # Initialize worker
    worker = AttentionWorker(args)
    
    # Register with coordinator
    worker.register_with_coordinator()
    
    # Main loop placeholder
    logger.info("AttentionWorker ready. Main loop not implemented in skeleton.")
    logger.info("Use worker.run_step(batch) to process batches.")
    
    # In production, this would enter a loop waiting for coordinator assignments
    # For now, just keep the process alive for manual testing
    try:
        import time
        while True:
            time.sleep(2.0)
            if worker.coord is not None:
                worker.coord.update_metrics({
                    "role": "attn",
                    "rank": args.rank,
                    "timestamp_us": int(time.time() * 1e6),
                })
            logger.debug("AttentionWorker heartbeat")
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    finally:
        worker.shutdown()


if __name__ == "__main__":
    main()
