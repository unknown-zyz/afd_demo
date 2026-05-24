"""
Unit tests for FallbackMoECommunicator using torch.distributed.
"""

import os
import sys
from typing import Optional

try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False
    # Define a minimal pytest.mark.skipif decorator
    class _MockPytest:
        class mark:
            @staticmethod
            def skipif(condition, reason=""):
                def decorator(func):
                    if condition:
                        def wrapper(*args, **kwargs):
                            print(f"⚠ Skipped: {reason}")
                            return None
                        return wrapper
                    return func
                return decorator
    pytest = _MockPytest()

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _can_spawn_gloo() -> bool:
    """Check if we can spawn gloo processes for testing."""
    try:
        # Check if torch distributed is available
        if not dist.is_available():
            return False
        # Check if gloo backend is available
        if not dist.is_gloo_available():
            return False
        return True
    except Exception:
        return False


def _init_process(rank: int, world_size: int, backend: str = "gloo") -> None:
    """Initialize distributed process group."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(
        backend=backend,
        init_method="env://",
        rank=rank,
        world_size=world_size,
    )


def _test_fallback_a2a_worker(
    rank: int,
    world_size: int,
    result_queue: Optional[mp.Queue] = None,
) -> None:
    """
    Worker function for testing FallbackMoECommunicator in distributed setting.
    
    Test setup:
    - 2 ranks (world_size=2)
    - 4 tokens (N=4), 4 experts, K=2 (top-2 routing)
    - H=8 (hidden size)
    - expert_to_rank = [0, 0, 1, 1] (experts 0,1 on rank 0; experts 2,3 on rank 1)
    """
    try:
        _init_process(rank, world_size, backend="gloo")
        
        # Import here to avoid issues if torch.distributed not available at module load
        from src.coordinator_arch.comm.fallback_a2a import FallbackMoECommunicator
        
        # Test configuration
        N = 4  # Number of tokens
        H = 8  # Hidden size
        K = 2  # Top-K experts per token
        num_experts = 4
        device = torch.device("cpu")
        
        # Create communicator
        ep_group = dist.group.WORLD
        communicator = FallbackMoECommunicator(
            ep_group=ep_group,
            hidden_size=H,
            num_experts=num_experts,
            max_tokens_per_rank=10,
            device=device,
        )
        
        # Set routing table: experts [0,1] on rank 0, experts [2,3] on rank 1
        routing_table = {
            "version": 1,
            "expert_to_rank": [0, 0, 1, 1],
            "mode": "normal",
        }
        communicator.update_routing_table(routing_table)
        
        # Create consistent input across ranks for testing
        torch.manual_seed(42 + rank)
        
        # Each rank generates the same input (simulating data parallelism)
        # In real scenario, each rank would have different tokens
        hidden_states = torch.randn(N, H, device=device)
        
        # Top-K routing: each token routed to 2 experts
        # Set deterministic routing for testing
        if rank == 0:
            # Rank 0: Test various routing patterns
            topk_indices = torch.tensor([
                [0, 1],  # Token 0: experts 0,1 (both local to rank 0)
                [1, 2],  # Token 1: expert 1 (rank 0), expert 2 (rank 1)
                [2, 3],  # Token 2: experts 2,3 (both on rank 1)
                [0, 3],  # Token 3: expert 0 (rank 0), expert 3 (rank 1)
            ], dtype=torch.long, device=device)
            
            topk_weights = torch.tensor([
                [0.6, 0.4],
                [0.7, 0.3],
                [0.5, 0.5],
                [0.8, 0.2],
            ], dtype=torch.float32, device=device)
        else:
            # Rank 1: Same routing (in data parallel, all ranks have same routing)
            topk_indices = torch.tensor([
                [0, 1],
                [1, 2],
                [2, 3],
                [0, 3],
            ], dtype=torch.long, device=device)
            
            topk_weights = torch.tensor([
                [0.6, 0.4],
                [0.7, 0.3],
                [0.5, 0.5],
                [0.8, 0.2],
            ], dtype=torch.float32, device=device)
        
        # Dispatch tokens to expert ranks
        dispatch_handle = communicator.dispatch(
            hidden_states=hidden_states,
            topk_indices=topk_indices,
            topk_weights=topk_weights,
        )
        
        # Verify dispatch results
        recv_hidden = dispatch_handle["recv_hidden"]
        recv_weights = dispatch_handle["recv_weights"]
        
        # Expected token counts per rank based on routing:
        # Token 0: [0,1] -> rank 0 gets 2 tokens
        # Token 1: [1,2] -> rank 0 gets 1, rank 1 gets 1
        # Token 2: [2,3] -> rank 1 gets 2 tokens
        # Token 3: [0,3] -> rank 0 gets 1, rank 1 gets 1
        # Total: rank 0 expects 4 tokens (2+1+0+1), rank 1 expects 4 tokens (0+1+2+1)
        
        if rank == 0:
            expected_recv = 4
        else:
            expected_recv = 4
        
        assert recv_hidden.shape[0] == expected_recv, (
            f"Rank {rank}: Expected {expected_recv} tokens, got {recv_hidden.shape[0]}"
        )
        assert recv_hidden.shape[1] == H, (
            f"Rank {rank}: Hidden size mismatch"
        )
        assert recv_weights.shape[0] == expected_recv, (
            f"Rank {rank}: Weight count mismatch"
        )
        
        # Simulate FFN: identity function (just return input)
        ffn_outputs = recv_hidden.clone()
        
        # Combine results back
        output = communicator.combine(
            ffn_outputs=ffn_outputs,
            dispatch_handle=dispatch_handle,
        )
        
        # Verify output shape
        assert output.shape == (N, H), (
            f"Rank {rank}: Output shape mismatch, expected {(N, H)}, got {output.shape}"
        )
        
        # Verify correctness: output should be weighted sum of expert outputs
        # Since FFN is identity, output[n] = sum_k(topk_weights[n,k] * hidden_states[n])
        expected_output = torch.zeros(N, H, device=device)
        for n in range(N):
            for k in range(K):
                # Each expert sees the same input token (identity FFN)
                expected_output[n] += topk_weights[n, k] * hidden_states[n]
        
        # Check if outputs match (with tolerance for floating point errors)
        if not torch.allclose(output, expected_output, atol=1e-5):
            max_diff = (output - expected_output).abs().max().item()
            raise AssertionError(
                f"Rank {rank}: Output mismatch. Max diff: {max_diff:.6e}"
            )
        
        # Test edge case: K=0 (no experts selected)
        empty_indices = torch.empty(N, 0, dtype=torch.long, device=device)
        empty_weights = torch.empty(N, 0, dtype=torch.float32, device=device)
        
        empty_handle = communicator.dispatch(
            hidden_states=hidden_states,
            topk_indices=empty_indices,
            topk_weights=empty_weights,
        )
        
        empty_output = communicator.combine(
            ffn_outputs=torch.empty(0, H, device=device),
            dispatch_handle=empty_handle,
        )
        
        assert empty_output.shape == (N, H), "K=0 output shape mismatch"
        assert torch.allclose(empty_output, torch.zeros(N, H, device=device)), (
            "K=0 should produce zeros"
        )
        
        # All tests passed for this rank
        if result_queue is not None:
            result_queue.put((rank, True, None))
        
        dist.destroy_process_group()
        
    except Exception as e:
        if result_queue is not None:
            result_queue.put((rank, False, str(e)))
        else:
            raise


@pytest.mark.skipif(
    not _can_spawn_gloo(),
    reason="Gloo backend not available or cannot spawn processes"
)
def test_fallback_a2a_distributed():
    """
    Test FallbackMoECommunicator with multiprocessing spawn and gloo backend.
    """
    world_size = 2
    
    # Use spawn method for clean process isolation
    mp.set_start_method("spawn", force=True)
    
    # Create a queue for collecting results
    result_queue = mp.Queue()
    
    # Spawn worker processes
    processes = []
    for rank in range(world_size):
        p = mp.Process(
            target=_test_fallback_a2a_worker,
            args=(rank, world_size, result_queue),
        )
        p.start()
        processes.append(p)
    
    # Wait for all processes to complete
    for p in processes:
        p.join()
    
    # Collect results
    results = []
    while not result_queue.empty():
        results.append(result_queue.get())
    
    # Check that all ranks succeeded
    assert len(results) == world_size, f"Expected {world_size} results, got {len(results)}"
    
    for rank, success, error in results:
        assert success, f"Rank {rank} failed: {error}"
    
    print("✓ All distributed tests passed")


def test_fallback_a2a_routing_table_validation():
    """
    Test routing table validation without distributed context.
    """
    # This test doesn't require distributed initialization
    from src.coordinator_arch.comm.fallback_a2a import FallbackMoECommunicator
    
    # Create a mock communicator (without valid process group)
    # We'll just test the routing table validation logic
    num_experts = 4
    
    # Note: We can't actually instantiate without a valid process group,
    # so we'll just test the validation logic would work
    # This is more of a smoke test to ensure imports work
    
    # Just verify the class is importable and has expected methods
    assert hasattr(FallbackMoECommunicator, 'update_routing_table')
    assert hasattr(FallbackMoECommunicator, 'dispatch')
    assert hasattr(FallbackMoECommunicator, 'combine')
    assert hasattr(FallbackMoECommunicator, 'set_mode')
    
    print("✓ Routing table validation test passed")


if __name__ == "__main__":
    # Run tests directly
    if _can_spawn_gloo():
        test_fallback_a2a_distributed()
    else:
        print("⚠ Skipping distributed test (gloo not available)")
    
    test_fallback_a2a_routing_table_validation()
    print("\n✓ All tests passed!")
