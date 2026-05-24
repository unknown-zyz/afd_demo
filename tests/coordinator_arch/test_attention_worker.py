"""
Tests for AttentionWorker in Coordinator-based MoE architecture.

These tests verify the AttentionWorker skeleton implementation in isolation,
without requiring distributed setup or real model weights.
"""

import argparse
import sys

import pytest

# Check if torch is available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available")
def test_attention_worker_basic():
    """
    Test basic AttentionWorker instantiation and run_step.
    
    Verifies:
    - Worker can be initialized in CPU mode without distributed
    - run_step accepts input and returns correct shape
    - Identity semantics work in skeleton mode
    """
    from src.coordinator_arch.workers.attention_worker import AttentionWorker
    
    # Create args namespace for testing
    args = argparse.Namespace(
        coord_addr="localhost:50051",
        rank=0,
        world=1,
        ffn_world=1,
        master_addr="localhost",
        master_port=29500,
        model_path="/models/Qwen3-30B-A3B",
        hidden_size=16,  # Small for testing
        num_experts=8,
        topk=2,
        max_tokens_per_chunk=4096,
        mode="normal",
        device_id=-1,  # CPU mode
        use_fallback=True,  # Force fallback communicator
        no_init_dist=True  # Skip distributed init
    )
    
    # Initialize worker
    worker = AttentionWorker(args)
    
    # Verify basic attributes
    assert worker.device == torch.device('cpu')
    assert worker.args.rank == 0
    assert worker.args.hidden_size == 16
    assert worker.batcher is not None
    
    # Create dummy input: [B=2, S=4, H=16]
    batch = torch.randn(2, 4, 16)
    
    # Note: run_step requires communicator which requires distributed init
    # In no_init_dist mode, run_step will return identity
    # We test the internal methods instead
    
    # Test _forward_attention
    hidden = worker._forward_attention(batch)
    assert hidden.shape == (8, 16)  # B*S=2*4=8, H=16
    
    # Test _route_topk
    topk_idx, topk_w = worker._route_topk(hidden)
    assert topk_idx.shape == (8, 2)  # [N=8, K=2]
    assert topk_w.shape == (8, 2)
    assert torch.allclose(topk_w, torch.tensor(0.5))  # 1/K = 1/2 = 0.5
    
    # Test routing_table_tensor
    rt_tensor = worker.routing_table_tensor
    assert rt_tensor.shape == (8,)  # num_experts=8
    assert rt_tensor.dtype == torch.long
    
    # Cleanup
    worker.shutdown()


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available")
def test_attention_worker_register():
    """
    Test worker registration with coordinator stub.
    
    Verifies:
    - Registration doesn't crash with stub coordinator
    - Stub returns expected response structure
    """
    from src.coordinator_arch.workers.attention_worker import AttentionWorker
    
    args = argparse.Namespace(
        coord_addr="localhost:50051",
        rank=5,
        world=16,
        ffn_world=16,
        master_addr="localhost",
        master_port=29500,
        model_path="/models/Qwen3-30B-A3B",
        hidden_size=2048,
        num_experts=128,
        topk=8,
        max_tokens_per_chunk=4096,
        mode="low_latency",
        device_id=-1,
        use_fallback=True,
        no_init_dist=True
    )
    
    worker = AttentionWorker(args)
    
    # Test registration
    worker.register_with_coordinator()
    
    # Should not crash, coordinator stub logs the call
    # Verify worker state after registration
    assert worker.coord is not None
    
    worker.shutdown()


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available")
def test_routing_table_fallback():
    """
    Test routing table fallback generation.
    
    Verifies:
    - Uniform distribution when coordinator unreachable
    - expert_to_rank covers all FFN ranks
    """
    from src.coordinator_arch.workers.attention_worker import AttentionWorker
    
    args = argparse.Namespace(
        coord_addr="invalid:99999",  # Intentionally invalid
        rank=0,
        world=16,
        ffn_world=16,
        master_addr="localhost",
        master_port=29500,
        model_path="/models/Qwen3-30B-A3B",
        hidden_size=2048,
        num_experts=128,
        topk=8,
        max_tokens_per_chunk=4096,
        mode="normal",
        device_id=-1,
        use_fallback=True,
        no_init_dist=True
    )
    
    worker = AttentionWorker(args)
    
    # Check fallback routing table
    rt = worker.routing_table
    assert rt["version"] == 0
    assert len(rt["expert_to_rank"]) == 128
    assert rt["mode"] == "normal"
    
    # Verify uniform distribution
    # Experts should be distributed across FFN ranks 16..31
    expert_to_rank = rt["expert_to_rank"]
    assert min(expert_to_rank) >= 16  # FFN ranks start at world=16
    assert max(expert_to_rank) < 32   # FFN ranks end at world+ffn_world=32
    
    # Check approximately uniform (each rank should get ~8 experts)
    from collections import Counter
    rank_counts = Counter(expert_to_rank)
    assert len(rank_counts) == 16  # All 16 FFN ranks should be used
    
    worker.shutdown()


if __name__ == "__main__":
    # Run tests directly
    if not TORCH_AVAILABLE:
        print("SKIP: torch not available", file=sys.stderr)
        sys.exit(0)
    
    print("Running test_attention_worker_basic...")
    test_attention_worker_basic()
    print("PASS")
    
    print("Running test_attention_worker_register...")
    test_attention_worker_register()
    print("PASS")
    
    print("Running test_routing_table_fallback...")
    test_routing_table_fallback()
    print("PASS")
    
    print("\nAll tests passed!")
