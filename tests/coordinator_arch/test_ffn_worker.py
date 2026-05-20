"""
Tests for FFNWorker in Coordinator-based MoE architecture.

Tests the FFN worker skeleton implementation with CPU fallback and
identity FFN computation.
"""

import argparse

try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False
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

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from src.coordinator_arch.workers.ffn_worker import FFNWorker
    IMPORTS_AVAILABLE = True
except ImportError:
    IMPORTS_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not IMPORTS_AVAILABLE or not HAS_TORCH,
    reason="Required imports not available (torch or FFNWorker)"
)


@pytest.fixture
def ffn_args():
    """Create FFNWorker arguments for testing."""
    args = argparse.Namespace(
        coord_addr="localhost:50051",
        rank=0,
        world=2,
        attn_world=2,
        master_addr="localhost",
        master_port=29500,
        model_path="/models/Qwen3-30B-A3B",
        hidden_size=16,
        num_experts=16,
        topk=2,
        max_batch=4,
        max_wait_ms=5.0,
        mode="normal",
        use_fallback=True,
        device_id=-1,
        no_init_dist=True,
    )
    return args


def test_ffn_worker_init(ffn_args):
    """Test FFNWorker initialization without distributed backend."""
    worker = FFNWorker(ffn_args)
    
    assert worker.args == ffn_args
    assert worker.device.type == "cpu"
    assert worker.args.hidden_size == 16
    assert worker.args.num_experts == 16
    assert worker.queue.max_batch == 4
    
    # Check routing table was initialized
    assert "version" in worker.routing_table
    assert "expert_to_rank" in worker.routing_table
    assert len(worker.routing_table["expert_to_rank"]) == 16


def test_ffn_worker_run_once(ffn_args):
    """Test run_once with fake dispatch handle (identity FFN)."""
    worker = FFNWorker(ffn_args)
    
    # Create fake dispatch handle
    batch_size = 8
    hidden_size = 16
    dispatch_handle = {
        "recv_hidden": torch.randn(batch_size, hidden_size),
        "topk_indices": torch.randint(0, 16, (batch_size, 2)),
        "topk_weights": torch.rand(batch_size, 2),
    }
    
    # Run once
    output = worker.run_once(dispatch_handle)
    
    # Check output shape matches input (identity function)
    assert output.shape == dispatch_handle["recv_hidden"].shape
    assert output.dtype == dispatch_handle["recv_hidden"].dtype
    
    # Check output is close to input (identity function)
    assert torch.allclose(output, dispatch_handle["recv_hidden"])


def test_compute_local_experts():
    """Test _compute_local_experts with specific routing table."""
    # Create args for rank 1, world 4, 16 experts
    args = argparse.Namespace(
        coord_addr="localhost:50051",
        rank=1,
        world=4,
        attn_world=4,
        master_addr="localhost",
        master_port=29500,
        model_path="/models/Qwen3-30B-A3B",
        hidden_size=16,
        num_experts=16,
        topk=2,
        max_batch=4,
        max_wait_ms=5.0,
        mode="normal",
        use_fallback=True,
        device_id=-1,
        no_init_dist=True,
    )
    
    worker = FFNWorker(args)
    
    # Override routing table to have uniform distribution
    # expert_to_rank[i] = i // 4, so:
    # rank 0: [0, 1, 2, 3]
    # rank 1: [4, 5, 6, 7]
    # rank 2: [8, 9, 10, 11]
    # rank 3: [12, 13, 14, 15]
    worker.routing_table = {
        "version": 1,
        "expert_to_rank": [i // 4 for i in range(16)],
        "mode": "normal",
    }
    
    # Recompute local experts
    local_expert_ids = worker._compute_local_experts()
    
    # Rank 1 should own experts [4, 5, 6, 7]
    assert local_expert_ids == [4, 5, 6, 7]


def test_ffn_worker_identity_forward():
    """Test _forward_ffn identity computation."""
    args = argparse.Namespace(
        coord_addr="localhost:50051",
        rank=0,
        world=2,
        attn_world=2,
        master_addr="localhost",
        master_port=29500,
        model_path="/models/Qwen3-30B-A3B",
        hidden_size=32,
        num_experts=16,
        topk=2,
        max_batch=8,
        max_wait_ms=5.0,
        mode="normal",
        use_fallback=True,
        device_id=-1,
        no_init_dist=True,
    )
    
    worker = FFNWorker(args)
    
    # Test with various input shapes
    for M in [1, 4, 16, 64]:
        hidden = torch.randn(M, 32)
        topk_indices = torch.randint(0, 16, (M, 2))
        
        output = worker._forward_ffn(hidden, topk_indices)
        
        # Identity: output should equal input
        assert output.shape == hidden.shape
        assert torch.allclose(output, hidden)


def test_device_initialization_cpu():
    """Test device initialization with CPU."""
    args = argparse.Namespace(
        coord_addr="localhost:50051",
        rank=0,
        world=1,
        attn_world=1,
        master_addr="localhost",
        master_port=29500,
        model_path="/models/Qwen3-30B-A3B",
        hidden_size=16,
        num_experts=8,
        topk=2,
        max_batch=4,
        max_wait_ms=5.0,
        mode="normal",
        use_fallback=True,
        device_id=-1,
        no_init_dist=True,
    )
    
    worker = FFNWorker(args)
    assert worker.device.type == "cpu"


def test_routing_table_update():
    """Test routing table update and local expert recomputation."""
    args = argparse.Namespace(
        coord_addr="localhost:50051",
        rank=2,
        world=4,
        attn_world=4,
        master_addr="localhost",
        master_port=29500,
        model_path="/models/Qwen3-30B-A3B",
        hidden_size=16,
        num_experts=12,
        topk=2,
        max_batch=4,
        max_wait_ms=5.0,
        mode="normal",
        use_fallback=True,
        device_id=-1,
        no_init_dist=True,
    )
    
    worker = FFNWorker(args)
    
    # Initial routing: uniform distribution
    initial_local = worker.local_expert_ids
    assert len(initial_local) > 0
    
    # Update to a different routing table
    new_routing = {
        "version": 2,
        "expert_to_rank": [i % 4 for i in range(12)],  # Round-robin
        "mode": "normal",
    }
    
    worker.routing_table = new_routing
    worker.comm.update_routing_table(new_routing)
    new_local = worker._compute_local_experts()
    
    # Rank 2 should own experts [2, 6, 10] with round-robin
    assert new_local == [2, 6, 10]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
