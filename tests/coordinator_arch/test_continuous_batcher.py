"""
Tests for ContinuousBatcher and FFNQueue.
"""

import time
import threading
import pytest
import torch

from src.coordinator_arch.batching.continuous_batcher import (
    ContinuousBatcher,
    MicroBatch
)
from src.coordinator_arch.batching.ffn_queue import FFNQueue, FFNBatch


class TestContinuousBatcher:
    """Tests for ContinuousBatcher split and merge operations."""
    
    def test_split_merge_roundtrip(self):
        """Test that split followed by merge recovers original tensor."""
        # Setup: 16 tokens, H=8, 4 EP ranks, 16 experts, K=2
        N, H, K = 16, 8, 2
        num_ep_ranks = 4
        num_experts = 16
        
        # Create batcher
        batcher = ContinuousBatcher(
            num_ep_ranks=num_ep_ranks,
            num_experts=num_experts,
            max_tokens_per_chunk=4096
        )
        
        # Create test data
        hidden_states = torch.randn(N, H)
        
        # expert_to_rank[i] = i // 4
        expert_to_rank = torch.arange(num_experts) // 4
        
        # Token i has top-1 = expert (i % 16), top-2 = expert ((i+1) % 16)
        topk_indices = torch.zeros(N, K, dtype=torch.long)
        for i in range(N):
            topk_indices[i, 0] = i % num_experts
            topk_indices[i, 1] = (i + 1) % num_experts
        
        topk_weights = torch.rand(N, K)
        
        # Split
        micro_batches = batcher.split(
            hidden_states=hidden_states,
            topk_indices=topk_indices,
            topk_weights=topk_weights,
            expert_to_rank=expert_to_rank
        )
        
        # Should have at least 4 MBs (one per rank)
        assert len(micro_batches) >= 4, f"Expected at least 4 MBs, got {len(micro_batches)}"
        
        # Verify all tokens are covered
        all_indices = torch.cat([mb.indices for mb in micro_batches])
        assert len(all_indices) == N
        assert set(all_indices.tolist()) == set(range(N))
        
        # Create identity outputs (same as inputs)
        outputs_per_mb = [mb.hidden for mb in micro_batches]
        
        # Merge
        merged = batcher.merge(
            outputs_per_mb=outputs_per_mb,
            micro_batches=micro_batches,
            total_tokens=N
        )
        
        # Verify round-trip
        assert merged.shape == hidden_states.shape
        assert torch.allclose(merged, hidden_states)
    
    def test_split_respects_rank_grouping(self):
        """Test that tokens with same primary rank are grouped together."""
        N, H, K = 12, 4, 1
        num_ep_ranks = 3
        num_experts = 9
        
        batcher = ContinuousBatcher(
            num_ep_ranks=num_ep_ranks,
            num_experts=num_experts,
            max_tokens_per_chunk=10
        )
        
        hidden_states = torch.randn(N, H)
        
        # expert_to_rank: experts 0-2 -> rank 0, 3-5 -> rank 1, 6-8 -> rank 2
        expert_to_rank = torch.arange(num_experts) // 3
        
        # Assign tokens deterministically to ranks
        # Tokens 0-3 -> expert 0 (rank 0)
        # Tokens 4-7 -> expert 4 (rank 1)
        # Tokens 8-11 -> expert 7 (rank 2)
        topk_indices = torch.zeros(N, K, dtype=torch.long)
        topk_indices[0:4, 0] = 0
        topk_indices[4:8, 0] = 4
        topk_indices[8:12, 0] = 7
        
        topk_weights = torch.ones(N, K)
        
        micro_batches = batcher.split(
            hidden_states=hidden_states,
            topk_indices=topk_indices,
            topk_weights=topk_weights,
            expert_to_rank=expert_to_rank
        )
        
        # Should have 3 MBs (one per rank)
        assert len(micro_batches) == 3
        
        # Verify primary ranks
        primary_ranks = [mb.primary_rank for mb in micro_batches]
        assert set(primary_ranks) == {0, 1, 2}


class TestFFNQueue:
    """Tests for FFNQueue batching behavior."""
    
    def test_size_trigger(self):
        """Test that queue returns batch when size threshold is reached."""
        max_batch = 2
        queue = FFNQueue(max_batch=max_batch, max_wait_ms=5.0)
        
        # Push 3 items
        for i in range(3):
            hidden = torch.randn(8)
            weights = torch.rand(1)
            queue.push(f"req_{i}", hidden, weights, src_rank=0, src_slot=i)
        
        # First pop should return batch of 2
        batch1 = queue.pop_batch()
        assert batch1 is not None
        assert len(batch1.request_ids) == 2
        assert batch1.request_ids == ["req_0", "req_1"]
        assert batch1.hidden.shape == (2, 8)
        
        # Immediate second pop should return None (only 1 item left)
        batch2 = queue.pop_batch()
        assert batch2 is None
        
        # Sleep and pop again with time trigger
        time.sleep(0.01)
        batch3 = queue.pop_batch()
        assert batch3 is not None
        assert len(batch3.request_ids) == 1
        assert batch3.request_ids == ["req_2"]
    
    def test_time_trigger(self):
        """Test that queue returns batch when time threshold is reached."""
        max_batch = 8
        max_wait_ms = 50.0  # 50ms
        queue = FFNQueue(max_batch=max_batch, max_wait_ms=max_wait_ms)
        
        # Push 1 item
        hidden = torch.randn(8)
        weights = torch.rand(1)
        queue.push("req_0", hidden, weights, src_rank=0, src_slot=0)
        
        # Immediate pop should return None
        batch1 = queue.pop_batch()
        assert batch1 is None
        
        # Sleep longer than max_wait_ms
        time.sleep(max_wait_ms / 1000.0 + 0.01)
        
        # Pop should now return batch of 1
        batch2 = queue.pop_batch()
        assert batch2 is not None
        assert len(batch2.request_ids) == 1
        assert batch2.request_ids == ["req_0"]
    
    def test_threading_safety(self):
        """Test that queue works correctly with concurrent access."""
        max_batch = 5
        queue = FFNQueue(max_batch=max_batch, max_wait_ms=10.0)
        num_items = 50
        
        # Track results
        pushed_ids = []
        popped_ids = []
        lock = threading.Lock()
        
        def producer():
            """Push items to queue."""
            for i in range(num_items):
                hidden = torch.randn(4)
                weights = torch.rand(1)
                request_id = f"req_{i}"
                queue.push(request_id, hidden, weights, src_rank=0, src_slot=i)
                with lock:
                    pushed_ids.append(request_id)
                time.sleep(0.001)  # Small delay
        
        def consumer():
            """Pop batches from queue."""
            count = 0
            while count < num_items:
                batch = queue.pop_batch()
                if batch is not None:
                    with lock:
                        popped_ids.extend(batch.request_ids)
                    count += len(batch.request_ids)
                else:
                    time.sleep(0.001)  # Small delay before retry
        
        # Start threads
        producer_thread = threading.Thread(target=producer)
        consumer_thread = threading.Thread(target=consumer)
        
        producer_thread.start()
        consumer_thread.start()
        
        # Wait for completion
        producer_thread.join()
        consumer_thread.join()
        
        # Verify all items recovered
        assert len(popped_ids) == num_items
        assert set(popped_ids) == set(pushed_ids)
        
        # Verify ordering is preserved (popped in same order as pushed)
        assert popped_ids == pushed_ids
    
    def test_empty_queue(self):
        """Test that empty queue returns None."""
        queue = FFNQueue(max_batch=5, max_wait_ms=5.0)
        batch = queue.pop_batch()
        assert batch is None
        assert len(queue) == 0
    
    def test_queue_length(self):
        """Test that __len__ returns correct queue size."""
        queue = FFNQueue(max_batch=10, max_wait_ms=5.0)
        
        assert len(queue) == 0
        
        for i in range(3):
            hidden = torch.randn(4)
            weights = torch.rand(1)
            queue.push(f"req_{i}", hidden, weights, src_rank=0, src_slot=i)
        
        assert len(queue) == 3
        
        batch = queue.pop_batch()
        assert batch is not None
        assert len(queue) == 0
