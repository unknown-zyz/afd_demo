"""
Tests for pipeline components.
"""

import pytest
import torch

from src.pipeline.micro_batch import (
    MicroBatch,
    MicroBatchManager,
    MicroBatchState,
    create_position_ids,
    create_causal_mask,
)


class TestMicroBatch:
    """Tests for MicroBatch class."""
    
    def test_micro_batch_creation(self):
        """Test basic micro-batch creation."""
        input_ids = torch.randint(0, 1000, (2, 10))
        mb = MicroBatch(id=0, input_ids=input_ids)
        
        assert mb.id == 0
        assert mb.batch_size == 2
        assert mb.seq_len == 10
        assert mb.state == MicroBatchState.WAITING
        assert mb.current_layer == 0
    
    def test_advance_layer(self):
        """Test layer advancement."""
        input_ids = torch.randint(0, 1000, (2, 10))
        mb = MicroBatch(id=0, input_ids=input_ids)
        
        mb.advance_layer()
        assert mb.current_layer == 1
        
        mb.advance_layer()
        assert mb.current_layer == 2


class TestMicroBatchManager:
    """Tests for MicroBatchManager class."""
    
    def test_split_batch_even(self):
        """Test splitting batch evenly."""
        manager = MicroBatchManager(num_micro_batches=2)
        input_ids = torch.randint(0, 1000, (4, 10))
        
        mbs = manager.split_batch(input_ids)
        
        assert len(mbs) == 2
        assert mbs[0].batch_size == 2
        assert mbs[1].batch_size == 2
    
    def test_split_batch_uneven(self):
        """Test splitting batch unevenly."""
        manager = MicroBatchManager(num_micro_batches=2)
        input_ids = torch.randint(0, 1000, (5, 10))
        
        mbs = manager.split_batch(input_ids)
        
        assert len(mbs) == 2
        # Remainder goes to first micro-batches
        assert mbs[0].batch_size == 3
        assert mbs[1].batch_size == 2
    
    def test_merge_results(self):
        """Test merging results back."""
        manager = MicroBatchManager(num_micro_batches=2)
        
        results = [
            torch.ones(2, 10, 100),
            torch.zeros(2, 10, 100),
        ]
        
        merged = manager.merge_results(results)
        
        assert merged.shape == (4, 10, 100)
        assert merged[:2].sum() == 2 * 10 * 100
        assert merged[2:].sum() == 0


class TestHelperFunctions:
    """Tests for helper functions."""
    
    def test_create_position_ids(self):
        """Test position ID creation."""
        pos_ids = create_position_ids(2, 10, torch.device("cpu"))
        
        assert pos_ids.shape == (2, 10)
        assert pos_ids[0].tolist() == list(range(10))
        assert pos_ids[1].tolist() == list(range(10))
    
    def test_create_causal_mask(self):
        """Test causal mask creation."""
        mask = create_causal_mask(2, 4, torch.device("cpu"))
        
        assert mask.shape == (2, 1, 4, 4)
        
        # Check lower triangular
        expected = torch.tensor([
            [1, 0, 0, 0],
            [1, 1, 0, 0],
            [1, 1, 1, 0],
            [1, 1, 1, 1],
        ], dtype=torch.float32)
        
        assert torch.allclose(mask[0, 0], expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
