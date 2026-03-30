"""
Correctness tests comparing disaggregated model with baseline.

These tests require a model to be loaded and are meant to run
on a single node for validation.
"""

import pytest
import torch

from src.utils.validation import compare_tensors


class TestTensorComparison:
    """Tests for tensor comparison utilities."""
    
    def test_identical_tensors(self):
        """Test comparison of identical tensors."""
        t1 = torch.randn(2, 10, 100)
        t2 = t1.clone()
        
        is_close, metrics = compare_tensors(t1, t2)
        
        assert is_close
        assert metrics["max_abs_diff"] == 0.0
        assert metrics["mean_abs_diff"] == 0.0
    
    def test_different_tensors(self):
        """Test comparison of different tensors."""
        t1 = torch.zeros(2, 10, 100)
        t2 = torch.ones(2, 10, 100)
        
        is_close, metrics = compare_tensors(t1, t2)
        
        assert not is_close
        assert metrics["max_abs_diff"] == 1.0
        assert metrics["mean_abs_diff"] == 1.0
    
    def test_close_tensors(self):
        """Test comparison of nearly identical tensors."""
        t1 = torch.randn(2, 10, 100)
        noise = torch.randn_like(t1) * 1e-6
        t2 = t1 + noise
        
        is_close, metrics = compare_tensors(t1, t2, rtol=1e-3, atol=1e-5)
        
        assert is_close
        assert metrics["max_abs_diff"] < 1e-4


class TestModelValidation:
    """Tests for model validation (requires model loading)."""
    
    @pytest.mark.skip(reason="Requires model download")
    def test_baseline_model(self):
        """Test that baseline model can be loaded and run."""
        from src.utils.validation import BaselineModel
        
        model = BaselineModel(
            model_name="Qwen/Qwen2-0.5B",  # Use smallest model for testing
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            dtype=torch.float32,
        )
        
        input_ids = torch.randint(0, 1000, (1, 10))
        output = model.forward(input_ids)
        
        assert output.shape[0] == 1
        assert output.shape[1] == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
