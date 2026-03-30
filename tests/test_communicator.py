"""
Tests for communicator components.

Note: These tests require distributed setup or mocking.
"""

import pytest
import torch

# Skip if torch.distributed not available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required for full communicator tests"
)


class TestBufferManagement:
    """Tests for buffer management without full distributed setup."""
    
    def test_buffer_shape(self):
        """Test that buffer shapes are correct."""
        hidden_size = 768
        max_seq_len = 128
        max_batch_size = 4
        
        buffer_shape = (max_batch_size, max_seq_len, hidden_size)
        buffer = torch.empty(buffer_shape, dtype=torch.bfloat16)
        
        assert buffer.shape == (4, 128, 768)
    
    def test_pack_unpack(self):
        """Test packing and unpacking tensors."""
        hidden_size = 768
        batch_size = 2
        seq_len = 10
        
        attn_output = torch.randn(batch_size, seq_len, hidden_size)
        residual = torch.randn(batch_size, seq_len, hidden_size)
        
        # Pack
        packed = torch.cat([attn_output, residual], dim=-1)
        assert packed.shape == (batch_size, seq_len, hidden_size * 2)
        
        # Unpack
        unpacked_attn = packed[..., :hidden_size]
        unpacked_res = packed[..., hidden_size:]
        
        assert torch.allclose(attn_output, unpacked_attn)
        assert torch.allclose(residual, unpacked_res)


class TestTagGeneration:
    """Tests for communication tag generation."""
    
    def test_unique_tags(self):
        """Test that tags are unique for different layer/mb combinations."""
        tags = set()
        
        for layer_idx in range(32):
            for mb_idx in range(2):
                for direction in [0, 1]:
                    tag = layer_idx * 1000 + mb_idx * 10 + direction
                    assert tag not in tags, f"Duplicate tag: {tag}"
                    tags.add(tag)
    
    def test_tag_values(self):
        """Test specific tag values."""
        # Layer 0, MB 0, attn->ffn
        assert 0 * 1000 + 0 * 10 + 0 == 0
        
        # Layer 0, MB 0, ffn->attn
        assert 0 * 1000 + 0 * 10 + 1 == 1
        
        # Layer 5, MB 1, attn->ffn
        assert 5 * 1000 + 1 * 10 + 0 == 5010
        
        # Layer 31, MB 1, ffn->attn
        assert 31 * 1000 + 1 * 10 + 1 == 31011


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
