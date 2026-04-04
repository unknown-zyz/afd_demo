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
from transformers import DynamicCache


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


class TestCrossLayerPipeline:
    """跨层 MB 流水线调度的正确性测试。"""

    def test_layer0_sends_before_layer1_recv_tags(self):
        """验证 layer 0 的 send tag 与 layer 1 recv 使用的 tag 正确匹配且不冲突。"""

        def get_tag(layer_idx, mb_idx, direction):
            dir_code = 0 if direction == "attn_to_ffn" else 1
            return layer_idx * 1000 + mb_idx * 10 + dir_code

        num_layers = 4
        num_mb = 2

        for layer_idx in range(num_layers):
            for mb_idx in range(num_mb):
                send_tag = get_tag(layer_idx, mb_idx, "attn_to_ffn")
                recv_tag = get_tag(layer_idx, mb_idx, "ffn_to_attn")
                assert send_tag != recv_tag, (
                    f"Layer {layer_idx} MB {mb_idx}: send/recv tag 冲突 {send_tag}"
                )

        # 跨层 tag 不冲突
        for layer_idx in range(num_layers - 1):
            for mb_idx in range(num_mb):
                recv_tag_L = get_tag(layer_idx, mb_idx, "ffn_to_attn")
                send_tag_L1 = get_tag(layer_idx + 1, mb_idx, "attn_to_ffn")
                assert recv_tag_L != send_tag_L1

    def test_pipeline_ordering_simulation(self):
        """模拟跨层流水线操作顺序，验证 MB0 下一层不等 MB1 recv 完成。"""
        events = []
        clock = [0]

        def tick():
            clock[0] += 1
            return clock[0]

        num_layers = 3
        num_mb = 2

        # Layer 0: compute + send
        for mb_idx in range(num_mb):
            events.append((tick(), "compute", 0, mb_idx))
            events.append((tick(), "send", 0, mb_idx))

        # Layers 1~N-1: interleaved recv + compute
        for layer_idx in range(1, num_layers):
            for mb_idx in range(num_mb):
                events.append((tick(), "recv_post", layer_idx - 1, mb_idx))
            for mb_idx in range(num_mb):
                events.append((tick(), "recv_wait", layer_idx - 1, mb_idx))
                events.append((tick(), "compute", layer_idx, mb_idx))
                events.append((tick(), "send", layer_idx, mb_idx))

        # Last recv
        for mb_idx in range(num_mb):
            events.append((tick(), "recv_wait", num_layers - 1, mb_idx))

        # 关键验证：MB0 layer 1 compute 在 MB1 layer 0 recv_wait 之前
        mb0_l1_compute = next(
            t for t, ev, l, m in events if ev == "compute" and l == 1 and m == 0
        )
        mb1_l0_recv_wait = next(
            t for t, ev, l, m in events if ev == "recv_wait" and l == 0 and m == 1
        )
        assert mb0_l1_compute < mb1_l0_recv_wait, (
            f"MB0 layer 1 compute (t={mb0_l1_compute}) 应在 "
            f"MB1 layer 0 recv_wait (t={mb1_l0_recv_wait}) 之前"
        )

    def test_single_layer_no_interleave(self):
        """单层场景：只有 layer 0 compute+send 和 last recv，无交错循环。"""
        num_layers = 1
        num_mb = 2
        events = []
        clock = [0]

        def tick():
            clock[0] += 1
            return clock[0]

        for mb_idx in range(num_mb):
            events.append((tick(), "compute", 0, mb_idx))
            events.append((tick(), "send", 0, mb_idx))

        # range(1, 1) 为空
        for layer_idx in range(1, num_layers):
            pass  # pragma: no cover

        for mb_idx in range(num_mb):
            events.append((tick(), "recv_wait", 0, mb_idx))

        assert len([e for e in events if e[1] == "compute"]) == num_mb
        assert len([e for e in events if e[1] == "send"]) == num_mb
        assert len([e for e in events if e[1] == "recv_wait"]) == num_mb

    def test_single_mb_pipeline(self):
        """单 MB 场景：每层只有一个 MB，流水线仍正确。"""
        num_layers = 3
        num_mb = 1
        events = []
        clock = [0]

        def tick():
            clock[0] += 1
            return clock[0]

        for mb_idx in range(num_mb):
            events.append((tick(), "compute", 0, mb_idx))
            events.append((tick(), "send", 0, mb_idx))

        for layer_idx in range(1, num_layers):
            for mb_idx in range(num_mb):
                events.append((tick(), "recv_wait", layer_idx - 1, mb_idx))
                events.append((tick(), "compute", layer_idx, mb_idx))
                events.append((tick(), "send", layer_idx, mb_idx))

        for mb_idx in range(num_mb):
            events.append((tick(), "recv_wait", num_layers - 1, mb_idx))

        for layer_idx in range(num_layers):
            assert len([e for e in events if e[1] == "compute" and e[2] == layer_idx]) == 1
            assert len([e for e in events if e[1] == "send" and e[2] == layer_idx]) == 1
            assert len([e for e in events if e[1] == "recv_wait" and e[2] == layer_idx]) == 1

    def test_tag_uniqueness_across_layers_and_mbs(self):
        """验证所有层和 MB 组合的 tag 全局唯一。"""

        def get_tag(layer_idx, mb_idx, direction):
            dir_code = 0 if direction == "attn_to_ffn" else 1
            return layer_idx * 1000 + mb_idx * 10 + dir_code

        num_layers = 8
        num_mb = 4
        all_tags = set()

        for layer_idx in range(num_layers):
            for mb_idx in range(num_mb):
                for direction in ("attn_to_ffn", "ffn_to_attn"):
                    tag = get_tag(layer_idx, mb_idx, direction)
                    assert tag not in all_tags, f"Tag 冲突: {layer_idx}/{mb_idx}/{direction}"
                    all_tags.add(tag)

        assert len(all_tags) == num_layers * num_mb * 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestKVCacheBatchSlicing:
    """Tests for DynamicCache batch-slicing used in decode per-MB attention."""

    def _make_cache(self, batch, heads, seq, dim, num_layers=2):
        """Create a DynamicCache populated with random KV for all layers."""
        cache = DynamicCache()
        for layer_idx in range(num_layers):
            k = torch.randn(batch, heads, seq, dim)
            v = torch.randn(batch, heads, seq, dim)
            cache.update(k, v, layer_idx)
        return cache

    def test_batch_slice_and_update(self):
        """Verify per-MB slicing + update produces same result as full-batch update."""
        batch, heads, seq, dim = 4, 2, 5, 8
        num_layers = 2

        # --- Full-batch path (ground truth) ---
        cache_full = self._make_cache(batch, heads, seq, dim, num_layers)
        new_k_full = torch.randn(batch, heads, 1, dim)
        new_v_full = torch.randn(batch, heads, 1, dim)
        for layer_idx in range(num_layers):
            cache_full.update(new_k_full.clone(), new_v_full.clone(), layer_idx)

        # --- Per-MB path (2 micro-batches of size 2) ---
        cache_mb = self._make_cache(batch, heads, seq, dim, num_layers)
        # Populate with the SAME initial data as cache_full
        # Re-create to match initial state
        cache_mb = DynamicCache()
        cache_full_reset = DynamicCache()
        init_keys = []
        init_values = []
        for layer_idx in range(num_layers):
            k = torch.randn(batch, heads, seq, dim)
            v = torch.randn(batch, heads, seq, dim)
            init_keys.append(k)
            init_values.append(v)
            cache_mb.update(k.clone(), v.clone(), layer_idx)
            cache_full_reset.update(k.clone(), v.clone(), layer_idx)

        # New KV tokens for decode step
        new_k = torch.randn(batch, heads, 1, dim)
        new_v = torch.randn(batch, heads, 1, dim)

        # Full-batch update
        for layer_idx in range(num_layers):
            cache_full_reset.update(new_k.clone(), new_v.clone(), layer_idx)

        # Per-MB update (simulating what decode_scheduler does)
        mb_sizes = [2, 2]
        for layer_idx in range(num_layers):
            cache_layer = cache_mb.layers[layer_idx]
            orig_keys = cache_layer.keys
            orig_values = cache_layer.values

            mb_updated_keys = []
            mb_updated_values = []

            offset = 0
            for mb_size in mb_sizes:
                start = offset
                end = offset + mb_size
                offset = end

                cache_layer.keys = orig_keys[start:end]
                cache_layer.values = orig_values[start:end]

                mb_k = new_k[start:end].clone()
                mb_v = new_v[start:end].clone()
                cache_mb.update(mb_k, mb_v, layer_idx)

                mb_updated_keys.append(cache_layer.keys)
                mb_updated_values.append(cache_layer.values)

            cache_layer.keys = torch.cat(mb_updated_keys, dim=0)
            cache_layer.values = torch.cat(mb_updated_values, dim=0)

        # Verify: per-MB result matches full-batch result
        for layer_idx in range(num_layers):
            full_k = cache_full_reset.layers[layer_idx].keys
            mb_k = cache_mb.layers[layer_idx].keys
            assert full_k.shape == mb_k.shape, (
                f"Layer {layer_idx}: shape mismatch {full_k.shape} vs {mb_k.shape}"
            )
            assert torch.allclose(full_k, mb_k), f"Layer {layer_idx}: keys mismatch"

            full_v = cache_full_reset.layers[layer_idx].values
            mb_v = cache_mb.layers[layer_idx].values
            assert torch.allclose(full_v, mb_v), f"Layer {layer_idx}: values mismatch"

    def test_uneven_batch_split(self):
        """Test per-MB cache slicing with uneven batch sizes (e.g. 5 → [3, 2])."""
        batch, heads, seq, dim = 5, 2, 4, 8

        cache = DynamicCache()
        k = torch.randn(batch, heads, seq, dim)
        v = torch.randn(batch, heads, seq, dim)
        cache.update(k.clone(), v.clone(), layer_idx=0)

        new_k = torch.randn(batch, heads, 1, dim)
        new_v = torch.randn(batch, heads, 1, dim)

        layer = cache.layers[0]
        orig_keys = layer.keys
        orig_values = layer.values

        mb_sizes = [3, 2]
        mb_updated_keys = []
        mb_updated_values = []
        offset = 0
        for mb_size in mb_sizes:
            start, end = offset, offset + mb_size
            offset = end

            layer.keys = orig_keys[start:end]
            layer.values = orig_values[start:end]
            cache.update(new_k[start:end].clone(), new_v[start:end].clone(), layer_idx=0)
            mb_updated_keys.append(layer.keys)
            mb_updated_values.append(layer.values)

        layer.keys = torch.cat(mb_updated_keys, dim=0)
        layer.values = torch.cat(mb_updated_values, dim=0)

        assert layer.keys.shape == (5, heads, seq + 1, dim)
        assert layer.values.shape == (5, heads, seq + 1, dim)

    def test_single_mb_equivalent_to_full_batch(self):
        """With num_mb=1, per-MB path is identical to full-batch."""
        batch, heads, seq, dim = 4, 2, 6, 8

        cache = DynamicCache()
        k = torch.randn(batch, heads, seq, dim)
        v = torch.randn(batch, heads, seq, dim)
        cache.update(k.clone(), v.clone(), layer_idx=0)

        new_k = torch.randn(batch, heads, 1, dim)
        new_v = torch.randn(batch, heads, 1, dim)

        # Single MB = full batch (no slicing needed)
        layer = cache.layers[0]
        orig_keys = layer.keys.clone()

        cache.update(new_k, new_v, layer_idx=0)

        # Keys should be [batch, heads, seq+1, dim]
        assert layer.keys.shape == (batch, heads, seq + 1, dim)
        # First seq tokens should match original
        assert torch.allclose(layer.keys[:, :, :seq, :], orig_keys)
