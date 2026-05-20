"""Unit tests for LoadAwareRouter (no protobuf required, uses dicts)."""

import pytest

from src.coordinator_arch.router import LoadAwareRouter


def _m(rank, role="ffn", queue_len_avg=0.0, dispatch_rate_tps=0.0, per_expert_load=None):
    return {
        "role": role, "rank": rank,
        "queue_len_avg": queue_len_avg,
        "dispatch_rate_tps": dispatch_rate_tps,
        "per_expert_load": per_expert_load or [],
    }


class TestLoadAwareRouter:
    def test_uniform_no_rebalance(self):
        r = LoadAwareRouter(num_experts=16, ffn_world=4, attn_world=0)
        prev = [i // 4 for i in range(16)]
        metrics = [_m(rank=i, queue_len_avg=1.0) for i in range(4)]
        assert r.rebalance(metrics, prev) is None

    def test_skewed_triggers_rebalance(self):
        r = LoadAwareRouter(
            num_experts=16, ffn_world=4, attn_world=0,
            imbalance_threshold=0.10, max_moves_per_rebalance=16,
        )
        prev = [i // 4 for i in range(16)]
        # rank 0 hot, others cold
        metrics = [
            _m(rank=0, queue_len_avg=10.0),
            _m(rank=1, queue_len_avg=1.0),
            _m(rank=2, queue_len_avg=1.0),
            _m(rank=3, queue_len_avg=1.0),
        ]
        new = r.rebalance(metrics, prev)
        assert new is not None
        assert len(new) == 16
        assert all(0 <= v < 4 for v in new)
        # Each rank should own 4 experts under uniform per-expert load.
        counts = [new.count(i) for i in range(4)]
        assert max(counts) - min(counts) <= 1

    def test_smoothing_caps_moves(self):
        r = LoadAwareRouter(
            num_experts=16, ffn_world=4, attn_world=0,
            imbalance_threshold=0.05, max_moves_per_rebalance=3,
        )
        prev = [i // 4 for i in range(16)]
        metrics = [
            _m(rank=0, queue_len_avg=100.0),
            _m(rank=1, queue_len_avg=1.0),
            _m(rank=2, queue_len_avg=1.0),
            _m(rank=3, queue_len_avg=1.0),
        ]
        new = r.rebalance(metrics, prev)
        assert new is not None
        moves = sum(1 for e in range(16) if new[e] != prev[e])
        assert moves <= 3, f"got {moves} moves, expected <= 3"

    def test_attn_world_offset(self):
        """FFN ranks start at attn_world; router must subtract correctly."""
        r = LoadAwareRouter(num_experts=8, ffn_world=2, attn_world=4)
        prev = [i // 4 for i in range(8)]  # 0,0,0,0,1,1,1,1
        metrics = [
            _m(rank=4, queue_len_avg=10.0),  # local 0
            _m(rank=5, queue_len_avg=1.0),   # local 1
        ]
        new = r.rebalance(metrics, prev)
        assert new is not None
        assert all(0 <= v < 2 for v in new)

    def test_invalid_prev_table(self):
        r = LoadAwareRouter(num_experts=8, ffn_world=2)
        with pytest.raises(ValueError):
            r.rebalance([], prev_table=[0, 0, 0])  # wrong length

    def test_invalid_init(self):
        with pytest.raises(ValueError):
            LoadAwareRouter(num_experts=0, ffn_world=2)
        with pytest.raises(ValueError):
            LoadAwareRouter(num_experts=8, ffn_world=0)
