"""
Load-aware routing algorithm for the Coordinator.

Given current FFN worker metrics, decides whether to rebalance the
expert→FFN-rank mapping. Uses a greedy bin-packing heuristic with
"max moves per rebalance" smoothing to avoid thrashing.

See doc/12-coordinator-arch.md §6.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence

logger = logging.getLogger(__name__)


class LoadAwareRouter:
    def __init__(
        self,
        num_experts: int,
        ffn_world: int,
        attn_world: int = 0,
        imbalance_threshold: float = 0.10,
        max_moves_per_rebalance: int = 10,
        dispatch_rate_weight: float = 0.01,
    ):
        if num_experts <= 0:
            raise ValueError("num_experts must be > 0")
        if ffn_world <= 0:
            raise ValueError("ffn_world must be > 0")
        self.num_experts = num_experts
        self.ffn_world = ffn_world
        self.attn_world = attn_world
        self.imbalance_threshold = imbalance_threshold
        self.max_moves = max_moves_per_rebalance
        self.dispatch_rate_weight = dispatch_rate_weight

    def rebalance(
        self,
        ffn_metrics,
        prev_table: Sequence[int],
    ) -> Optional[List[int]]:
        """
        Returns new expert_to_rank, or None if no rebalance needed.

        ffn_metrics: list of pb.WorkerMetrics OR list of dicts with the same
            fields (for unit tests without protobuf).
        prev_table: current expert_to_rank (len = num_experts).
        """
        if len(prev_table) != self.num_experts:
            raise ValueError(
                f"prev_table len {len(prev_table)} != num_experts {self.num_experts}"
            )

        # Aggregate per-rank cost.
        cost = [0.0] * self.ffn_world
        per_expert_load_acc: List[float] = [0.0] * self.num_experts
        local_experts_by_rank: dict = {}

        for m in ffn_metrics:
            role = _get(m, "role")
            if role != "ffn":
                continue
            # rank field is "global" — FFN ranks start at attn_world.
            global_rank = int(_get(m, "rank"))
            local_rank = global_rank - self.attn_world
            if local_rank < 0 or local_rank >= self.ffn_world:
                logger.warning(
                    "FFN metric rank=%d out of bounds (attn_world=%d, ffn_world=%d), skipping",
                    global_rank, self.attn_world, self.ffn_world,
                )
                continue
            q = float(_get(m, "queue_len_avg", 0.0))
            r = float(_get(m, "dispatch_rate_tps", 0.0))
            cost[local_rank] = q + self.dispatch_rate_weight * r

            # Accumulate per-expert load if reported.
            per_expert = _get(m, "per_expert_load", None) or []
            # Need to know which absolute experts this rank owns to map.
            owned = [e for e, r in enumerate(prev_table) if r == local_rank]
            if per_expert and len(per_expert) == len(owned):
                for idx, abs_expert in enumerate(owned):
                    per_expert_load_acc[abs_expert] = float(per_expert[idx])

        # Imbalance check.
        if not cost or max(cost) <= 0:
            return None
        lo = max(min(cost), 1e-3)
        imb = (max(cost) - min(cost)) / lo
        if imb < self.imbalance_threshold:
            return None
        logger.info(
            "Rebalance triggered: cost=%s imbalance=%.3f threshold=%.3f",
            [round(c, 3) for c in cost], imb, self.imbalance_threshold,
        )

        # If no per-expert load reported, fall back to uniform per-expert weight.
        if all(v == 0 for v in per_expert_load_acc):
            per_expert_load_acc = [1.0] * self.num_experts

        # Greedy bin-pack: experts in descending load → least-loaded bin.
        order = sorted(
            range(self.num_experts), key=lambda e: -per_expert_load_acc[e]
        )
        bins = [0.0] * self.ffn_world
        new_table: List[int] = [-1] * self.num_experts
        for e in order:
            r = min(range(self.ffn_world), key=lambda i: bins[i])
            new_table[e] = r
            bins[r] += per_expert_load_acc[e]

        # Smoothing: limit number of expert reassignments per rebalance.
        moved = [
            e for e in range(self.num_experts) if new_table[e] != prev_table[e]
        ]
        if len(moved) > self.max_moves:
            # Keep the K moves with largest per-expert load (most impactful).
            moved_sorted = sorted(
                moved, key=lambda e: -per_expert_load_acc[e]
            )
            keep = set(moved_sorted[: self.max_moves])
            for e in moved:
                if e not in keep:
                    new_table[e] = prev_table[e]
            logger.info(
                "Smoothed rebalance: capped %d moves at max=%d",
                len(moved), self.max_moves,
            )

        if all(new_table[e] == prev_table[e] for e in range(self.num_experts)):
            return None
        return new_table


def _get(obj, name, default=None):
    """Get attribute from protobuf message or key from dict, with default."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)
