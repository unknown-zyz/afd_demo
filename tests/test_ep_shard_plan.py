import pytest

from src.model.ep_moe import ExpertShardPlan


@pytest.mark.parametrize("ep_size", [1, 4, 8, 15])
@pytest.mark.parametrize("policy", ["round_robin", "contiguous"])
def test_expert_assignments_cover_all_experts(ep_size, policy):
    assignments = ExpertShardPlan.all_assignments(128, ep_size, policy)
    flattened = [expert for shard in assignments for expert in shard]

    assert sorted(flattened) == list(range(128))
    assert len(flattened) == len(set(flattened))


@pytest.mark.parametrize("ep_size", [4, 8, 15])
def test_round_robin_assignment_is_balanced(ep_size):
    assignments = ExpertShardPlan.all_assignments(128, ep_size, "round_robin")
    sizes = [len(shard) for shard in assignments]

    assert max(sizes) - min(sizes) <= 1


def test_ep15_round_robin_matches_expected_prefix():
    assignments = ExpertShardPlan.all_assignments(128, 15, "round_robin")

    assert assignments[0] == [0, 15, 30, 45, 60, 75, 90, 105, 120]
    assert assignments[7] == [7, 22, 37, 52, 67, 82, 97, 112, 127]
    assert assignments[14] == [14, 29, 44, 59, 74, 89, 104, 119]


def test_invalid_shard_plan_rejected():
    with pytest.raises(ValueError):
        ExpertShardPlan(128, 0, 0)
    with pytest.raises(ValueError):
        ExpertShardPlan(128, 4, 4)
    with pytest.raises(ValueError):
        ExpertShardPlan(128, 4, 0, "bad")
