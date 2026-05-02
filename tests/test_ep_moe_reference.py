import torch

from src.model.ep_moe import ExpertShardPlan, ShardedExperts


class TinyExperts:
    def __init__(self, num_experts=6, hidden=4, intermediate=5):
        generator = torch.Generator().manual_seed(123)
        self.num_experts = num_experts
        self.gate_up_proj = torch.randn(num_experts, intermediate * 2, hidden, generator=generator)
        self.down_proj = torch.randn(num_experts, hidden, intermediate, generator=generator)
        self.act_fn = torch.nn.SiLU()


def reference_forward(experts, hidden_2d, selected_experts, routing_weights):
    final = torch.zeros_like(hidden_2d)
    for expert_id in range(experts.num_experts):
        token_idx, topk_idx = torch.where(selected_experts == expert_id)
        if token_idx.numel() == 0:
            continue
        x = hidden_2d[token_idx]
        gate_up = torch.nn.functional.linear(x, experts.gate_up_proj[expert_id])
        gate, up = gate_up.chunk(2, dim=-1)
        out = torch.nn.functional.linear(experts.act_fn(gate) * up, experts.down_proj[expert_id])
        final.index_add_(0, token_idx, out * routing_weights[token_idx, topk_idx, None])
    return final


def test_sharded_experts_sum_matches_reference():
    experts = TinyExperts()
    generator = torch.Generator().manual_seed(456)
    hidden_2d = torch.randn(7, 4, generator=generator)
    selected_experts = torch.tensor(
        [
            [0, 1],
            [2, 3],
            [4, 5],
            [0, 5],
            [1, 4],
            [2, 0],
            [3, 1],
        ],
        dtype=torch.long,
    )
    routing_weights = torch.rand(7, 2, generator=generator)
    routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)

    partials = []
    for ep_rank in range(3):
        plan = ExpertShardPlan(experts.num_experts, 3, ep_rank, "round_robin")
        shard = ShardedExperts(experts, plan, device=torch.device("cpu"), dtype=torch.float32)
        partial, active, assignments = shard.forward_local(hidden_2d, selected_experts, routing_weights)
        assert active > 0
        assert assignments > 0
        partials.append(partial)

    actual = torch.stack(partials).sum(dim=0)
    expected = reference_forward(experts, hidden_2d, selected_experts, routing_weights)
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)
