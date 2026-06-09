import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from src.model.ep_moe import EPFFNLayer, ExpertShardPlan, ShardedExperts


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


def reference_dispatched(experts, recv_hidden, recv_experts):
    out = torch.empty_like(recv_hidden)
    for pos, expert_id in enumerate(recv_experts.tolist()):
        x = recv_hidden[pos:pos + 1]
        gate_up = torch.nn.functional.linear(x, experts.gate_up_proj[expert_id])
        gate, up = gate_up.chunk(2, dim=-1)
        out[pos:pos + 1] = torch.nn.functional.linear(
            experts.act_fn(gate) * up,
            experts.down_proj[expert_id],
        )
    return out


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


def test_sharded_experts_dispatched_matches_reference():
    experts = TinyExperts()
    generator = torch.Generator().manual_seed(789)
    recv_hidden = torch.randn(6, 4, generator=generator)
    recv_experts = torch.tensor([0, 3, 0, 3, 0, 3], dtype=torch.long)

    plan = ExpertShardPlan(experts.num_experts, 3, 0, "round_robin")
    shard = ShardedExperts(experts, plan, device=torch.device("cpu"), dtype=torch.float32)
    actual, active, assignments = shard.forward_dispatched(recv_hidden, recv_experts)
    expected = reference_dispatched(experts, recv_hidden, recv_experts)

    assert active == 2
    assert assignments == recv_hidden.shape[0]
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)


class _SparseConfig:
    ffn_ep_backend = "sparse_p2p_overlap"


class _SparseCtx:
    def __init__(self, rank, world_size):
        self.config = _SparseConfig()
        self.rank = rank
        self.ffn_ep_group = dist.group.WORLD
        self.ffn_coordinator_rank = 0
        self.ffn_ranks = list(range(world_size))
        self.ffn_ep_size = world_size
        self.ffn_ep_rank = rank
        self.is_ffn_coordinator = rank == 0


def _sparse_p2p_worker(rank, world_size, queue):
    try:
        dist.init_process_group(
            backend="gloo",
            init_method="tcp://127.0.0.1:29531",
            rank=rank,
            world_size=world_size,
        )
        experts = TinyExperts(num_experts=6, hidden=4, intermediate=5)
        plan = ExpertShardPlan(experts.num_experts, world_size, rank, "round_robin")
        shard = ShardedExperts(experts, plan, device=torch.device("cpu"), dtype=torch.float32)
        layer = EPFFNLayer(
            post_attention_layernorm=None,
            gate=None,
            sharded_experts=shard,
            hidden_size=4,
            top_k=2,
            layer_idx=0,
            layer_device=torch.device("cpu"),
            ctx=_SparseCtx(rank, world_size),
        )

        if rank == 0:
            generator = torch.Generator().manual_seed(321)
            hidden = torch.randn(5, 4, generator=generator)
            selected = torch.tensor(
                [[0, 1], [2, 3], [4, 5], [0, 2], [3, 5]],
                dtype=torch.long,
            )
            weights = torch.rand(5, 2, generator=generator)
            weights = weights / weights.sum(dim=-1, keepdim=True)
        else:
            hidden = torch.empty(0, 4)
            selected = torch.empty(0, 2, dtype=torch.long)
            weights = torch.empty(0, 2)

        handle = layer._sparse_p2p_dispatch_async(hidden, selected, weights)
        handle = layer._sparse_p2p_wait_dispatch(handle)
        ffn_outputs, _active, _assignments = shard.forward_dispatched(
            handle["recv_hidden"],
            handle["recv_experts"],
        )
        combine_handle = layer._sparse_p2p_combine_async(ffn_outputs, handle)
        combined = layer._sparse_p2p_wait_combine(combine_handle)

        if rank == 0:
            expected = reference_forward(experts, hidden, selected, weights)
            assert combined is not None
            assert torch.allclose(combined, expected, atol=1e-5, rtol=1e-5)
        else:
            assert combined is None
        queue.put((rank, True, None))
    except Exception as exc:
        queue.put((rank, False, str(exc)))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def test_sparse_p2p_assignment_combine_matches_reference():
    if not dist.is_available() or not dist.is_gloo_available():
        return
    world_size = 3
    mp.set_start_method("spawn", force=True)
    queue = mp.Queue()
    processes = [
        mp.Process(target=_sparse_p2p_worker, args=(rank, world_size, queue))
        for rank in range(world_size)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(30)
    results = []
    while not queue.empty():
        results.append(queue.get())
    assert len(results) == world_size
    for rank, success, error in results:
        assert success, f"rank {rank} failed: {error}"
