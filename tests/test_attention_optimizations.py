import os

import torch
import torch.nn as nn

from src.distributed import DistributedConfig, DistributedContext
from src.model.attention_worker import (
    AttentionLayer,
    AttentionLayerInputCache,
    AttentionLayerInputs,
    AttentionOptimizationConfig,
)
from src.utils.timing import PipelineTiming


class FakeAttention(nn.Module):
    def forward(self, hidden_states, attention_mask=None, position_ids=None):
        return hidden_states


def test_attention_optimization_config_to_dict():
    cfg = AttentionOptimizationConfig(
        attn_kernel="hf",
        precopy_layer_inputs=True,
        attn_tp_size=2,
        fused_rmsnorm=True,
        fused_rope=False,
        stream_overlap=True,
    )

    assert cfg.to_dict() == {
        "attn_kernel": "hf",
        "attn_precopy_layer_inputs": True,
        "attn_tp_size": 2,
        "attn_fused_rmsnorm": True,
        "attn_fused_rope": False,
        "attn_stream_overlap": True,
    }


def test_pipeline_timing_serializes_attention_optimizations():
    timing = PipelineTiming(
        node="attention",
        attention_optimizations={"attn_kernel": "hf", "attn_precopy_layer_inputs": True},
    )

    data = timing.to_dict()

    assert data["attention_optimizations"] == {
        "attn_kernel": "hf",
        "attn_precopy_layer_inputs": True,
    }


def test_attention_layer_uses_prepositioned_inputs():
    layer = AttentionLayer(
        input_layernorm=nn.Identity(),
        self_attn=FakeAttention(),
        hidden_size=4,
        layer_idx=0,
        layer_device=torch.device("cpu"),
        output_device=torch.device("cpu"),
    )
    hidden_states = torch.randn(2, 3, 4)
    attention_mask = torch.zeros(2, 1, 3, 3)
    position_ids = torch.arange(3).unsqueeze(0).expand(2, -1)

    attn_output, residual = layer(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        inputs_prepositioned=True,
    )

    assert torch.equal(attn_output, hidden_states)
    assert torch.equal(residual, hidden_states)


def test_attention_layer_input_cache_gets_by_device():
    inputs = AttentionLayerInputs(
        attention_mask=torch.zeros(1, 1, 2, 2),
        position_ids=torch.arange(2).unsqueeze(0),
    )
    cache = AttentionLayerInputCache({"cpu": inputs})

    assert cache.get(torch.device("cpu")) is inputs


def test_distributed_context_reports_reserved_and_active_npus():
    old_active = os.environ.get("AFD_ACTIVE_NPUS")
    os.environ["AFD_ACTIVE_NPUS"] = "0,1"
    try:
        ctx = DistributedContext()
        ctx.config = DistributedConfig(
            rank=0,
            world_size=1,
            local_rank=0,
            reserved_npus="2,3",
        )

        assert ctx.reserved_npus == [2, 3]
        assert ctx.active_npus == [0, 1]
    finally:
        if old_active is None:
            os.environ.pop("AFD_ACTIVE_NPUS", None)
        else:
            os.environ["AFD_ACTIVE_NPUS"] = old_active
