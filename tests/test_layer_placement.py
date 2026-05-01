import torch

from src.model.layer_placement import select_layer_device, summarize_layer_devices


def test_legacy_layer_placement_preserves_two_device_behavior(monkeypatch):
    monkeypatch.delenv("AFD_LAYER_SHARD_POLICY", raising=False)
    devices = [torch.device(f"cuda:{idx}") for idx in range(8)]

    selected = [select_layer_device(idx, 48, devices) for idx in range(48)]

    assert summarize_layer_devices(selected) == {"cuda:0": 3, "cuda:1": 45}


def test_contiguous_layer_placement_uses_all_requested_devices(monkeypatch):
    monkeypatch.setenv("AFD_LAYER_SHARD_POLICY", "contiguous")
    devices = [torch.device(f"cuda:{idx}") for idx in range(4)]

    selected = [select_layer_device(idx, 48, devices) for idx in range(48)]

    assert summarize_layer_devices(selected) == {
        "cuda:0": 12,
        "cuda:1": 12,
        "cuda:2": 12,
        "cuda:3": 12,
    }
