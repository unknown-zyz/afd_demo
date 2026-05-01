"""Helpers for role-local layer placement across visible accelerator devices."""

import math
import os
from collections import Counter
from typing import Sequence

import torch


def resolve_role_devices(primary_device: torch.device, role: str) -> list[torch.device]:
    """Resolve the accelerator devices this role may use for layer placement."""
    from ..utils import device as devmod

    if primary_device.type not in ("cuda", "npu") or not devmod.is_available():
        return [primary_device]

    devices = [torch.device(f"{primary_device.type}:{idx}") for idx in range(devmod.device_count())]
    limit_env = (
        os.environ.get("AFD_ROLE_LAYER_DEVICES")
        or os.environ.get(f"AFD_{role.upper()}_LAYER_DEVICES")
        or os.environ.get("AFD_LAYER_DEVICES")
    )
    if not limit_env:
        return devices

    try:
        limit = int(limit_env)
    except ValueError as exc:
        raise ValueError(f"Invalid AFD layer device count for {role}: {limit_env!r}") from exc
    if limit < 1:
        raise ValueError(f"AFD layer device count for {role} must be >= 1, got {limit}")
    return devices[: min(limit, len(devices))]


def select_layer_device(
    layer_idx: int,
    num_layers: int,
    role_devices: Sequence[torch.device],
) -> torch.device:
    """Select a device for one transformer layer according to the configured policy."""
    if not role_devices:
        raise ValueError("role_devices must not be empty")

    policy = os.environ.get("AFD_LAYER_SHARD_POLICY", "legacy").strip().lower()
    if policy in ("", "legacy"):
        if len(role_devices) >= 2:
            layers_on_dev0 = max(1, num_layers // len(role_devices) - 3)
            device_idx = 0 if layer_idx < layers_on_dev0 else min(1, len(role_devices) - 1)
        else:
            device_idx = 0
        return role_devices[device_idx]

    if policy == "contiguous":
        layers_per_device = math.ceil(num_layers / len(role_devices))
        device_idx = min(layer_idx // layers_per_device, len(role_devices) - 1)
        return role_devices[device_idx]

    if policy == "round_robin":
        return role_devices[layer_idx % len(role_devices)]

    raise ValueError(
        f"Unsupported AFD_LAYER_SHARD_POLICY={policy!r}; "
        "expected legacy, contiguous, or round_robin"
    )


def summarize_layer_devices(layer_devices: Sequence[torch.device]) -> dict[str, int]:
    """Return a compact count of how many layers are placed on each device."""
    return dict(Counter(str(device) for device in layer_devices))
