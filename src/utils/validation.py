"""Small tensor validation helpers used by tests and debugging scripts."""

import logging
from typing import Dict, Tuple

import torch

logger = logging.getLogger(__name__)


def compare_tensors(
    tensor1: torch.Tensor,
    tensor2: torch.Tensor,
    rtol: float = 1e-3,
    atol: float = 1e-5,
    name: str = "tensor",
) -> Tuple[bool, Dict[str, float]]:
    """Compare two tensors and return allclose status plus diff metrics."""
    if tensor1.device != tensor2.device:
        tensor2 = tensor2.to(tensor1.device)

    abs_diff = torch.abs(tensor1 - tensor2)
    rel_diff = abs_diff / (torch.abs(tensor1) + 1e-10)
    metrics = {
        "max_abs_diff": abs_diff.max().item(),
        "mean_abs_diff": abs_diff.mean().item(),
        "max_rel_diff": rel_diff.max().item(),
        "mean_rel_diff": rel_diff.mean().item(),
    }
    is_close = torch.allclose(tensor1, tensor2, rtol=rtol, atol=atol)

    logger.info(
        "Comparison for %s: is_close=%s, max_abs_diff=%.6f, mean_abs_diff=%.6f",
        name,
        is_close,
        metrics["max_abs_diff"],
        metrics["mean_abs_diff"],
    )
    return is_close, metrics


def validate_output(
    output: torch.Tensor,
    reference: torch.Tensor,
    rtol: float = 1e-3,
    atol: float = 1e-5,
) -> bool:
    """Return whether ``output`` matches ``reference`` within tolerance."""
    is_close, _ = compare_tensors(reference, output, rtol=rtol, atol=atol, name="model_output")
    return is_close
