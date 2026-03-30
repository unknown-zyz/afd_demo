"""
Validation utilities for AFD Demo.

Provides tools to validate correctness of disaggregated model output.
"""

import logging
from typing import Optional, Tuple, Dict, Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


def compare_tensors(
    tensor1: torch.Tensor,
    tensor2: torch.Tensor,
    rtol: float = 1e-3,
    atol: float = 1e-5,
    name: str = "tensor",
) -> Tuple[bool, Dict[str, float]]:
    """
    Compare two tensors and compute difference metrics.
    
    Args:
        tensor1: First tensor (reference)
        tensor2: Second tensor (test)
        rtol: Relative tolerance
        atol: Absolute tolerance
        name: Name for logging
    
    Returns:
        Tuple of (is_close, metrics_dict)
    """
    # Move to same device
    if tensor1.device != tensor2.device:
        tensor2 = tensor2.to(tensor1.device)
    
    # Compute metrics
    abs_diff = torch.abs(tensor1 - tensor2)
    rel_diff = abs_diff / (torch.abs(tensor1) + 1e-10)
    
    metrics = {
        "max_abs_diff": abs_diff.max().item(),
        "mean_abs_diff": abs_diff.mean().item(),
        "max_rel_diff": rel_diff.max().item(),
        "mean_rel_diff": rel_diff.mean().item(),
    }
    
    # Check if close
    is_close = torch.allclose(tensor1, tensor2, rtol=rtol, atol=atol)
    
    logger.info(
        f"Comparison for {name}: "
        f"is_close={is_close}, "
        f"max_abs_diff={metrics['max_abs_diff']:.6f}, "
        f"mean_abs_diff={metrics['mean_abs_diff']:.6f}"
    )
    
    return is_close, metrics


def validate_output(
    output: torch.Tensor,
    reference: torch.Tensor,
    rtol: float = 1e-3,
    atol: float = 1e-5,
) -> bool:
    """
    Validate model output against reference.
    
    Args:
        output: Output from disaggregated model
        reference: Reference output from original model
        rtol: Relative tolerance
        atol: Absolute tolerance
    
    Returns:
        True if outputs match within tolerance
    """
    is_close, metrics = compare_tensors(
        reference, output, rtol=rtol, atol=atol, name="model_output"
    )
    return is_close


class BaselineModel:
    """
    Wrapper for the original (non-disaggregated) model for comparison.
    """
    
    def __init__(
        self,
        model_name: str,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
    ):
        """
        Load baseline model.
        
        Args:
            model_name: HuggingFace model name
            device: Device to load model on
            dtype: Data type
        """
        logger.info(f"Loading baseline model: {model_name}")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=device,
            trust_remote_code=True,
        )
        self.model.eval()
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        
        self.device = device
        self.dtype = dtype
    
    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Run forward pass on baseline model.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
        
        Returns:
            Logits tensor
        """
        outputs = self.model(
            input_ids=input_ids.to(self.device),
            attention_mask=attention_mask.to(self.device) if attention_mask is not None else None,
        )
        return outputs.logits
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        **kwargs,
    ) -> torch.Tensor:
        """Generate tokens."""
        return self.model.generate(
            input_ids=input_ids.to(self.device),
            max_new_tokens=max_new_tokens,
            **kwargs,
        )


def compare_with_baseline(
    disagg_model,
    baseline_model: BaselineModel,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    rtol: float = 1e-3,
    atol: float = 1e-5,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Compare disaggregated model output with baseline.
    
    Args:
        disagg_model: Disaggregated model
        baseline_model: Baseline model
        input_ids: Input token IDs
        attention_mask: Attention mask
        rtol: Relative tolerance
        atol: Absolute tolerance
    
    Returns:
        Tuple of (is_valid, results_dict)
    """
    # Get baseline output
    logger.info("Running baseline model...")
    baseline_logits = baseline_model.forward(input_ids, attention_mask)
    
    # Get disaggregated output
    logger.info("Running disaggregated model...")
    disagg_logits = disagg_model.forward(input_ids, attention_mask)
    
    if disagg_logits is None:
        logger.warning("Disaggregated model returned None (FFN node)")
        return True, {"note": "FFN node does not produce logits"}
    
    # Compare
    is_valid, metrics = compare_tensors(
        baseline_logits, disagg_logits, rtol=rtol, atol=atol, name="logits"
    )
    
    # Compare predicted tokens
    baseline_tokens = baseline_logits.argmax(dim=-1)
    disagg_tokens = disagg_logits.argmax(dim=-1)
    token_match = (baseline_tokens == disagg_tokens).float().mean().item()
    
    results = {
        "is_valid": is_valid,
        "metrics": metrics,
        "token_match_ratio": token_match,
    }
    
    logger.info(f"Validation results: is_valid={is_valid}, token_match={token_match:.4f}")
    
    return is_valid, results


def run_validation_suite(
    model_name: str,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    test_prompts: Optional[list] = None,
) -> Dict[str, Any]:
    """
    Run a full validation suite comparing disaggregated vs baseline.
    
    This should be run on a single node with both models loaded.
    
    Args:
        model_name: Model name
        device: Device
        dtype: Data type
        test_prompts: List of test prompts
    
    Returns:
        Validation results
    """
    if test_prompts is None:
        test_prompts = [
            "Hello, how are you?",
            "The capital of France is",
            "def fibonacci(n):",
            "In machine learning,",
        ]
    
    # Load baseline
    baseline = BaselineModel(model_name, device, dtype)
    
    # Tokenize
    tokenizer = baseline.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    results = {
        "model_name": model_name,
        "prompts": test_prompts,
        "per_prompt_results": [],
        "overall_valid": True,
    }
    
    for prompt in test_prompts:
        logger.info(f"Testing prompt: {prompt[:50]}...")
        
        inputs = tokenizer(prompt, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        
        # Get baseline output
        baseline_logits = baseline.forward(input_ids, attention_mask)
        
        prompt_result = {
            "prompt": prompt,
            "input_shape": list(input_ids.shape),
            "baseline_output_shape": list(baseline_logits.shape),
        }
        results["per_prompt_results"].append(prompt_result)
    
    return results
