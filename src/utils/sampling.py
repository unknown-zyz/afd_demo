"""
Sampling utilities for text generation.

Supports temperature scaling, top-k, and top-p (nucleus) sampling.
"""

from typing import Optional

import torch
import torch.nn.functional as F


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.9,
    do_sample: bool = True,
) -> torch.Tensor:
    """
    Sample next token from logits.
    
    Args:
        logits: Logits for next token [batch_size, vocab_size]
        temperature: Temperature for scaling (1.0 = no change)
        top_k: Keep only top-k tokens (0 = disabled)
        top_p: Keep tokens with cumulative prob <= top_p (1.0 = disabled)
        do_sample: If False, use greedy decoding
    
    Returns:
        Sampled token IDs [batch_size, 1]
    """
    if not do_sample:
        # Greedy decoding
        return logits.argmax(dim=-1, keepdim=True)
    
    # Apply temperature
    if temperature != 1.0:
        logits = logits / temperature
    
    # Apply top-k filtering
    if top_k > 0:
        logits = top_k_filtering(logits, top_k)
    
    # Apply top-p (nucleus) filtering
    if top_p < 1.0:
        logits = top_p_filtering(logits, top_p)
    
    # Sample from filtered distribution
    probs = F.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    
    return next_token


def top_k_filtering(
    logits: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    """
    Keep only top-k tokens, set others to -inf.
    
    Args:
        logits: [batch_size, vocab_size]
        top_k: Number of tokens to keep
    
    Returns:
        Filtered logits
    """
    top_k = min(top_k, logits.size(-1))
    
    # Get the k-th largest value
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits = logits.masked_fill(indices_to_remove, float('-inf'))
    
    return logits


def top_p_filtering(
    logits: torch.Tensor,
    top_p: float,
    min_tokens_to_keep: int = 1,
) -> torch.Tensor:
    """
    Keep tokens with cumulative probability <= top_p (nucleus sampling).
    
    Args:
        logits: [batch_size, vocab_size]
        top_p: Cumulative probability threshold
        min_tokens_to_keep: Minimum tokens to keep regardless of top_p
    
    Returns:
        Filtered logits
    """
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    
    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > top_p
    
    # Shift the indices to the right to keep the first token above threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False
    
    # Keep at least min_tokens_to_keep
    sorted_indices_to_remove[..., :min_tokens_to_keep] = False
    
    # Scatter back to original indices
    indices_to_remove = sorted_indices_to_remove.scatter(
        dim=-1, index=sorted_indices, src=sorted_indices_to_remove
    )
    logits = logits.masked_fill(indices_to_remove, float('-inf'))
    
    return logits


def repetition_penalty(
    logits: torch.Tensor,
    generated_ids: torch.Tensor,
    penalty: float = 1.2,
) -> torch.Tensor:
    """
    Apply repetition penalty to discourage repeating tokens.
    
    Args:
        logits: [batch_size, vocab_size]
        generated_ids: Previously generated token IDs [batch_size, seq_len]
        penalty: Penalty factor (> 1.0 discourages repetition)
    
    Returns:
        Logits with penalty applied
    """
    if penalty == 1.0:
        return logits
    
    # Gather logits for generated tokens
    for i in range(logits.shape[0]):
        for token_id in generated_ids[i].unique():
            if logits[i, token_id] < 0:
                logits[i, token_id] *= penalty
            else:
                logits[i, token_id] /= penalty
    
    return logits


class StoppingCriteria:
    """Criteria for stopping generation."""
    
    def __init__(
        self,
        max_length: int,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
    ):
        self.max_length = max_length
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
    
    def should_stop(
        self,
        input_ids: torch.Tensor,
        current_length: int,
    ) -> torch.Tensor:
        """
        Check which sequences should stop.
        
        Args:
            input_ids: Current generated IDs [batch_size, seq_len]
            current_length: Current total length including prompt
        
        Returns:
            Boolean tensor indicating which sequences should stop [batch_size]
        """
        batch_size = input_ids.shape[0]
        stop_mask = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)
        
        # Stop if max length reached
        if current_length >= self.max_length:
            stop_mask[:] = True
            return stop_mask
        
        # Stop if EOS generated
        if self.eos_token_id is not None:
            last_token = input_ids[:, -1]
            stop_mask = stop_mask | (last_token == self.eos_token_id)
        
        return stop_mask
    
    def all_stopped(
        self,
        input_ids: torch.Tensor,
        current_length: int,
    ) -> bool:
        """Check if all sequences in batch have stopped."""
        return self.should_stop(input_ids, current_length).all().item()
