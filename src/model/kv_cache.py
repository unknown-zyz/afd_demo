"""
KV Cache Manager for autoregressive generation.

Manages key-value cache for all transformer layers during decode phase.
The cache is stored locally on the Attention node - never transmitted.
"""

from typing import Optional, Tuple, List
from dataclasses import dataclass

import torch


@dataclass
class KVCacheConfig:
    """Configuration for KV Cache."""
    num_layers: int
    num_heads: int
    head_dim: int
    max_batch_size: int
    max_seq_len: int
    device: torch.device
    dtype: torch.dtype = torch.bfloat16


class KVCache:
    """
    KV Cache for a single layer.
    
    Stores key and value tensors with shape [batch, num_heads, seq_len, head_dim].
    Supports incremental updates during decode phase.
    """
    
    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        max_batch_size: int,
        max_seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.device = device
        self.dtype = dtype
        
        # Pre-allocate cache tensors [batch, heads, max_seq, head_dim]
        self.k_cache = torch.zeros(
            max_batch_size, num_heads, max_seq_len, head_dim,
            device=device, dtype=dtype
        )
        self.v_cache = torch.zeros(
            max_batch_size, num_heads, max_seq_len, head_dim,
            device=device, dtype=dtype
        )
        
        # Current sequence length per batch item
        self.seq_len = 0
    
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update cache with new key/value states and return full cache.
        
        Args:
            key_states: [batch, num_heads, new_seq_len, head_dim]
            value_states: [batch, num_heads, new_seq_len, head_dim]
        
        Returns:
            Tuple of (full_keys, full_values) including cached + new
        """
        batch_size = key_states.shape[0]
        new_seq_len = key_states.shape[2]
        
        # Store new KV in cache
        end_pos = self.seq_len + new_seq_len
        self.k_cache[:batch_size, :, self.seq_len:end_pos, :] = key_states
        self.v_cache[:batch_size, :, self.seq_len:end_pos, :] = value_states
        
        # Update sequence length
        self.seq_len = end_pos
        
        # Return full cache up to current position
        return (
            self.k_cache[:batch_size, :, :self.seq_len, :],
            self.v_cache[:batch_size, :, :self.seq_len, :],
        )
    
    def get(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get current cache contents."""
        return (
            self.k_cache[:batch_size, :, :self.seq_len, :],
            self.v_cache[:batch_size, :, :self.seq_len, :],
        )
    
    def reset(self):
        """Reset cache for new generation."""
        self.seq_len = 0
        # Note: We don't zero the tensors - just reset the length pointer
    
    @property
    def seen_tokens(self) -> int:
        """Number of tokens seen so far."""
        return self.seq_len


class KVCacheManager:
    """
    Manages KV caches for all transformer layers.
    
    Used by AttentionWorker to store and retrieve KV during generation.
    The cache stays on the Attention node - never transmitted to FFN node.
    """
    
    def __init__(self, config: KVCacheConfig):
        """
        Initialize KV cache manager.
        
        Args:
            config: Cache configuration
        """
        self.config = config
        self.num_layers = config.num_layers
        
        # Create cache for each layer
        self.caches: List[KVCache] = []
        for _ in range(config.num_layers):
            cache = KVCache(
                num_heads=config.num_heads,
                head_dim=config.head_dim,
                max_batch_size=config.max_batch_size,
                max_seq_len=config.max_seq_len,
                device=config.device,
                dtype=config.dtype,
            )
            self.caches.append(cache)
    
    def get_layer_cache(self, layer_idx: int) -> KVCache:
        """Get cache for a specific layer."""
        return self.caches[layer_idx]
    
    def update_layer(
        self,
        layer_idx: int,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update cache for a layer and return full KV."""
        return self.caches[layer_idx].update(key_states, value_states)
    
    def get_layer(
        self,
        layer_idx: int,
        batch_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get cached KV for a layer."""
        return self.caches[layer_idx].get(batch_size)
    
    def reset(self):
        """Reset all layer caches."""
        for cache in self.caches:
            cache.reset()
    
    @property
    def seen_tokens(self) -> int:
        """Number of tokens seen (from first layer)."""
        return self.caches[0].seen_tokens if self.caches else 0
    
    @classmethod
    def from_model_config(
        cls,
        model_config,
        max_batch_size: int,
        max_seq_len: int,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
    ) -> "KVCacheManager":
        """
        Create KVCacheManager from HuggingFace model config.
        
        Args:
            model_config: HuggingFace model configuration
            max_batch_size: Maximum batch size to support
            max_seq_len: Maximum sequence length to support
            device: Device for cache tensors
            dtype: Data type for cache tensors
        """
        # Extract config values (works for Qwen2)
        num_layers = model_config.num_hidden_layers
        num_heads = model_config.num_key_value_heads  # Use KV heads for GQA
        head_dim = model_config.hidden_size // model_config.num_attention_heads
        
        config = KVCacheConfig(
            num_layers=num_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            device=device,
            dtype=dtype,
        )
        
        return cls(config)
    
    def get_memory_usage(self) -> int:
        """Get total memory usage in bytes."""
        if not self.caches:
            return 0
        
        cache = self.caches[0]
        per_layer = cache.k_cache.numel() + cache.v_cache.numel()
        element_size = cache.k_cache.element_size()
        
        return self.num_layers * per_layer * element_size
    
    def __repr__(self) -> str:
        cfg = self.config
        mem_mb = self.get_memory_usage() / (1024 * 1024)
        return (
            f"KVCacheManager(layers={cfg.num_layers}, "
            f"heads={cfg.num_heads}, head_dim={cfg.head_dim}, "
            f"max_batch={cfg.max_batch_size}, max_seq={cfg.max_seq_len}, "
            f"memory={mem_mb:.1f}MB)"
        )
