"""
Workers package for Coordinator-based MoE architecture.

Contains AttentionWorker and FFNWorker entrypoints for the distributed
Coordinator-based Mixture of Experts system.
"""

__all__ = ["AttentionWorker", "FFNWorker"]

from .attention_worker import AttentionWorker
from .ffn_worker import FFNWorker
