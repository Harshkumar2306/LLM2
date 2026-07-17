"""
Abstract Base Classes defining the mathematical and structural contracts 
for the Axiom model components.
"""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Optional, Tuple

class BaseFFN(nn.Module, ABC):
    """
    Contract for Feed-Forward Networks.
    Invariants:
    - Input shape must equal output shape: (Batch, Time, Channels) -> (Batch, Time, Channels)
    - Must not alter the sequence length or batch size.
    """
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, T, C)
        Returns:
            Tensor of shape (B, T, C)
        """
        pass


class BaseAttention(nn.Module, ABC):
    """
    Contract for Attention mechanisms (MHA, GQA, MQA).
    Invariants:
    - Must support an optional KV cache for autoregressive generation.
    - Input shape must equal output shape: (Batch, Time, Channels) -> (Batch, Time, Channels)
    """
    @abstractmethod
    def forward(
        self, 
        x: torch.Tensor, 
        kv_cache: Optional[object] = None,
        use_flash: bool = True
    ) -> Tuple[torch.Tensor, Optional[object]]:
        """
        Args:
            x: Tensor of shape (B, T, C)
            kv_cache: Optional state object for rolling key/value caching.
            use_flash: Toggle for hardware-optimized attention kernels.
        Returns:
            Tuple containing:
                - Output tensor of shape (B, T, C)
                - Updated kv_cache object (or None if not using cache)
        """
        pass


class BasePositionStrategy(nn.Module, ABC):
    """
    Contract for Positional Encoding injections (Absolute, RoPE, ALiBi).
    Invariants:
    - Must cleanly separate additive embeddings (Absolute) from multiplicative/rotary (RoPE).
    - If a strategy does not use additive embeddings, `get_embeddings` must return 0.
    - If a strategy does not use rotary embeddings, `apply_rotary` must return q, k unmodified.
    """
    @abstractmethod
    def get_embeddings(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Provides additive absolute embeddings.
        Args:
            seq_len: Current length of the sequence (T).
            device: Target device.
        Returns:
            Tensor of shape (T, C) or scalar 0 if not applicable.
        """
        pass

    @abstractmethod
    def apply_rotary(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Applies rotary/multiplicative transformations directly to Queries and Keys.
        Args:
            q: Query tensor of shape (B, n_heads, T, head_dim)
            k: Key tensor of shape (B, n_heads, T, head_dim)
        Returns:
            Tuple of modified (q, k) tensors.
        """
        pass


class BaseNorm(nn.Module, ABC):
    """
    Contract for Normalization layers (LayerNorm, RMSNorm).
    Invariants:
    - Input shape must equal output shape.
    - Must normalize across the channel dimension (C).
    """
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, T, C)
        Returns:
            Tensor of shape (B, T, C)
        """
        pass
