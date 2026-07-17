import torch
import torch.nn as nn
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.gpt_config import GPTConfig
from core.attention import CausalSelfAttention
from core.ffn import FeedForward

class Block(nn.Module):
    """
    A single Transformer Block.
    Uses the modern Pre-Norm architecture (LayerNorm -> Attention -> Add)
    for improved training stability.
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        
        # Layer Normalizations
        self.ln_1 = nn.LayerNorm(config.d_model, bias=config.bias)
        self.ln_2 = nn.LayerNorm(config.d_model, bias=config.bias)
        
        # Core mathematical primitives
        self.attn = CausalSelfAttention(config)
        self.mlp = FeedForward(config)

    def forward(self, x: torch.Tensor, use_flash: bool = True) -> torch.Tensor:
        """
        Forward pass with residual connections.
        Args:
            x: Tensor of shape (Batch, Time, Channels)
            use_flash: Whether to use optimized FlashAttention.
        """
        # The Residual Stream (x = x + f(x))
        # 1. We read from the stream, normalize it, and pass it to Attention.
        # 2. Attention figures out what information to move between tokens.
        # 3. We ADD that information back into the stream.
        x = x + self.attn(self.ln_1(x), use_flash=use_flash)
        
        # 1. We read from the updated stream, normalize it, and pass it to the FFN.
        # 2. The FFN retrieves facts/patterns for each token individually.
        # 3. We ADD that information back into the stream.
        x = x + self.mlp(self.ln_2(x))
        
        return x
