import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import sys
import os

# Ensure config can be imported if run directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.gpt_config import GPTConfig

class CausalSelfAttention(nn.Module):
    """
    Multi-Head Causal Self-Attention mechanism.
    Routes information between tokens while strictly preventing future-peeking.
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.d_model % config.n_heads == 0, "d_model must be divisible by n_heads"
        
        self.n_heads = config.n_heads
        self.d_model = config.d_model
        self.head_dim = config.d_model // config.n_heads
        self.dropout_p = config.dropout
        
        # Systems Optimization: Packed QKV projection
        # Instead of 3 sequential GPU kernel launches for Q, K, and V, we launch 1.
        self.c_attn = nn.Linear(config.d_model, 3 * config.d_model, bias=config.bias)
        
        # Output projection
        self.c_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        # Scale initialization for residual stability
        self.c_proj.RESIDUAL_SCALE_INIT = 1
        
        # Regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        # Causal mask for manual attention
        # register_buffer ensures it is moved to the correct device (GPU/CPU) alongside the module
        # shape: (1, 1, T, T) to broadcast across Batch and Heads
        mask = torch.tril(torch.ones(config.context_length, config.context_length))
        self.register_buffer("bias", mask.view(1, 1, config.context_length, config.context_length))

    def _manual_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, T: int) -> torch.Tensor:
        """
        Calculates attention mathematically from scratch.
        Useful for education, debugging, and inspecting raw probabilities.
        """
        # Calculate affinity scores: (B, nh, T, hs) @ (B, nh, hs, T) -> (B, nh, T, T)
        scores = q @ k.transpose(-2, -1) * (1.0 / math.sqrt(self.head_dim))
        
        # Apply causal mask: set values above the diagonal to -infinity
        # We slice the bias buffer up to T to handle sequences shorter than context_length
        scores = scores.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        
        # Softmax to get probabilities
        probs = F.softmax(scores, dim=-1)
        probs = self.attn_dropout(probs)
        
        # Weighted sum of values: (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)
        out = probs @ v
        return out

    def _flash_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Uses PyTorch's optimized scaled_dot_product_attention.
        Automatically invokes FlashAttention on supported GPUs, fusing the 
        matmul, masking, softmax, and dropout operations into a single kernel.
        """
        return F.scaled_dot_product_attention(
            q, k, v, 
            dropout_p=self.dropout_p if self.training else 0.0, 
            is_causal=True # Handles the lower-triangular masking automatically
        )

    def forward(self, x: torch.Tensor, use_flash: bool = True) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Tensor of shape (Batch, Time, Channels)
            use_flash: Toggle between optimized FlashAttention and educational Manual Attention.
        """
        B, T, C = x.size()
        
        # 1. Packed QKV projection
        # (B, T, C) -> (B, T, 3 * C)
        qkv = self.c_attn(x)
        
        # 2. Slice the packed tensor into Query, Key, Value
        q, k, v = qkv.split(self.d_model, dim=2)
        
        # 3. Reshape for Multi-Head Attention
        # (B, T, C) -> (B, T, n_heads, head_dim) -> transpose -> (B, n_heads, T, head_dim)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        
        # 4. Dispatch to the selected implementation
        if use_flash and hasattr(F, 'scaled_dot_product_attention'):
            y = self._flash_attention(q, k, v)
        else:
            y = self._manual_attention(q, k, v, T)
            
        # 5. Re-assemble heads
        # (B, n_heads, T, head_dim) -> (B, T, n_heads, head_dim) -> (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # 6. Output projection and dropout
        y = self.resid_dropout(self.c_proj(y))
        
        return y
