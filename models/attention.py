import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import sys
import os
from typing import Tuple

# Ensure config can be imported if run directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.gpt_config import GPTConfig
from config.enums import AttentionType

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    t = torch.arange(end, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[2], x.shape[-1]) # (T, head_dim//2)
    shape = [d if i == 2 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # xq, xk shape: (B, num_heads, T, head_dim)
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    
    freqs_cis_q = reshape_for_broadcast(freqs_cis, xq_)
    freqs_cis_k = reshape_for_broadcast(freqs_cis, xk_)
    
    xq_out = torch.view_as_real(xq_ * freqs_cis_q).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis_k).flatten(3)
    
    return xq_out.type_as(xq), xk_out.type_as(xk)

class CausalSelfAttention(nn.Module):
    """
    Multi-Head & Grouped-Query Causal Self-Attention mechanism.
    Routes information between tokens while strictly preventing future-peeking.
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.d_model % config.n_heads == 0, "d_model must be divisible by n_heads"
        
        self.attention_type = config.attention_type
        self.n_heads = config.n_heads
        self.d_model = config.d_model
        self.head_dim = config.d_model // config.n_heads
        
        if self.attention_type == AttentionType.GQA:
            self.n_kv_heads = config.n_kv_heads
        else:
            self.n_kv_heads = self.n_heads
            
        self.num_repeats = self.n_heads // self.n_kv_heads
        self.dropout_p = config.dropout
        self.position_type = config.position_type
        
        if self.position_type.value == "rope":
            # Precompute freqs_cis for RoPE
            freqs_cis = precompute_freqs_cis(self.head_dim, config.context_length)
            self.register_buffer("freqs_cis", freqs_cis, persistent=False)
        
        # QKV projection
        # Q size = n_heads * head_dim (d_model)
        # K size = n_kv_heads * head_dim
        # V size = n_kv_heads * head_dim
        self.qkv_dim = self.d_model + 2 * (self.n_kv_heads * self.head_dim)
        self.c_attn = nn.Linear(config.d_model, self.qkv_dim, bias=config.bias)
        
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
        # (B, T, C) -> (B, T, d_model + 2*kv_dim)
        qkv = self.c_attn(x)
        
        # 2. Slice the packed tensor into Query, Key, Value
        q, k, v = qkv.split([self.d_model, self.n_kv_heads * self.head_dim, self.n_kv_heads * self.head_dim], dim=2)
        
        # 3. Reshape for Attention
        # (B, T, n_heads, head_dim) -> transpose -> (B, n_heads, T, head_dim)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        
        if self.position_type.value == "rope":
            q, k = apply_rotary_emb(q, k, self.freqs_cis[:T])
            
        # If GQA, repeat K and V to match Q's n_heads
        if self.num_repeats > 1:
            k = torch.repeat_interleave(k, repeats=self.num_repeats, dim=1)
            v = torch.repeat_interleave(v, repeats=self.num_repeats, dim=1)
        
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
