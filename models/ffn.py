import torch
import torch.nn as nn
import sys
import os

# Ensure config can be imported if run directly (though usually run via train.py)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.gpt_config import GPTConfig


class FeedForward(nn.Module):
    """
    The Feed-Forward Network (FFN).
    Acts as a per-token Key-Value memory bank.
    Uses the GPT-2 standard 4x expansion and GELU non-linearity.
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ffn_type = config.ffn_type
        
        if self.ffn_type == "swiglu":
            # For SwiGLU, we use 8/3 expansion instead of 4x to match parameter count
            hidden_dim = int(8 * config.d_model / 3)
            # Make sure it's divisible by 2 for good measure, or 256 depending on implementation, 
            # but simple int cast is fine for research.
            
            self.w1 = nn.Linear(config.d_model, hidden_dim, bias=config.bias)
            self.w3 = nn.Linear(config.d_model, hidden_dim, bias=config.bias)
            self.w2 = nn.Linear(hidden_dim, config.d_model, bias=config.bias)
            
            # Use SiLU (Swish with beta=1)
            self.silu = nn.SiLU()
            
            self.w2.RESIDUAL_SCALE_INIT = 1
            
        else:
            # Project up to 4x the embedding dimension
            self.c_fc = nn.Linear(config.d_model, 4 * config.d_model, bias=config.bias)
            
            # GELU approximation used by OpenAI (tanh approximation)
            self.gelu = nn.GELU(approximate="tanh")
            
            # Project back down to the model dimension
            self.c_proj = nn.Linear(4 * config.d_model, config.d_model, bias=config.bias)
            self.c_proj.RESIDUAL_SCALE_INIT = 1
            
        # Regularization
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Tensor of shape (Batch, Time, Channels)
        Returns:
            Tensor of shape (Batch, Time, Channels)
        """
        if self.ffn_type == "swiglu":
            x = self.silu(self.w1(x)) * self.w3(x)
            x = self.w2(x)
        else:
            x = self.c_fc(x)
            x = self.gelu(x)
            x = self.c_proj(x)
            
        x = self.dropout(x)
        return x
