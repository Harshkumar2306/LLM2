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
        # Project up to 4x the embedding dimension
        self.c_fc = nn.Linear(config.d_model, 4 * config.d_model, bias=config.bias)
        
        # GELU approximation used by OpenAI (tanh approximation)
        self.gelu = nn.GELU(approximate="tanh")
        
        # Project back down to the model dimension
        self.c_proj = nn.Linear(4 * config.d_model, config.d_model, bias=config.bias)
        
        # Regularization
        self.dropout = nn.Dropout(config.dropout)
        
        # --- Initialization Flag ---
        # We flag this layer so that our custom initialization logic (in the main GPT model)
        # knows to scale its weights by 1/sqrt(2 * n_layers). This prevents variance explosion
        # because this layer writes directly into the residual stream.
        self.c_proj.RESIDUAL_SCALE_INIT = 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Tensor of shape (Batch, Time, Channels)
        Returns:
            Tensor of shape (Batch, Time, Channels)
        """
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
