import torch
import torch.nn as nn
from config.gpt_config import GPTConfig


class GPTEmbeddings(nn.Module):
    """
    Handles both Token Embeddings and Learned Absolute Positional Embeddings.
    This injects both the 'what' (token meaning) and the 'where' (sequence position)
    into the residual stream.
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        
        self.config = config
        
        # Word Token Embeddings (wte)
        # Maps token integers [0, vocab_size-1] to dense vectors of size d_model
        self.wte = nn.Embedding(config.vocab_size, config.d_model)
        
        # Word Position Embeddings (wpe)
        # Maps position integers [0, context_length-1] to dense vectors of size d_model
        # Only used if position_type is ABSOLUTE
        if config.position_type == "absolute":
            self.wpe = nn.Embedding(config.context_length, config.d_model)
        else:
            self.register_parameter("wpe", None)
        
        # Regularization applied immediately after combining embeddings
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            idx: Tensor of shape (Batch, Time) containing integer token IDs.
        Returns:
            Tensor of shape (Batch, Time, Channels)
        """
        batch_size, seq_len = idx.size()
        
        # Extract the dense vectors
        tok_emb = self.wte(idx) # shape: (Batch, Time, Channels)
        x = tok_emb
        
        # Combine token meaning with positional information if using absolute embeddings
        if self.config.position_type == "absolute":
            pos = torch.arange(0, seq_len, dtype=torch.long, device=idx.device)
            pos_emb = self.wpe(pos) # shape: (Time, Channels)
            x = x + pos_emb
        
        return self.dropout(x)
