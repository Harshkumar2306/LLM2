"""
Centralized Configuration for the GPT Model.
This module defines the architectural blueprint of the model.
"""
from dataclasses import dataclass
from typing import Optional
from config.enums import FFNType, AttentionType, PositionType, NormType

@dataclass
class GPTConfig:
    """
    Hyperparameters defining the physical structure of the GPT model.
    """
    vocab_size: int
    context_length: int
    d_model: int
    n_layers: int
    n_heads: int
    ffn_type: FFNType
    attention_type: AttentionType
    position_type: PositionType
    norm_type: NormType
    
    dropout: float = 0.1
    bias: bool = True
    fingerprint: Optional[str] = None
    
    # Training Specific parameters commonly passed around with config
    batch_size: int = 16
    max_iters: int = 10000

    def __post_init__(self) -> None:
        """
        Validates hyperparameters immediately upon instantiation.
        Fails fast if the architectural blueprint is mathematically invalid.
        """
        # Type checking for critical numerical constraints
        if not isinstance(self.vocab_size, int):
            raise TypeError(f"vocab_size must be an int, got {type(self.vocab_size)}")
            
        # 1. Attention dimension validation
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"Embedding dimension (d_model={self.d_model}) must be perfectly "
                f"divisible by the number of heads (n_heads={self.n_heads}). "
                f"This ensures we can split the embedding vector equally across heads."
            )

        # 2. Structural requirements
        if self.d_model <= 0:
            raise ValueError("d_model must be greater than 0.")
        if self.n_layers <= 0:
            raise ValueError("n_layers must be greater than 0.")
        if self.n_heads <= 0:
            raise ValueError("n_heads must be greater than 0.")
        if self.context_length <= 0:
            raise ValueError("context_length must be greater than 0.")
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be greater than 0.")

        # 3. Probability constraints
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError(f"dropout must be in range [0.0, 1.0), got {self.dropout}")
