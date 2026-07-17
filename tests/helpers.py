import os
from config.gpt_config import GPTConfig
from config.enums import FFNType, AttentionType, PositionType, NormType

def get_test_config(**kwargs) -> GPTConfig:
    """Provides a valid baseline configuration for unit tests."""
    default_kwargs = {
        "vocab_size": 100,
        "context_length": 16,
        "d_model": 64,
        "n_layers": 2,
        "n_heads": 4,
        "ffn_type": FFNType.STANDARD,
        "attention_type": AttentionType.MHA,
        "position_type": PositionType.ABSOLUTE,
        "norm_type": NormType.LAYER_NORM,
        "dropout": 0.0,
        "bias": True
    }
    default_kwargs.update(kwargs)
    return GPTConfig(**default_kwargs)
