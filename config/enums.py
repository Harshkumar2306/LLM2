"""
Strict Enumerations defining the architectural configurations allowed in Axiom.
Prevents combinatorial explosion of boolean flags.
"""

from enum import Enum, auto

class FFNType(Enum):
    """Types of Feed-Forward Networks."""
    STANDARD = "standard"  # Standard GELU FFN
    SWIGLU = "swiglu"      # Swish-Gated Linear Unit FFN


class AttentionType(Enum):
    """Types of Attention Mechanisms."""
    MHA = "mha"  # Multi-Head Attention
    GQA = "gqa"  # Grouped-Query Attention


class PositionType(Enum):
    """Types of Positional Embeddings."""
    ABSOLUTE = "absolute"  # Learned absolute positional embeddings
    ROPE = "rope"          # Rotary Positional Embeddings
    # Future: ALIBI, YARN, XPOS


class NormType(Enum):
    """Types of Normalization Layers."""
    LAYER_NORM = "layer_norm"  # Standard nn.LayerNorm
    RMS_NORM = "rms_norm"      # Root Mean Square Normalization
