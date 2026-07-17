from tests.helpers import get_test_config
import torch
import math
import sys
import os

# Ensure config can be imported if run directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.gpt_config import GPTConfig
from core.attention import CausalSelfAttention

def test_attention_shape():
    """
    Test that the Attention module strictly preserves the tensor shape (B, T, C).
    """
    config = get_test_config(d_model=64, n_heads=4, context_length=16)
    attn = CausalSelfAttention(config)
    x = torch.randn(2, 10, 64)
    
    # Test Flash
    out_flash = attn(x, use_flash=True)
    assert out_flash.shape == (2, 10, 64), f"Flash output shape incorrect: {out_flash.shape}"
    
    # Test Manual
    out_manual = attn(x, use_flash=False)
    assert out_manual.shape == (2, 10, 64), f"Manual output shape incorrect: {out_manual.shape}"

def test_causal_masking_correctness():
    """
    Ensure that tokens cannot attend to future tokens.
    If we change the input at T=4, the outputs at T=0, 1, 2, 3 must remain absolutely identical.
    """
    config = get_test_config(d_model=64, n_heads=4, context_length=16, dropout=0.0)
    attn = CausalSelfAttention(config)
    attn.eval() # Disable dropout for deterministic outputs
    
    x1 = torch.randn(1, 5, 64)
    
    # Create x2 identical to x1 EXCEPT at the final timestep
    x2 = x1.clone()
    x2[0, 4, :] = torch.randn(64)
    
    out1 = attn(x1, use_flash=False)
    out2 = attn(x2, use_flash=False)
    
    # Outputs at T=0, 1, 2, 3 must be exactly equal
    assert torch.allclose(out1[:, :4, :], out2[:, :4, :], atol=1e-6), "Causality broken: past tokens changed when future token changed."
    
    # Outputs at T=4 should differ
    assert not torch.allclose(out1[:, 4, :], out2[:, 4, :], atol=1e-6), "Outputs at T=4 should differ."

def test_manual_vs_flash_equivalence():
    """
    Verify that our manual math exactly matches the C++ optimized flash attention.
    """
    config = get_test_config(d_model=64, n_heads=4, context_length=16, dropout=0.0)
    attn = CausalSelfAttention(config)
    attn.eval()
    
    x = torch.randn(2, 10, 64)
    
    out_flash = attn(x, use_flash=True)
    out_manual = attn(x, use_flash=False)
    
    # Floating point precision requires atol=1e-5
    diff = (out_flash - out_manual).abs().max().item()
    assert torch.allclose(out_flash, out_manual, atol=1e-5), f"Manual and Flash Attention differ by {diff}"
