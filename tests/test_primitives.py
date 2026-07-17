import torch
from config.gpt_config import GPTConfig
from core.ffn import FeedForward
from core.embeddings import GPTEmbeddings

def test_ffn_shape():
    """
    Test that the Feed-Forward Network strictly preserves the tensor shape (B, T, C).
    """
    config = GPTConfig(d_model=64, n_heads=4)
    ffn = FeedForward(config)
    
    # Batch=2, Time=10, Channels=64
    x = torch.randn(2, 10, 64) 
    out = ffn(x)
    
    assert out.shape == (2, 10, 64), f"Expected shape (2, 10, 64), got {out.shape}"

def test_ffn_initialization_flag():
    """
    Test that the FFN marks its final projection layer for residual scaling.
    """
    config = GPTConfig()
    ffn = FeedForward(config)
    assert hasattr(ffn.c_proj, "RESIDUAL_SCALE_INIT"), "FFN projection layer missing initialization flag"
    assert ffn.c_proj.RESIDUAL_SCALE_INIT == 1, "Flag value must be 1"

def test_embeddings_shape():
    """
    Test that the Embedding module converts a 2D integer tensor (B, T) 
    into a 3D float tensor (B, T, C).
    """
    config = GPTConfig(vocab_size=100, context_length=16, d_model=64)
    emb = GPTEmbeddings(config)
    
    # Simulate a batch of 2 sequences, each length 10
    idx = torch.randint(0, 100, (2, 10))
    out = emb(idx)
    
    assert out.shape == (2, 10, 64), f"Expected shape (2, 10, 64), got {out.shape}"

def test_embeddings_context_length_error():
    """
    Test that passing a sequence longer than context_length raises an IndexError
    because the positional embedding table is too small.
    """
    config = GPTConfig(vocab_size=100, context_length=16, d_model=64)
    emb = GPTEmbeddings(config)
    
    # Try to pass a sequence of length 20 (max is 16)
    idx = torch.randint(0, 100, (2, 20))
    
    try:
        out = emb(idx)
        assert False, "Should have raised an IndexError"
    except IndexError:
        pass # Expected behavior
