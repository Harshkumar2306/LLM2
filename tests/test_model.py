from tests.helpers import get_test_config
import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.gpt_config import GPTConfig
from core.model import GPT

def test_model_shape():
    """
    Ensure the full model compiles and outputs the correct shape (B, T, Vocab_Size).
    """
    config = get_test_config(vocab_size=100, d_model=32, n_heads=2, n_layers=2, context_length=16)
    model = GPT(config)
    
    idx = torch.randint(0, 100, (2, 10)) # Batch 2, Time 10
    logits, loss = model(idx, use_flash=False)
    
    assert logits.shape == (2, 10, 100), f"Expected shape (2, 10, 100), got {logits.shape}"
    assert loss is None, "Loss should be None when no targets are provided"

def test_weight_tying():
    """
    Verify that the embedding layer and LM head point to the exact same memory in RAM.
    """
    config = get_test_config(vocab_size=100, d_model=32, n_heads=2, n_layers=2, context_length=16)
    model = GPT(config)
    
    # Check that they are the identical Python object (memory address)
    assert model.lm_head.weight is model.embeddings.wte.weight, "Weight tying failed: Tensors are not identical."

def test_residual_initialization():
    """
    Verify that the residual layers have been scaled down correctly during initialization.
    """
    config = get_test_config(vocab_size=100, d_model=32, n_heads=2, n_layers=2, context_length=16)
    model = GPT(config)
    
    # We expect standard deviation to be 0.02 * (1 / sqrt(2 * n_layers))
    expected_std = 0.02 * ((2 * config.n_layers) ** -0.5)
    
    # Sample standard deviation (will vary slightly, but should be close to expected_std)
    actual_std = model.blocks[0].mlp.c_proj.weight.std().item()
    
    # Check that it's noticeably smaller than 0.02
    assert actual_std < 0.015, f"Residual scaling failed: actual std is {actual_std}, expected ~{expected_std}"
