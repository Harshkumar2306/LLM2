"""
Reproducibility utilities for deterministic training and debugging.
"""
import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Sets the seed for all random number generators to ensure reproducible results.
    
    Args:
        seed (int): The seed value to lock RNGs.
    """
    # 1. Set Python's built-in random module seed
    random.seed(seed)
    
    # 2. Set NumPy's random seed
    np.random.seed(seed)
    
    # 3. Set PyTorch's CPU random seed
    torch.manual_seed(seed)
    
    # 4. Set PyTorch's GPU/Accelerator random seed (if available)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # For multi-GPU
        
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
        
    # 5. Enforce deterministic algorithms in cuDNN (may impact performance slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
