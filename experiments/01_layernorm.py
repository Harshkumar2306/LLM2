"""
Educational Module: Deriving Layer Normalization from Scratch.

This script demonstrates the exact mathematics behind PyTorch's `nn.LayerNorm`.
It is crucial to understand this before we rely on the optimized C++ implementation.
"""
import torch
import torch.nn as nn

class ManualLayerNorm(nn.Module):
    """
    A manual implementation of Layer Normalization for educational purposes.
    """
    def __init__(self, ndim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        # Learnable scale (gamma) and shift (beta)
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is of shape (Batch, Time, Channels)
        
        # 1. Calculate the mean across the Channel dimension 
        # keepdim=True ensures the output shape is (Batch, Time, 1) so it broadcasts correctly.
        mean = x.mean(dim=-1, keepdim=True)
        
        # 2. Calculate the variance across the Channel dimension
        # PyTorch uses unbiased=False (population variance) for LayerNorm by default
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        
        # 3. Normalize to mean 0, variance 1
        # We add epsilon to the variance to prevent division by zero
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        
        # 4. Scale and shift using learnable parameters
        out = self.weight * x_norm + self.bias
        return out

if __name__ == "__main__":
    torch.manual_seed(42)
    
    # Simulate a batch of 2 sequences, 3 tokens long, with 4 embedding channels
    x = torch.randn(2, 3, 4) 
    
    manual_ln = ManualLayerNorm(ndim=4)
    pytorch_ln = nn.LayerNorm(4)
    
    out_manual = manual_ln(x)
    out_pytorch = pytorch_ln(x)
    
    # Verify mathematical equivalence
    diff = (out_manual - out_pytorch).abs().max().item()
    print(f"Max difference between Manual and PyTorch LayerNorm: {diff:.8e}")
    
    if diff < 1e-6:
        print("Success: Manual implementation perfectly matches PyTorch nn.LayerNorm.")
    else:
        print("Error: Implementations do not match!")
