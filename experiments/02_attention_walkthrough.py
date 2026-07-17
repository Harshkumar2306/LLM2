"""
Educational Module: Causal Self-Attention Step-by-Step Walkthrough.
This script manually computes attention for a tiny example to build intuition.
"""
import torch
import torch.nn.functional as F

def attention_walkthrough():
    # Lock seed for reproducibility
    torch.manual_seed(42)
    
    # Tiny parameters for readability
    B = 1       # Batch size
    T = 4       # Sequence length (Time)
    C = 6       # Embedding dimension (Channels)
    n_heads = 2
    head_dim = C // n_heads # 3
    
    # Simulate an input tensor (e.g. word embeddings + positional embeddings)
    print("--- 1. INPUT (X) ---")
    x = torch.randn(B, T, C)
    print(f"Shape: {x.shape} (Batch, Time, Channels)")
    print(x[0].round(decimals=2))
    print()

    # Create dummy linear projections for Q, K, V
    # In practice, these are learned weights in nn.Linear
    W_q = torch.randn(C, C)
    W_k = torch.randn(C, C)
    W_v = torch.randn(C, C)
    
    # Calculate Q, K, V (The packed projection does this simultaneously)
    q = x @ W_q
    k = x @ W_k
    v = x @ W_v
    
    # Reshape for multi-head attention
    # (B, T, C) -> (B, T, n_heads, head_dim) -> (B, n_heads, T, head_dim)
    q = q.view(B, T, n_heads, head_dim).transpose(1, 2)
    k = k.view(B, T, n_heads, head_dim).transpose(1, 2)
    v = v.view(B, T, n_heads, head_dim).transpose(1, 2)
    
    print("--- 2. Q, K, V (For Head 0) ---")
    print(f"Shape: {q.shape} (Batch, Heads, Time, Head_Dim)")
    print("Query (What I'm looking for):")
    print(q[0, 0].round(decimals=2))
    print("Key (What I contain):")
    print(k[0, 0].round(decimals=2))
    print("Value (What I will output):")
    print(v[0, 0].round(decimals=2))
    print()
    
    # Calculate Attention Scores
    print("--- 3. ATTENTION SCORES (Raw Affinities) ---")
    # k.transpose(-2, -1) swaps Time and Head_Dim for the dot product
    scores = q @ k.transpose(-2, -1)
    
    # Scale by 1/sqrt(head_dim) to control variance
    scores = scores / (head_dim ** 0.5)
    print(f"Shape: {scores.shape} (Batch, Heads, Time, Time)")
    print(scores[0, 0].round(decimals=2))
    print()
    
    # Apply Causal Mask
    print("--- 4. CAUSAL MASKING ---")
    # torch.tril creates a lower triangular matrix
    mask = torch.tril(torch.ones(T, T)).view(1, 1, T, T)
    print("Boolean Mask (1 means allowed to attend):")
    print(mask[0, 0])
    
    # Replace 0s with -infinity
    scores = scores.masked_fill(mask == 0, float('-inf'))
    print("\nMasked Scores (-inf prevents looking into the future):")
    print(scores[0, 0].round(decimals=2))
    print()
    
    # Calculate Softmax Probabilities
    print("--- 5. SOFTMAX PROBABILITIES ---")
    # Softmax across the last dimension
    probs = F.softmax(scores, dim=-1)
    print("Probabilities (Rows sum to 1):")
    print(probs[0, 0].round(decimals=2))
    print()
    
    # Compute Final Weighted Output
    print("--- 6. WEIGHTED OUTPUT ---")
    out = probs @ v
    print(f"Shape: {out.shape} (Batch, Heads, Time, Head_Dim)")
    print(out[0, 0].round(decimals=2))
    
if __name__ == "__main__":
    attention_walkthrough()
