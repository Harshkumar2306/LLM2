import torch
from config.gpt_config import GPTConfig
from core.model import GPT

def test_generate():
    torch.manual_seed(42)
    config = GPTConfig(vocab_size=100, d_model=32, n_heads=2, n_layers=2, context_length=8)
    model = GPT(config)
    
    # Initial prompt: batch 2, time 3
    idx = torch.randint(0, 100, (2, 3))
    
    # Generate 5 new tokens
    out = model.generate(idx, max_new_tokens=5, temperature=1.0, top_k=10)
    
    # Shape should be (2, 3 + 5)
    assert out.shape == (2, 8), f"Expected shape (2, 8), got {out.shape}"
    
    # Check greedy decoding
    out_greedy = model.generate(idx, max_new_tokens=2, temperature=0.0)
    assert out_greedy.shape == (2, 5)
    
    print("✅ Generate tests passed.")
    
if __name__ == "__main__":
    test_generate()
