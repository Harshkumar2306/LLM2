"""
Sanity Check: Overfitting a single batch.
This is the ultimate test of a neural network's architecture and training loop.
If the model cannot memorize a single batch (reaching near 0.0 loss), 
there is a fundamental bug in the code.
"""
import sys
import os
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.gpt_config import GPTConfig
from core.model import GPT
from engine.trainer import Trainer, TrainerConfig

def overfit_single_batch():
    # 1. Reproducibility
    torch.manual_seed(42)
    device = 'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
        
    print(f"Using device: {device}")
    
    # 2. Tiny Model Configuration (so it trains in seconds)
    config = GPTConfig(
        vocab_size=100, 
        d_model=64, 
        n_heads=2, 
        n_layers=2, 
        context_length=16
    )
    model = GPT(config)
    
    # 3. Generate a single deterministic batch of dummy data
    # Shape: (Batch=4, Time=16)
    B, T = 4, 16
    x = torch.randint(0, config.vocab_size, (B, T)).to(device)
    # The target is just shifted (autoregressive modeling)
    # But for a basic overfit test, we can just memorize y = random tokens
    y = torch.randint(0, config.vocab_size, (B, T)).to(device)
    
    # 4. Configure Trainer
    # We use a high learning rate and short warmup to overfit quickly
    t_config = TrainerConfig(
        max_iters=150, 
        learning_rate=3e-3, 
        warmup_iters=10, 
        lr_decay_iters=150,
        device=device,
        weight_decay=0.0 # Disable weight decay to allow perfect memorization
    )
    
    trainer = Trainer(t_config, model, train_dataloader=None)
    
    # 5. The Training Loop
    print("Starting overfitting test...")
    initial_loss = None
    final_loss = None
    
    for it in range(t_config.max_iters):
        # We pass the same exact batch (x, y) every single iteration
        loss = trainer.train_step(x, y, it)
        
        if it == 0:
            initial_loss = loss
            # Expected initial loss is roughly -ln(1/vocab_size) = -ln(0.01) = ~4.6
            print(f"Iter {it:03d}: Initial Loss = {loss:.4f} (Expected: ~4.60)")
            
        if it % 20 == 0 and it > 0:
            print(f"Iter {it:03d}: Loss = {loss:.4f}")
            
        final_loss = loss
        
    print(f"Iter {t_config.max_iters-1:03d}: Final Loss = {final_loss:.4f}")
    
    # 6. Validation
    assert final_loss < 0.1, f"Model failed to overfit! Final loss {final_loss:.4f} is too high."
    print("\n✅ SUCCESS: Model perfectly memorized the batch. The architecture and training loop are fundamentally sound.")

if __name__ == "__main__":
    overfit_single_batch()
