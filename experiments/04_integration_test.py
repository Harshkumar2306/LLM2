"""
Integration Test: End-to-End Pipeline
Validates Tokenizer -> DataLoader -> Model -> Trainer -> Checkpointing.
"""
import sys
import os
import torch
from torch.utils.data import Dataset, DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.gpt_config import GPTConfig
from models.model import GPT
from trainer.trainer import Trainer, TrainerConfig

# A tiny excerpt of real text
TINY_CORPUS = """
O Romeo, Romeo! wherefore art thou Romeo?
Deny thy father and refuse thy name;
Or, if thou wilt not, be but sworn my love,
And I'll no longer be a Capulet.
"""

class DummyShakespeareDataset(Dataset):
    def __init__(self, text, context_length):
        self.context_length = context_length
        # 1. Tokenizer
        try:
            import tiktoken
            enc = tiktoken.get_encoding("gpt2")
            tokens = enc.encode(text)
            self.vocab_size = 50257
        except ImportError:
            # Fallback to simple character-level tokenizer if tiktoken is not installed
            chars = sorted(list(set(text)))
            self.vocab_size = len(chars)
            stoi = { ch:i for i,ch in enumerate(chars) }
            tokens = [stoi[c] for c in text]
            
        self.data = torch.tensor(tokens, dtype=torch.long)

    def __len__(self):
        return len(self.data) - self.context_length

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.context_length]
        y = self.data[idx + 1 : idx + self.context_length + 1]
        return x, y

def integration_test():
    torch.manual_seed(42)
    
    context_length = 8
    # 3. Data Pipeline
    dataset = DummyShakespeareDataset(TINY_CORPUS, context_length)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # 2. Config
    config = GPTConfig(
        vocab_size=dataset.vocab_size,
        d_model=64, 
        n_heads=2, 
        n_layers=2, 
        context_length=context_length
    )
    
    # 4. Model
    model = GPT(config)
    
    # 5. Trainer
    t_config = TrainerConfig(
        max_iters=50, 
        learning_rate=3e-3, 
        warmup_iters=5,
        lr_decay_iters=50,
        checkpoint_dir='checkpoints_test',
        log_file='logs/integration_test.jsonl'
    )
    trainer = Trainer(t_config, model, train_dataloader=dataloader)
    
    # 6. Training Loop
    print("Starting integration test training...")
    for it in range(t_config.max_iters):
        try:
            x, y = next(iter(dataloader))
        except StopIteration:
            pass # We just reuse batches for this tiny test
            
        x, y = x.to(trainer.device), y.to(trainer.device)
        loss = trainer.train_step(x, y)
        
    print(f"Final Loss: {loss:.4f}")
    
    # 7. Checkpointing Test
    ckpt_path = os.path.join(t_config.checkpoint_dir, 'ckpt_test.pt')
    trainer.save_checkpoint(ckpt_path)
    print(f"Checkpoint saved to {ckpt_path}")
    
    # 8. Load Checkpoint Test
    new_model = GPT(config)
    new_trainer = Trainer(t_config, new_model, train_dataloader=dataloader)
    new_trainer.load_checkpoint(ckpt_path)
    print(f"Checkpoint successfully loaded. Restored iteration: {new_trainer.current_iter}")
    
    assert new_trainer.current_iter == 50, "Checkpoint loading failed to restore iteration count."
    print("✅ Integration test passed! Data Pipeline -> Model -> Trainer -> I/O works perfectly.")

if __name__ == "__main__":
    integration_test()
