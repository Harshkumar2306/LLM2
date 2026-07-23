"""
Supervised Fine-Tuning (SFT) Training Script
- Loads the pre-trained best.pt model.
- Expands the embedding matrix to fit the new chat tokens.
- Trains on the packed SFT dataset aligned to conversation boundaries.
- Uses Mixed Precision (AMP) for speed on T4 GPUs.
"""
import os
import sys
import json
import torch
import torch.nn as nn
import numpy as np
import time
import argparse
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.gpt_config import GPTConfig
from config.enums import FFNType, AttentionType, PositionType, NormType
from models.model import GPT

class SFTDataset(Dataset):
    def __init__(self, data_dir, split, context_length):
        self.context_length = context_length
        tokens_path = os.path.join(data_dir, f"{split}_tokens.bin")
        labels_path = os.path.join(data_dir, f"{split}_labels.bin")
        
        self.tokens = np.memmap(tokens_path, dtype=np.int32, mode='r')
        self.labels = np.memmap(labels_path, dtype=np.int32, mode='r')
        
        with open(os.path.join(data_dir, "meta.json"), "r") as f:
            meta = json.load(f)
            
        raw_offsets = meta[f"{split}_offsets"]
        
        # Only keep offsets where we can fit a full context_length window
        self.valid_offsets = [off for off in raw_offsets if off + context_length <= len(self.tokens)]
        
    def __len__(self):
        return len(self.valid_offsets)

    def __getitem__(self, idx):
        i = self.valid_offsets[idx]
        x = torch.from_numpy(self.tokens[i:i+self.context_length].astype(np.int64))
        y = torch.from_numpy(self.labels[i:i+self.context_length].astype(np.int64))
        return x, y

def expand_embeddings(model, old_vocab_size, new_vocab_size, device):
    if new_vocab_size <= old_vocab_size:
        return model
        
    print(f"Expanding vocabulary from {old_vocab_size} to {new_vocab_size}...")
    
    old_embeddings = model.embeddings.wte
    new_embeddings = nn.Embedding(new_vocab_size, model.config.d_model)
    with torch.no_grad():
        new_embeddings.weight[:old_vocab_size] = old_embeddings.weight
        # Mentor Tip: Initialize new chat tokens using the <|endoftext|> embedding (ID 50256 for GPT-2)
        eot_vector = old_embeddings.weight[50256]
        for i in range(old_vocab_size, new_vocab_size):
            new_embeddings.weight[i] = eot_vector.clone()
        
    model.embeddings.wte = new_embeddings
    
    old_lm_head = model.lm_head
    new_lm_head = nn.Linear(model.config.d_model, new_vocab_size, bias=False)
    with torch.no_grad():
        new_lm_head.weight[:old_vocab_size] = old_lm_head.weight
        new_lm_head.weight[old_vocab_size:] = new_embeddings.weight[old_vocab_size:]
        
    model.lm_head = new_lm_head
    model.lm_head.weight = model.embeddings.wte.weight
    
    model.config.vocab_size = new_vocab_size
    model.to(device)
    return model

@torch.no_grad()
def estimate_loss(model, train_loader, val_loader, eval_iters, device):
    model.eval()
    out = {}
    
    for split_name, loader in [("train", train_loader), ("val", val_loader)]:
        losses = []
        iterator = iter(loader)
        for _ in range(min(eval_iters, len(loader))):
            try:
                x, y = next(iterator)
            except StopIteration:
                break
            
            x, y = x.to(device), y.to(device)
            # Use AMP for evaluation too
            if device == "cuda":
                with autocast(device_type="cuda", dtype=torch.float16):
                    _, loss = model(x, targets=y)
            else:
                _, loss = model(x, targets=y)
            losses.append(loss.item())
            
        out[split_name] = np.mean(losses) if losses else float('inf')
        
    model.train()
    return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to best.pt")
    parser.add_argument('--yaml', type=str, required=True, help="Path to run_config.yaml")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"Number of GPUs available: {torch.cuda.device_count()}")
    
    print(f"Loading pretrained model from {args.checkpoint}...")
    import yaml
    import inspect
    with open(args.yaml, "r") as f:
        raw_config = yaml.safe_load(f)
        
    valid_keys = inspect.signature(GPTConfig).parameters.keys()
    config_kwargs = {k: v for k, v in raw_config.items() if k in valid_keys}
    
    config_kwargs['attention_type'] = AttentionType(config_kwargs['attention_type'])
    config_kwargs['ffn_type'] = FFNType(config_kwargs['ffn_type'])
    config_kwargs['norm_type'] = NormType(config_kwargs['norm_type'])
    config_kwargs['position_type'] = PositionType(config_kwargs['position_type'])
    
    config = GPTConfig(**config_kwargs)
    model = GPT(config)
    
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state'])
    model.to(device)
    
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "sft_dataset")
    with open(os.path.join(data_dir, "meta.json"), "r") as f:
        meta = json.load(f)
        
    model = expand_embeddings(model, config.vocab_size, meta['new_vocab_size'], device)
    
    batch_size = 8
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
        batch_size = 16 # Double batch size for 2 GPUs # T4 AMP allows slightly higher batch sizes!
    context_length = config.context_length
    
    train_ds = SFTDataset(data_dir, "train", context_length)
    val_ds = SFTDataset(data_dir, "val", context_length)
    
    # Use PyTorch DataLoader for multiprocessing and shuffling
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    steps_per_epoch = len(train_loader)
    print(f"\nTotal SFT Train Conversations: {len(train_ds):,}")
    print(f"Steps per epoch: {steps_per_epoch:,}")
    
    # Explicitly creating a new optimizer and ignoring pretraining state
    learning_rate = 1e-5
    print(f"Initializing fresh AdamW optimizer with lr={learning_rate}...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.1)
    
    # Mixed precision scaler
    scaler = GradScaler("cuda", enabled=torch.cuda.is_available())
    
    max_steps = steps_per_epoch
    eval_interval = max(1, min(500, max_steps // 5))
    
    out_dir = os.path.join(os.path.dirname(args.checkpoint), "sft")
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"\nStarting SFT for {max_steps} steps...")
    model.train()
    
    t0 = time.time()
    best_val_loss = float('inf')
    
    train_iter = iter(train_loader)
    
    for step in range(max_steps + 1):
        if step % eval_interval == 0 or step == max_steps:
            losses = estimate_loss(model, train_loader, val_loader, eval_iters=50, device=device)
            dt = time.time() - t0
            print(f"Step {step:5d} | train loss {losses['train']:.4f} | val loss {losses['val']:.4f} | time: {dt:.2f}s")
            
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                save_path = os.path.join(out_dir, "sft_best.pt")
                model_to_save = model.module if hasattr(model, 'module') else model
                torch.save({'model_state': model_to_save.state_dict()}, save_path)
                print(f"--> Saved best model to {save_path}")
            
            t0 = time.time()
            
        if step == max_steps:
            break
            
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)
            
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad(set_to_none=True)
        
        # Forward pass with Automatic Mixed Precision
        if device == "cuda":
            with autocast(device_type="cuda", dtype=torch.float16):
                logits, loss = model(x, targets=y)
        else:
            logits, loss = model(x, targets=y)
            
        # Backward pass using Scaler
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        scaler.step(optimizer)
        scaler.update()

    print("\nPhase 3: SFT Training Complete!")

if __name__ == "__main__":
    main()
