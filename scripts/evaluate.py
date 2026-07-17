import os
import sys
import argparse
import math
import time
import json
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.gpt_config import GPTConfig
from core.model import GPT
from data.dataset import GPTDataset, InMemoryStorage
from engine.trainer import Trainer, TrainerConfig

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to best.pt")
    parser.add_argument('--val_bin', type=str, default='data/val.bin', help="Path to val.bin")
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--out_file', type=str, default='results/evaluate.json')
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"Loading checkpoint {args.checkpoint}...")
    device = args.device if args.device else ('cuda' if torch.cuda.is_available() else ('mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'cpu'))
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config_dict = checkpoint['config']
    
    # Checkpoint doesn't explicitly store gpt_config, but we can reconstruct it
    # wait, trainer saves its own config, where did we put gpt_config?
    # Ah, in train.py we wrote it to config.json. If it's not in the checkpoint,
    # let's try to infer from model_state.
    # Actually, we can read config.json from the checkpoint directory!
    ckpt_dir = os.path.dirname(args.checkpoint)
    config_path = os.path.join(ckpt_dir, "config.json")
    if not os.path.exists(config_path):
        print(f"Error: Could not find config.json in {ckpt_dir}. Needed to reconstruct model architecture.")
        sys.exit(1)
        
    with open(config_path, 'r') as f:
        meta = json.load(f)
        gpt_kwargs = meta['gpt_config']
        
    gpt_config = GPTConfig(**gpt_kwargs)
    model = GPT(gpt_config)
    model.load_state_dict(checkpoint['model_state'])
    model.to(device)
    model.eval()
    
    print(f"Loading validation data from {args.val_bin}...")
    val_storage = InMemoryStorage(args.val_bin)
    val_dataset = GPTDataset(val_storage, gpt_config.context_length)
    
    # We will evaluate over the entire validation set.
    # Total sequences = len(val_dataset) // args.batch_size
    pin_memory = device == 'cuda'
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=pin_memory)
    
    print("Evaluating...")
    losses = []
    t0 = time.time()
    
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            # In trainer we use ctx (autocast). We can do that here too.
            ptdtype = torch.float16 if device == 'cuda' else torch.float32
            ctx = torch.autocast(device_type=device, dtype=ptdtype) if device in ['cuda', 'mps'] else contextlib.nullcontext()
            
            # Since we didn't import contextlib, let's just do it directly or without autocast for eval
            # (eval doesn't strictly need autocast but it's faster on CUDA)
            if device == 'cuda':
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
                    logits, loss = model(x, targets=y)
            else:
                logits, loss = model(x, targets=y)
                
            losses.append(loss.item())

    t1 = time.time()
    total_time = t1 - t0
    
    avg_loss = sum(losses) / len(losses)
    perplexity = math.exp(avg_loss)
    
    total_tokens = len(val_dataset) * gpt_config.context_length
    tokens_per_sec = total_tokens / total_time
    
    results = {
        "validation_loss": round(avg_loss, 4),
        "perplexity": round(perplexity, 4),
        "evaluation_tokens_per_sec": round(tokens_per_sec, 2),
        "total_eval_time_sec": round(total_time, 2),
        "device": device
    }
    
    print("\n--- Evaluation Results ---")
    for k, v in results.items():
        print(f"{k}: {v}")
        
    os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
    with open(args.out_file, 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"\nSaved results to {args.out_file}")

if __name__ == "__main__":
    main()
