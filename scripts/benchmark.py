import os
import sys
import argparse
import time
import json
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.gpt_config import GPTConfig
from models.model import GPT

def get_device():
    if torch.cuda.is_available():
        return 'cuda'
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--out_file', type=str, default='results/benchmark.json')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--context_length', type=int, default=256)
    parser.add_argument('--gen_tokens', type=int, default=100)
    return parser.parse_args()

def benchmark_training_step(model, device, batch_size, context_length, iters=20):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Dummy data
    x = torch.randint(0, 50257, (batch_size, context_length), device=device)
    y = torch.randint(0, 50257, (batch_size, context_length), device=device)
    
    # Warmup
    for _ in range(3):
        optimizer.zero_grad()
        logits, loss = model(x, targets=y)
        loss.backward()
        optimizer.step()
        
    if device == 'cuda':
        torch.cuda.synchronize()
    elif device == 'mps':
        torch.mps.synchronize()
        
    t0 = time.time()
    for _ in range(iters):
        optimizer.zero_grad()
        logits, loss = model(x, targets=y)
        loss.backward()
        optimizer.step()
        
    if device == 'cuda':
        torch.cuda.synchronize()
    elif device == 'mps':
        torch.mps.synchronize()
        
    t1 = time.time()
    
    total_tokens = iters * batch_size * context_length
    dt = t1 - t0
    return total_tokens / dt

def benchmark_generation(model, device, gen_tokens=100):
    model.eval()
    
    # Dummy prompt (batch size 1)
    x = torch.randint(0, 50257, (1, 10), device=device)
    
    # Warmup
    _ = model.generate(x, max_new_tokens=2)
    
    if device == 'cuda':
        torch.cuda.synchronize()
    elif device == 'mps':
        torch.mps.synchronize()
        
    t0 = time.time()
    _ = model.generate(x, max_new_tokens=gen_tokens)
    
    if device == 'cuda':
        torch.cuda.synchronize()
    elif device == 'mps':
        torch.mps.synchronize()
        
    t1 = time.time()
    dt = t1 - t0
    return gen_tokens / dt

def get_memory_usage(device):
    if device == 'cuda':
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    elif device == 'mps':
        # mps does not expose max_memory_allocated, we approximate with current_allocated_memory
        return torch.mps.current_allocated_memory() / (1024 ** 2)
    return 0.0

def main():
    args = parse_args()
    device = args.device if args.device else get_device()
    print(f"Benchmarking on {device}...")
    
    # Initialize default model
    config = GPTConfig(
        vocab_size=50257,
        d_model=256,
        n_heads=8,
        n_layers=6,
        context_length=args.context_length
    )
    
    model = GPT(config)
    model.to(device)
    
    params = sum(p.numel() for p in model.parameters())
    
    print("Benchmarking training step...")
    train_tps = benchmark_training_step(model, device, args.batch_size, args.context_length)
    
    print("Benchmarking generation...")
    gen_tps = benchmark_generation(model, device, args.gen_tokens)
    
    mem_mb = get_memory_usage(device)
    
    results = {
        "parameters": params,
        "training_tokens_per_sec": round(train_tps, 2),
        "generation_tokens_per_sec": round(gen_tps, 2),
        "peak_memory_mb": round(mem_mb, 2),
        "device": device
    }
    
    print("\n--- Benchmark Results ---")
    for k, v in results.items():
        print(f"{k}: {v}")
        
    os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
    with open(args.out_file, 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"\nSaved results to {args.out_file}")

if __name__ == "__main__":
    main()
