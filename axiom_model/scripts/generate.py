"""
Text Generation Script
Loads a checkpoint and generates text autoregressively.
Supports streaming output to the console.
"""
import os
import sys
import argparse
import json
import torch
import tiktoken

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.gpt_config import GPTConfig
from models.model import GPT

def parse_args():
    parser = argparse.ArgumentParser(description="GPT Generation Script")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to best.pt or latest.pt")
    parser.add_argument('--prompt', type=str, default="\n", help="Initial text to prompt the model")
    parser.add_argument('--max_new_tokens', type=int, default=100, help="Number of tokens to generate")
    parser.add_argument('--temperature', type=float, default=0.8, help="Temperature for sampling (0.0 = greedy)")
    parser.add_argument('--top_k', type=int, default=50, help="Top-K sampling")
    parser.add_argument('--device', type=str, default=None, help="cpu, mps, cuda")
    return parser.parse_args()

def get_default_device():
    if torch.cuda.is_available():
        return 'cuda'
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'

def main():
    args = parse_args()
    device = args.device if args.device else get_default_device()
    print(f"Using device: {device}")
    
    # 1. Load Checkpoint Metadata
    run_dir = os.path.dirname(args.checkpoint)
    config_path = os.path.join(run_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Missing config.json in {run_dir}. Cannot reconstruct model.")
        
    with open(config_path, "r") as f:
        metadata = json.load(f)
        
    # 2. Initialize Model
    gpt_config = GPTConfig(**metadata['gpt_config'])
    model = GPT(gpt_config)
    
    # 3. Load Weights
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    model.to(device)
    model.eval()
    
    # 4. Tokenizer
    # We default to tiktoken GPT-2 encoding for MVP
    enc = tiktoken.get_encoding("gpt2")
    
    # Encode prompt
    prompt_tokens = enc.encode(args.prompt)
    x = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
    
    print("\n--- GENERATION START ---")
    print(args.prompt, end="", flush=True)
    
    # Track time
    import time
    t0 = time.time()
    
    # 5. Generation Loop (Streaming)
    with torch.no_grad():
        for _ in range(args.max_new_tokens):
            x_cond = x if x.size(1) <= model.config.context_length else x[:, -model.config.context_length:]
            out = model(x_cond, targets=None)
                
            logits = out[0]
            logits = logits[:, -1, :] # (Batch, Vocab_Size)
            
            if args.temperature == 0.0:
                _, next_token_tensor = torch.topk(logits, k=1, dim=-1)
            else:
                logits = logits / args.temperature
                if args.top_k is not None:
                    v, _ = torch.topk(logits, min(args.top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                probs = torch.nn.functional.softmax(logits, dim=-1)
                next_token_tensor = torch.multinomial(probs, num_samples=1)
                
            next_token = next_token_tensor[0, -1].item()
            x = torch.cat((x, next_token_tensor), dim=1)
            
            # Decode and stream to console
            text_chunk = enc.decode([next_token])
            print(text_chunk, end="", flush=True)
            
    t1 = time.time()
    dt = t1 - t0
    tokens_per_sec = args.max_new_tokens / dt
    
    print("\n--- GENERATION END ---\n")
    print(f"Generated {args.max_new_tokens} tokens in {dt:.2f} seconds ({tokens_per_sec:.2f} tok/sec)")

if __name__ == "__main__":
    main()
