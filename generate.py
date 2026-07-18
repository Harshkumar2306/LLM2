import torch
import argparse
from core.model import AxiomModel
from data.tokenizer import Tokenizer
import yaml

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def generate(model, tokenizer, prompt, max_new_tokens=100, temperature=0.8, top_k=50, device='cuda'):
    model.eval()
    
    # Encode prompt
    input_ids = tokenizer.encode(prompt)
    x = torch.tensor([input_ids], dtype=torch.long, device=device)
    
    # Generate
    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            for _ in range(max_new_tokens):
                # Crop context if it gets too long
                x_cond = x if x.size(1) <= model.config.context_length else x[:, -model.config.context_length:]
                
                # Forward pass
                logits = model(x_cond)
                logits = logits[:, -1, :] # Pluck the logits at the final step
                
                # Temperature scaling
                logits = logits / temperature
                
                # Top-K filtering
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                    
                # Sample
                probs = torch.nn.functional.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                
                # Append to sequence
                x = torch.cat((x, idx_next), dim=1)
                
                # If we hit the End of Text token, stop early
                if idx_next.item() == tokenizer.eot_token:
                    break

    # Decode
    generated_text = tokenizer.decode(x[0].tolist())
    return generated_text

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best.pt")
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    parser.add_argument("--prompt", type=str, default="The future of artificial intelligence is", help="Prompt to start generation")
    parser.add_argument("--tokens", type=int, default=150, help="Number of tokens to generate")
    parser.add_argument("--temp", type=float, default=0.8, help="Temperature for sampling")
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Loading model on {device}...")

    # Load config and initialize model architecture
    config = load_config(args.config)
    model = AxiomModel(config['model'])
    
    # Load weights
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    
    # Handle DDP prefixes if they exist
    state_dict = checkpoint['model_state']
    unwrapped_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('_orig_mod.'):
            unwrapped_state_dict[k[10:]] = v
        elif k.startswith('module.'):
            unwrapped_state_dict[k[7:]] = v
        else:
            unwrapped_state_dict[k] = v
            
    model.load_state_dict(unwrapped_state_dict)
    model.to(device)
    
    tokenizer = Tokenizer()

    print("\n" + "="*50)
    print("PROMPT:", args.prompt)
    print("="*50)
    
    output = generate(model, tokenizer, args.prompt, max_new_tokens=args.tokens, temperature=args.temp)
    
    print("\nGENERATED TEXT:")
    print("-" * 50)
    print(output)
    print("-" * 50)
