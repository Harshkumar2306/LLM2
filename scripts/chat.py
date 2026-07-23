import os
import sys
import argparse
import torch
import yaml
import inspect

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.gpt_config import GPTConfig
from config.enums import AttentionType, PositionType, FFNType, NormType
from models.model import GPT
from tokenizer.tokenizer import Tokenizer, SPECIAL_TOKENS

def get_default_device():
    if torch.cuda.is_available(): return 'cuda'
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(): return 'mps'
    return 'cpu'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--yaml', type=str, required=True)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--max_new_tokens', type=int, default=256)
    args = parser.parse_args()
    
    device = args.device if args.device else get_default_device()
    print(f"Loading checkpoint on {device}...")
    
    with open(args.yaml, "r") as f:
        raw_config = yaml.safe_load(f)
    
    valid_keys = inspect.signature(GPTConfig).parameters.keys()
    config_kwargs = {k: v for k, v in raw_config.items() if k in valid_keys}
    if 'attention_type' in config_kwargs and isinstance(config_kwargs['attention_type'], str):
        config_kwargs['attention_type'] = AttentionType(config_kwargs['attention_type'])
    if 'position_type' in config_kwargs and isinstance(config_kwargs['position_type'], str):
        config_kwargs['position_type'] = PositionType(config_kwargs['position_type'])
    if 'ffn_type' in config_kwargs and isinstance(config_kwargs['ffn_type'], str):
        config_kwargs['ffn_type'] = FFNType(config_kwargs['ffn_type'])
    if 'norm_type' in config_kwargs and isinstance(config_kwargs['norm_type'], str):
        config_kwargs['norm_type'] = NormType(config_kwargs['norm_type'])
    gpt_config = GPTConfig(**config_kwargs)
    gpt_config.vocab_size = 50261
    
    model = GPT(gpt_config)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    model.to(device)
    model.eval()
    
    tokenizer = Tokenizer(special_tokens=SPECIAL_TOKENS)
    end_token_id = SPECIAL_TOKENS["<|end|>"]
    
    print("\n" + "="*50)
    print("🤖 Axiom Chat Interface (Type 'quit' to exit)")
    print("="*50 + "\n")
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ['quit', 'exit']:
            break
            
        prompt = f"<|system|>\nYou are Axiom, a helpful AI assistant.\n<|user|>\n{user_input}\n<|assistant|>\n"
        tokens = tokenizer.encode(prompt)
        x = torch.tensor([tokens], dtype=torch.long, device=device)
        
        print("Axiom: ", end="", flush=True)
        
        with torch.no_grad():
            for _ in range(args.max_new_tokens):
                x_cond = x if x.size(1) <= gpt_config.context_length else x[:, -gpt_config.context_length:]
                out = model(x_cond, targets=None)
                logits = out[0] if isinstance(out, tuple) else out
                logits = logits[:, -1, :]
                
                logits = logits / args.temperature
                probs = torch.nn.functional.softmax(logits, dim=-1)
                next_token_tensor = torch.multinomial(probs, num_samples=1)
                next_token = next_token_tensor[0, -1].item()
                
                if next_token == end_token_id:
                    break
                    
                x = torch.cat((x, next_token_tensor), dim=1)
                text_chunk = tokenizer.decode([next_token])
                print(text_chunk, end="", flush=True)
        print()

if __name__ == "__main__":
    main()
