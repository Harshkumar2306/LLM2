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
from scripts.retrievers import LocalRetriever, WebRetriever, HybridRetriever

def get_default_device():
    if torch.cuda.is_available(): return 'cuda'
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(): return 'mps'
    return 'cpu'

def build_rag_prompt(ctx, query):
    if ctx:
        return (
            f"<|system|>\nYou are Axiom, a helpful AI assistant.\n"
            f"Use the retrieved context below only if it is relevant to the user's question. "
            f"If the context is insufficient, answer from your general knowledge and clearly state when you are not certain.\n"
            f"Context:\n========\n{ctx.strip()}\n<|end|>\n"
            f"<|user|>\nQuestion:\n========\n{query}\nAnswer:\n======<|end|>\n<|assistant|>\n"
        )
    else:
        return f"<|system|>\nYou are Axiom, a helpful AI assistant.<|end|>\n<|user|>\n{query}<|end|>\n<|assistant|>\n"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--yaml', type=str, required=True)
    parser.add_argument('--mode', type=str, choices=['none', 'local', 'web', 'hybrid'], default='none', help='Retrieval mode')
    parser.add_argument('--rag_db', type=str, default='experiments/rag_db', help='FAISS db directory for local/hybrid')
    parser.add_argument('--min_similarity', type=float, default=0.3, help='Threshold for local retrieval')
    parser.add_argument('--top_k', type=int, default=3, help='Max retrieved chunks to return')
    parser.add_argument('--reddit_only', action='store_true', help='Only search Reddit in web/hybrid mode')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--max_new_tokens', type=int, default=256)
    parser.add_argument('--prompt', type=str, default=None)
    parser.add_argument('--debug_retrieval', action='store_true')
    args = parser.parse_args()
    
    device = args.device if args.device else get_default_device()
    
    # 1. Initialize Retriever based on mode
    retriever = None
    if args.mode == 'local':
        retriever = LocalRetriever(db_dir=args.rag_db, top_k=args.top_k, min_similarity=args.min_similarity)
    elif args.mode == 'web':
        retriever = WebRetriever(top_k=args.top_k, reddit_only=args.reddit_only)
    elif args.mode == 'hybrid':
        local_r = LocalRetriever(db_dir=args.rag_db, top_k=args.top_k, min_similarity=args.min_similarity)
        web_r = WebRetriever(top_k=args.top_k, reddit_only=args.reddit_only)
        retriever = HybridRetriever(local_r, web_r, top_k=args.top_k)
        
    print(f"Loading GPT checkpoint on {device}...")
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
    checkpoint = torch.load(args.checkpoint, map_location=device)
    gpt_config.vocab_size = checkpoint['model_state']['embeddings.wte.weight'].shape[0]
    
    model = GPT(gpt_config)
    model.load_state_dict(checkpoint['model_state'])
    model.to(device)
    model.eval()
    
    tokenizer = Tokenizer(special_tokens=SPECIAL_TOKENS)
    end_token_id = SPECIAL_TOKENS["<|end|>"]
    
    print("\n" + "="*50)
    print(f"🤖 Axiom Unified Chat [Mode: {args.mode.upper()}]")
    print("="*50 + "\n")
    
    def generate_response(user_text):
        context_text = ""
        retrieved_sources = []
        
        # 1. Retrieval Phase
        if retriever:
            results = retriever.retrieve(user_text)
            debug_output = []
            
            for i, res in enumerate(results):
                source = res['source']
                text = res['text']
                score = res['score']
                
                context_text += f"[Source: {source}]\n{text}\n\n"
                retrieved_sources.append(source)
                
                if args.debug_retrieval:
                    debug_output.append(f"{i+1}.\nSource: {source}\nScore: {score:.4f}\n{text[:150]}...")
                    
            if args.debug_retrieval:
                print("\n=== [DEBUG: RETRIEVAL RESULTS] ===")
                if results:
                    print("\n------------------\n".join(debug_output))
                else:
                    print("No chunks retrieved.")
                print("==================================\n")
                
        # 2. Prompting Phase
        prompt = build_rag_prompt(context_text, user_text)
        tokens = tokenizer.encode(prompt)
        x = torch.tensor([tokens], dtype=torch.long, device=device)
        
        print("Axiom: ", end="", flush=True)
        
        # 3. Generation Phase
        with torch.no_grad():
            for _ in range(args.max_new_tokens):
                x_cond = x if x.size(1) <= gpt_config.context_length else x[:, -gpt_config.context_length:]
                out = model(x_cond, targets=None)
                logits = out[0] if isinstance(out, tuple) else out
                logits = logits[:, -1, :] / args.temperature
                
                probs = torch.nn.functional.softmax(logits, dim=-1)
                next_token_tensor = torch.multinomial(probs, num_samples=1)
                next_token = next_token_tensor[0, -1].item()
                
                if next_token == end_token_id:
                    break
                    
                x = torch.cat((x, next_token_tensor), dim=1)
                text_chunk = tokenizer.decode([next_token])
                print(text_chunk, end="", flush=True)
        print("\n")
        
        if retrieved_sources:
            print("\nSources:")
            for i, src in enumerate(set(retrieved_sources)):
                print(f"{i+1}. {src}")
            print("\n")

    if args.prompt:
        print(f"You: {args.prompt}")
        generate_response(args.prompt)
    else:
        print("Type 'quit' to exit")
        while True:
            try:
                user_input = input("\nYou: ")
            except EOFError:
                break
            if user_input.lower() in ['quit', 'exit']:
                break
            generate_response(user_input)

if __name__ == '__main__':
    main()
