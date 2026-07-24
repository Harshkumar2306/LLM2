import os
import sys
import json
import argparse
import torch
import yaml
import faiss
import numpy as np
import inspect
from sentence_transformers import SentenceTransformer

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
    parser.add_argument('--rag_db', type=str, default='experiments/rag_db')
    parser.add_argument('--embedding_model', type=str, default='all-MiniLM-L6-v2')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--max_new_tokens', type=int, default=256)
    parser.add_argument('--prompt', type=str, default=None)
    parser.add_argument('--top_k_retrieval', type=int, default=2, help='Number of chunks to retrieve')
    parser.add_argument('--threshold', type=float, default=0.3, help='Minimum similarity score to use context')
    args = parser.parse_args()
    
    device = args.device if args.device else get_default_device()
    
    print("Loading FAISS Vector Database...")
    index_path = os.path.join(args.rag_db, "vector.index")
    meta_path = os.path.join(args.rag_db, "meta.json")
    
    if not os.path.exists(index_path) or not os.path.exists(meta_path):
        print(f"Error: RAG database not found at {args.rag_db}")
        print("Please run build_vector_db.py first!")
        return
        
    faiss_index = faiss.read_index(index_path)
    with open(meta_path, "r") as f:
        rag_meta = json.load(f)
        
    embedder = SentenceTransformer(args.embedding_model)
    
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
    print("🤖 Axiom RAG Chat Interface")
    print("="*50 + "\n")
    
    def generate_response(user_text):
        # 1. Retrieve context
        query_vec = embedder.encode([user_text])
        query_vec = query_vec / np.linalg.norm(query_vec, axis=1, keepdims=True)
        D, I = faiss_index.search(query_vec.astype(np.float32), k=args.top_k_retrieval)
        
        context_text = ""
        valid_chunks = 0
        for distance, idx in zip(D[0], I[0]):
            if idx != -1 and idx < len(rag_meta):
                if distance >= args.threshold:
                    source = rag_meta[idx].get('source', 'Unknown')
                    context_text += f"[Source: {source}]\n{rag_meta[idx]['text']}\n\n"
                    valid_chunks += 1
                
        if valid_chunks > 0:
            print(f"\n[Retrieved {valid_chunks} relevant chunks from FAISS (Threshold: {args.threshold})]")
            print(context_text.strip()[:200] + "...\n")
            # 2. Build explicit augmented prompt
            prompt = f"<|system|>\nYou are Axiom. Use the following retrieved information to answer.\nContext:\n----------------\n{context_text.strip()}\n----------------<|end|>\n<|user|>\nQuestion:\n{user_text}\nAnswer:<|end|>\n<|assistant|>\n"
        else:
            print(f"\n[No relevant documents found above threshold {args.threshold}]\n")
            prompt = f"<|system|>\nYou are Axiom, a helpful AI assistant.<|end|>\n<|user|>\n{user_text}<|end|>\n<|assistant|>\n"
        
        tokens = tokenizer.encode(prompt)
        x = torch.tensor([tokens], dtype=torch.long, device=device)
        
        print("Axiom: ", end="", flush=True)
        
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
