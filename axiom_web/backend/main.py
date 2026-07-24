import os
import sys
import torch
import yaml
import inspect
import json
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# Append repo root to sys.path to import from axiom_model
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT_DIR)

from axiom_model.config.gpt_config import GPTConfig
from axiom_model.config.enums import AttentionType, PositionType, FFNType, NormType
from axiom_model.models.model import GPT
from axiom_model.tokenizer.tokenizer import Tokenizer, SPECIAL_TOKENS
from axiom_model.scripts.retrievers import LocalRetriever, WebRetriever, HybridRetriever

app = FastAPI(title="Axiom API")

# Enable CORS for the React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state for models
model = None
gpt_config = None
tokenizer = None
end_token_id = None
device = 'cpu'

class ChatRequest(BaseModel):
    message: str
    mode: str = "none"  # none, local, web, hybrid
    reddit_only: bool = False
    temperature: float = 0.7
    max_tokens: int = 256

def get_default_device():
    if torch.cuda.is_available(): return 'cuda'
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(): return 'mps'
    return 'cpu'

def load_model():
    global model, gpt_config, tokenizer, end_token_id, device
    print("Loading Axiom model into memory...")
    device = get_default_device()
    
    checkpoint_path = os.path.join(ROOT_DIR, "axiom_model", "sft_best.pt")
    yaml_path = os.path.join(ROOT_DIR, "axiom_model", "configs", "phase2", "axiom_v1.0.yaml")
    
    if not os.path.exists(checkpoint_path):
        print(f"Warning: Checkpoint not found at {checkpoint_path}. Please place sft_best.pt in axiom_model/")
        return
        
    with open(yaml_path, "r") as f:
        raw_config = yaml.safe_load(f)
    
    valid_keys = inspect.signature(GPTConfig).parameters.keys()
    config_kwargs = {k: v for k, v in raw_config.items() if k in valid_keys}
    if 'attention_type' in config_kwargs: config_kwargs['attention_type'] = AttentionType(config_kwargs['attention_type'])
    if 'position_type' in config_kwargs: config_kwargs['position_type'] = PositionType(config_kwargs['position_type'])
    if 'ffn_type' in config_kwargs: config_kwargs['ffn_type'] = FFNType(config_kwargs['ffn_type'])
    if 'norm_type' in config_kwargs: config_kwargs['norm_type'] = NormType(config_kwargs['norm_type'])
    
    gpt_config = GPTConfig(**config_kwargs)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    gpt_config.vocab_size = checkpoint['model_state']['embeddings.wte.weight'].shape[0]
    
    model = GPT(gpt_config)
    model.load_state_dict(checkpoint['model_state'])
    model.to(device)
    model.eval()
    
    tokenizer = Tokenizer(special_tokens=SPECIAL_TOKENS)
    end_token_id = SPECIAL_TOKENS["<|end|>"]
    print("Model loaded successfully!")

@app.on_event("startup")
def startup_event():
    load_model()

def build_rag_prompt(ctx, query):
    if ctx:
        return (
            f"<|system|>\nYou are Axiom, a helpful AI assistant.\n"
            f"Use the retrieved context below only if it is relevant to the user's question. "
            f"If the context is insufficient, answer from your general knowledge and clearly state when you are not certain.\n"
            f"Context:\n========\n{ctx.strip()}\n<|end|>\n"
            f"<|user|>\nQuestion:\n========\n{query}\nAnswer:\n======<|end|>\n<|assistant|>\n"
        )
    return f"<|system|>\nYou are Axiom, a helpful AI assistant.<|end|>\n<|user|>\n{query}<|end|>\n<|assistant|>\n"

async def stream_generation(req: ChatRequest):
    if model is None:
        yield f"data: {json.dumps({'error': 'Model not loaded'})}\n\n"
        return
        
    # 1. Retrieval Phase
    context_text = ""
    sources = []
    
    rag_db_path = os.path.join(ROOT_DIR, "axiom_model", "experiments", "rag_db")
    
    if req.mode != "none":
        retriever = None
        try:
            if req.mode == 'local':
                retriever = LocalRetriever(db_dir=rag_db_path)
            elif req.mode == 'web':
                retriever = WebRetriever(reddit_only=req.reddit_only)
            elif req.mode == 'hybrid':
                local_r = LocalRetriever(db_dir=rag_db_path)
                web_r = WebRetriever(reddit_only=req.reddit_only)
                retriever = HybridRetriever(local_r, web_r)
                
            if retriever:
                results = retriever.retrieve(req.message)
                for res in results:
                    context_text += f"[Source: {res['source']}]\n{res['text']}\n\n"
                    if res['source'] not in sources:
                        sources.append(res['source'])
                        
                # Yield sources first so UI can display them immediately
                yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"
        except Exception as e:
            print(f"Retrieval error: {e}")
            
    # 2. Prompting Phase
    prompt = build_rag_prompt(context_text, req.message)
    tokens = tokenizer.encode(prompt)
    x = torch.tensor([tokens], dtype=torch.long, device=device)
    
    # 3. Generation Phase
    with torch.no_grad():
        for _ in range(req.max_tokens):
            x_cond = x if x.size(1) <= gpt_config.context_length else x[:, -gpt_config.context_length:]
            out = model(x_cond, targets=None)
            logits = out[0] if isinstance(out, tuple) else out
            logits = logits[:, -1, :] / req.temperature
            
            probs = torch.nn.functional.softmax(logits, dim=-1)
            next_token_tensor = torch.multinomial(probs, num_samples=1)
            next_token = next_token_tensor[0, -1].item()
            
            if next_token == end_token_id:
                break
                
            x = torch.cat((x, next_token_tensor), dim=1)
            text_chunk = tokenizer.decode([next_token])
            
            # Send SSE chunk
            yield f"data: {json.dumps({'type': 'chunk', 'text': text_chunk})}\n\n"
            await asyncio.sleep(0.01) # Yield control back to event loop
            
    yield f"data: {json.dumps({'type': 'done'})}\n\n"

@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
    return StreamingResponse(stream_generation(req), media_type="text/event-stream")
