import os
import sys
import json
import hashlib
import numpy as np
from datasets import load_dataset
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tokenizer.tokenizer import Tokenizer, SPECIAL_TOKENS, DEFAULT_SYSTEM_PROMPT

# Configurable Targets
TARGET_CONVERSATIONS = 80000
CONTEXT_LENGTH = 1024
SEED = 42

DATASET_MIX = {
    "teknium/OpenHermes-2.5": {"weight": 0.4375, "format": "hermes"},
    "HuggingFaceH4/ultrachat_200k": {"weight": 0.3125, "format": "ultrachat"},
    "shibing624/sharegpt_gpt4": {"weight": 0.125, "format": "sharegpt"},
    "sahil2801/CodeAlpaca-20k": {"weight": 0.125, "format": "alpaca"}
}

def get_chat_tokenizer():
    tok = Tokenizer("gpt2", special_tokens=SPECIAL_TOKENS)
    return tok, SPECIAL_TOKENS

def passes_heuristics(turns, context_length, tokenizer):
    if len(turns) < 2:
        return False
        
    has_assistant = False
    total_tokens = 0
    
    for turn in turns:
        role = turn['role']
        content = turn['content']
        
        if not content or len(content.strip()) < 5:
            return False
            
        if role == "assistant":
            has_assistant = True
            if len(content) < 15:
                return False
                
            words = content.split()
            if len(words) > 10:
                unique_words = len(set(words))
                if unique_words / len(words) < 0.5:
                    return False
                    
        total_tokens += len(content) // 4
        
    if not has_assistant:
        return False
        
    if total_tokens > context_length * 2:
        return False
        
    return True

def parse_turns(example, format_type):
    turns = []
    if format_type == "hermes":
        for msg in example.get("conversations", []):
            role = msg.get("from", "")
            if role == "human": role = "user"
            elif role == "gpt": role = "assistant"
            turns.append({"role": role, "content": msg.get("value", "")})
            
    elif format_type == "ultrachat":
        for msg in example.get("messages", []):
            turns.append({"role": msg.get("role", ""), "content": msg.get("content", "")})
            
    elif format_type == "sharegpt":
        for msg in example.get("conversations", []):
            role = msg.get("from", "")
            if role == "human": role = "user"
            elif role == "gpt": role = "assistant"
            turns.append({"role": role, "content": msg.get("value", "")})
            
    elif format_type == "alpaca":
        instruction = example.get('instruction', '')
        inp = example.get('input', '')
        output = example.get('output', '')
        user_text = f"{instruction}\n\n{inp}" if inp else instruction
        turns.append({"role": "user", "content": user_text})
        turns.append({"role": "assistant", "content": output})
        
    return turns

def tokenize_conversation(turns, tokenizer):
    tokens = []
    labels = []
    
    sys_prompt = f"<|system|>\n{DEFAULT_SYSTEM_PROMPT}<|end|>\n"
    sys_toks = tokenizer.encode(sys_prompt, allowed_special="all")
    tokens.extend(sys_toks)
    labels.extend([-100] * len(sys_toks))
    
    for turn in turns:
        role = turn["role"]
        content = turn["content"]
        
        if role == "user":
            prompt = f"<|user|>\n{content}<|end|>\n"
            toks = tokenizer.encode(prompt, allowed_special="all")
            tokens.extend(toks)
            labels.extend([-100] * len(toks))
            
        elif role == "assistant":
            marker = "<|assistant|>\n"
            marker_toks = tokenizer.encode(marker, allowed_special="all")
            tokens.extend(marker_toks)
            labels.extend([-100] * len(marker_toks))
            
            resp = f"{content}<|end|>\n"
            resp_toks = tokenizer.encode(resp, allowed_special="all")
            tokens.extend(resp_toks)
            labels.extend(resp_toks)
            
    return tokens, labels

def main():
    import random
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry_run', action='store_true')
    args = parser.parse_args()

    random.seed(SEED)
    tokenizer, special_tokens = get_chat_tokenizer()
    
    target_convos = 40 if args.dry_run else TARGET_CONVERSATIONS

    print(f"Building Axiom Phase 3.1 SFT Dataset...")
    print(f"Target Conversations: {target_convos}")
    
    master_train_tokens = []
    master_train_labels = []
    master_val_tokens = []
    master_val_labels = []
    
    train_offsets = []
    val_offsets = []
    
    metadata = {
        "datasets": {},
        "context_length": CONTEXT_LENGTH,
        "vocab_size": tokenizer.n_vocab,
        "assistant_only_loss": True,
        "packing": True,
        "seed": SEED,
        "total_train_conversations": 0,
        "total_val_conversations": 0
    }
    
    seen_hashes = set()
    
    for dataset_name, info in DATASET_MIX.items():
        target_count = int(target_convos * info["weight"])
        print(f"\nProcessing {dataset_name} (Target: {target_count})")
        
        try:
            if dataset_name == "HuggingFaceH4/ultrachat_200k":
                ds = load_dataset(dataset_name, split="train_sft")
            else:
                ds = load_dataset(dataset_name, split="train")
        except Exception as e:
            print(f"Failed to load {dataset_name}: {e}")
            continue
            
        ds = list(ds)
        random.shuffle(ds)
        
        accepted_train = 0
        accepted_val = 0
        target_val = int(target_count * 0.05)
        target_train = target_count - target_val
        
        pbar = tqdm(total=target_count)
        
        for example in ds:
            if accepted_train >= target_train and accepted_val >= target_val:
                break
                
            turns = parse_turns(example, info["format"])
            
            if turns:
                first_prompt = next((t["content"] for t in turns if t["role"] == "user"), "")
                prompt_hash = hashlib.md5(first_prompt.encode('utf-8')).hexdigest()
                if prompt_hash in seen_hashes:
                    continue
                seen_hashes.add(prompt_hash)
            
            if not passes_heuristics(turns, CONTEXT_LENGTH, tokenizer):
                continue
                
            t, l = tokenize_conversation(turns, tokenizer)
            
            if len(t) > CONTEXT_LENGTH * 1.5:
                continue
                
            is_val = False
            if accepted_val < target_val:
                is_val = True
                val_offsets.append(len(master_val_tokens))
                master_val_tokens.extend(t)
                master_val_labels.extend(l)
                accepted_val += 1
            elif accepted_train < target_train:
                train_offsets.append(len(master_train_tokens))
                master_train_tokens.extend(t)
                master_train_labels.extend(l)
                accepted_train += 1
                
            if is_val or accepted_train <= target_train:
                pbar.update(1)
                
        pbar.close()
        metadata["datasets"][dataset_name] = {"train": accepted_train, "val": accepted_val}
        metadata["total_train_conversations"] += accepted_train
        metadata["total_val_conversations"] += accepted_val

    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "sft_v2_dataset")
    os.makedirs(output_dir, exist_ok=True)
    
    print("\nSaving binary datasets...")
    np.array(master_train_tokens, dtype=np.int32).tofile(os.path.join(output_dir, "train_tokens.bin"))
    np.array(master_train_labels, dtype=np.int32).tofile(os.path.join(output_dir, "train_labels.bin"))
    np.array(master_val_tokens, dtype=np.int32).tofile(os.path.join(output_dir, "val_tokens.bin"))
    np.array(master_val_labels, dtype=np.int32).tofile(os.path.join(output_dir, "val_labels.bin"))
    
    metadata["train_offsets"] = train_offsets
    metadata["val_offsets"] = val_offsets
    
    with open(os.path.join(output_dir, "meta.json"), "w") as f:
        json.dump(metadata, f, indent=4)
        
    print(f"Successfully generated Phase 3.1 dataset in {output_dir}")

if __name__ == "__main__":
    main()
