"""
Prepare Stanford Alpaca for Supervised Fine-Tuning (SFT)
- Uses Custom Chat Tokens (<|system|>, <|user|>, <|assistant|>, <|end|>)
- Applies -100 ignore index to prompt tokens so loss is only calculated on the assistant's response.
- Saves tokens as int32 to be future-proof for vocab expansions.
"""
import os
import sys
import json
import numpy as np
from datasets import load_dataset
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tokenizer.tokenizer import Tokenizer, SPECIAL_TOKENS, DEFAULT_SYSTEM_PROMPT

def get_chat_tokenizer():
    tok = Tokenizer("gpt2", special_tokens=SPECIAL_TOKENS)
    return tok, SPECIAL_TOKENS

def format_example(example, tokenizer):
    """
    Formats a single Alpaca example into tokens and labels.
    Tokens: The actual token IDs to feed into the model.
    Labels: The targets. -100 for system/user prompts, and the actual tokens for the assistant.
    """
    instruction = example['instruction']
    inp = example.get('input', '')
    output = example['output']
    
    if inp:
        user_text = f"{instruction}\n\n{inp}"
    else:
        user_text = instruction

    sys_prompt = f"<|system|>\n{DEFAULT_SYSTEM_PROMPT}<|end|>\n"
    user_prompt = f"<|user|>\n{user_text}<|end|>\n"
    
    tokens = []
    labels = []
    
    # 1. System Prompt (Loss = -100)
    sys_toks = tokenizer.encode(sys_prompt, allowed_special="all")
    tokens.extend(sys_toks)
    labels.extend([-100] * len(sys_toks))
    
    # 2. User Prompt (Loss = -100)
    user_toks = tokenizer.encode(user_prompt, allowed_special="all")
    tokens.extend(user_toks)
    labels.extend([-100] * len(user_toks))
    
    # 3. Assistant Marker (Loss = -100)
    ast_marker = "<|assistant|>\n"
    ast_marker_toks = tokenizer.encode(ast_marker, allowed_special="all")
    tokens.extend(ast_marker_toks)
    labels.extend([-100] * len(ast_marker_toks))

    # 4. Actual Assistant Response (Loss = Actual Token IDs)
    ast_response = f"{output}<|end|>\n"
    ast_response_toks = tokenizer.encode(ast_response, allowed_special="all")
    tokens.extend(ast_response_toks)
    labels.extend(ast_response_toks)
    
    return tokens, labels

def main():
    print("Loading Stanford Alpaca...")
    dataset = load_dataset("tatsu-lab/alpaca", split="train")
    
    # Split into 95% train, 5% val
    dataset = dataset.train_test_split(test_size=0.05, seed=42)
    train_ds = dataset['train']
    val_ds = dataset['test']
    
    tokenizer, special_tokens = get_chat_tokenizer()
    new_vocab_size = tokenizer.n_vocab
    
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "sft_dataset")
    os.makedirs(output_dir, exist_ok=True)
    
    meta_offsets = {}
    
    for split_name, ds in zip(["train", "val"], [train_ds, val_ds]):
        print(f"\nProcessing {split_name} split ({len(ds)} examples)...")
        all_tokens = []
        all_labels = []
        offsets = []
        current_offset = 0
        
        for example in tqdm(ds):
            offsets.append(current_offset)
            t, l = format_example(example, tokenizer)
            all_tokens.extend(t)
            all_labels.extend(l)
            current_offset += len(t)
            
        meta_offsets[split_name] = offsets
        print(f"Total tokens for {split_name}: {len(all_tokens):,}")
        
        # Save tokens as int32 (future-proof)
        tokens_np = np.array(all_tokens, dtype=np.int32)
        tokens_np.tofile(os.path.join(output_dir, f"{split_name}_tokens.bin"))
        
        # Save labels as int32 because it contains -100
        labels_np = np.array(all_labels, dtype=np.int32)
        labels_np.tofile(os.path.join(output_dir, f"{split_name}_labels.bin"))
        
    # Save meta.json
    meta = {
        "dataset_name": "tatsu-lab/alpaca",
        "tokenizer": "gpt2_custom",
        "chat_template": "<|system|>\n...<|end|>\n<|user|>\n...<|end|>\n<|assistant|>\n...<|end|>\n",
        "base_vocab_size": 50257,
        "special_token_order": ["<|system|>", "<|user|>", "<|assistant|>", "<|end|>"],
        "special_tokens": special_tokens,
        "new_vocab_size": new_vocab_size,
        "train_examples": len(train_ds),
        "val_examples": len(val_ds),
        "train_offsets": meta_offsets['train'],
        "val_offsets": meta_offsets['val']
    }
    
    with open(os.path.join(output_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=4)
        
    print(f"\nSFT Dataset successfully prepared in {output_dir}")
    print(f"New Vocab Size required: {new_vocab_size}")

if __name__ == "__main__":
    main()
