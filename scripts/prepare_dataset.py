"""
Deterministic Streaming Preprocessing Pipeline for Axiom Phase 0.

Reads data/datasets.yaml, streams from HuggingFace, cleans, deduplicates, tokenizes,
and outputs independent train.bin, val.bin, and metadata.json into isolated dataset folders.
"""

import os
import sys
import time
import json
import hashlib
import argparse
import datetime
import yaml
import numpy as np
from datasets import load_dataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tokenizer.tokenizer import Tokenizer

def clean_text(text: str, min_chars: int = 100) -> bool:
    """Lightweight cleaning rule."""
    if not text:
        return False
    if len(text.strip()) < min_chars:
        return False
    try:
        text.encode('utf-8').decode('utf-8')
    except UnicodeDecodeError:
        return False
    return True

def hash_document(text: str) -> str:
    """Generates a SHA-256 hash of the document."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def process_dataset(config: dict, base_outdir: str, tokenizer: Tokenizer, context_length: int):
    name = config['name']
    hf_path = config['hf']
    subset = config.get('subset', 'default')
    split = config.get('split', 'train')
    target_tokens = config['tokens']
    dedup = config.get('dedup', True)
    min_chars = config.get('min_chars', 100)

    out_dir = os.path.join(base_outdir, name)
    os.makedirs(out_dir, exist_ok=True)
    
    train_path = os.path.join(out_dir, "train.bin")
    val_path = os.path.join(out_dir, "val.bin")
    meta_path = os.path.join(out_dir, "metadata.json")
    
    # Check if already processed
    if os.path.exists(meta_path):
        try:
            with open(meta_path, 'r') as f:
                meta = json.load(f)
                if meta.get("tokens", 0) >= target_tokens:
                    print(f"\n[Skipping] {name} - already fully processed ({meta['tokens']:,} tokens).")
                    return
        except json.JSONDecodeError:
            pass
    
    # Reset files if starting fresh
    open(train_path, 'wb').close()
    open(val_path, 'wb').close()
    
    stats = {
        "docs_streamed": 0,
        "docs_accepted": 0,
        "docs_rejected_empty": 0,
        "docs_rejected_short": 0,
        "docs_rejected_duplicates": 0,
        "train_tokens": 0,
        "val_tokens": 0,
    }
    
    seen_hashes = set() if dedup else None
    
    chunk_size = 1000000
    train_buffer = []
    val_buffer = []
    
    print(f"\n==================================================")
    print(f"Processing Dataset: {name}")
    print(f"Source: {hf_path} ({subset} | {split})")
    print(f"Target Tokens: {target_tokens:,}")
    print(f"==================================================")
    
    start_time = time.time()
    
    try:
        kwargs = {'name': subset} if subset != 'default' else {}
        dataset = load_dataset(hf_path, split=split, streaming=True, **kwargs)
    except Exception as e:
        print(f"Error loading {hf_path}: {e}")
        return

    def flush_buffer(buffer, filepath):
        if not buffer: return
        arr = np.array(buffer, dtype=np.uint16)
        with open(filepath, 'ab') as f:
            f.write(arr.tobytes())
        buffer.clear()

    total_tokens = 0
    
    for doc in dataset:
        if total_tokens >= target_tokens:
            break
            
        stats["docs_streamed"] += 1
        
        # Try to extract text field (hf datasets use various keys)
        text = doc.get("text", doc.get("content", doc.get("document", doc.get("code", ""))))
        
        # 1. Cleaning
        if not text or not text.strip():
            stats["docs_rejected_empty"] += 1
            continue
            
        if len(text.strip()) < min_chars:
            stats["docs_rejected_short"] += 1
            continue
            
        # 2. Deduplication
        if dedup:
            doc_hash = hash_document(text)
            if doc_hash in seen_hashes:
                stats["docs_rejected_duplicates"] += 1
                continue
            seen_hashes.add(doc_hash)
            
        stats["docs_accepted"] += 1
        
        # 3. Tokenization
        tokens = tokenizer.encode(text)
        tokens.append(tokenizer.eot_token)
        
        token_count = len(tokens)
        
        # 4. Deterministic Split (Index % 100 == 0 -> Validation)
        is_val = (stats["docs_accepted"] % 100 == 0)
        
        if is_val:
            val_buffer.extend(tokens)
            stats["val_tokens"] += token_count
        else:
            train_buffer.extend(tokens)
            stats["train_tokens"] += token_count
            
        total_tokens += token_count
        
        if len(train_buffer) >= chunk_size:
            flush_buffer(train_buffer, train_path)
        if len(val_buffer) >= chunk_size:
            flush_buffer(val_buffer, val_path)
            
        if stats["docs_streamed"] % 100000 == 0:
            elapsed = time.time() - start_time
            tps = total_tokens/elapsed if elapsed > 0 else 0
            print(f"  Processed {stats['docs_streamed']:,} docs | Tokens: {total_tokens:,}/{target_tokens:,} | TPS: {tps:,.0f}")

    # Final flush
    flush_buffer(train_buffer, train_path)
    flush_buffer(val_buffer, val_path)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    tps = total_tokens / elapsed_time if elapsed_time > 0 else 0
    
    train_size_mb = os.path.getsize(train_path) / (1024 * 1024)
    val_size_mb = os.path.getsize(val_path) / (1024 * 1024)
    
    meta = {
        "dataset": name,
        "source": hf_path,
        "subset": subset,
        "version": "1.0",
        "tokens": total_tokens,
        "vocab_size": tokenizer.n_vocab,
        "tokenizer": "tiktoken (gpt2)",
        "created": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "statistics": stats
    }
    
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=4)
        
    print(f"\n[Done] {name}")
    print(f"Total Tokens: {total_tokens:,}")
    print(f"Train Tokens: {stats['train_tokens']:,} ({train_size_mb:.2f} MB)")
    print(f"Val Tokens: {stats['val_tokens']:,} ({val_size_mb:.2f} MB)")
    print(f"Speed: {tps:,.0f} tokens/sec")

def main(args):
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
        
    tokenizer = Tokenizer()
    base_outdir = args.outdir
    
    for ds_config in cfg.get('datasets', []):
        process_dataset(ds_config, base_outdir, tokenizer, args.context_length)
        
    print("\nAll datasets processed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deterministic Streaming Preprocessing Pipeline")
    parser.add_argument("--config", type=str, default="data/datasets.yaml", help="Path to YAML config")
    parser.add_argument("--outdir", type=str, default="data", help="Base output directory")
    parser.add_argument("--context-length", type=int, default=1024, help="Target context length for metadata")
    
    args = parser.parse_args()
    main(args)
