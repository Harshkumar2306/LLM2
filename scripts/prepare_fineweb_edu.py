"""
Deterministic Streaming Preprocessing Pipeline for Axiom.

Streams from HuggingFaceFW/fineweb-edu, cleans, optionally deduplicates, tokenizes,
and chunks binary output deterministically into train.bin and val.bin.
"""

import os
import sys
import time
import pickle
import hashlib
import argparse
import datetime
import numpy as np
from datasets import load_dataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.tokenizer import Tokenizer

def clean_text(text: str, min_chars: int = 100) -> bool:
    """Lightweight cleaning rule."""
    if not text:
        return False
    if len(text.strip()) < min_chars:
        return False
    # Check for valid utf-8 decoding (Huggingface usually handles this, but just to be safe)
    try:
        text.encode('utf-8').decode('utf-8')
    except UnicodeDecodeError:
        return False
    return True

def hash_document(text: str) -> str:
    """Generates a SHA-256 hash of the document."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def main(args):
    start_time = time.time()
    
    out_dir = args.outdir
    os.makedirs(out_dir, exist_ok=True)
    
    train_path = os.path.join(out_dir, "train.bin")
    val_path = os.path.join(out_dir, "val.bin")
    meta_path = os.path.join(out_dir, "meta.pkl")
    
    # Reset files if starting fresh
    open(train_path, 'wb').close()
    open(val_path, 'wb').close()
    
    tokenizer = Tokenizer()
    
    # Pipeline Statistics
    stats = {
        "docs_streamed": 0,
        "docs_accepted": 0,
        "docs_rejected_empty": 0,
        "docs_rejected_short": 0,
        "docs_rejected_duplicates": 0,
        "train_tokens": 0,
        "val_tokens": 0,
    }
    
    seen_hashes = set() if args.dedup else None
    
    # We buffer tokens in RAM up to a chunk size to minimize I/O overhead
    chunk_size = 1000000  # 1 million tokens before writing
    train_buffer = []
    val_buffer = []
    
    print(f"Streaming HuggingFaceFW/fineweb-edu (sample-10BT)...")
    print(f"Target Tokens: {args.target_tokens:,}")
    print(f"Deduplication: {'Enabled' if args.dedup else 'Disabled'}")
    
    dataset = load_dataset('HuggingFaceFW/fineweb-edu', name='sample-10BT', split='train', streaming=True)
    
    # Progress tracking
    def flush_buffer(buffer, filepath):
        if not buffer: return
        arr = np.array(buffer, dtype=np.uint16)
        with open(filepath, 'ab') as f:
            f.write(arr.tobytes())
        buffer.clear()

    total_tokens = 0
    
    for doc in dataset:
        if total_tokens >= args.target_tokens:
            break
            
        stats["docs_streamed"] += 1
        text = doc.get("text", "")
        
        # 1. Cleaning
        if not text or not text.strip():
            stats["docs_rejected_empty"] += 1
            continue
            
        if len(text.strip()) < args.min_chars:
            stats["docs_rejected_short"] += 1
            continue
            
        # 2. Deduplication (Optional)
        if args.dedup:
            doc_hash = hash_document(text)
            if doc_hash in seen_hashes:
                stats["docs_rejected_duplicates"] += 1
                continue
            seen_hashes.add(doc_hash)
            
        stats["docs_accepted"] += 1
        
        # 3. Tokenization
        tokens = tokenizer.encode(text)
        # Add EOT token to separate documents
        tokens.append(tokenizer.eot_token)
        
        token_count = len(tokens)
        
        # 4. Deterministic Split (Index % 100 == 0 -> Validation)
        # Using stats["docs_accepted"] ensures split is independent of how many docs were rejected
        is_val = (stats["docs_accepted"] % 100 == 0)
        
        if is_val:
            val_buffer.extend(tokens)
            stats["val_tokens"] += token_count
        else:
            train_buffer.extend(tokens)
            stats["train_tokens"] += token_count
            
        total_tokens += token_count
        
        # Chunking I/O
        if len(train_buffer) >= chunk_size:
            flush_buffer(train_buffer, train_path)
        if len(val_buffer) >= chunk_size:
            flush_buffer(val_buffer, val_path)
            
        # Print progress every ~100k docs
        if stats["docs_streamed"] % 100000 == 0:
            elapsed = time.time() - start_time
            print(f"Processed {stats['docs_streamed']} docs | Tokens: {total_tokens:,}/{args.target_tokens:,} | TPS: {total_tokens/elapsed:.0f}")

    # Final flush
    flush_buffer(train_buffer, train_path)
    flush_buffer(val_buffer, val_path)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    tps = total_tokens / elapsed_time if elapsed_time > 0 else 0
    
    # Get final file sizes
    train_size_mb = os.path.getsize(train_path) / (1024 * 1024)
    val_size_mb = os.path.getsize(val_path) / (1024 * 1024)
    
    # Compile metadata
    meta = {
        "dataset_name": "FineWeb-Edu sample-10BT",
        "dataset_version": "1.0",
        "tokenizer_name": "tiktoken (gpt2)",
        "vocabulary_size": tokenizer.n_vocab,
        "total_tokens": total_tokens,
        "train_tokens": stats["train_tokens"],
        "validation_tokens": stats["val_tokens"],
        "train_validation_ratio": f"{stats['train_tokens']/total_tokens*100:.2f}% / {stats['val_tokens']/total_tokens*100:.2f}%",
        "context_length": args.context_length,
        "preprocessing_version": "1.0",
        "creation_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "deduplication_enabled": args.dedup,
        "statistics": stats
    }
    
    with open(meta_path, 'wb') as f:
        pickle.dump(meta, f)
        
    print("\n====================================")
    print("PREPROCESSING SUMMARY")
    print("====================================")
    print(f"Documents Streamed   : {stats['docs_streamed']:,}")
    print(f"Documents Accepted   : {stats['docs_accepted']:,}")
    print(f"Rejected (Empty)     : {stats['docs_rejected_empty']:,}")
    print(f"Rejected (Short)     : {stats['docs_rejected_short']:,}")
    if args.dedup:
        print(f"Rejected (Duplicate) : {stats['docs_rejected_duplicates']:,}")
    print("------------------------------------")
    print(f"Total Tokens         : {total_tokens:,}")
    print(f"Training Tokens      : {stats['train_tokens']:,} ({train_size_mb:.2f} MB)")
    print(f"Validation Tokens    : {stats['val_tokens']:,} ({val_size_mb:.2f} MB)")
    print("------------------------------------")
    print(f"Processing Time      : {elapsed_time:.2f} seconds")
    print(f"Tokens Per Second    : {tps:,.0f} tokens/sec")
    print("====================================\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deterministic Streaming Preprocessing Pipeline")
    parser.add_argument("--outdir", type=str, default="data/fineweb_edu", help="Output directory")
    parser.add_argument("--target-tokens", type=int, default=2500000000, help="Target total tokens (e.g., 2.5B for ~5GB)")
    parser.add_argument("--min-chars", type=int, default=100, help="Minimum character length for cleaning")
    parser.add_argument("--dedup", action="store_true", help="Enable SHA-256 document deduplication")
    parser.add_argument("--context-length", type=int, default=1024, help="Target context length for metadata")
    
    args = parser.parse_args()
    main(args)
