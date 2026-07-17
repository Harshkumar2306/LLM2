"""
Offline preprocessing script for GPT-style Large Language Models.

This script separates the heavy lifting of string tokenization from the PyTorch DataLoader.
It reads a raw text corpus, tokenizes it into integers, splits it into train/val sets,
and saves them as flat binary files.
"""
import os
import argparse
import sys
import numpy as np

# Ensure we can import from the parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.tokenizer import Tokenizer


def prepare_data(input_file: str, output_dir: str, val_ratio: float = 0.1) -> None:
    print(f"Loading corpus from {input_file}...")
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Cannot find input file: {input_file}")
        
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()
        
    # The Preprocessing Layer: Catching empty documents
    if not text.strip():
        raise ValueError(f"Input file {input_file} is empty or contains only whitespace.")
        
    print(f"Tokenizing {len(text)} characters...")
    tokenizer = Tokenizer()
    tokens = tokenizer.encode(text)
    
    total_tokens = len(tokens)
    print(f"Total tokens after compression: {total_tokens:,}")
    
    # Catching documents shorter than our expected context length
    # Note: We hardcode a generic warning here. In a production pipeline, this 
    # would read the GPTConfig.context_length to warn appropriately.
    if total_tokens < 1024:
        print("WARNING: Dataset contains fewer than 1024 tokens. "
              "This may be too small for the model's context length.")
    
    # Splitting logic
    val_size = int(total_tokens * val_ratio)
    train_tokens = tokens[:-val_size] if val_size > 0 else tokens
    val_tokens = tokens[-val_size:] if val_size > 0 else []
    
    # Structural definition:
    # GPT-2 has a vocab size of 50,257. This perfectly fits inside an unsigned 16-bit integer (max 65,535).
    # Saving as uint16 cuts disk space in half compared to int32, and by 4x compared to int64.
    train_ids = np.array(train_tokens, dtype=np.uint16)
    
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, 'train.bin')
    train_ids.tofile(train_path)
    print(f"Saved {len(train_ids):,} training tokens to {train_path}")
    
    if val_size > 0:
        val_ids = np.array(val_tokens, dtype=np.uint16)
        val_path = os.path.join(output_dir, 'val.bin')
        val_ids.tofile(val_path)
        print(f"Saved {len(val_ids):,} validation tokens to {val_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess text data into binary tokens.")
    parser.add_argument("--input", type=str, required=True, help="Path to raw text file (e.g. dataset.txt).")
    parser.add_argument("--outdir", type=str, required=True, help="Directory to save train.bin and val.bin.")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio (0.0 to 1.0).")
    args = parser.parse_args()
    
    prepare_data(args.input, args.outdir, args.val_ratio)
