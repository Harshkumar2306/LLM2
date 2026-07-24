"""
Downloads the TinyShakespeare dataset for the Kaggle baseline phase.
Does NOT perform any tokenization or preprocessing.
"""
import os
import urllib.request
import argparse

def download_tinyshakespeare(output_path: str):
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    print(f"Downloading TinyShakespeare from {url}...")
    
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    urllib.request.urlretrieve(url, output_path)
    
    file_size = os.path.getsize(output_path)
    print(f"Downloaded {file_size / 1024 / 1024:.2f} MB to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download TinyShakespeare")
    parser.add_argument("--output", type=str, default="data/tinyshakespeare.txt", help="Output file path")
    args = parser.parse_args()
    
    download_tinyshakespeare(args.output)
