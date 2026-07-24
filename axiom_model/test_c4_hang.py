import time
from datasets import load_dataset

print("Starting to load c4-en-html-with-metadata...")
start_time = time.time()
try:
    dataset = load_dataset("bs-modeling-metadata/c4-en-html-with-metadata", split="train", streaming=True, trust_remote_code=True)
    
    # Force it to resolve the first document
    iterator = iter(dataset)
    print(f"Dataset object created in {time.time() - start_time:.2f} seconds. Attempting to fetch first row...")
    
    fetch_start = time.time()
    doc = next(iterator)
    print(f"First row fetched in {time.time() - fetch_start:.2f} seconds!")
    print(f"Keys: {list(doc.keys())}")
except Exception as e:
    print(f"FAILED: {e}")
