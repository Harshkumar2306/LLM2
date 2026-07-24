from datasets import load_dataset
dataset = load_dataset("vikp/starcoder_cleaned", split="train", streaming=True)
for i, doc in enumerate(dataset):
    print(f"Doc {i} keys: {list(doc.keys())}")
    break
