from datasets import load_dataset
try:
    dataset = load_dataset("togethercomputer/RedPajama-Data-1T", name="arxiv", split="train", streaming=True)
    for i, doc in enumerate(dataset):
        print(f"Doc {i} keys: {list(doc.keys())}")
        if i >= 1: break
    print("SUCCESS")
except Exception as e:
    print("FAIL:", e)
