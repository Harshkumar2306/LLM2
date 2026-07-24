from datasets import load_dataset
try:
    dataset = load_dataset("Open-Orca/OpenOrca", split="train", streaming=True)
    iterator = iter(dataset)
    for i in range(2):
        doc = next(iterator)
        print(f"Doc {i} keys: {list(doc.keys())}")
    print("SUCCESS")
except Exception as e:
    print("FAIL:", e)
