from datasets import load_dataset
import yaml

with open('data/datasets.yaml', 'r') as f:
    cfg = yaml.safe_load(f)

for ds_config in cfg.get('datasets', []):
    name = ds_config['name']
    hf_path = ds_config['hf']
    subset = ds_config.get('subset', 'default')
    split = ds_config.get('split', 'train')
    
    print(f"Testing {name}: {hf_path} ({subset})")
    
    try:
        kwargs = {'name': subset} if subset != 'default' else {}
        dataset = load_dataset(hf_path, split=split, streaming=True, trust_remote_code=True, **kwargs)
        
        for i, doc in enumerate(dataset):
            text = doc.get("text", doc.get("content", doc.get("document", doc.get("code", ""))))
            if not text:
                print(f"  [ERROR] No text found! Keys available: {list(doc.keys())}")
            else:
                print(f"  [SUCCESS] Found text! Length: {len(text)}")
            break
    except Exception as e:
        print(f"  [FATAL ERROR] {e}")
