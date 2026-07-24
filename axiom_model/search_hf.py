import requests

def search_datasets(query):
    url = f"https://huggingface.co/api/datasets?search={query}&limit=20&full=True"
    r = requests.get(url)
    for ds in r.json():
        if ds.get('gated', False):
            continue
        print(f"Name: {ds['id']}, Size: {ds.get('size_categories', [])}, Tags: {ds.get('tags', [])}")

search_datasets("starcoder")
search_datasets("github")
