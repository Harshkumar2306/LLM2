import os
import json
import argparse
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader

def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
    return text

def chunk_text(text, chunk_size=500, overlap=50):
    # Simple word-based chunking
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--docs_dir', type=str, required=True, help='Directory containing text or PDF files')
    parser.add_argument('--out_dir', type=str, default='experiments/rag_db', help='Output directory for the FAISS index')
    parser.add_argument('--model', type=str, default='all-MiniLM-L6-v2', help='SentenceTransformer model name')
    parser.add_argument('--chunk_size', type=int, default=500, help='Number of words per chunk')
    parser.add_argument('--overlap', type=int, default=50, help='Number of overlapping words between chunks')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    
    print(f"Loading embedding model: {args.model}...")
    embedder = SentenceTransformer(args.model)
    
    all_chunks = []
    
    print(f"Scanning directory: {args.docs_dir}...")
    for root, dirs, files in os.walk(args.docs_dir):
        for file in files:
            file_path = os.path.join(root, file)
            text = ""
            if file.endswith(".txt") or file.endswith(".md"):
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
            elif file.endswith(".pdf"):
                text = extract_text_from_pdf(file_path)
            else:
                continue
                
            if text.strip():
                chunks = chunk_text(text, chunk_size=args.chunk_size, overlap=args.overlap)
                for chunk in chunks:
                    if len(chunk.strip()) > 20: # Ignore tiny chunks
                        all_chunks.append({
                            "source": file,
                            "text": chunk
                        })
                        
    if not all_chunks:
        print("No valid text found in the provided directory!")
        return
        
    print(f"Extracted {len(all_chunks)} chunks. Generating embeddings...")
    texts = [c["text"] for c in all_chunks]
    embeddings = embedder.encode(texts, show_progress_bar=True)
    
    # Normalize embeddings for cosine similarity
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    print("Building FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension) # Inner product (cosine similarity since normalized)
    index.add(embeddings.astype(np.float32))
    
    index_path = os.path.join(args.out_dir, "vector.index")
    meta_path = os.path.join(args.out_dir, "meta.json")
    
    faiss.write_index(index, index_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2)
        
    print(f"Successfully saved FAISS index and metadata to {args.out_dir}")

if __name__ == '__main__':
    main()
