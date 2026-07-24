import os
import json
import faiss
import numpy as np
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from .base import Retriever

class LocalRetriever(Retriever):
    def __init__(self, db_dir: str = 'experiments/rag_db', 
                 embedding_model: str = 'all-MiniLM-L6-v2', 
                 top_k: int = 2, 
                 min_similarity: float = 0.3):
        self.db_dir = db_dir
        self.top_k = top_k
        self.min_similarity = min_similarity
        
        index_path = os.path.join(db_dir, "vector.index")
        meta_path = os.path.join(db_dir, "meta.json")
        
        if not os.path.exists(index_path) or not os.path.exists(meta_path):
            raise FileNotFoundError(f"RAG database not found at {db_dir}. Please run build_vector_db.py first!")
            
        self.faiss_index = faiss.read_index(index_path)
        with open(meta_path, "r") as f:
            self.rag_meta = json.load(f)
            
        self.embedder = SentenceTransformer(embedding_model)
        
    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        query_vec = self.embedder.encode([query])
        query_vec = query_vec / np.linalg.norm(query_vec, axis=1, keepdims=True)
        
        D, I = self.faiss_index.search(query_vec.astype(np.float32), k=self.top_k)
        
        results = []
        for distance, idx in zip(D[0], I[0]):
            if idx != -1 and idx < len(self.rag_meta):
                if distance >= self.min_similarity:
                    results.append({
                        "source": self.rag_meta[idx].get('source', 'Unknown'),
                        "text": self.rag_meta[idx]['text'],
                        "score": float(distance)
                    })
        return results
