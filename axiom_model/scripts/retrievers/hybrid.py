from typing import List, Dict, Any
from .base import Retriever

class HybridRetriever(Retriever):
    def __init__(self, local_retriever: Retriever, web_retriever: Retriever, top_k: int = 4):
        self.local_retriever = local_retriever
        self.web_retriever = web_retriever
        self.top_k = top_k
        
    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        local_results = []
        web_results = []
        
        try:
            local_results = self.local_retriever.retrieve(query)
        except Exception as e:
            print(f"[Hybrid] Local retrieval failed: {e}")
            
        try:
            web_results = self.web_retriever.retrieve(query)
        except Exception as e:
            print(f"[Hybrid] Web retrieval failed: {e}")
            
        # Simple merging strategy: alternate picking from local and web until top_k is reached
        merged_results = []
        max_len = max(len(local_results), len(web_results))
        
        for i in range(max_len):
            if i < len(local_results):
                merged_results.append(local_results[i])
            if i < len(web_results):
                merged_results.append(web_results[i])
                
            if len(merged_results) >= self.top_k:
                break
                
        return merged_results[:self.top_k]
