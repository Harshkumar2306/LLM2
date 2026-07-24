from typing import List, Dict, Any
from duckduckgo_search import DDGS
from .base import Retriever

class WebRetriever(Retriever):
    def __init__(self, top_k: int = 3, reddit_only: bool = False):
        self.top_k = top_k
        self.reddit_only = reddit_only
        self.ddgs = DDGS()
        
    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        search_query = query
        if self.reddit_only:
            search_query += " site:reddit.com"
            
        try:
            results = self.ddgs.text(search_query, max_results=self.top_k)
            if not results:
                return []
                
            formatted_results = []
            for i, res in enumerate(results):
                formatted_results.append({
                    "source": res.get('href', 'Web'),
                    "text": res.get('body', ''),
                    "score": 1.0 - (i * 0.1) # Fake score for ranking (1.0, 0.9, 0.8...)
                })
            return formatted_results
        except Exception as e:
            print(f"[WebRetriever Error] {e}")
            return []
