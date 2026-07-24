from abc import ABC, abstractmethod
from typing import List, Dict, Any

class Retriever(ABC):
    """
    Abstract base class for all retrievers.
    """
    
    @abstractmethod
    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a given query.
        
        Args:
            query (str): The search query.
            
        Returns:
            List[Dict[str, Any]]: A list of dictionaries representing retrieved chunks.
            Each dictionary should minimally contain:
            {
                "source": str,  # URL or filename
                "text": str,    # The text content
                "score": float  # Relevance score (higher is better)
            }
        """
        pass
