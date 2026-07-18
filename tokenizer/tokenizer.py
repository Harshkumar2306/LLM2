import tiktoken
from typing import List, Union, Set


class Tokenizer:
    """
    A wrapper around the underlying tokenizer (currently tiktoken).
    This interface ensures that if we build a custom BPE tokenizer later,
    the rest of the codebase will not need to change.
    """
    def __init__(self, encoding_name: str = "gpt2"):
        try:
            self._tokenizer = tiktoken.get_encoding(encoding_name)
        except Exception as e:
            raise RuntimeError(f"Failed to load tiktoken encoding '{encoding_name}'.") from e
            
        self.n_vocab = self._tokenizer.n_vocab
        # The End-Of-Text token used to separate distinct documents
        self.eot_token = self._tokenizer.eot_token

    def encode(self, text: str, allowed_special: Union[Set[str], str] = "all") -> List[int]:
        """
        Converts a raw text string into a list of integer token IDs.
        """
        if not text:
            return []
        return self._tokenizer.encode(text, allowed_special=allowed_special)

    def decode(self, tokens: List[int]) -> str:
        """
        Converts a list of integer token IDs back into a raw text string.
        """
        if not tokens:
            return ""
        return self._tokenizer.decode(tokens)
