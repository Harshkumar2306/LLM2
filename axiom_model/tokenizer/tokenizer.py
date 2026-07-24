import tiktoken
from typing import List, Union, Set

BASE_VOCAB_SIZE = tiktoken.get_encoding("gpt2").n_vocab
SPECIAL_TOKENS = {
    "<|system|>": BASE_VOCAB_SIZE,
    "<|user|>": BASE_VOCAB_SIZE + 1,
    "<|assistant|>": BASE_VOCAB_SIZE + 2,
    "<|end|>": BASE_VOCAB_SIZE + 3,
}

DEFAULT_SYSTEM_PROMPT = "You are Axiom, a helpful AI assistant."

class Tokenizer:
    """
    A wrapper around the underlying tokenizer (currently tiktoken).
    This interface ensures that if we build a custom BPE tokenizer later,
    the rest of the codebase will not need to change.
    """
    def __init__(self, encoding_name: str = "gpt2", special_tokens: dict = None):
        try:
            base_enc = tiktoken.get_encoding(encoding_name)
            if special_tokens is not None:
                enc = tiktoken.Encoding(
                    name=f"{encoding_name}_custom",
                    pat_str=base_enc._pat_str,
                    mergeable_ranks=base_enc._mergeable_ranks,
                    special_tokens={**base_enc._special_tokens, **special_tokens}
                )
                self._tokenizer = enc
            else:
                self._tokenizer = base_enc
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
