import os
from typing import Protocol, Tuple
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class TokenSequence:
    """
    A Data Transfer Object (DTO) container for a sequence of tokens.
    Designed for extensibility: future iterations can add 'attention_mask', 
    'document_ids', or 'loss_mask' without breaking the Protocol signature.
    """
    tokens: torch.Tensor


class TokenStorage(Protocol):
    """
    Interface for dataset storage backends. 
    Guarantees that the Dataset can swap InMemory for Memmap seamlessly.
    """
    def __len__(self) -> int:
        """Returns the total number of valid tokens in storage."""
        ...
        
    def get_sequence(self, index: int, length: int) -> TokenSequence:
        """Retrieves a TokenSequence of the specified length starting at 'index'."""
        ...


class InMemoryStorage:
    """
    Loads the entire preprocessed binary dataset into RAM.
    Excellent for debugging and small datasets.
    """
    def __init__(self, filepath: str):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Dataset file not found: {filepath}")
        
        # Structural Integrity Check (Defense in Depth)
        try:
            # We assume the preprocessing script outputs uint16 for GPT-2 vocab (<65536)
            tokens_np = np.fromfile(filepath, dtype=np.uint16)
            
            # PyTorch's nn.Embedding requires int64 (LongTensor) or int32
            self.data = torch.from_numpy(tokens_np.astype(np.int64))
        except Exception as e:
            raise RuntimeError(f"Failed to load dataset file {filepath}. Corrupted format?") from e
            
        if len(self.data) == 0:
            raise ValueError(f"Dataset file {filepath} is completely empty.")

    def __len__(self) -> int:
        return len(self.data)
        
    def get_sequence(self, index: int, length: int) -> TokenSequence:
        # Bounds checking to prevent silent truncation
        if index < 0 or index + length > len(self.data):
            raise IndexError(
                f"Cannot slice length {length} at index {index}. "
                f"Storage capacity is {len(self.data)}."
            )
        return TokenSequence(tokens=self.data[index : index + length])


class MemmapStorage:
    """
    Memory-maps the preprocessed binary dataset directly from the hard drive.
    Crucial for massive datasets (e.g., 2GB+) that cannot fit in standard RAM.
    """
    def __init__(self, filepath: str):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Dataset file not found: {filepath}")
        
        # Load via numpy memmap (does not load into RAM)
        self.data_np = np.memmap(filepath, dtype=np.uint16, mode='r')
        self._len = len(self.data_np)
        
        if self._len == 0:
            raise ValueError(f"Dataset file {filepath} is completely empty.")

    def __len__(self) -> int:
        return self._len
        
    def get_sequence(self, index: int, length: int) -> TokenSequence:
        if index < 0 or index + length > self._len:
            raise IndexError(
                f"Cannot slice length {length} at index {index}. "
                f"Storage capacity is {self._len}."
            )
        # Slicing a memmap returns a standard numpy array, which we then convert to a tensor
        chunk = self.data_np[index : index + length]
        return TokenSequence(tokens=torch.from_numpy(chunk.astype(np.int64)))


class GPTDataset(Dataset):
    """
    The PyTorch Dataset for Language Modeling.
    Relies on Dependency Inversion: it doesn't know *how* tokens are stored,
    it only queries the TokenStorage interface.
    """
    def __init__(self, storage: TokenStorage, context_length: int):
        self.storage = storage
        self.context_length = context_length
        
        # We need context_length + 1 tokens from storage to create an (x, y) target pair
        if len(self.storage) < self.context_length + 1:
            raise ValueError(
                f"Storage contains {len(self.storage)} tokens, but we need at least "
                f"{self.context_length + 1} to create a single context_length example. "
                f"The dataset is smaller than the model's receptive field."
            )

    def __len__(self) -> int:
        # The number of valid starting indices that can provide a full context_length + 1 chunk
        return len(self.storage) - self.context_length
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Fetch the DTO
        seq = self.storage.get_sequence(idx, self.context_length + 1)
        
        # Contract Enforcement: Ensure the storage didn't break its promise
        if seq.tokens.size(0) != self.context_length + 1:
            raise RuntimeError("TokenStorage returned a sequence of incorrect length.")
            
        # x is the input sequence [0, 1, ..., T-1]
        # y is the target sequence [1, 2, ..., T] shifted by one step into the future
        x = seq.tokens[:-1]
        y = seq.tokens[1:]
        
        return x, y
