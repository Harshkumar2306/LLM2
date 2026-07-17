import os
import torch
from typing import Dict, Any, Tuple, Iterator, List
from torch.utils.data import DataLoader

from data.dataset import MemmapStorage, GPTDataset
from data.tokenizer import Tokenizer

def _infinite_iterator(dataloader: DataLoader) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    """Yields batches forever."""
    while True:
        for batch in dataloader:
            yield batch

class DataManager:
    """
    Subsystem 7: Data Manager
    Encapsulates datasets, DataLoaders, and tokenizer logic.
    Provides an infinite batch generator to the Trainer, hiding all PyTorch DataLoader details.
    """
    def __init__(self, data_dir: str, config: Dict[str, Any]):
        """
        Why it exists: To cleanly separate dataset fetching and iteration from the orchestration loop.
        Design decision: Relies strictly on `MemmapStorage` to support massive out-of-core datasets.
        """
        self.data_dir = data_dir
        self.config = config
        self.tokenizer = Tokenizer()
        
        self.batch_size = self.config.get("batch_size", 12)
        
        self.context_length = self.config.get("context_length", 1024)
        self.num_workers = self.config.get("num_workers", 0)
        self.pin_memory = self.config.get("pin_memory", True)
        self.persistent_workers = self.config.get("persistent_workers", False)
        self.prefetch_factor = self.config.get("prefetch_factor", 2)
        
        self.train_loader = None
        self.val_loader = None
        self._train_iter = None
        self._val_iter = None

    def prepare(self):
        """
        Loads the data from disk and constructs the PyTorch DataLoaders.
        Design decision: This is explicitly called so the manager can be instantiated before disk I/O.
        """
        train_path = os.path.join(self.data_dir, "train.bin")
        val_path = os.path.join(self.data_dir, "val.bin")
        
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Missing train data: {train_path}")
        if not os.path.exists(val_path):
            raise FileNotFoundError(f"Missing val data: {val_path}")
            
        train_storage = MemmapStorage(train_path)
        val_storage = MemmapStorage(val_path)
        
        train_dataset = GPTDataset(train_storage, self.context_length)
        val_dataset = GPTDataset(val_storage, self.context_length)
        
        # Trade-off: If num_workers > 0, we need persistent_workers=True to avoid spawning 
        # overheads for every epoch. However, prefetch_factor requires num_workers > 0.
        kwargs = {
            "batch_size": self.batch_size,
            "pin_memory": self.pin_memory,
            "num_workers": self.num_workers,
        }
        if self.num_workers > 0:
            kwargs["persistent_workers"] = self.persistent_workers
            kwargs["prefetch_factor"] = self.prefetch_factor

        self.train_loader = DataLoader(train_dataset, shuffle=True, **kwargs)
        self.val_loader = DataLoader(val_dataset, shuffle=False, **kwargs)
        
        self._train_iter = _infinite_iterator(self.train_loader)
        self._val_iter = _infinite_iterator(self.val_loader)

    def get_train_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the next training batch (x, y).
        Hides DataLoader orchestration from the Trainer.
        """
        if self._train_iter is None:
            raise RuntimeError("DataManager.prepare() must be called before fetching batches.")
        return next(self._train_iter)

    def get_val_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the next validation batch (x, y).
        """
        if self._val_iter is None:
            raise RuntimeError("DataManager.prepare() must be called before fetching batches.")
        return next(self._val_iter)

    def num_train_tokens(self) -> int:
        """Returns the total number of tokens in the training set."""
        if self.train_loader is None:
            return 0
        return len(self.train_loader.dataset)

    def num_val_tokens(self) -> int:
        """Returns the total number of tokens in the validation set."""
        if self.val_loader is None:
            return 0
        return len(self.val_loader.dataset)

    def get_context_length(self) -> int:
        """Returns the sequence context length."""
        return self.context_length

    def dataset_name(self) -> str:
        """Returns the name/path of the dataset."""
        return self.data_dir

    def vocab_size(self) -> int:
        """Returns the total vocabulary size of the tokenizer."""
        return self.tokenizer.n_vocab

    def encode(self, text: str) -> List[int]:
        """Encodes text to a list of token IDs."""
        return self.tokenizer.encode(text)

    def decode(self, tokens: List[int]) -> str:
        """Decodes a list of token IDs to text."""
        return self.tokenizer.decode(tokens)
