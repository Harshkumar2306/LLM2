import os
import yaml
import torch
import torch.distributed as dist
from typing import Dict, Any, Tuple, Iterator, List
from torch.utils.data import DataLoader, ConcatDataset

from data.dataset import MemmapStorage, GPTDataset
from tokenizer.tokenizer import Tokenizer

def _infinite_iterator(dataloader: DataLoader, sampler=None) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    """Yields batches forever."""
    epoch = 0
    while True:
        if sampler is not None and hasattr(sampler, 'set_epoch'):
            sampler.set_epoch(epoch)
        for batch in dataloader:
            yield batch
        epoch += 1

class WeightedDistributedSampler(torch.utils.data.Sampler):
    """
    Samples from multiple datasets based on predefined weights.
    Supports Distributed Data Parallel (DDP).
    """
    def __init__(self, dataset_lengths: List[int], weights: List[float], num_replicas=1, rank=0, seed=0):
        self.dataset_lengths = dataset_lengths
        self.weights = torch.tensor(weights, dtype=torch.float32)
        
        # Validate weights
        assert torch.all(self.weights > 0), "All weights must be > 0"
        self.weights = self.weights / self.weights.sum() # Normalize
        
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.seed = seed
        self.total_size = sum(dataset_lengths)
        
        # Precompute starting offsets for each dataset
        self.offsets = [0]
        for length in dataset_lengths[:-1]:
            self.offsets.append(self.offsets[-1] + length)
            
    def set_epoch(self, epoch):
        self.epoch = epoch
        
    def __iter__(self):
        # We yield indices forever.
        while True:
            # 1. Choose a dataset based on weight
            dataset_idx = torch.multinomial(self.weights, 1).item()
            
            # 2. Pick a random index WITHIN that dataset
            length = self.dataset_lengths[dataset_idx]
            offset = self.offsets[dataset_idx]
            
            local_idx = torch.randint(0, length, (1,)).item()
            
            # 3. Return the absolute global index for the ConcatDataset
            yield int(offset + local_idx)
            
    def __len__(self):
        return self.total_size // self.num_replicas


class WeightedDatasetManager:
    """
    Reads weights.yaml, validates paths, and loads datasets.
    """
    def __init__(self, weights_path: str, context_length: int):
        self.weights_path = weights_path
        self.context_length = context_length
        self.datasets_info = []
        
        self._load_and_validate()
        
    def _load_and_validate(self):
        if not os.path.exists(self.weights_path):
            raise FileNotFoundError(f"Missing configuration: {self.weights_path}")
            
        with open(self.weights_path, 'r') as f:
            cfg = yaml.safe_load(f)
            
        datasets = cfg.get('datasets', {})
        if not datasets:
            raise ValueError(f"No datasets defined in {self.weights_path}")
            
        total_weight = 0.0
        seen_paths = set()
        
        for name, info in datasets.items():
            path = info.get('path')
            weight = info.get('weight', 0.0)
            
            # Validation: Path exists
            train_bin = os.path.join(path, "train.bin")
            val_bin = os.path.join(path, "val.bin")
            
            if not os.path.exists(train_bin):
                raise FileNotFoundError(f"Missing train dataset: {train_bin}")
            if not os.path.exists(val_bin):
                raise FileNotFoundError(f"Missing val dataset: {val_bin}")
                
            # Validation: Weight > 0
            if weight <= 0:
                raise ValueError(f"Dataset {name} must have weight > 0")
                
            # Validation: No duplicates
            if path in seen_paths:
                raise ValueError(f"Duplicate dataset path: {path}")
            seen_paths.add(path)
            
            # Validation: Not empty
            if os.path.getsize(train_bin) == 0:
                raise ValueError(f"Dataset {name} is empty: {train_bin}")
                
            total_weight += weight
            self.datasets_info.append({
                'name': name,
                'train_path': train_bin,
                'val_path': val_bin,
                'weight': weight
            })
            
        # Normalize weights if they don't exactly sum to 1.0 (float imprecision)
        for info in self.datasets_info:
            info['normalized_weight'] = info['weight'] / total_weight
            
    def get_datasets_and_weights(self) -> Tuple[ConcatDataset, ConcatDataset, List[int], List[float]]:
        train_datasets = []
        val_datasets = []
        lengths = []
        weights = []
        
        for info in self.datasets_info:
            train_storage = MemmapStorage(info['train_path'])
            val_storage = MemmapStorage(info['val_path'])
            
            train_ds = GPTDataset(train_storage, self.context_length)
            val_ds = GPTDataset(val_storage, self.context_length)
            
            train_datasets.append(train_ds)
            val_datasets.append(val_ds)
            
            lengths.append(len(train_ds))
            weights.append(info['normalized_weight'])
            
        # We concatenate them. The Sampler will yield absolute indices covering this concatenated space.
        return ConcatDataset(train_datasets), ConcatDataset(val_datasets), lengths, weights


class DataManager:
    """
    Subsystem 7: Data Manager
    Encapsulates datasets, DataLoaders, and tokenizer logic.
    Provides an infinite batch generator to the Trainer, hiding all PyTorch DataLoader details.
    """
    def __init__(self, data_dir: str, config: Dict[str, Any]):
        """
        Uses WeightedDatasetManager to support modular datasets.
        """
        # We assume data_dir points to the weights.yaml directly or the directory containing it
        if os.path.isdir(data_dir):
            self.weights_path = os.path.join(data_dir, "weights.yaml")
        else:
            self.weights_path = data_dir
            
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
        """
        wdm = WeightedDatasetManager(self.weights_path, self.context_length)
        train_dataset, val_dataset, lengths, weights = wdm.get_datasets_and_weights()
        
        kwargs = {
            "batch_size": self.batch_size,
            "pin_memory": self.pin_memory,
            "num_workers": self.num_workers,
        }
        if self.num_workers > 0:
            kwargs["persistent_workers"] = self.persistent_workers
            kwargs["prefetch_factor"] = self.prefetch_factor

        is_ddp = dist.is_initialized()
        world_size = dist.get_world_size() if is_ddp else 1
        rank = dist.get_rank() if is_ddp else 0
        
        # Training is randomly sampled using the weights
        train_sampler = WeightedDistributedSampler(lengths, weights, world_size, rank) if is_ddp else WeightedDistributedSampler(lengths, weights, 1, 0)
        
        # Validation is deterministic and sequential (we evaluate everything)
        # We use standard DistributedSampler to chunk the validation set across GPUs
        val_sampler = torch.utils.data.DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False) if is_ddp else None

        self.train_loader = DataLoader(
            train_dataset, 
            shuffle=False, # Handled by sampler
            sampler=train_sampler, 
            **kwargs
        )
        
        self.val_loader = DataLoader(
            val_dataset, 
            shuffle=False, 
            sampler=val_sampler, 
            **kwargs
        )
        
        self.train_sampler = train_sampler
        self.val_sampler = val_sampler
        
        self._train_iter = _infinite_iterator(self.train_loader, self.train_sampler)
        self._val_iter = _infinite_iterator(self.val_loader, self.val_sampler)

    def get_train_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._train_iter is None:
            raise RuntimeError("DataManager.prepare() must be called before fetching batches.")
        return next(self._train_iter)

    def get_val_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._val_iter is None:
            raise RuntimeError("DataManager.prepare() must be called before fetching batches.")
        return next(self._val_iter)

    def num_train_tokens(self) -> int:
        if self.train_loader is None:
            return 0
        return len(self.train_loader.dataset) * self.context_length

    def num_val_tokens(self) -> int:
        if self.val_loader is None:
            return 0
        return len(self.val_loader.dataset) * self.context_length

    def get_context_length(self) -> int:
        return self.context_length

    def dataset_name(self) -> str:
        return self.weights_path

    def vocab_size(self) -> int:
        return self.tokenizer.n_vocab

    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text)

    def decode(self, tokens: List[int]) -> str:
        return self.tokenizer.decode(tokens)
