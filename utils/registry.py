"""
Component Registry for the Axiom framework.
Decouples the instantiation of architectural components from the model structure.
"""

from typing import Dict, Type, Any, Callable
import torch.nn as nn
from enum import Enum

class Registry:
    """
    A strict registry system that maps architectural Enums to concrete PyTorch Module classes.
    """
    def __init__(self, name: str):
        self.name = name
        self._registry: Dict[Enum, Type[nn.Module]] = {}

    def register(self, enum_key: Enum) -> Callable:
        """
        Decorator to register a class to a specific Enum key.
        Args:
            enum_key: The Enum value representing this architecture.
        """
        def wrapper(cls: Type[nn.Module]) -> Type[nn.Module]:
            if not isinstance(enum_key, Enum):
                raise TypeError(f"Registry keys must be Enums. Got {type(enum_key)}")
            if enum_key in self._registry:
                raise ValueError(f"Key {enum_key} is already registered in {self.name} registry.")
            
            self._registry[enum_key] = cls
            return cls
        return wrapper

    def build(self, enum_key: Enum, *args, **kwargs) -> nn.Module:
        """
        Instantiates the registered class.
        Args:
            enum_key: The Enum value mapping to the desired architecture.
            *args, **kwargs: Arguments passed to the class constructor (e.g., config).
        Returns:
            An instantiated PyTorch Module.
        """
        if enum_key not in self._registry:
            raise KeyError(
                f"Architecture {enum_key} not found in {self.name} registry. "
                f"Available keys: {list(self._registry.keys())}"
            )
        
        cls = self._registry[enum_key]
        return cls(*args, **kwargs)


# Global Registries for Axiom Components
FFN_REGISTRY = Registry("FFN")
ATTENTION_REGISTRY = Registry("Attention")
POSITION_REGISTRY = Registry("PositionStrategy")
NORM_REGISTRY = Registry("Normalization")
