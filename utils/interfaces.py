"""
Abstract Base Classes defining the structural contracts for systems engineering,
including Configuration Loading and Experiment Tracking.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict

class BaseConfigLoader(ABC):
    """
    Contract for Configuration Parsers.
    Invariants:
    - Must be capable of parsing a structured file (YAML/JSON) into a Python dictionary.
    - Must strictly validate against the architecture Enums before returning.
    """
    @abstractmethod
    def load(self, filepath: str) -> Dict[str, Any]:
        """
        Parses the configuration file and returns a validated dictionary.
        Args:
            filepath: Path to the configuration file.
        Returns:
            Dict representing the validated hyperparameters.
        """
        pass


class BaseExperiment(ABC):
    """
    Contract for the Local Experiment Manager.
    Invariants:
    - Must enforce reproducibility by saving config and git state.
    - Must provide structured logging mechanisms (e.g. JSONL).
    """
    @abstractmethod
    def log_metric(self, step: int, key: str, value: Any) -> None:
        """
        Logs a single scalar metric.
        """
        pass

    @abstractmethod
    def save_checkpoint(self, state_dict: Dict[str, Any], is_best: bool = False) -> None:
        """
        Saves the model weights to the experiment directory.
        """
        pass

    @abstractmethod
    def finish(self) -> None:
        """
        Finalizes the experiment (e.g., closing file handles, logging total time).
        """
        pass
