from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class TrainingState:
    """Encapsulates all mutable runtime variables for an experiment."""
    iteration: int = 0
    epoch: int = 0
    global_step: int = 0
    best_val_loss: float = float('inf')
    current_learning_rate: float = 0.0
    gradient_norm: float = 0.0
    tokens_processed: int = 0
    elapsed_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the training state for checkpointing."""
        return {
            "iteration": self.iteration,
            "epoch": self.epoch,
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
            "current_learning_rate": self.current_learning_rate,
            "gradient_norm": self.gradient_norm,
            "tokens_processed": self.tokens_processed,
            "elapsed_time": self.elapsed_time
        }
        
    def load_dict(self, state_dict: Dict[str, Any]):
        """Restores the training state from a checkpoint."""
        self.iteration = state_dict.get("iteration", 0)
        self.epoch = state_dict.get("epoch", 0)
        self.global_step = state_dict.get("global_step", 0)
        self.best_val_loss = state_dict.get("best_val_loss", float('inf'))
        self.current_learning_rate = state_dict.get("current_learning_rate", 0.0)
        self.gradient_norm = state_dict.get("gradient_norm", 0.0)
        self.tokens_processed = state_dict.get("tokens_processed", 0)
        self.elapsed_time = state_dict.get("elapsed_time", 0.0)
