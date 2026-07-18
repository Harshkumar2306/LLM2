import math
import torch
from typing import Dict, Any

from trainer.training_state import TrainingState

class LearningRateScheduler:
    """
    Subsystem: Learning Rate Scheduler
    Implements Linear Warmup and Cosine Decay for the GPT training loop.
    Decouples scheduling logic from the Trainer.
    """
    def __init__(self, optimizer: torch.optim.Optimizer, config: Dict[str, Any]):
        self.optimizer = optimizer
        self.config = config
        
        self.warmup_iters = self.config.get("warmup_iters", 0)
        self.max_iters = self.config.get("max_iters", 10000)
        self.lr_decay_iters = self.config.get("lr_decay_iters", self.max_iters)
        
        self.base_lr = self.config.get("learning_rate", 3e-4)
        self.min_lr = self.config.get("min_lr", 3e-5)
        
        # We track the internal step to decouple from potentially changing TrainingState definitions,
        # but usually we sync it with state.iteration in step().
        self.current_step = 0

    def step(self, state: TrainingState) -> None:
        """
        Calculates and applies the learning rate for the current iteration.
        Updates the current_learning_rate in the TrainingState.
        """
        self.current_step = state.iteration
        lr = self._compute_lr(self.current_step)
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            
        state.current_learning_rate = lr

    def _compute_lr(self, it: int) -> float:
        """
        Mathematical computation of the learning rate.
        1. Linear warmup
        2. Cosine decay down to min_lr
        3. Constant min_lr thereafter
        """
        # 1. Linear warmup
        if it < self.warmup_iters:
            if self.warmup_iters == 0:
                return self.base_lr
            return self.base_lr * (it + 1) / self.warmup_iters
            
        # 3. Constant min_lr after decay finishes
        if it > self.lr_decay_iters:
            return self.min_lr
            
        # 2. Cosine decay
        decay_ratio = (it - self.warmup_iters) / (self.lr_decay_iters - self.warmup_iters)
        # assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges from 1 to 0
        return self.min_lr + coeff * (self.base_lr - self.min_lr)

    def state_dict(self) -> Dict[str, Any]:
        """Returns the state for checkpointing."""
        return {
            "current_step": self.current_step
        }
        
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Restores the state from a checkpoint."""
        self.current_step = state_dict.get("current_step", 0)
