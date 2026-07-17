import torch
import torch.nn as nn
from typing import Dict, Any, Callable
from engine.device_manager import DeviceManager

class ValidationManager:
    """
    Dedicated manager for evaluating the model.
    Responsibilities: Switch to eval mode, run no_grad, manage autocast, 
    compute structured metrics, and restore train mode.
    """
    
    def __init__(self, device_manager: DeviceManager):
        self.device_manager = device_manager

    def evaluate(
        self, 
        model: nn.Module, 
        eval_iters: int, 
        batch_fetcher: Callable[[], tuple[torch.Tensor, torch.Tensor]]
    ) -> Dict[str, float]:
        """
        Runs validation and returns structured metrics.
        Guarantees that the model's training mode is restored exactly as it was.
        """
        was_training = model.training
        model.eval()
        
        metrics = {}
        total_loss = 0.0
        
        # Explicit no_grad prevents gradient allocation during evaluation
        with torch.no_grad():
            for _ in range(eval_iters):
                x, y = batch_fetcher()
                x, y = self.device_manager.to_device(x, y)
                
                # Mixed precision inference
                with self.device_manager.autocast():
                    _, loss = model(x, targets=y)
                
                total_loss += loss.item()
                
        metrics["val_loss"] = total_loss / eval_iters
        
        # Future extension points (e.g. perplexity, accuracy, BLEU) can be added here
        # metrics["perplexity"] = math.exp(metrics["val_loss"])
        
        if was_training:
            model.train()
            
        return metrics
