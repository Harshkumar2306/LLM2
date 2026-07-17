import time
import torch
import torch.nn as nn
from typing import Callable, Optional, Tuple, Dict, Any

from engine.training_state import TrainingState
from engine.device_manager import DeviceManager
from engine.checkpoint_manager import CheckpointManager
from engine.experiment_manager import ExperimentManager
from engine.validation_manager import ValidationManager
from utils.reproducibility import ReproducibilityEngine

class Trainer:
    """
    The Orchestrator. 
    Strictly coordinates the specialized managers and orchestrates the event-driven training loop.
    Does NOT contain file I/O, device placement logic, or ad-hoc validation loops.
    """
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        device_manager: DeviceManager,
        checkpoint_manager: CheckpointManager,
        experiment_manager: ExperimentManager,
        validation_manager: ValidationManager,
        config: Dict[str, Any]
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        
        self.device_manager = device_manager
        self.checkpoint_manager = checkpoint_manager
        self.experiment_manager = experiment_manager
        self.validation_manager = validation_manager
        self.config = config
        
        self.state = TrainingState()
        self.scaler = self.device_manager.get_grad_scaler()
        
        # Extract training configurations (from flat config)
        self.max_iters = self.config.get("max_iters", 1000)
        self.eval_interval = self.config.get("eval_interval", 100)
        self.eval_iters = self.config.get("eval_iters", 10)
        self.save_interval = self.config.get("save_interval", 100)
        self.log_interval = self.config.get("log_interval", 10)
        self.grad_clip = self.config.get("grad_clip", 1.0)
        
        self.resume_mode = self.config.get("resume_mode", "none")
        if self.resume_mode != "none":
            self._resume_training()

    def _resume_training(self):
        """Coordinates resume logic via CheckpointManager."""
        try:
            checkpoint = self.checkpoint_manager.load(mode=self.resume_mode)
            
            # Restore model and optimizer
            self.model.load_state_dict(checkpoint["model_state"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
            
            # Restore scheduler
            if self.scheduler and checkpoint["scheduler_state"]:
                self.scheduler.load_state_dict(checkpoint["scheduler_state"])
                
            # Restore scaler
            if self.scaler and checkpoint["grad_scaler_state"]:
                self.scaler.load_state_dict(checkpoint["grad_scaler_state"])
                
            # Restore RNG
            if checkpoint["rng_states"]:
                ReproducibilityEngine.restore_rng_states(checkpoint["rng_states"])
                
            # Restore Training State
            if "training_progress" in checkpoint:
                self.state.load_dict(checkpoint["training_progress"])
                
            print(f"[Trainer] Resumed from iteration {self.state.iteration} with best val loss {self.state.best_val_loss}")
        except Exception as e:
            print(f"[Trainer] Could not resume: {e}. Starting fresh.")

    # =========================================================================
    # EVENT HOOKS
    # =========================================================================

    def before_train(self):
        self.model.train()
        self.start_time = time.time()

    def before_step(self):
        self.step_start_time = time.time()

    def after_step(self):
        # Update timings
        elapsed = time.time() - self.step_start_time
        self.state.elapsed_time += elapsed
        
        # Calculate rates
        # (Assuming batch_size and block_size could be passed in to compute tokens_per_second if needed)
        tokens_per_sec = 0 # Placeholder for now

        # 1. Logging Trigger
        if self.state.iteration > 0 and self.state.iteration % self.log_interval == 0:
            metrics = {
                "iteration": self.state.iteration,
                "train_loss": getattr(self, "current_loss", None),
                "learning_rate": self.state.current_learning_rate,
                "gradient_norm": self.state.gradient_norm,
                "elapsed_time": self.state.elapsed_time,
                "best_val_loss": self.state.best_val_loss
            }
            self.experiment_manager.log_metrics(metrics)
            
        # 2. Validation Trigger
        if self.state.iteration > 0 and self.state.iteration % self.eval_interval == 0:
            self._trigger_validation()
            
        # 3. Checkpoint Trigger
        if self.state.iteration > 0 and self.state.iteration % self.save_interval == 0:
            self._trigger_checkpoint(is_best=False)

    def after_train(self):
        # Ensure final state is saved
        self._trigger_checkpoint(is_best=False)
        self.experiment_manager.generate_summary()

    # =========================================================================
    # ACTIONS
    # =========================================================================

    def _trigger_validation(self):
        metrics = self.validation_manager.evaluate(
            self.model, 
            self.eval_iters, 
            self.val_batch_fetcher
        )
        
        val_loss = metrics.get("val_loss", float('inf'))
        
        is_best = val_loss < self.state.best_val_loss
        if is_best:
            self.state.best_val_loss = val_loss
            self._trigger_checkpoint(is_best=True)
            
        # Log validation metrics
        val_record = {
            "iteration": self.state.iteration,
            "val_loss": val_loss,
            "elapsed_time": self.state.elapsed_time,
            "best_val_loss": self.state.best_val_loss
        }
        self.experiment_manager.log_metrics(val_record)

    def _trigger_checkpoint(self, is_best: bool):
        scaler_state = self.scaler.state_dict() if self.scaler else None
        scheduler_state = self.scheduler.state_dict() if self.scheduler else None
        
        self.checkpoint_manager.save(
            model_state=self.model.state_dict(),
            optimizer_state=self.optimizer.state_dict(),
            scheduler_state=scheduler_state,
            grad_scaler_state=scaler_state,
            rng_states=ReproducibilityEngine.capture_rng_states(),
            iteration=self.state.iteration,
            metrics={"best_val_loss": self.state.best_val_loss},
            is_best=is_best
        )
        
        # Sync experiment state
        self.state.to_dict()

    # =========================================================================
    # CORE LOOP
    # =========================================================================

    def train(self, train_batch_fetcher: Callable, val_batch_fetcher: Callable):
        """
        The orchestrator execution block.
        Receives abstract fetchers yielding (x, y) tensors.
        """
        self.val_batch_fetcher = val_batch_fetcher
        self.before_train()
        
        while self.state.iteration < self.max_iters:
            self.before_step()
            
            # Fetch and place data
            x, y = train_batch_fetcher()
            x, y = self.device_manager.to_device(x, y)
            
            # --- THE OPTIMIZATION PIPELINE ---
            
            # 1. Forward
            with self.device_manager.autocast():
                logits, loss = self.model(x, targets=y)
                
            self.current_loss = loss.item()
                
            # 2. Loss Scaling
            if self.scaler:
                self.scaler.scale(loss).backward() # 3. Backward
                
                # 4. Gradient Unscaling & Clipping
                if self.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    self.state.gradient_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip
                    ).item()
                    
                # 5. Optimizer Step
                scale_before = self.scaler.get_scale()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                scale_after = self.scaler.get_scale()
                optimizer_step_was_skipped = scale_after < scale_before
            else:
                loss.backward() # 3. Backward
                if self.grad_clip > 0:
                    self.state.gradient_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip
                    ).item()
                self.optimizer.step() # 5. Optimizer Step
                optimizer_step_was_skipped = False
                
            # 6. Scheduler Step
            # Only advance the learning rate if the optimizer successfully updated weights.
            if self.scheduler and not optimizer_step_was_skipped:
                self.scheduler.step(self.state)
            
            # 7. Zero Grad
            self.optimizer.zero_grad(set_to_none=True)
            
            self.state.iteration += 1
            self.state.global_step += 1
            self.after_step()
            
        self.after_train()
