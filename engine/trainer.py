import os
import math
import time
import json
import contextlib
import torch
import torch.nn as nn
from torch.optim import AdamW

def get_default_device():
    """Single device abstraction for CPU, MPS (Apple Silicon), and CUDA."""
    if torch.cuda.is_available():
        return 'cuda'
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'

def get_default_dtype(device):
    """Sensible defaults for mixed precision."""
    if device == 'cuda':
        return 'bfloat16' if torch.cuda.is_bf16_supported() else 'float16'
    # MPS runs fine in fp32 implicitly, and PyTorch MPS autocast is sometimes experimental.
    # We default to float32 for safety, but allow user override.
    return 'float32'

class TrainerConfig:
    def __init__(self, 
                 # Optimization
                 max_iters=1000, batch_size=4, learning_rate=6e-4, 
                 weight_decay=1e-1, beta1=0.9, beta2=0.95, 
                 # Features
                 grad_clip=1.0, grad_accum_steps=1,
                 # LR Schedule
                 warmup_iters=100, lr_decay_iters=1000, min_lr=6e-5,
                 # Systems
                 device=None, dtype=None, 
                 # I/O
                 checkpoint_dir='checkpoints', log_file='logs/training.jsonl'):
        self.device = device if device else get_default_device()
        self.dtype = dtype if dtype else get_default_dtype(self.device)
        self.max_iters = max_iters
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.beta1 = beta1
        self.beta2 = beta2
        
        self.grad_clip = grad_clip
        self.grad_accum_steps = grad_accum_steps
        
        self.warmup_iters = warmup_iters
        self.lr_decay_iters = lr_decay_iters
        self.min_lr = min_lr
        
        self.checkpoint_dir = checkpoint_dir
        self.log_file = log_file

class Trainer:
    """
    The main training engine prioritizing correctness over optimization.
    Features like AMP, gradient accumulation, and clipping are strictly configurable.
    """
    def __init__(self, config: TrainerConfig, model: nn.Module, train_dataloader=None):
        self.config = config
        self.device = config.device
        self.model = model.to(self.device)
        self.train_dataloader = train_dataloader
        
        self.optimizer = self._configure_optimizers()
        
        # 1. Mixed Precision (Optional)
        ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}.get(config.dtype, torch.float32)
        if config.dtype in ['float16', 'bfloat16'] and self.device == 'cuda':
            self.ctx = torch.autocast(device_type='cuda', dtype=ptdtype)
        elif config.dtype in ['float16', 'bfloat16'] and self.device == 'mps' and hasattr(torch.amp, 'autocast'):
            self.ctx = torch.autocast(device_type='mps', dtype=ptdtype)
        else:
            self.ctx = contextlib.nullcontext()
            
        # Scaler is only needed for float16 (to prevent underflow), not bfloat16 or float32
        self.scaler = torch.cuda.amp.GradScaler(enabled=(config.dtype == 'float16')) if self.device == 'cuda' else None
        
        # 2. I/O Setup
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        if os.path.dirname(self.config.log_file):
            os.makedirs(os.path.dirname(self.config.log_file), exist_ok=True)
            
        self.current_iter = 0

    def _configure_optimizers(self):
        param_dict = {pn: p for pn, p in self.model.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': self.config.weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        return AdamW(optim_groups, lr=self.config.learning_rate, betas=(self.config.beta1, self.config.beta2))

    def get_lr(self, it):
        if it < self.config.warmup_iters:
            return self.config.learning_rate * (it + 1) / (self.config.warmup_iters + 1)
        if it > self.config.lr_decay_iters:
            return self.config.min_lr
        decay_ratio = (it - self.config.warmup_iters) / (self.config.lr_decay_iters - self.config.warmup_iters)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.config.min_lr + coeff * (self.config.learning_rate - self.config.min_lr)
        
    def save_checkpoint(self, path):
        checkpoint = {
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'config': self.config.__dict__,
            'current_iter': self.current_iter,
            'rng_state_cpu': torch.get_rng_state(),
        }
        if self.device == 'cuda':
            checkpoint['rng_state_gpu'] = torch.cuda.get_rng_state()
        elif self.device == 'mps':
            # MPS currently doesn't have a specific get_rng_state(), it relies on CPU seed usually
            pass 
            
        torch.save(checkpoint, path)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.current_iter = checkpoint['current_iter']
        # Map_location moves everything to device, but set_rng_state expects CPU ByteTensor
        torch.set_rng_state(checkpoint['rng_state_cpu'].cpu().byte())
        if self.device == 'cuda' and 'rng_state_gpu' in checkpoint:
            torch.cuda.set_rng_state(checkpoint['rng_state_gpu'].cpu().byte())

    @torch.no_grad()
    def evaluate(self, val_dataloader, eval_iters=100) -> float:
        """
        Runs validation loop. Disables gradients, computes average loss.
        """
        self.model.eval()
        losses = []
        
        # In a real training run, we might want to iterate sequentially.
        # But for large datasets, random sampling `eval_iters` batches is faster.
        iterator = iter(val_dataloader)
        
        for _ in range(eval_iters):
            try:
                x, y = next(iterator)
            except StopIteration:
                iterator = iter(val_dataloader)
                x, y = next(iterator)
                
            x, y = x.to(self.device), y.to(self.device)
            with self.ctx:
                logits, loss = self.model(x, targets=y)
            losses.append(loss.item())
            
        self.model.train()
        return sum(losses) / len(losses) if losses else 0.0

    def log_metrics(self, epoch, loss, val_loss, lr, grad_norm, dt, tokens_processed):
        eta = (self.config.max_iters - self.current_iter) * dt
        log_entry = {
            "epoch": round(epoch, 4),
            "iter": self.current_iter,
            "loss": round(loss, 4),
            "val_loss": round(val_loss, 4) if val_loss is not None else None,
            "lr": f"{lr:.2e}",
            "grad_norm": round(grad_norm, 4),
            "tokens_processed": tokens_processed,
            "iter_time_ms": round(dt * 1000, 2),
            "eta_sec": round(eta, 2)
        }
        # Print to console
        val_str = f" | Val {log_entry['val_loss']:.4f}" if val_loss is not None else ""
        print(f"Ep {log_entry['epoch']:.2f} | It {self.current_iter:04d} | Loss {log_entry['loss']:.4f}{val_str} | LR {log_entry['lr']} | Norm {log_entry['grad_norm']:.2f} | dt {log_entry['iter_time_ms']:.2f}ms | ETA {log_entry['eta_sec']:.0f}s")
        
        # Write to structured JSONL
        with open(self.config.log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> float:
        self.model.train()
        t0 = time.time()
        
        # 1. Learning Rate Scheduler
        lr = self.get_lr(self.current_iter)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            
        self.optimizer.zero_grad(set_to_none=True)
        
        # 2. Gradient Accumulation & Forward/Backward (with optional AMP)
        accum_loss = 0.0
        # In a real scenario with DataLoader, we would fetch grad_accum_steps micro-batches.
        # For simplicity in this step API, we'll just divide the loss by grad_accum_steps 
        # assuming the caller is managing the micro-batches. If they pass a full batch, 
        # grad_accum_steps=1 is standard.
        with self.ctx:
            logits, loss = self.model(x, targets=y)
            loss = loss / self.config.grad_accum_steps
            
        # Backward
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
            
        accum_loss = loss.item() * self.config.grad_accum_steps
        
        # 3. Gradient Clipping
        if self.scaler is not None:
            self.scaler.unscale_(self.optimizer)
            
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
        
        # 4. Optimizer Step
        if self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
            
        t1 = time.time()
        dt = t1 - t0
        
        self.current_iter += 1
        return accum_loss, lr, grad_norm.item(), dt
