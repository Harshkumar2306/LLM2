import os
import torch
from typing import Dict, Any, Tuple

from utils.config_loader import ConfigLoader
from engine.device_manager import DeviceManager
from engine.experiment_manager import ExperimentManager
from engine.checkpoint_manager import CheckpointManager
from engine.validation_manager import ValidationManager
from engine.data_manager import DataManager
from engine.trainer import Trainer
from engine.scheduler import LearningRateScheduler
from core.model import GPT
from config.gpt_config import GPTConfig

def bootstrap_training(config_path: str, data_dir: str = "data") -> Tuple[Trainer, DataManager]:
    """
    Constructs the entire dependency graph for the training pipeline.
    This separates the orchestration of components from the runtime training loop.
    """
    # 1. Configuration
    config_loader = ConfigLoader()
    config = config_loader.load(config_path)
    
    # DDP Initialization
    is_ddp = int(os.environ.get('RANK', -1)) != -1
    if is_ddp:
        import torch.distributed as dist
        dist.init_process_group(backend='nccl' if torch.cuda.is_available() else 'gloo')
        local_rank = int(os.environ['LOCAL_RANK'])
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            config["device"] = f"cuda:{local_rank}"
    else:
        local_rank = 0

    # 2. Hardware and Device
    device_manager = DeviceManager(config)
    if local_rank == 0:
        device_manager.print_status()
    
    # 3. Experiment and Logging
    exp_name = config.get("experiment_name", "baseline")
    experiment_manager = ExperimentManager("experiments", exp_name, config)
    
    # 4. Checkpoints
    checkpoint_manager = CheckpointManager(experiment_manager.get_experiment_dir(), config)
    
    # 5. Validation
    validation_manager = ValidationManager(device_manager)
    
    # 6. Data
    data_manager = DataManager(data_dir, config)
    data_manager.prepare()
    
    # 7. Model Architecture
    config["vocab_size"] = data_manager.vocab_size()
    
    # Filter config for GPTConfig kwargs
    from dataclasses import fields
    valid_keys = {f.name for f in fields(GPTConfig)}
    gpt_kwargs = {k: v for k, v in config.items() if k in valid_keys}
    
    gpt_config = GPTConfig(**gpt_kwargs)
    model = GPT(gpt_config)
    model = device_manager.to_device(model)
    
    if is_ddp:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[local_rank] if "cuda" in device_manager.device else None)
        raw_model = model.module
    else:
        raw_model = model
    
    # Print parameter statistics and FLOPs
    if local_rank == 0:
        raw_model.print_parameter_statistics()
        batch_size = config.get("batch_size", 12)
        flops = raw_model.estimate_flops(batch_size)
        print(f"Estimated FLOPs per step: {flops:.2e}")
    
    # 8. Optimizer
    # Group parameters to exclude 1D params (LayerNorm, biases) from weight decay
    decay_params = []
    no_decay_params = []
    for n, p in raw_model.named_parameters():
        if not p.requires_grad:
            continue
        # Biases and 1D parameters (like LayerNorm weights) typically don't use weight decay
        if p.dim() < 2 or "bias" in n or "ln_" in n:
            no_decay_params.append(p)
        else:
            decay_params.append(p)
            
    lr = config.get("learning_rate", 3e-4)
    weight_decay = config.get("weight_decay", 0.1)
    beta1 = config.get("beta1", 0.9)
    beta2 = config.get("beta2", 0.95)
    eps = config.get("eps", 1e-8)
    
    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    
    # Use fused AdamW if available (PyTorch >= 2.0 and CUDA)
    use_fused = torch.cuda.is_available()
    import inspect
    if "fused" in inspect.signature(torch.optim.AdamW).parameters and use_fused:
        optimizer = torch.optim.AdamW(optim_groups, lr=lr, betas=(beta1, beta2), eps=eps, fused=True)
    else:
        optimizer = torch.optim.AdamW(optim_groups, lr=lr, betas=(beta1, beta2), eps=eps)
    
    # 9. Scheduler
    scheduler = LearningRateScheduler(optimizer, config)
    
    # 10. Trainer Orchestrator
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device_manager=device_manager,
        checkpoint_manager=checkpoint_manager,
        experiment_manager=experiment_manager,
        validation_manager=validation_manager,
        config=config
    )
    
    # Optional print for baseline startup check
    print("\n====================================")
    print("BASELINE TRAINING STARTUP")
    print("====================================")
    print(f"Experiment: {exp_name}")
    print(f"Device: {device_manager.device.upper()}")
    print(f"Precision: {device_manager.dtype_str}")
    print(f"Dataset: {data_dir}")
    print(f"Context Length: {gpt_config.context_length}")
    print(f"Batch Size: {batch_size}")
    print(f"Checkpoint Dir: {experiment_manager.get_experiment_dir()}")
    print("====================================\n")
    
    return trainer, data_manager
