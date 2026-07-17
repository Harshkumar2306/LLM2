"""
Full-Scale Training Pipeline for GPT Model
Supports evaluation, checkpointing, and comprehensive logging.
"""
import os
import sys
import json
import time
import argparse
import datetime
import subprocess
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.gpt_config import GPTConfig
from core.model import GPT
from engine.trainer import Trainer, TrainerConfig
from data.dataset import GPTDataset

def get_git_commit():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL).decode('utf-8').strip()
    except Exception:
        return "Not available"

def setup_experiment_dir(base_dir="runs"):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(base_dir, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

def parse_args():
    parser = argparse.ArgumentParser(description="GPT Training Script")
    # Dataset
    parser.add_argument('--train_bin', type=str, required=True, help="Path to training .bin dataset")
    parser.add_argument('--val_bin', type=str, required=True, help="Path to validation .bin dataset")
    # Model
    parser.add_argument('--vocab_size', type=int, default=50257)
    parser.add_argument('--d_model', type=int, default=384)
    parser.add_argument('--n_heads', type=int, default=12)
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--context_length', type=int, default=1024)
    # Trainer
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_iters', type=int, default=10000)
    parser.add_argument('--learning_rate', type=float, default=6e-4)
    parser.add_argument('--grad_accum_steps', type=int, default=1)
    parser.add_argument('--eval_interval', type=int, default=500)
    parser.add_argument('--eval_iters', type=int, default=50)
    parser.add_argument('--log_interval', type=int, default=10)
    # System
    parser.add_argument('--device', type=str, default=None, help="cpu, mps, cuda")
    parser.add_argument('--resume', type=str, default=None, help="Path to run directory to resume from")
    parser.add_argument('--out_dir', type=str, default=None, help="Static output directory (useful for Colab/Drive)")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. Setup Directories
    if args.resume:
        run_dir = args.resume
    elif args.out_dir:
        run_dir = args.out_dir
        os.makedirs(run_dir, exist_ok=True)
    else:
        run_dir = setup_experiment_dir()
        
    print(f"Experiment Directory: {run_dir}")
    
    # Auto-resume logic: if not explicitly resuming but the out_dir has a latest.pt, auto-resume
    auto_resume = False
    if not args.resume and args.out_dir:
        if os.path.exists(os.path.join(run_dir, "latest.pt")):
            auto_resume = True
            print(f"Auto-resuming from existing checkpoint in {run_dir}")
    
    # 2. Configs
    gpt_config = GPTConfig(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        context_length=args.context_length
    )
    
    trainer_config = TrainerConfig(
        max_iters=args.max_iters,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        grad_accum_steps=args.grad_accum_steps,
        device=args.device,
        checkpoint_dir=run_dir,
        log_file=os.path.join(run_dir, "metrics.jsonl")
    )
    
    # 3. Model
    model = GPT(gpt_config)
    
    # 3.5 Wrap Model in DataParallel for Multi-GPU Kaggle T4x2
    if torch.cuda.device_count() > 1 and trainer_config.device == 'cuda':
        print(f"Let's use {torch.cuda.device_count()} GPUs with DataParallel!")
        model = torch.nn.DataParallel(model)

    # Calculate parameters (works for both DataParallel and standard model)
    model_to_count = model.module if hasattr(model, 'module') else model
    param_count = sum(p.numel() for p in model_to_count.parameters())
    
    # 4. Metadata Logging
    metadata = {
        "gpt_config": gpt_config.__dict__,
        "trainer_config": trainer_config.__dict__,
        "total_parameters": param_count,
        "pytorch_version": torch.__version__,
        "device": trainer_config.device,
        "seed": torch.initial_seed(),
        "timestamp": datetime.datetime.now().isoformat(),
        "git_commit": get_git_commit()
    }
    
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(metadata, f, indent=4)
        
    print(f"Model Parameters: {param_count/1e6:.2f}M")
    print(f"Device: {trainer_config.device}")
    
    # 5. Dataloaders
    from data.dataset import MemmapStorage
    train_storage = MemmapStorage(args.train_bin)
    val_storage = MemmapStorage(args.val_bin)
    train_dataset = GPTDataset(train_storage, args.context_length)
    val_dataset = GPTDataset(val_storage, args.context_length)
    
    # For large memmap datasets, pin_memory is fast on CUDA, irrelevant on MPS
    pin_memory = trainer_config.device == 'cuda'
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=pin_memory)
    
    # 6. Trainer Init
    trainer = Trainer(trainer_config, model, train_dataloader=train_loader)
    
    if args.resume or auto_resume:
        ckpt_path = os.path.join(run_dir, "latest.pt")
        print(f"Resuming from checkpoint: {ckpt_path}")
        trainer.load_checkpoint(ckpt_path)
        
    # 7. Training Loop
    best_val_loss = float('inf')
    t0 = time.time()
    tokens_processed = trainer.current_iter * args.batch_size * args.context_length * args.grad_accum_steps
    
    print("Starting training...")
    try:
        train_iter = iter(train_loader)
        
        while trainer.current_iter < trainer.config.max_iters:
            # A. Validation
            val_loss = None
            if trainer.current_iter % args.eval_interval == 0 or trainer.current_iter == trainer.config.max_iters - 1:
                val_loss = trainer.evaluate(val_loader, eval_iters=args.eval_iters)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    trainer.save_checkpoint(os.path.join(run_dir, "best.pt"))
                    print(f"--> Saved new best model: val_loss={val_loss:.4f}")
                trainer.save_checkpoint(os.path.join(run_dir, "latest.pt"))

            # B. Training Step
            # Simulate gradient accumulation
            trainer.model.train()
            # We don't accumulate in the step manually here since our trainer MVP 
            # currently does `loss = loss / grad_accum_steps` and `.backward()`.
            # We will pass a single batch and let trainer handle it for now.
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)
                
            x, y = x.to(trainer.device), y.to(trainer.device)
            
            # Forward + Backward
            loss, lr, grad_norm, dt = trainer.train_step(x, y)
            tokens_processed += (x.numel() * trainer.config.grad_accum_steps)
            epoch = tokens_processed / (len(train_dataset) * args.context_length)
            
            # C. Logging
            if trainer.current_iter % args.log_interval == 0:
                trainer.log_metrics(epoch, loss, val_loss, lr, grad_norm, dt, tokens_processed)
                
    except KeyboardInterrupt:
        print("\nTraining interrupted manually. Saving current state...")
        trainer.save_checkpoint(os.path.join(run_dir, "latest.pt"))

    # 8. End of Training Report
    total_time = time.time() - t0
    report = f"""
    ================ END OF TRAINING REPORT ================
    Run Directory: {run_dir}
    Total Time: {total_time/60:.2f} minutes
    Total Tokens Processed: {tokens_processed}
    Average Throughput: {tokens_processed/total_time:.2f} tokens/sec
    Final Training Loss: {loss:.4f}
    Best Validation Loss: {best_val_loss:.4f}
    ========================================================
    """
    with open(os.path.join(run_dir, "report.txt"), "w") as f:
        f.write(report)
        
    print(report)

if __name__ == "__main__":
    main()
