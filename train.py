import argparse
from engine.bootstrap import bootstrap_training

def main():
    parser = argparse.ArgumentParser(description="Axiom Training Entrypoint")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--data-dir", type=str, default="data", help="Directory containing train.bin and val.bin")
    parser.add_argument("--resume", type=str, choices=["latest", "best"], default="none", help="Resume from a checkpoint")
    parser.add_argument("--smoke-test", action="store_true", help="Overrides max_iters to 5 for a quick smoke test")
    args = parser.parse_args()

    # Build the entire dependency graph
    trainer, data_manager = bootstrap_training(args.config, args.data_dir, args.resume)
    
    # Smoke test override
    if args.smoke_test:
        print("\n[SMOKE TEST] Overriding max_iters to 5 and eval_interval to 2.")
        trainer.max_iters = 5
        trainer.eval_interval = 2
        trainer.save_interval = 2

    # Start the event-driven training loop
    trainer.train(
        train_batch_fetcher=data_manager.get_train_batch,
        val_batch_fetcher=data_manager.get_val_batch
    )
    
    # Graceful DDP cleanup
    import torch.distributed as dist
    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
