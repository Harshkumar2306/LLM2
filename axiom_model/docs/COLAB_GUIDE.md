# Training Guide: Local vs Google Colab

This document outlines how to execute training across different environments while maintaining reproducibility. Our core design philosophy guarantees that the same Python code runs identically on macOS, Linux, and Google Colab.

## 1. Local Training (macOS / Linux)

On local machines, we use timestamped run directories.

```bash
# Prepare dataset
python scripts/prepare_data.py

# Train (Creates runs/YYYY-MM-DD_HH-MM-SS)
python scripts/train.py --train_bin data/train.bin --val_bin data/val.bin --vocab_size 50257 --d_model 256 --n_heads 8 --n_layers 6 --context_length 256 --batch_size 16 --max_iters 5000 --eval_interval 500 --learning_rate 6e-4

# Resume a specific local run
python scripts/train.py --train_bin data/train.bin --val_bin data/val.bin --resume runs/2026-07-15_21-10-37
```

## 2. Google Colab Training

Google Colab instances are ephemeral. If your browser disconnects, the kernel dies and you lose all unsaved progress. To combat this, we use the `colab_train.ipynb` orchestrator and the `--out_dir` flag.

### Colab Best Practices

1. **Google Drive Mounting**: Always run the Google Drive mount cell in `colab_train.ipynb`. This ensures checkpoints are saved securely outside the ephemeral Colab disk.
2. **Auto-Resume**: The `--out_dir` argument (e.g., `--out_dir /content/drive/MyDrive/GPT2-Run`) will automatically resume from `latest.pt` if it exists. If your session disconnects, simply reconnect, run the cells from top to bottom, and training will pick up exactly where it left off.
3. **GPU Selection**:
   - Free Tier usually provides T4 GPUs.
   - Pro/Pro+ provides L4, V100, or A100 GPUs. 
   - Increase `--batch_size` (e.g., to 64 or 128) if you get a powerful GPU (like A100) to maximize throughput.
4. **Checkpoint Frequency**: Colab sessions can drop randomly. Set `--eval_interval` (which also controls checkpoint saving) to a number that triggers every ~15-30 minutes.

### Common Colab Issues

- **Out of Memory (OOM)**: Reduce `--batch_size` or `--context_length`.
- **Runtime Disconnected**: Click "Reconnect". Re-run all cells. Thanks to `--out_dir`, no progress is lost.
- **Drive Not Mounted**: Ensure you accept the permissions prompt. If it fails, the script falls back to local Colab storage, meaning your data will be lost when the instance shuts down.
