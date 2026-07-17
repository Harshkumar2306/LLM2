# ADR 0004: Experiment Manager & Training Resilience

## Context
Axiom is designed to train on Kaggle Tesla T4x2 environments. These environments suffer from strict runtime limits, idle timeouts, and unexpected disconnections. A standard PyTorch training loop that assumes a single continuous session will fail catastrophically and lose days of compute.

## Problem
Currently, the training script dumps `.pt` files to a `runs/` directory blindly. It does not strictly enforce configuration immutability (you could resume a SwiGLU checkpoint using a Baseline config by mistake). It also does not securely preserve all RNG (Random Number Generator) states.

## Alternatives Considered
1. **Cloud Sync Solutions (W&B, MLflow):** Heavy dependencies, requires internet access, overkill for a strict local script.
2. **Basic PyTorch `torch.save` / `torch.load`:** Insufficient. Does not track configuration parameters, git state, or RNG state securely across restarts.

## Decision
We implemented a strict, fault-tolerant **BaseExperimentManager** interface.
- It strictly enforces checkpoint recovery as a core architectural feature.
- It serializes not just model weights, but Optimizer state, Scheduler state, GradScaler state, Python/Numpy/PyTorch/CUDA RNG states, and Git Commit hashes.
- On resumption, it automatically verifies that the architecture, vocabulary, and model dimensions perfectly match the checkpoint before continuing.

## Consequences
- **Positive:** Maximum resilience. If Kaggle crashes at iteration 24,999, the script can seamlessly restart and compute iteration 25,000 as if nothing happened, yielding mathematically identical results (within floating-point tolerance).
- **Negative:** Checkpoints become significantly larger due to the inclusion of full optimizer states and RNG seeds. Saving checkpoints incurs a slight IO latency penalty.

## Future Revisions
If checkpoint size becomes an issue during the 32M run, we may implement asynchronous saving or model weight chunking to prevent I/O blocking during training.
