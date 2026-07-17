# ADR 0005: Configuration Fingerprinting and Reproducibility

## Context
Research experiments require 100% reproducibility. If an experiment trained with RoPE produces a 4.2 validation loss, we must be able to recreate that exact result six months later. Additionally, Kaggle training environments can be volatile, requiring the capability to resume models securely.

## Problem
Relying on human memory or file names (like `model_v2_final_final.pt`) to track experiments inevitably leads to corruption. Furthermore, resuming a model checkpoint with a mathematically incompatible configuration (e.g., resuming a 25M checkpoint but accidentally setting `d_model=512`) will silently corrupt the optimizer state and weights if not explicitly caught.

## Alternatives Considered
1. **Third-party experiment trackers (W&B, MLFlow)**: Adds internet dependencies and bloat.
2. **Simple JSON dumps**: Dumps the `argparse` args to a json file, but doesn't track Git state or RNG seeds, and doesn't prevent incompatible resumes.

## Decision
We implemented a strict, deterministic **Configuration Fingerprinting System (`ConfigLoader.generate_fingerprint`)** and a **Reproducibility Engine (`utils/reproducibility.py`)**.
- The `ConfigLoader` sorts all architectural keys alphabetically, dumps them to JSON, and generates a SHA256 hash. This hash strictly identifies the architectural blueprint.
- The `ReproducibilityEngine` automatically captures the Git Commit Hash, Dirty Tree Status, Python/PyTorch versions, OS, and GPU hardware.
- It also captures and restores the exact internal byte states of the `random`, `numpy`, and `torch` (CPU/CUDA) random number generators.

## Consequences
- **Positive:** Unbreakable reproducibility. The checkpoint loader will now compare the current SHA256 config fingerprint against the checkpoint's fingerprint and hard-crash if they do not match perfectly.
- **Negative:** A minor change (e.g., changing `dropout` from `0.1` to `0.2`) generates a new fingerprint, meaning it will refuse to resume from the old checkpoint unless explicitly forced (which is mathematically correct, as changing dropout alters the scale of the gradients).

## Future Revisions
None planned. This is a fundamental invariant of the Axiom platform.
