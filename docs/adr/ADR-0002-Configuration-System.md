# ADR 0002: Configuration System

## Context
Axiom requires a mechanism to define and transport hyperparameter blueprints across the training system, dataloaders, and model initialization logic.

## Problem
The previous configuration system relied on a flat Python `dataclass` combined with CLI `argparse` overrides. As the number of architectural toggles increases (e.g. SwiGLU, RoPE, ALiBi), relying on boolean flags creates a combinatorial explosion of illegal states (e.g., trying to use both Absolute and ALiBi positional embeddings simultaneously). Furthermore, passing 20+ arguments via CLI is error-prone and hurts reproducibility.

## Alternatives Considered
1. **Hierarchical JSON configs**: Standard but lacks comments.
2. **Polymorphic Python Dataclasses**: Secure, but requires writing python code to create a new preset.

## Decision
We implemented a strict **Enum-based Configuration System** wrapped by a **YAML File Loader**. 
Architectural components are defined exclusively by Enums (e.g., `FFNType.SWIGLU`), preventing illegal combinations. Training runs are defined by static YAML files (`configs/stage1_baseline.yaml`).

## Consequences
- **Positive:** Configuration becomes data, not code. Guaranteed reproducibility. No illegal states.
- **Negative:** Requires a YAML parsing dependency. Requires validation logic to map YAML strings to Python Enums safely.

## Future Revisions
If configs become overly massive, we may need to introduce "config inheritance" (e.g., `base: default.yaml`) to prevent repeating identical parameters.
