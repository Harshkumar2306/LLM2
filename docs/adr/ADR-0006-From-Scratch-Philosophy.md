# ADR 0006: "From Scratch" Core Project Philosophy

## Context
Axiom is not merely an engineering effort to produce a 32M parameter language model; it is fundamentally a research and educational platform. The goal is complete, transparent mastery over every mathematical operation, optimization technique, and architectural component of a modern GPT-style LLM.

## Problem
Modern ML engineering often defaults to wrapping heavyweight frameworks (like HuggingFace Trainer, PyTorch Lightning, DeepSpeed, or Accelerate) to save time. While this speeds up initial development, it treats the underlying mathematics and systems engineering as black boxes. This directly violates the educational and research goals of the project.

## Alternatives Considered
1. **Framework-Heavy Approach:** Using HuggingFace `Trainer` and `Accelerate`. This would save ~5,000 lines of code but would hide the actual mechanics of gradient accumulation, optimizer grouping, and DDP setup.
2. **Strict "From Scratch" Approach:** Utilizing only Python, PyTorch, and NumPy as the foundational dependencies, building all orchestrators, schedulers, and checkpoints manually.

## Decision
We officially adopt the **"From Scratch First"** philosophy. 
- Axiom will be built exclusively on fundamental libraries (`torch`, `numpy`).
- We will NOT introduce heavyweight training frameworks.
- We will study industry leaders (Megatron-LM, nanoGPT, Llama 3) for their mathematical and structural solutions, but we will implement those solutions ourselves from first principles.
- Every major implementation will include deep docstrings detailing the *What*, *Why*, *Mathematics*, and *Complexity* of the component.

## Consequences
- **Positive:** Ultimate control over the codebase. Complete transparency. Massive educational value for anyone studying the Axiom repository. No dependency hell when a third-party framework updates.
- **Negative:** Increased development time. We must manually implement and thoroughly test complex systems (like Checkpoint serialization and Multi-GPU DDP orchestration) that frameworks normally provide for free.

## Future Revisions
If we scale beyond a single node (e.g., training a 7B parameter model across multiple Kaggle machines), the complexity of fully-sharded data parallel (FSDP) communication may eventually require adopting `DeepSpeed` or PyTorch's native `FSDP`. We will delay this until the physics of the problem absolutely mandates it.
