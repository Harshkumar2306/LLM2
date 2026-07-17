# ADR 0001: Component Registry System

## Context
Axiom is transitioning from a static tutorial implementation to a dynamic research platform. The model requires the ability to seamlessly swap architectural components (e.g., FFN vs. SwiGLU, LayerNorm vs. RMSNorm) without modifying the main `model.py` or block initialization code. 

## Problem
Tightly coupling implementations to the model structure violates the Open/Closed Principle. Adding a new architecture (like MoE) would currently require invasive changes to the core `TransformerBlock` and `GPT` classes, resulting in a fragile, branching codebase.

## Alternatives Considered
1. **Dependency Injection**: Passing instantiated module objects deep down the hierarchy. (Creates messy initialization scripts).
2. **If/Else Factory inside Model**: The `Block` class checks a string/boolean and instantiates the correct class. (Violates single-responsibility; the `Block` shouldn't know about `SwiGLU`).

## Decision
We implemented a strict **Decorator-based Component Registry Pattern** in `utils/registry.py`. The registry maps concrete `Enum` values to architectural `nn.Module` classes. The Model asks the registry for an implementation during `__init__`.

## Consequences
- **Positive:** Maximum decoupling. New architectures can be added simply by creating a new file and decorating the class. `model.py` remains perfectly pristine.
- **Negative:** Adds a layer of indirection. Developers cannot Ctrl+Click to immediately jump to the concrete class being used.

## Future Revisions
If future components require highly specialized, non-standard initialization arguments (like routing tables for MoE), we may need to upgrade the Registry to support a Builder Pattern interface.
