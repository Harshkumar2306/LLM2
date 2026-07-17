# ADR 0003: Positional Embeddings Strategy

## Context
Standard Transformers use Absolute Positional Embeddings (learned vectors added to the word embeddings at the very beginning of the model). Modern models (LLaMA) use Rotary Positional Embeddings (RoPE), which are multiplicative transformations applied deep inside the Attention blocks. Axiom must support ablating between these fundamentally different approaches.

## Problem
Because Absolute and Rotary embeddings operate at different stages of the computational graph (Embedding layer vs. Attention layer), standard implementations rely on messy `if/else` checks scattered throughout the codebase to turn them on or off.

## Alternatives Considered
1. **Scattered Booleans:** Add `if config.use_rope` inside the attention mechanism and `if not config.use_rope` in the embedding layer.
2. **Unified Base Strategy Interface:** Create a single contract that dictates exactly how positional operations occur across the entire graph.

## Decision
We implemented a **Decoupled Strategy Pattern (`BasePositionStrategy`)**. The interface enforces two separate methods:
- `get_embeddings(seq_len)`: For additive strategies.
- `apply_rotary(q, k)`: For multiplicative strategies.

Every specific positional implementation (e.g. `AbsolutePosition`, `RotaryPosition`) implements both methods. If a strategy doesn't use additive math (like RoPE), it simply returns `0` for `get_embeddings`. If it doesn't use multiplicative math (like Absolute), it simply returns `q, k` unmodified for `apply_rotary`.

## Consequences
- **Positive:** Mathematically respects the computational graph. Completely isolates the `Attention` and `Embedding` layers from knowing *what* positional strategy is active.
- **Negative:** Adds a slight function-call overhead for operations that pass through as no-ops.

## Future Revisions
This interface perfectly supports future extensions like ALiBi (which adds a bias matrix to attention scores). We will just add an `apply_bias(scores)` method to the interface contract.
