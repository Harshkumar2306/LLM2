# ADR 0001: Configuration Management

## Problem
A Large Language Model relies on numerous hyperparameters (embedding dimensions, layer counts, vocabulary size, etc.). Hardcoding these values throughout the codebase leads to bugs, shape mismatches, and poor maintainability. We need a central, type-safe "source of truth" for the model's physical blueprint.

## Alternatives Considered

1. **Python Dictionaries (`dict`)**
   - *Advantages*: Built-in, extremely flexible.
   - *Disadvantages*: No type safety, no autocompletion, prone to typos (e.g., `config["d_mdel"]` will fail at runtime). Hard to validate.

2. **YAML / JSON Files**
   - *Advantages*: Language agnostic, great for static configuration.
   - *Disadvantages*: Requires writing a parser, loses IDE type hinting without a backing Python class.

3. **Pydantic**
   - *Advantages*: Industry standard for API validation. Out-of-the-box runtime validation (e.g., `Field(gt=0)`). Very robust.
   - *Disadvantages*: Introduces an external dependency. Can feel slightly heavy/magical for a foundational educational project where we want to understand how things work under the hood.

4. **Hydra / OmegaConf**
   - *Advantages*: Extremely powerful for combinatorial hyperparameter sweeps in massive ML projects.
   - *Disadvantages*: High learning curve, bloats the codebase, obscures the simple control flow we want for a 10M parameter model.

5. **Python `dataclass` with Custom `__post_init__` Validation**
   - *Advantages*: Built-in standard library module. Provides strong type hinting and autocompletion. We can write explicit, readable validation rules in Python.
   - *Disadvantages*: Lacks the automatic coercion and complex nested validation of Pydantic.

## Recommendation
We will use **Python `dataclass` with custom `__post_init__` validation**.

## Advantages of Recommendation
- **Zero Dependencies**: Keeps the core model code pure Python/PyTorch.
- **Educational Value**: By writing our own `__post_init__` validation logic, we force ourselves to mathematically define the constraints of the Transformer architecture (e.g., why `d_model` must be divisible by `n_heads`).
- **Type Safety**: Full IDE support.

## Disadvantages
- We must manually write `if / raise ValueError` boilerplate for validation.

## Validation Rules Implemented
If the configuration becomes invalid, it must fail instantly upon instantiation (Fail Fast principle).
1. `d_model % n_heads == 0`: Essential for Multi-Head Attention splitting.
2. `d_model > 0`, `n_heads > 0`, `n_layers > 0`: Structural requirements.
3. `vocab_size > 0`: Must have tokens to embed.
4. `0.0 <= dropout < 1.0`: Standard probability constraint.
5. `context_length > 0`: Must have a sequence length.

## Future Implications
If this project scales to massive hyperparameter sweeps across clusters, we may need to migrate to `Hydra`. However, for a single 10M parameter model run, a `dataclass` is the most elegant and readable solution.
