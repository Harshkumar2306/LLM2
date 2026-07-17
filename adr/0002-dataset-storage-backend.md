# ADR 0002: Dataset Storage Abstraction

## Problem
A naive PyTorch `Dataset` implementation that loads all tokens into RAM works well for small educational datasets but crashes immediately when scaled to gigabytes of text. However, introducing `numpy.memmap` immediately adds complexity and makes debugging harder. We need an architecture that supports both in-memory and disk-backed storage seamlessly without requiring modifications to the core `Dataset` or training loop.

## Alternatives Considered

1. **Hardcoding an In-Memory Tensor in the Dataset**
   - *Advantages*: Simple, fast to write.
   - *Disadvantages*: Does not scale. Requires rewriting the dataset later.

2. **Hardcoding `numpy.memmap` in the Dataset**
   - *Advantages*: Scales to massive datasets out-of-the-box.
   - *Disadvantages*: Harder to debug, requires managing file pointers, adds unnecessary complexity for unit testing.

3. **Storage Strategy Pattern (Dependency Inversion)**
   - *Advantages*: We define a `TokenStorage` interface. The `GPTDataset` receives a `TokenStorage` instance and queries it. We can easily swap `InMemoryStorage` with `MemmapStorage`.
   - *Disadvantages*: Requires slightly more upfront boilerplate (defining the Protocol/Interface).

## Recommendation
We will use the **Storage Strategy Pattern** (Alternative 3). We will define a standard interface (`Protocol`) that requires `__len__` and `get_slice`. 

## Advantages of Recommendation
- **Testability**: We can pass an `InMemoryStorage` mock during unit tests.
- **Scalability**: We can swap in `MemmapStorage` for the 10M parameter training run without touching the PyTorch DataLoader or training loop.
- **Extensibility**: If we later want to stream data over the network (e.g., Hugging Face streaming), we just write a `StreamingStorage` class.

## Future Implications
The `GPTDataset` is entirely decoupled from how the data is stored. We will implement `InMemoryStorage` first, verify it works, and introduce `MemmapStorage` as an optimization step later.
