# ADR 0003: Core Model Primitives Architecture

## Problem
The core primitives (FFN, Activations, Positional Embeddings, and Initialization) define the mathematical capacity and optimization landscape of the model. We must choose designs that balance educational simplicity with modern performance.

## Feed-Forward Network (FFN) Expansion
- **Standard (4x)**: Uses 2 matrices. Expands $d$ to $4d$. Simple, historically standard (GPT-1, GPT-2, GPT-3).
- **Gated (SwiGLU, 8/3x)**: Uses 3 matrices. Modern standard (LLaMA, Mistral). Better performance per parameter, but more complex and requires more VRAM for activations.
- **Recommendation**: We will use a standard $4\times$ expansion to maintain structural similarity to GPT-2 and keep the codebase simple for our initial 10M parameter run.

## Activation Functions
- **ReLU**: $\max(0, x)$. Prone to "dead neurons".
- **GELU**: $x \Phi(x)$. Smooth, allows negative gradients. Standard in GPT-2/3.
- **SiLU**: $x \sigma(x)$. Standard in LLaMA.
- **Recommendation**: We will use **GELU** (specifically the `tanh` approximation used by OpenAI). It provides smooth gradients and avoids dead neurons without the complexity of a Gated Linear Unit.

## Positional Information
- **Learned Absolute**: A simple lookup table $E_{pos} \in \mathbb{R}^{T \times C}$. Very fast, easy to implement. Fails to extrapolate beyond training context length.
- **RoPE (Rotary)**: Rotates Queries and Keys. Excellent relative distance preservation and extrapolation. Mathematically dense.
- **Recommendation**: We will use **Learned Absolute Positional Embeddings** for the base model to maximize understanding of how information is injected into the residual stream. RoPE is better for production, but obscures the basic addition of position into the residual stream that we want to study first.

## Initialization Strategy
- **Standard Normal**: PyTorch defaults often use Kaiming or uniform. We will explicitly initialize weights from $\mathcal{N}(0, \sigma^2=0.02)$, which is the GPT-2 standard.
- **Residual Scaling**: We will scale the initialization of layers that write directly to the residual stream (the final projection in Attention and FFN) by $\frac{1}{\sqrt{2 \times n\_layers}}$. This prevents variance explosion deep in the network.
