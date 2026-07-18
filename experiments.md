# Axiom Research Journal

This journal tracks all experiments in the Axiom Foundation project to ensure reproducibility and provide a clear comparison of architectural upgrades and scaling laws.

## Phase 1: Architecture Research (37M Scale)

**Dataset:** FineWeb-Edu (Sample-10BT)
**Context Length:** 1024
**Batch Size:** 8 (per GPU, DDP across 2x T4 GPUs)
**Max Iterations:** 10,000

| Metric                | V1 (Vanilla GPT) | V2 (RoPE) | V3 (Llama-2 Arch) |
| :-------------------- | :--- | :--- | :--- |
| **Final Train Loss**  |      |      |      |
| **Validation Loss**   |      |      |      |
| **Best Val Loss**     |      | 4.3166 |      |
| **Perplexity**        |      | ~74.93 |      |
| **Tokens/sec**        |      |      |      |
| **Peak GPU Memory**   |      |      |      |
| **Parameters**        | 37.56M | 37.13M |      |
| **Training Time (s)** |      | 8650 |      |

### Notes & Observations
* **V1:** Started training with an unfair advantage (resumed from pre-trained Shakespeare weights rather than from scratch). Validations metrics cannot be fairly compared.
* **V2:** Swapped Absolute Positional Embeddings for Rotary Positional Embeddings (RoPE). Trained from absolute scratch (random noise) on FineWeb-Edu. Achieved strong convergence.
* **V3:** *(Currently Running)* Introduces RMSNorm and SwiGLU activation functions to match the Llama-2 architecture. Expecting improved loss curve stability and slightly better final perplexity.
