# 🚀 Axiom Foundation 

A research-first roadmap to build a production-quality foundation language model.

---

# Pipeline

### Objective

Build a **high-quality, modular, reproducible 15 GB dataset** that can be dynamically mixed during training.

---

## Dataset Composition

| Dataset                       |  Weight |       Size |
| ----------------------------- | ------: | ---------: |
| FineWeb-Edu                   | **60%** | **9.0 GB** |
| The Stack v2                  | **20%** | **3.0 GB** |
| Wikipedia                     |  **8%** | **1.2 GB** |
| API & Framework Documentation |  **8%** | **1.2 GB** |
| Project Gutenberg             |  **4%** | **0.6 GB** |

**Total Dataset Size:** **15 GB**

## Foundation Training

### Objective

Train the production-quality foundation model.

---

## Configuration

| Setting        | Value                     |
| -------------- | ------------------------- |
| Dataset        | Axiom Dataset v1 (15 GB)  |
| Architecture   | Winner of Phase 1         |
| Model Size     | 100M Parameters           |
| Training Steps | 100,000                   |
| Optimizer      | AdamW                     |
| Scheduler      | Warmup + Cosine Decay     |
| Precision      | Mixed Precision           |
| Training       | Distributed Data Parallel |

