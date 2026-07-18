# 🚀 Axiom Foundation v1.0 Roadmap

A research-first roadmap to build a production-quality foundation language model.

---

# 📌 Phase 0 — Dataset Pipeline

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

---

## Data Pipeline

```text
Raw Datasets
      │
      ▼
Download
      │
      ▼
Cleaning
      │
      ▼
Deduplication
      │
      ▼
Tokenization
      │
      ▼
Separate Binary Files

fineweb/
stack/
wikipedia/
api_docs/
books/

      │
      ▼
metadata.json

      │
      ▼
weights.yaml

      │
      ▼
Weighted DataLoader

      │
      ▼
Axiom Dataset v1
```

---

## Project Structure

```text
Axiom/
├── configs/
├── data/
│   ├── fineweb/
│   ├── stack/
│   ├── wikipedia/
│   ├── api_docs/
│   ├── books/
│   └── weights.yaml
├── tokenizer/
├── models/
├── trainer/
├── evaluation/
├── inference/
├── checkpoints/
├── logs/
└── scripts/
```

---

# 📌 Phase 1A — Architecture Screening

### Objective

Find the strongest architectures before committing to longer training runs.

---

## Screening Configuration

| Setting        | Value                     |
| -------------- | ------------------------- |
| Dataset        | Axiom Dataset v1 (15 GB)  |
| Model Size     | 37M Parameters            |
| Training Steps | 2,000                     |
| Tokenizer      | GPT-2 / tiktoken          |
| Optimizer      | AdamW                     |
| Scheduler      | Warmup + Cosine Decay     |
| Precision      | Mixed Precision           |
| Training       | Distributed Data Parallel |

---

## Experiments

### V1

Vanilla GPT

* Learned Positional Embeddings
* LayerNorm
* GELU
* Bias=True

---

### V2

RoPE GPT

* RoPE
* LayerNorm
* GELU
* Bias=True

---

### V3

Modern GPT (Llama-style)

* RoPE
* RMSNorm
* SwiGLU
* Grouped-Query Attention (GQA)
* Bias=False

---

## Evaluation Metrics

Measure

* Validation Loss
* Perplexity
* Training Stability
* Throughput (tokens/sec)

---

### Outcome

↓

**Eliminate the weakest architecture**

---

# 📌 Phase 1B — Architecture Confirmation

### Objective

Run the top 2 architectures for longer to confidently confirm the winner.

---

## Confirmation Configuration

| Setting        | Value                     |
| -------------- | ------------------------- |
| Architectures  | Top 2 from Phase 1A       |
| Model Size     | 37M Parameters            |
| Training Steps | 5,000                     |

---

## Evaluation Metrics

Measure again:

* Validation Loss
* Perplexity
* Stability
* GPU Memory
* Training Speed

---

### Winner

↓

**Freeze Best Architecture**

---

# 📌 Phase 2 — Foundation Training

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

---

## Final Output

```text
15 GB Dataset

↓

Best Architecture

↓

100M Parameters

↓

100K Iterations

↓

Axiom Foundation v1.0
```
