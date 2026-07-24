---
title: Axiom LLM
emoji: 🚀
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
app_port: 8000
---

<div align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/1/10/PyTorch_logo_icon.svg" width="80" alt="PyTorch Logo">
  <h1 align="center">Axiom AI: The Monolithic Ecosystem</h1>
  <p align="center">
    <strong>An end-to-end, fully custom 114M parameter Large Language Model and Hybrid-RAG Web Ecosystem.</strong>
  </p>
  <p align="center">
    <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
    <img src="https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi" alt="FastAPI">
    <img src="https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB" alt="React">
    <img src="https://img.shields.io/badge/Vite-B73BFE?style=for-the-badge&logo=vite&logoColor=FFD62E" alt="Vite">
  </p>
  <p>
    <img src="https://img.shields.io/badge/Platform-Web%20%7C%20Local-lightgrey?style=flat-square" alt="Platforms">
    <img src="https://img.shields.io/badge/Architecture-RAG%20%2B%20SSE%20Streaming-success?style=flat-square" alt="Architecture">
    <img src="https://img.shields.io/badge/License-MIT-blue?style=flat-square" alt="License">
  </p>
</div>

<br/>

> **Axiom AI** is an end-to-end ecosystem that bridges raw PyTorch machine learning with a blazing-fast web streaming architecture. It features a custom 114M parameter neural network hooked into a live internet-scraping Hybrid RAG system, all delivered through a premium glassmorphism React interface.

<br/>

## 🌐 Live Environments
| Resource | Link |
| :--- | :--- |
| **💻 Web Dashboard (React)** | [https://axiom0-gamma.vercel.app](https://axiom0-gamma.vercel.app) |
| **⚙️ Backend API (FastAPI)** | `https://harsh0o23-smart-agro-api.hf.space/api/chat` |
| **🗄️ Model Weights (LFS)** | [View sft_best.pt on GitHub](https://github.com/Harshkumar2306/LLM2/tree/main/axiom_model) |
| **📁 Source Code** | [GitHub Repository](https://github.com/Harshkumar2306/LLM2) |

---

## 📖 Table of Contents
1. [Project Philosophy](#-project-philosophy)
2. [System Architecture](#-system-architecture)
3. [The Neural Network (Axiom v1.0)](#-the-neural-network-axiom-v10)
4. [The 7.5B Training Curriculum](#-the-75b-training-curriculum)
5. [Hybrid Brain (RAG) Architecture](#-hybrid-brain-rag-architecture)
6. [Performance Metrics](#-performance-metrics)
7. [Local Setup & Deployment](#-local-setup--deployment)

---

## 🧠 Project Philosophy

Unlike standard API wrappers that simply forward calls to OpenAI, **Axiom is a proprietary neural network written entirely from scratch**. We constructed the tensor mathematics, compiled the dataset, trained the model over 7.5 Billion tokens, and wrapped it in a production-ready streaming architecture.

This repository is a **Monorepo** containing three core pillars:
1. **`axiom_model`**: The raw PyTorch training engine, weights, and inference logic.
2. **`axiom_web/backend`**: A blazing-fast FastAPI server utilizing Server-Sent Events (SSE).
3. **`axiom_web/frontend`**: A premium Glassmorphism React.js UI that consumes the live token stream.

---

## 🏗️ System Architecture

```mermaid
graph TB
    subgraph Frontend["Frontend (React + Vite)"]
        UI["Axiom Dashboard (Glassmorphism UI)"]
        UI --> |"REST API + SSE Stream"| API
    end

    subgraph Backend["Backend (FastAPI)"]
        API["FastAPI Server"]
        
        subgraph RAG["Hybrid Brain Retrievers"]
            R1["🌐 WebRetriever (DuckDuckGo Live Scraper)"]
            R2["📚 LocalRetriever (FAISS Vector DB)"]
            R1 & R2 --> R3["🧠 HybridRetriever"]
        end
        
        API --> |"Dynamic Queries"| RAG
        API --> |"Context + Augmented Prompt"| MODEL["PyTorch Engine"]
    end

    subgraph Model["The Neural Network (114M)"]
        MODEL --> |"Autoregressive Generation"| LLM["sft_best.pt (GQA, SwiGLU, RoPE)"]
        LLM --> |"Real-Time Token Flush"| API
    end

    style Frontend fill:#dbeafe,stroke:#3b82f6,color:#1e3a5f
    style Backend fill:#d1fae5,stroke:#10b981,color:#064e3b
    style Model fill:#ede9fe,stroke:#8b5cf6,color:#4c1d95
    style RAG fill:#fef3c7,stroke:#f59e0b,color:#78350f
```

---

## 🔬 The Neural Network (Axiom v1.0)

Axiom relies on a modern, highly optimized Autoregressive Transformer architecture.

### Hyperparameters & Tensor Mathematics
*   **Parameters:** 114 Million
*   **Layers:** 12 Transformer Blocks
*   **Dimensionality (`d_model`):** 768
*   **Attention Mechanism:** 12 Query Heads, 4 KV Heads
*   **Context Window:** 2048 Tokens

### Architectural Enhancements
1.  **Grouped-Query Attention (GQA):** By sharing Keys and Values across multiple Query heads (3:1 ratio), we dramatically reduced KV-Cache memory bandwidth during inference.
2.  **SwiGLU Activations:** Replaced standard ReLU/GELU with a Swish-Gated Linear Unit (`Swish(xW) * xV`), allowing for richer representations and faster convergence.
3.  **Rotary Positional Embeddings (RoPE):** Eliminated absolute positional embeddings in favor of RoPE, which encodes relative distances directly into the Q and K vectors via complex plane rotations.
4.  **RMSNorm:** Replaced standard LayerNorm with Root Mean Square Normalization to eliminate mean-centering computation, boosting GPU throughput.

---

## 📚 The 7.5B Training Curriculum

To teach the model human language, coding logic, and conversational alignment, we engineered a carefully balanced **7.5 Billion token curriculum** during the Phase 1 Pre-Training loop.

| Dataset | Subset / Source Repo | Percentage | Core Objective |
| :--- | :--- | :--- | :--- |
| **FineWeb-Edu** | `HuggingFaceFW/fineweb-edu` | 55% | Broad educational web data for foundational world knowledge. |
| **StarCoder** | `vikp/starcoder_cleaned` | 20% | Cleaned programming data for logical reasoning and syntax structure. |
| **Wikipedia** | `wikimedia/wikipedia` | 10% | Encyclopedic facts, dates, and historical data. |
| **OpenOrca** | `Open-Orca/OpenOrca` | 10% | Technical instruction-following and chain-of-thought data. |
| **MiniPile Books** | `JeanKaddour/minipile` | 5% | Long-form literature to develop narrative coherence. |

Following Phase 1, the model underwent **Phase 2: Supervised Fine-Tuning (SFT)** on a high-quality conversational dataset to align it as an AI assistant, resulting in the final `sft_best.pt` deployment weights.

---

## 🌐 Hybrid Brain (RAG) Architecture

Because 114M parameters cannot memorize the entire internet, we augmented Axiom with a multi-modal Retrieval-Augmented Generation (RAG) pipeline.

1.  **WebRetriever (`ddgs`):** When a user asks about current events, the backend silently halts generation, executes a live DuckDuckGo search, scrapes the HTML of the top 3 results, and compiles the text.
2.  **LocalRetriever (`FAISS`):** Retrieves domain-specific context from local documents using highly optimized vector embeddings.
3.  **Context Injection:** The retrieved text is injected into the `<|system|>` prompt wrapper before the tokens reach the PyTorch engine, allowing Axiom to "read" the internet before answering.

---

## 📊 Performance Metrics

Axiom was engineered specifically for edge-deployment and local-first execution.

| Metric | Result | Hardware Target |
| :--- | :--- | :--- |
| **Inference Speed** | `35-45 tokens/sec` | Standard Apple Silicon M-Series CPUs / Nvidia T4. |
| **RAG Web Latency** | `~1.2 seconds` | Includes DuckDuckGo query, HTML scraping, and parsing. |
| **Base VRAM/RAM** | `~450 MB` | Idle memory footprint (FP32 weights only). |
| **Peak VRAM/RAM** | `~800 MB` | Under maximum load during heavy 2048-token KV-Caching. |
| **SFT Validation Loss** | `~2.85` | Achieved upon completion of the conversational alignment phase. |

---

## 🚀 Local Setup & Deployment

### 1. Run Backend (FastAPI + PyTorch)
Navigate to the backend directory, install the dependencies, and start the Uvicorn server:
```bash
cd axiom_web/backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```
*The API will be available at http://localhost:8000.*

### 2. Run Frontend (React Dashboard)
Navigate to the frontend directory, install Node modules, and boot the Vite server:
```bash
cd axiom_web/frontend
npm install
npm run dev
```
*The dashboard will be available at http://localhost:5173.*

### 3. Cloud Deployment (Hugging Face + Vercel)
This repository is configured for automated CI/CD deployments:
1. **GitHub Actions (Backend):** Pushing to the `main` branch automatically triggers `.github/workflows/sync_to_hf.yml`, which forces a sync to Hugging Face Spaces. The Space automatically builds the `Dockerfile` and boots the FastAPI server.
2. **Vercel (Frontend):** Vercel watches the `axiom_web/frontend` directory. Ensure your Vercel Environment Variables contain `VITE_API_URL` pointing to your deployed Hugging Face Space.

---
<div align="center">
  <i>Engineered from scratch by Harsh Kumar.</i>
</div>
