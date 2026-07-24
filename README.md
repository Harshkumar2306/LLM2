<div align="center">
  <h1>🧠 Axiom AI</h1>
  <p><b>A custom-built, 114M parameter Large Language Model with a full-stack Live Web-Search RAG pipeline.</b></p>
</div>

---

## 📖 Overview

Axiom is an end-to-end, fully custom AI ecosystem built entirely from scratch. Unlike wrappers around OpenAI's API, **Axiom is a proprietary neural network written in pure PyTorch**. 

This project documents the entire journey of building a Large Language Model—from designing the core Transformer architecture and benchmarking modern techniques (like RoPE and SwiGLU), to pre-training, Supervised Fine-Tuning (SFT), and finally deploying the weights into a blazing-fast, production-ready React web application with live internet browsing capabilities.

## ✨ Key Features

- **Custom PyTorch LLM**: A 114M parameter autoregressive transformer built from the ground up.
- **Winning Architecture**: Benchmarked and implemented modern techniques including **Grouped-Query Attention (GQA)**, **SwiGLU** activation, **Rotary Positional Embeddings (RoPE)**, and **RMSNorm**.
- **Live Web Search (RAG)**: The AI can break out of its static weights. It intercepts queries, silently scrapes DuckDuckGo using `ddgs`, and injects live internet data into its context window before generating an answer.
- **Local FAISS Database**: Seamless integration with a local vector database for querying internal documents.
- **Hybrid Brain Mode**: Simultaneously queries local documents and the live web, concatenating the best results.
- **Server-Sent Events (SSE)**: Blazing fast token-by-token streaming from the PyTorch backend to the React frontend.
- **Premium UI/UX**: A highly polished, custom React frontend featuring a glassmorphism aesthetic, custom animated dropdowns, Markdown rendering, and clickable source citations.

---

## 🏗 System Architecture

The Axiom ecosystem is divided into three distinct layers:

### 1. The Core Model (`axiom_model/`)
The beating heart of the system. This directory contains the pure PyTorch implementation of the GPT architecture.
- **`config/`**: Highly modular YAML configuration system mapped to Pydantic-style Python classes.
- **`core/`**: The neural network components (`attention.py`, `ffn.py`, `model.py`).
- **`scripts/`**: Training loops, dataset preparation, and evaluation harnesses.

### 2. The API Backend (`axiom_web/backend/`)
A high-performance **FastAPI** server that bridges the neural network with the outside world.
- **Inference Engine**: Loads `sft_best.pt` into memory and manages tensor generation.
- **RAG Retrievers**: Implements `LocalRetriever` (FAISS), `WebRetriever` (DuckDuckGo), and `HybridRetriever`.
- **SSE Streaming**: Yields generated tokens incrementally to reduce perceived latency.

### 3. The React Frontend (`axiom_web/frontend/`)
A stunning user interface built with **React** and **Vite**.
- **Message Rendering**: Parses Markdown and displays code blocks beautifully.
- **Dynamic Mode Selector**: A custom-built dropdown allowing users to switch between "Local Only", "Database", "Live Web Search", and "Hybrid Brain".
- **Source Citations**: Displays clickable URL pills for any information retrieved via RAG.

---

## 🚀 Development Phases

### Phase 1: Architecture & Pre-training
We began by building a flexible, modular GPT architecture. To ensure we were using the most efficient design, we ran rigorous benchmarks against various architectural choices:
- **Attention**: Multi-Head Attention vs. Grouped-Query Attention (GQA). *Winner: GQA.*
- **Activations**: Standard GeLU vs. SwiGLU. *Winner: SwiGLU.*
- **Positional Encoding**: Absolute Positional Embeddings vs. Rotary Positional Embeddings (RoPE). *Winner: RoPE.*

The winning configuration was compiled into `axiom_v1.0.yaml`, and the model was pre-trained on a massive corpus of text to learn the fundamental structure of human language.

### Phase 2: Supervised Fine-Tuning (SFT)
A base model only knows how to predict the next word—it doesn't know how to act like an assistant. In Phase 2, we formatted high-quality instruction datasets into conversational `<|user|>` and `<|assistant|>` formats. We then ran a Supervised Fine-Tuning training loop to align the model, resulting in our final deployable weights: `sft_best.pt`.

### Phase 3: RAG Implementation
To solve the problem of LLM hallucinations and static knowledge cutoff dates, we built a Retrieval-Augmented Generation (RAG) pipeline.
- If the user selects **Web Mode**, the backend pauses the model, executes a live Python DuckDuckGo search, scrapes the top articles, and formats them into a strictly structured system prompt. The model is then instructed to read the scraped text and answer the user's question, citing its sources.

### Phase 4: Full-Stack Deployment
Finally, we built a FastAPI backend to serve the model via an SSE stream, and constructed a beautiful React frontend to consume it. We implemented a sleek glassmorphism aesthetic with rigid flex-box layouts to ensure a perfectly stable, jitter-free streaming experience.

---

## 🛠 Tech Stack

- **AI/ML**: PyTorch, FAISS, Transformers, Tiktoken
- **Backend**: Python, FastAPI, Uvicorn, DuckDuckGo-Search (`ddgs`)
- **Frontend**: React, Vite, CSS3, Lucide-React, React-Markdown

## 🎮 How to Run

1. **Start the Backend**
   ```bash
   cd axiom_web/backend
   pip install -r requirements.txt
   uvicorn main:app --reload --port 8000
   ```

2. **Start the Frontend**
   ```bash
   cd axiom_web/frontend
   npm install
   npm run dev
   ```

3. Open `http://localhost:5173` in your browser and start chatting!

---
*Built from scratch with ❤️*
