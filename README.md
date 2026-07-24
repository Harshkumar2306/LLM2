<div align="center">
  <h1>🧠 Axiom AI: Developer Deep Dive</h1>
  <p><b>An exhaustive breakdown of a fully custom 114M parameter LLM and its Hybrid-RAG Web Ecosystem.</b></p>
</div>

---

## 📖 Executive Overview

Axiom is a proprietary ecosystem built entirely from scratch. Unlike standard AI wrappers, this project encompasses the fundamental mathematics of the neural network, the engineering of the data pipeline, the architecture of a Live-RAG FastAPI backend, and the polishing of a reactive UI.

This document serves as a complete "Deep Dive" into the entire `LLM2` project, breaking down every folder, pipeline, and critical piece of code.

---

## 📂 Master Repository Map

```text
LLM2/
├── axiom_model/          # The PyTorch Neural Network & Training Engine
│   ├── configs/          # YAML configurations for architecture & training
│   ├── data/             # Datasets, tokenization, and data loaders
│   ├── models/           # Core PyTorch Modules (Attention, FFN, Block)
│   ├── scripts/          # Training loops (train.py), eval, and RAG tools
│   ├── sft_best.pt       # The final Supervised Fine-Tuned model weights
│   └── trainer/          # The engine that handles the PyTorch backward pass
├── axiom_web/            # The Full-Stack Web Application
│   ├── backend/          # FastAPI Server (Bridging PyTorch & React)
│   │   ├── main.py       # API Routes and Server-Sent Events (SSE) logic
│   │   └── requirements.txt
│   └── frontend/         # The React.js / Vite Application
│       ├── src/
│       │   ├── App.jsx   # Main Chat Interface (SSE consumption, RAG parsing)
│       │   └── index.css # Premium Glassmorphism styling and custom UI elements
└── README.md             # You are here
```

---

## 🧠 Part 1: The Core Neural Network (`axiom_model/`)

The model is a 114M parameter autoregressive transformer written in pure PyTorch.

### 🔬 Architectural Choices
We rigorously benchmarked and implemented modern, state-of-the-art transformer techniques:
1. **Grouped-Query Attention (GQA):** Replaced standard Multi-Head Attention. By grouping key-value heads (`n_kv_heads: 4`), we drastically reduced VRAM overhead during generation, allowing for massive batch sizes without OOM errors.
2. **SwiGLU Activations:** Replaced the standard GeLU activation in the Feed-Forward Network. SwiGLU provides superior convergence rates by utilizing a learned gating mechanism.
3. **Rotary Positional Embeddings (RoPE):** Replaced absolute positional embeddings. RoPE encodes relative positional information directly into the attention calculations by rotating the query and key vectors in a complex plane.
4. **RMSNorm:** Replaced standard LayerNorm to improve computational speed without sacrificing stability.

### 📚 The Pre-Training Pipeline (Phase 1)
To teach the model the fundamental rules of human language, we designed a perfectly balanced **7.5 Billion token curriculum** (`data/datasets.yaml`):
- **55% FineWeb-Edu** (`HuggingFaceFW/fineweb-edu`): High-quality educational data.
- **20% StarCoder** (`vikp/starcoder_cleaned`): Cleaned code to teach logical reasoning.
- **10% Wikipedia**: Encyclopedic facts and history.
- **10% OpenOrca**: Technical instructions.
- **5% MiniPile Books**: Long-form literature for narrative coherence.

**Pipeline Execution:**
1. `scripts/download_dataset.py` fetches the streams.
2. `trainer/trainer.py` executes the distributed PyTorch training loop using AdamW, a Cosine Annealing Learning Rate scheduler, and mixed-precision (FP16/BF16) training.

### 🗣 The Fine-Tuning Pipeline (Phase 2)
A raw base model only predicts text. To turn Axiom into an assistant, we performed Supervised Fine-Tuning (SFT).
- **Code:** `scripts/train_sft.py`
- **Process:** We injected `<|user|>` and `<|assistant|>` control tokens into a high-quality conversational dataset. The model learned to stop predicting random internet text and instead adopt an assistant persona. 
- **Output:** The final weights are saved as `sft_best.pt`.

### 📂 File-by-File Breakdown (`axiom_model/models/`)
- `attention.py`: Contains the `CausalSelfAttention` class. Implements the mathematically complex RoPE rotations and the GQA tensor reshaping (`einops` style).
- `ffn.py`: Implements the `SwiGLU` Feed-Forward network.
- `model.py`: The `GPT` wrapper that ties everything together. Contains the `generate()` function which handles Top-K sampling, temperature scaling, and KV-caching.

---

## ⚡️ Part 2: The Backend & RAG System (`axiom_web/backend/`)

To serve the model to users, we built a high-performance **FastAPI** server that intercepts API requests, runs inference, and streams tokens.

### 🌐 The "Hybrid Brain" RAG System
LLMs hallucinate and lack real-time data. We solved this by building a multi-modal Retrieval-Augmented Generation (RAG) pipeline (`scripts/retrievers/`):
1. **Local FAISS Database (`local.py`):** Uses vector embeddings to retrieve context from private documents.
2. **Live Web Search (`web.py`):** The crown jewel. When a user asks a question, this script silently executes a live Python DuckDuckGo search using the `ddgs` library. It scrapes the HTML of the top 3 results, cleans the text, and injects it directly into the LLM's system prompt before generation begins.
3. **Hybrid Mode (`hybrid.py`):** Concurrently searches both local FAISS databases and the live internet.

### 🌊 Server-Sent Events (SSE) Streaming
Waiting 10 seconds for an LLM to generate a full response is terrible UX. 
- **Code:** `main.py`
- **Process:** As the PyTorch `generate()` function yields a single token (e.g., `["Hello"]`), FastAPI immediately flushes that chunk down an open HTTP connection to the frontend using the `text/event-stream` protocol.

---

## 🎨 Part 3: The Frontend Application (`axiom_web/frontend/`)

A completely custom React.js application powered by Vite, designed to look and feel like a $100M enterprise product.

### 🏗 Architecture & State Management
- **Code:** `src/App.jsx`
- **Process:** The app maintains a `messages` array in React state. When the user sends a message, it opens an asynchronous `fetch` connection to the FastAPI server and reads the SSE stream chunk-by-chunk. As chunks arrive, React re-renders the specific message bubble in real-time, creating a smooth typing effect.

### 💎 Premium UI/UX Features
- **Glassmorphism Aesthetic (`index.css`):** Built with complex CSS radial gradients, `backdrop-filter: blur()`, and soft semi-transparent borders to create a deep, futuristic look.
- **Custom Dropdown Selector:** We bypassed native HTML `<select>` elements and built a fully custom React dropdown using `Lucide-React` icons to let the user select between "Local Only", "Web Search", and "Hybrid" modes.
- **Rigid Flex-Box Layout:** We engineered the CSS so that the header and input boxes are pinned with `flex-shrink: 0`. This guarantees that the UI never jitters, stretches, or jumps around wildly while the LLM is generating long paragraphs of text.
- **Source Citations:** When the web scraper pulls data, the backend sends the URLs to the frontend, which renders them as beautiful, clickable pills beneath the message.

---

## 🚀 How to Run the Ecosystem

Because this is a decoupled Full-Stack application, you need to run both the API and the UI simultaneously.

**Terminal 1: Start the AI Backend**
```bash
cd axiom_web/backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```
*Note: This will load the `sft_best.pt` weights into your CPU/GPU memory and initialize the DuckDuckGo web scrapers.*

**Terminal 2: Start the React Frontend**
```bash
cd axiom_web/frontend
npm install
npm run dev
```

Finally, open your browser to `http://localhost:5173` to interact with Axiom!
