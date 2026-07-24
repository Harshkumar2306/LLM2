<div align="center">
  <h1>🧠 Axiom Foundation Model</h1>
  <p><i>A full-stack, from-scratch 114M parameter Large Language Model and Web Application.</i></p>
</div>

---

## 🌟 Overview

**Axiom** is a complete, end-to-end AI ecosystem built entirely from scratch. This project doesn't just wrap an OpenAI API—it contains a custom-built PyTorch transformer architecture, a supervised fine-tuning (SFT) pipeline, a high-performance FastAPI streaming backend, and a beautiful React UI.

This monorepo serves as a masterclass in modern LLM architecture, showcasing how to build, train, and deploy an instruction-following model on consumer hardware.

## ✨ Features

### 🧠 Modern Architecture (Axiom v1.0)
- **114M Parameters**: Engineered for rapid training and inference on standard hardware.
- **Grouped-Query Attention (GQA)**: drastically reduces KV cache memory footprint.
- **SwiGLU Activation**: State-of-the-art non-linear activation function for better representation.
- **Rotary Positional Embeddings (RoPE)**: Enhanced sequence extrapolation.
- **RMSNorm & Weight Tying**: Stabilized training dynamics and parameter efficiency.

### ⚡ High-Performance Backend
- **FastAPI**: Asynchronous Python backend serving the PyTorch model.
- **Server-Sent Events (SSE)**: Real-time, token-by-token streaming back to the client.
- **Hybrid RAG Pipeline**: Built-in FAISS vector database integration for grounding the model in external knowledge.

### 🎨 Stunning Frontend
- **React + Vite**: Lightning-fast UI compilation.
- **Glassmorphism UI**: A premium, modern dark-mode aesthetic with blur effects and sleek gradients.
- **Markdown Rendering**: Full support for code blocks and formatted AI responses via `react-markdown`.

---

## 📂 Repository Structure

```text
LLM2/
├── axiom_model/            # Core PyTorch Neural Network & Training Pipeline
│   ├── models/             # Transformer blocks, embeddings, and attention
│   ├── scripts/            # Training, validation, and SFT scripts
│   ├── configs/            # YAML hyperparameter configurations
│   └── data/               # Datasets and tokenization logic
│
├── axiom_web/              # Full-Stack Application
│   ├── backend/            # FastAPI Server (main.py)
│   └── frontend/           # React + Vite Web UI
│
└── Dockerfile              # Production deployment configuration
```

---

## 🚀 Quick Start (Local Deployment)

### Prerequisites
- Python 3.10+
- Node.js 18+
- `sft_best.pt` checkpoint file (Place in `axiom_model/`)

### 1. Start the Backend (Terminal 1)

The backend loads the 114M parameter PyTorch model into RAM and exposes a streaming inference API.

```bash
cd axiom_web/backend
pip install -r requirements.txt
pip install -r ../../axiom_model/requirements.txt
uvicorn main:app --reload --port 8000
```
*Wait until you see `Model loaded successfully!` in the console.*

### 2. Start the Frontend (Terminal 2)

The frontend connects to the backend and provides a beautiful chat interface.

```bash
cd axiom_web/frontend
npm install
npm run dev
```

Open `http://localhost:5173` in your browser and start chatting with Axiom!

---

## 🛠️ Training the Model

If you want to train your own version of Axiom from scratch:

1. Configure your hyperparameters in `axiom_model/configs/base.yaml`.
2. Run the pre-training loop:
   ```bash
   cd axiom_model/scripts/training
   python train.py
   ```
3. Fine-tune for instruction following (SFT):
   ```bash
   python sft.py
   ```

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
<div align="center">
  <p>Built with ❤️ by Harsh</p>
</div>
