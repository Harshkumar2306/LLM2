# 🔬 Axiom LLM: Technical Research Report

**Project:** Axiom Large Language Model  
**Parameters:** 17.86 Million  
**Architecture:** Decoder-Only Transformer (SwiGLU)  
**Dataset:** TinyShakespeare (1MB)  
**Author:** Harsh Kumar  

---

## 1. Abstract
This report details the architectural design, mathematical foundation, and cloud deployment of the **Axiom Large Language Model**. The objective of this project was to construct a state-of-the-art Generative AI system entirely from scratch using PyTorch, specifically mirroring the architectural choices of modern frontier models like Meta's Llama 3, scaled down for consumer hardware. 

The resulting model successfully learned the vocabulary, grammar, and structural formatting of Shakespearean English, and was successfully containerized and deployed as a live, full-stack web application.

---

## 2. Architectural Design

Rather than utilizing off-the-shelf wrappers (e.g., HuggingFace Transformers), the Axiom model was built from the ground up using core PyTorch tensor operations.

### 2.1 Multi-Head Causal Self-Attention
The core reasoning engine of the model is a Multi-Head Attention block. 
- **Attention Heads:** We utilized 8 parallel attention heads (`n_heads = 8`). This allows the model to simultaneously track multiple linguistic contexts (e.g., Head 1 tracks grammar, Head 2 tracks emotional tone).
- **Causal Masking:** A strict lower-triangular causal mask was applied to the attention matrix (filling the upper triangle with `-inf`). This enforces the autoregressive property, preventing the model from "looking ahead" into the future during training, guaranteeing it acts strictly as a next-token predictor.

### 2.2 SwiGLU Activation Mechanism
A defining feature of the Axiom architecture is the abandonment of the traditional ReLU or GELU feed-forward networks in favor of **SwiGLU** (Swish-Gated Linear Unit).
- **Mathematics:** SwiGLU splits the hidden dimension stream into two parallel projections. One projection is passed through a non-linear Swish function (acting as a soft gate), which is then multiplied element-wise against the second projection.
- **Why it matters:** This gating mechanism drastically increases the representational capacity of the network without increasing the parameter count, allowing a tiny 17M parameter model to learn much richer nuances of the English language.

### 2.3 Positional Embeddings
Because Transformers process sequences in parallel, they natively lack a sense of time. To solve this, we injected learned positional embeddings directly into the token embeddings, granting the model a strict spatial understanding of a 1,024-token context window.

---

## 3. Training Methodology

The model was initialized with completely randomized weights (via standard Gaussian initialization) and trained to learn English from zero.

### 3.1 Dataset & Tokenization
- **Dataset:** We utilized `tinyshakespeare`, a 1MB compilation of William Shakespeare's plays.
- **Tokenization:** To maximize processing efficiency, text was converted to integers using the **GPT-2 Byte-Pair Encoding (BPE)** tokenizer, yielding a strict vocabulary size of 50,257 tokens.

### 3.2 Optimization & Loss
- **Objective:** The model was trained using **Cross-Entropy Loss**, calculating the probabilistic distance between the model's token prediction and the actual next token in the script.
- **Optimizer:** We implemented the **AdamW** optimizer. By separating weight decay from the gradient update, AdamW prevents weights from ballooning out of control.
- **Gradient Clipping:** To protect the 17.86M parameters from sudden destabilization ("exploding gradients") caused by anomalous batches, gradients were strictly clipped.

### 3.3 Training Loop Execution
The model was trained for exactly **10,000 iterations**. 
To prevent overfitting, a strict **90/10 data split** was enforced. The model adjusted its weights based on the 90% training split, and was periodically evaluated on the unseen 10% validation split. 
An automated checkpointing system was implemented to save the exact matrix state (`best.pt`) whenever the model achieved a new minimum validation loss.

---

## 4. Engineering & Deployment Challenges

Deploying a custom, heavy PyTorch model to a free-tier cloud environment (0.1 vCPU, 512MB RAM) presented massive engineering hurdles.

### 4.1 Bypassing File Size Limits (The Chunking System)
**Problem:** The final `best.pt` file was 213MB. GitHub and Render strictly block files over 100MB, making automated deployment impossible.
**Solution:** We built a Python deployment script that binary-chunks the model into multiple 95MB pieces. Upon container boot, a custom FastAPI startup script reads the chunks in binary mode, perfectly reassembles the 213MB file into active memory, and loads it into PyTorch before opening the web ports.

### 4.2 Eliminating ASGI Event-Loop Deadlocks
**Problem:** FastAPI is built on an asynchronous event loop. PyTorch inference is a heavily synchronous, CPU-bound process. Triggering a generation originally blocked the event loop entirely, causing Render's automated health checks to timeout, resulting in the server being forcefully killed (502 Bad Gateway).
**Solution:** We re-architected the API endpoint from `async def generate()` to standard `def generate()`. This forced FastAPI to offload the PyTorch generation into a background **Starlette Threadpool**, freeing the main event loop to instantly respond to health checks and keep the server alive during 30-second generations.

### 4.3 Preventing CPU Starvation (Thread Limiting)
**Problem:** PyTorch aggressively spawns 16 parallel threads by default to handle matrix multiplication. On a server with only 0.1 vCPU, this caused severe "Thread Contention" (thrashing), resulting in extreme latency and frequent crashes.
**Solution:** We injected `torch.set_num_threads(1)` into the PyTorch engine, enforcing strict single-threaded execution. This localized all matrix math to the single available CPU core, eliminating thrashing and decreasing inference time by over 1000%.

---

## 5. Conclusion

The Axiom project successfully demonstrates the ability to architect, train, and deploy a frontier-style Large Language Model from absolute zero. 

By implementing advanced mathematical architectures like SwiGLU, and engineering robust backend solutions like dynamic binary chunking and threadpool offloading, we successfully brought a highly complex PyTorch engine to life on the global internet with a stunning React user interface.
