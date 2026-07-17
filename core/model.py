import torch
import torch.nn as nn
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.gpt_config import GPTConfig
from core.embeddings import GPTEmbeddings
from core.block import Block

class GPT(nn.Module):
    """
    The full GPT-style Large Language Model.
    Assembles the Embeddings, Transformer Blocks, and Language Modeling Head.
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        
        # 1. Embeddings (Injects meaning and geometry into the residual stream)
        self.embeddings = GPTEmbeddings(config)
        
        # 2. Transformer Blocks (The deep processing layers)
        # We use nn.ModuleList so PyTorch properly registers the parameters
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layers)])
        
        # 3. Final Normalization
        self.ln_f = nn.LayerNorm(config.d_model, bias=config.bias)
        
        # 4. Language Modeling Head (Classifier)
        # Maps the final continuous d_model vector back to discrete vocabulary probabilities
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # --- Weight Tying ---
        # The embedding matrix and the output projection matrix are forced to share memory.
        # This reduces parameters by ~30% and enforces symmetry (understanding a word is the 
        # same mathematical operation as predicting a word).
        self.lm_head.weight = self.embeddings.wte.weight
        
        # Initialize all weights using our specific Gaussian strategy
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        Custom weight initialization strategy for GPT-2.
        """
        if isinstance(module, nn.Linear):
            std = 0.02
            # Apply residual scaling to prevent variance explosion deep in the network
            if hasattr(module, 'RESIDUAL_SCALE_INIT'):
                std *= (2 * self.config.n_layers) ** -0.5
                
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
                
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None, use_flash: bool = True):
        """
        The full forward pass.
        Args:
            idx: Tensor of shape (Batch, Time) containing integer token IDs.
            targets: Optional Tensor of shape (Batch, Time) containing integer target IDs.
            use_flash: Whether to use optimized FlashAttention.
        Returns:
            logits: Tensor of shape (Batch, Time, Vocab_Size) containing unnormalized predictions.
            loss: Scalar tensor if targets are provided, else None.
        """
        B, T = idx.size()
        assert T <= self.config.context_length, f"Sequence length {T} exceeds context {self.config.context_length}"

        # 1. Start the residual stream: (B, T) -> (B, T, C)
        x = self.embeddings(idx)

        # 2. Pass through all blocks (Information is added to the stream sequentially)
        for block in self.blocks:
            x = block(x, use_flash=use_flash)
                
        # 3. Final LayerNorm
        x = self.ln_f(x)
        
        # 4. Output Projection
        # (B, T, Vocab_Size)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            # Flatten (B, T, Vocab_Size) -> (B * T, Vocab_Size)
            logits_flat = logits.view(-1, logits.size(-1))
            # Flatten (B, T) -> (B * T)
            targets_flat = targets.view(-1)
            
            # CrossEntropyLoss expects logits and integer targets
            loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
        
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: int = None):
        """
        Auto-regressively generates new tokens.
        Args:
            idx: (Batch, Time) sequence of current contexts.
            max_new_tokens: Number of tokens to generate.
            temperature: Scales logits before softmax. T < 1.0 reduces randomness. T=0.0 is greedy.
            top_k: If set, only samples from the top K most probable tokens.
        Returns:
            idx: (Batch, Time + max_new_tokens) sequence of generated tokens.
        """
        self.eval()
        for _ in range(max_new_tokens):
            # Crop the sequence to the maximum context length
            idx_cond = idx if idx.size(1) <= self.config.context_length else idx[:, -self.config.context_length:]
            
            # Forward pass to get predictions
            logits, _ = self(idx_cond, targets=None)
            
            # Pluck the logits at the final step and scale by temperature
            logits = logits[:, -1, :] # (Batch, Vocab_Size)
            
            if temperature == 0.0:
                # Greedy Decoding (argmax)
                _, idx_next = torch.topk(logits, k=1, dim=-1)
            else:
                logits = logits / temperature
                
                # Optionally crop the logits to only the top k options
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                
                # Apply softmax to convert logits to (normalized) probabilities
                probs = torch.nn.functional.softmax(logits, dim=-1)
                
                # Sample from the distribution
                idx_next = torch.multinomial(probs, num_samples=1)
                
            # Append the sampled token to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)
            
        return idx
