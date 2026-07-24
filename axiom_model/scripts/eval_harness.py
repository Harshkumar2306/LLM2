"""
Axiom LLM Evaluation Harness
Custom wrapper to bridge our pure PyTorch model with the industry-standard lm-evaluation-harness.
"""
import os
import sys
import json
import torch
import tiktoken
import argparse
import yaml
import inspect

# Ensure we can import from the parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.gpt_config import GPTConfig
from config.enums import FFNType, AttentionType, PositionType, NormType
from models.model import GPT

# Import lm-eval
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
import lm_eval

@register_model("axiom")
class AxiomLM(LM):
    def __init__(self, checkpoint_path, yaml_path, device="cuda", max_length=1024):
        super().__init__()
        self._device = device
        self._max_length = max_length
        self.enc = tiktoken.get_encoding("gpt2")
        
        print(f"Loading checkpoint: {checkpoint_path}")
        with open(yaml_path, "r") as f:
            raw_config = yaml.safe_load(f)
            
        valid_keys = inspect.signature(GPTConfig).parameters.keys()
        config_kwargs = {k: v for k, v in raw_config.items() if k in valid_keys}
        
        config_kwargs['attention_type'] = AttentionType(config_kwargs['attention_type'])
        config_kwargs['ffn_type'] = FFNType(config_kwargs['ffn_type'])
        config_kwargs['norm_type'] = NormType(config_kwargs['norm_type'])
        config_kwargs['position_type'] = PositionType(config_kwargs['position_type'])
        
        config = GPTConfig(**config_kwargs)
        self.model = GPT(config)
        
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state'])
        self.model.to(device)
        self.model.eval()

    @property
    def eot_token_id(self):
        return self.enc.eot_token

    @property
    def max_length(self):
        return self._max_length

    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        return 1

    @property
    def device(self):
        return self._device

    def tok_encode(self, string: str, **kwargs):
        return self.enc.encode(string)

    def tok_decode(self, tokens):
        return self.enc.decode(tokens)

    def loglikelihood(self, requests):
        """
        Required for multiple-choice benchmarks (HellaSwag, PIQA, ARC)
        Calculates the probability of a continuation given a context.
        """
        res = []
        with torch.inference_mode():
            for req in requests:
                context, continuation = req.args
                
                ctx_enc = self.tok_encode(context)
                cont_enc = self.tok_encode(continuation)
                
                # If context is empty, model predicts from start
                if len(ctx_enc) == 0:
                    ctx_enc = [self.eot_token_id]
                    
                inp = torch.tensor([ctx_enc + cont_enc], dtype=torch.long, device=self.device)
                
                if inp.size(1) > self.max_length:
                    inp = inp[:, -self.max_length:]
                    
                logits = self.model(inp, targets=None)[0] # (1, seq_len, vocab)
                
                cont_start = len(ctx_enc) - 1
                cont_end = cont_start + len(cont_enc)
                
                logits_for_cont = logits[0, cont_start:cont_end, :]
                log_probs = torch.nn.functional.log_softmax(logits_for_cont, dim=-1)
                
                cont_tensor = torch.tensor(cont_enc, dtype=torch.long, device=self.device)
                token_log_probs = log_probs.gather(dim=-1, index=cont_tensor.unsqueeze(-1)).squeeze(-1)
                
                is_greedy = (logits_for_cont.argmax(dim=-1) == cont_tensor).all().item()
                
                res.append((token_log_probs.sum().item(), is_greedy))
                
        return res

    def loglikelihood_rolling(self, requests):
        """Required for Perplexity (WikiText)"""
        res = []
        with torch.inference_mode():
            for req in requests:
                string = req.args[0]
                tokens = self.tok_encode(string)
                if len(tokens) == 0:
                    res.append(0.0)
                    continue
                    
                inp = torch.tensor([tokens], dtype=torch.long, device=self.device)
                if inp.size(1) > self.max_length:
                    inp = inp[:, -self.max_length:]
                    
                logits = self.model(inp, targets=None)[0]
                log_probs = torch.nn.functional.log_softmax(logits[0, :-1, :], dim=-1)
                target_tokens = inp[0, 1:]
                
                token_log_probs = log_probs.gather(dim=-1, index=target_tokens.unsqueeze(-1)).squeeze(-1)
                res.append(token_log_probs.sum().item())
                
        return res

    def generate_until(self, requests):
        """Required for HumanEval (Code Generation)"""
        res = []
        with torch.inference_mode():
            for req in requests:
                context = req.args[0]
                until = req.args[1].get("until", [self.enc.decode([self.eot_token_id])])
                if isinstance(until, str):
                    until = [until]
                    
                x = torch.tensor([self.tok_encode(context)], dtype=torch.long, device=self.device)
                generated = []
                
                for _ in range(self.max_gen_toks):
                    x_cond = x if x.size(1) <= self.max_length else x[:, -self.max_length:]
                    logits = self.model(x_cond, targets=None)[0][:, -1, :]
                    
                    next_token = logits.argmax(dim=-1).item()
                    generated.append(next_token)
                    x = torch.cat((x, torch.tensor([[next_token]], device=self.device)), dim=1)
                    
                    gen_text = self.tok_decode(generated)
                    if any(stop in gen_text for stop in until):
                        break
                        
                res.append(self.tok_decode(generated))
        return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Axiom Evaluation Harness")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to best.pt")
    parser.add_argument('--yaml', type=str, required=True, help="Path to run_config.yaml")
    parser.add_argument('--tasks', type=str, default="hellaswag", help="Comma-separated list of tasks")
    parser.add_argument('--limit', type=int, default=None, help="Limit number of examples")
    args = parser.parse_args()

    print(f"Initializing AxiomLM with tasks: {args.tasks}")
    
    axiom_lm = AxiomLM(
        checkpoint_path=args.checkpoint,
        yaml_path=args.yaml,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    results = lm_eval.simple_evaluate(model=axiom_lm, tasks=args.tasks.split(","), limit=args.limit, batch_size=1, confirm_run_unsafe_code=True)
    
    from lm_eval.utils import make_table
    print(make_table(results))
    
    os.makedirs("evaluation", exist_ok=True)
    with open("evaluation/summary.json", "w") as f:
        json.dump(results["results"], f, indent=4)
        
    print("Evaluation complete! Results saved to evaluation/summary.json")
