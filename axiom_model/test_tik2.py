import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from tokenizer.tokenizer import Tokenizer, SPECIAL_TOKENS

tok = Tokenizer(special_tokens=SPECIAL_TOKENS)
prompt = "<|system|>\nYou are Axiom, a helpful AI assistant.<|end|>\n<|user|>\nGive me three tips for staying healthy.<|end|>\n<|assistant|>\n"
tokens = tok.encode(prompt)
print('Tokens:', tokens)
print('Decoded:', tok.decode(tokens))
