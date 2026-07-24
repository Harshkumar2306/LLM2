import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from tokenizer.tokenizer import Tokenizer, SPECIAL_TOKENS

tok = Tokenizer(special_tokens=SPECIAL_TOKENS)
try:
    print('Decoding standard token:', tok.decode([198]))
except Exception as e:
    print('Error standard:', e)

try:
    print('Decoding special token:', tok.decode([50260]))
except Exception as e:
    print('Error special:', type(e), e)
