#!/usr/bin/env python3
from transformers import AutoTokenizer

import os
import subprocess
import sys
import tempfile

def get_token_strings(tokenizer, vocab_size):
    token_strings = []
    example_single_char_token_id = -1
    for i in range(vocab_size):
        if tokenizer.convert_ids_to_tokens(i) == '{':
            example_single_char_token_id = i
    for i in range(vocab_size):
        s = tokenizer.decode([example_single_char_token_id, i, example_single_char_token_id])[1:-1]
        if i in tokenizer.all_special_ids or i == tokenizer.eos_token_id:
            s = ''
        token_strings.append(s.replace('\u0000', '\u0001').replace('\r\n', '\n').replace('\r','\n'))
    token_strings[len(token_strings):vocab_size] = ['\u0001'] * max(0, vocab_size - len(token_strings))
    return token_strings

if __name__=='__main__':
    tok = AutoTokenizer.from_pretrained(sys.argv[1])
    strs = get_token_strings(tok, len(tok))
    #import json
    #print(json.dumps(strs))
    combined_strs = '\u0000'.join(strs).encode('UTF-32LE')
    with open(sys.argv[2], 'wb') as outf:
        outf.write(combined_strs)
        outf.flush()
