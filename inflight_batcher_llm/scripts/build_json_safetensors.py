#!/usr/bin/env python3
# build_json_safetensors.py
# Copyright 2024 DeepInfra, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

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
        s = tokenizer.decode([example_single_char_token_id, i])[1:]
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
    with tempfile.NamedTemporaryFile(suffix='.strings', mode='ab') as f:
        f.write(combined_strs)
        f.flush()
        json_safetensors_cli = os.path.join(os.path.dirname(sys.argv[0]), "../build/json_safetensors_cli")
        out_safetensors_file = sys.argv[1] + "/json.safetensors"
        cmdline = [json_safetensors_cli, f.name, out_safetensors_file]
        print("Running " + " ".join(cmdline))
        retval = subprocess.check_call(cmdline)
        print("\n *** Building %s finished: %s" % (out_safetensors_file, retval))
        cmdline = [json_safetensors_cli, out_safetensors_file]
        print("\nTesting " + " ".join(cmdline))
        test_out_bytes = subprocess.check_output(cmdline)
        retval = b"Completed Test" in test_out_bytes
        if retval:
            print("\n *** Testing %s succeeded" % (out_safetensors_file))
        else:
            print("\n *** Test %s failed: %s" % (out_safetensors_file, test_out_bytes.decode("UTF-8", "ignore")))
