import os
import requests
import tiktoken
import numpy as np
import pandas as pd

t = 0

# override if explicitly given in terminal
if 't' in globals().items():
    t = globals()['t']

if type(t) is not int or (t > 9):
    print('t incorrect')
    t = 0

# prepare the training data
# collect dataset{t} and samples text files -> encode them using gpt2

direc = 'data/movies/'
name = direc + 'dataset' + str(t) + '.txt'
f = open(name, "r")
text1 = f.read()
print(f'true part has {len(text1.split())} words')

f = open(direc + 'samples.txt', "r")
text2 = f.read()
print(f'synthetic part has {len(text2.split())} words')

text = text1 + '\n' + text2

enc = tiktoken.get_encoding("gpt2")
ids = enc.encode_ordinary(text)
print(f"text has {len(ids):,} tokens")
ids = np.array(ids, dtype=np.uint16)
ids.tofile(os.path.join(direc + 'train.bin'))