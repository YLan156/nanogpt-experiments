import os
import requests
import tiktoken
import numpy as np
from collections import Counter

# download the tiny shakespeare dataset
input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')
if not os.path.exists(input_file_path):
    data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    with open(input_file_path, 'w', encoding='utf-8') as f:
        f.write(requests.get(data_url).text)

with open(input_file_path, 'r', encoding='utf-8') as f:
    data = f.read()
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

# encode with tiktoken gpt2 bpe
enc = tiktoken.get_encoding("gpt2")
train_ids = enc.encode_ordinary(train_data)
val_ids = enc.encode_ordinary(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

def print_token_frequencies(name, token_ids, limit=20):
    counts = Counter(token_ids)
    print(f"{name} top {limit} token frequencies:")
    for token_id, count in counts.most_common(limit):
        token_text = enc.decode([token_id]).replace("\n", "\\n")
        print(f"  id={token_id:>5} count={count:>6} token={token_text!r}")

print_token_frequencies("train", train_ids)
print_token_frequencies("val", val_ids)

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

train_bin_path = os.path.join(os.path.dirname(__file__), 'train.bin')
val_bin_path = os.path.join(os.path.dirname(__file__), 'val.bin')
print(f"train.bin size: {os.path.getsize(train_bin_path):,} bytes")
print(f"val.bin size: {os.path.getsize(val_bin_path):,} bytes")

# train.bin has 301,966 tokens
# val.bin has 36,059 tokens
