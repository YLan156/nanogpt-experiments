import os
import pickle
import csv
from collections import Counter
import matplotlib.pyplot as plt

# read the input text
with open('input.txt', 'r', encoding='utf-8') as f:
    data = f.read()

print(f"总原始字符数：{len(data)}")

# get all unique characters
chars = sorted(list(set(data)))
vocab_size = len(chars)
print(f"vocabulary size: {vocab_size}")

# create mapping
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}

# ========= encode / decode 函数 =========
def encode(text):
    """text -> token id list"""
    return [stoi[c] for c in text]

def decode(ids):
    """token id list -> text"""
    return ''.join([itos[i] for i in ids])

# -------- 测试encode decode --------
test_text = "Hello World!"
test_ids = encode(test_text)
decoded_text = decode(test_ids)
print("\n==== encode/decode 双向测试 ====")
print(f"原始文本: {test_text}")
print(f"encode结果: {test_ids}")
print(f"decode还原: {decoded_text}")
assert decoded_text == test_text, "encode decode 双向转换失败！"

# full token ids
full_ids = encode(data)

# split train / val
n = int(0.9 * len(full_ids))
train_ids = full_ids[:n]
val_ids = full_ids[n:]

print(f"\ntrain token数量: {len(train_ids)}")
print(f"val token数量: {len(val_ids)}")
print(f"train占比: {len(train_ids)/len(full_ids):.4f}，val占比: {len(val_ids)/len(full_ids):.4f}")

# =========统计每个token id频次========
counter = Counter(full_ids)
total_tokens = len(full_ids)
token_stats = []
for token_id in range(vocab_size):
    cnt = counter.get(token_id, 0)
    freq = cnt / total_tokens
    char = itos[token_id]
    token_stats.append({"token_id":token_id, "char":char, "count":cnt, "frequency":freq})

# 按count降序排序
token_stats_sorted = sorted(token_stats, key=lambda x:x["count"], reverse=True)

# 打印Top30
print("\n==== Top30 token ====")
print(f"{'token_id':<10}{'char':<10}{'count':<12}{'frequency'}")
for item in token_stats_sorted[:30]:
    print(f"{item['token_id']:<10}{repr(item['char']):<10}{item['count']:<12}{item['frequency']:.6f}")

# =========保存csv reports/token_frequency_char.csv =========
csv_path = "../../reports/token_frequency_char.csv"
with open(csv_path, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["token_id","char","count","frequency"])
    writer.writeheader()
    writer.writerows(token_stats_sorted)
print(f"\n完整统计已保存到 {csv_path}")

# =========绘制Top30柱状图 reports/token_frequency_char.png =========
top30 = token_stats_sorted[:30]
ids_top30 = [x["token_id"] for x in top30]
counts_top30 = [x["count"] for x in top30]
chars_top30 = [repr(x["char"]) for x in top30]

plt.figure(figsize=(14,6))
plt.bar(range(len(top30)), counts_top30)
plt.xticks(range(len(top30)), chars_top30, rotation=60)
plt.xlabel("token char")
plt.ylabel("count")
plt.title("Top30 token frequency (char‑level)")
plt.tight_layout()
plt.savefig("../../reports/token_frequency_char.png", dpi=150)
plt.close()
print("Top30柱状图已保存 reports/token_frequency_char.png")

# =========保存原始meta.pkl（原有逻辑不动）========
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}
with open('meta.pkl', 'wb') as f:
    pickle.dump(meta, f)

import numpy as np
train_ids_np = np.array(train_ids, dtype=np.uint16)
val_ids_np = np.array(val_ids, dtype=np.uint16)
train_ids_np.tofile('train.bin')
val_ids_np.tofile('val.bin')
print("\n prepare.py执行完毕，train.bin val.bin meta.pkl生成完成")
