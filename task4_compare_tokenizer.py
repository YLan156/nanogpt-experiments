import json
import tiktoken
from transformers import GPT2Tokenizer

# -------- 加载tokenizer --------
enc_tiktoken = tiktoken.get_encoding("gpt2")
hf_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# -------- 读取nanoGPT数据集 --------
with open("data/shakespeare/meta.json", "r", encoding="utf-8") as f:
    meta = json.load(f)
stoi = meta["stoi"]
itos = meta["itos"]

with open("data/shakespeare/train.bin", "rb") as f:
    data = f.read()

# 取10段，每段120字符
sample_len = 120
sample_count = 10

sample_texts = []
for i in range(sample_count):
    start = i * sample_len
    end = start + sample_len
    bytes_slice = data[start:end]
    ids = [int(b) for b in bytes_slice]
    text = "".join([itos[idx] for idx in ids])
    sample_texts.append(text)

# -------- 循环对比 --------
for idx, raw_text in enumerate(sample_texts):
    # 关键修复：只清除开头空白，保留中间和末尾
    text_clean = raw_text.lstrip()

    tiktoken_ids = enc_tiktoken.encode(text_clean)
    hf_ids = hf_tokenizer.encode(
        text_clean,
        add_special_tokens=False,
        add_prefix_space=False
    )

    print(f"=====样本 {idx} =====")
    show = text_clean[:60].replace("\n", "\\n")
    print(f"原文片段：{show}...")
    print(f"字符token数量：{len(raw_text)}")
    print(f"tiktoken(gpt2)数量：{len(tiktoken_ids)}")
    print(f"HF‑GPT2数量：{len(hf_ids)}")
    print(f"tiktoken与HF id是否一致：{tiktoken_ids == hf_ids}")
    print()

# -------- 全局统计 --------
total_char = sum(len(t) for t in sample_texts)
total_tok_tik = sum(len(enc_tiktoken.encode(t.lstrip())) for t in sample_texts)
total_tok_hf = sum(len(hf_tokenizer.encode(t.lstrip(), add_special_tokens=False)) for t in sample_texts)

print("=====全局统计====")
print(f"字符tokenizer平均序列长度：{total_char / sample_count:.2f}")
print(f"tiktoken‑gpt2平均序列长度：{total_tok_tik / sample_count:.2f}")
print(f"HF‑GPT2平均序列长度：{total_tok_hf / sample_count:.2f}")
print(f"BPE压缩比(tik/char)：{total_tok_tik / total_char:.4f}")

# 简单调试测试
test_text = "Hello world"
print("\n【调试测试】hf encode test:", hf_tokenizer.encode(test_text, add_special_tokens=False))
# Review note: keep tokenizer comparison outputs deterministic for PR validation.