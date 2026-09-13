"""Sample from a randomly initialized model."""

import json
import os
import pickle
import time
from contextlib import nullcontext

import torch
import tiktoken

from model import GPTConfig, GPT


# -----------------------------------------------------------------------------
# configuration

# 濞达綀娉曢弫銈夋⒕韫囨梹绨氶柛鎺撶箓椤劙宕犻弽顭屼線宕圭€ｅ墎绀夊☉鎾崇Т婵偞娼?checkpoint
init_from = "random"

out_dir = "out"

start = "\n"

# 濞寸姾顕ф慨鐔烘啺娴ｅ湱婀撮柨娑欐皑閺佹捇骞?5 濞戞搩浜濋悧閬嶅嫉?
num_samples = 5
max_new_tokens = 500
temperature = 0.8
top_k = 200
seed = 1337

device = "cpu"
dtype = (
    "bfloat16"
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else "float16"
)

compile = False

# 濞寸姾顕ф慨鐔烘啺娴ｅ湱婀撮柣銊ュ缁额參宕欓悜妯荤€ù?
report_path = "reports/samples_scratch_before.jsonl"

# 闂傚懎绻戝┃鈧柛鎺撶箓椤劙宕犻弽顭屼線宕圭€ｎ剚鐣遍梺鏉跨Ф閻?
# 濠碘€冲€归悘澶嬫媴閻樺灚鐣遍悹渚囧幘缁矂鏌婂鍥╂瀭濞达綀娉曢弫銈嗙閸℃洜鐟濋柛姘灱濞堟垵螣閳ュ磭鈧攱寰勮閻剟鏁嶅畝鍐惧殲闁硅泛锕ㄧ换鏍煂鐏炵偓鏆柟瀛樺姌椤斿嫮绱掗崘顔煎赋缂傚喚鍠曢懙鎴︽儍閸曨偄妫橀柡浣藉焽閳?
random_model_args = dict(
    block_size=256,
    vocab_size=65,
    n_layer=4,
    n_head=4,
    n_embd=128,
    dropout=0.2,
    bias=False,
)

# -----------------------------------------------------------------------------

exec(open("configurator.py").read())

# 闁搞儱鎼悾楣冩⒕韫囨梹绨氱紒澶婄Т閻?
torch.manual_seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

device_type = "cuda" if "cuda" in device else "cpu"

ptdtype = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}[dtype]

ctx = (
    nullcontext()
    if device_type == "cpu"
    else torch.amp.autocast(
        device_type=device_type,
        dtype=ptdtype,
    )
)

# -----------------------------------------------------------------------------
# model

checkpoint = None

if init_from == "random":
    # 闁烩晛鐡ㄧ敮鎾几閸曨垪鍋?GPTConfig闁挎稑鏈啯闁搞劌顑呭顒勫极娴煎瓨顓归柡鍫濇惈閸ㄥ灚鎱ㄧ€ｎ亜顕?
    gptconf = GPTConfig(**random_model_args)
    model = GPT(gptconf)

elif init_from == "resume":
    # 濞?checkpoint 闁告梻濮惧ù鍥熼垾宕団偓?
    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    checkpoint = torch.load(
        ckpt_path,
        map_location=torch.device("cpu"),
    )

    gptconf = GPTConfig(**checkpoint["model_args"])
    model = GPT(gptconf)

    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."

    for key, value in list(state_dict.items()):
        if key.startswith(unwanted_prefix):
            state_dict[key[len(unwanted_prefix):]] = state_dict.pop(key)

    model.load_state_dict(state_dict)

elif init_from.startswith("gpt2"):
    # 濞?GPT-2 濡澘瀚鍕磼閸愨敔渚€宕圭€ｎ亜顫ｉ弶?
    model = GPT.from_pretrained(
        init_from,
        dict(dropout=0.0),
    )

else:
    raise ValueError(f"Unknown init_from: {init_from}")

model.eval()
model.to(device)

if compile:
    model = torch.compile(model)

# -----------------------------------------------------------------------------
# tokenizer

if (
    init_from == "resume"
    and checkpoint is not None
    and "config" in checkpoint
    and "dataset" in checkpoint["config"]
):
    dataset_name = checkpoint["config"]["dataset"]
else:
    dataset_name = "shakespeare_char"

meta_path = os.path.join(
    "data",
    dataset_name,
    "meta.pkl",
)
load_meta = os.path.exists(meta_path)

if load_meta:
    print(f"Loading meta from {meta_path}...")

    with open(meta_path, "rb") as file:
        meta = pickle.load(file)

    stoi = meta["stoi"]
    itos = meta["itos"]

    encode = lambda text: [stoi[c] for c in text]
    decode = lambda tokens: "".join([itos[i] for i in tokens])

else:
    print("No meta.pkl found, assuming GPT-2 encodings...")

    enc = tiktoken.get_encoding("gpt2")

    encode = lambda text: enc.encode(
        text,
        allowed_special={"<|endoftext|>"},
    )
    decode = lambda tokens: enc.decode(tokens)

# -----------------------------------------------------------------------------
# prompt

if start.startswith("FILE:"):
    with open(start[5:], "r", encoding="utf-8") as file:
        start = file.read()

prompt_text = start

start_ids = encode(start)

x = torch.tensor(
    start_ids,
    dtype=torch.long,
    device=device,
)[None, ...]

# -----------------------------------------------------------------------------
# generate samples before training

os.makedirs("reports", exist_ok=True)

records = []

with torch.no_grad():
    with ctx:
        for sample_id in range(num_samples):
            # 婵絽绻嬮柌婊堝冀闁垮鎷卞ù锝堟硶閺併倗娑甸鑲╂毎闁汇劌瀚€氼厾绮?seed
            sample_seed = seed + sample_id
            torch.manual_seed(sample_seed)

            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(sample_seed)
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()

            start_time = time.perf_counter()

            y = model.generate(
                x,
                max_new_tokens,
                temperature=temperature,
                top_k=top_k,
            )

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            elapsed_sec = time.perf_counter() - start_time

            # 闁告瑯浜炵划铏规媼閳╁啯鐓€闁汇垻鍠愰崹姘舵儍?token闁挎稑濂旂粭澶岀磼閻旀椿鍚€ prompt
            generated_ids = y[0, x.shape[1]:].tolist()
            generated_text = decode(generated_ids)

            generated_tokens = len(generated_ids)

            if elapsed_sec > 0:
                tokens_per_sec = generated_tokens / elapsed_sec
            else:
                tokens_per_sec = 0.0

            if device_type == "cuda":
                peak_memory_mb = (
                    torch.cuda.max_memory_allocated() / 1024 / 1024
                )
            else:
                peak_memory_mb = None

            record = {
                "sample_id": sample_id,
                "seed": sample_seed,
                "prompt": prompt_text,
                "text": generated_text,
                "temperature": temperature,
                "top_k": top_k,
                "max_new_tokens": max_new_tokens,
                "generated_tokens": generated_tokens,
                "elapsed_sec": elapsed_sec,
                "tokens_per_sec": tokens_per_sec,
                "peak_memory_mb": peak_memory_mb,
            }

            records.append(record)

            print(prompt_text + generated_text)
            print(
                f"sample={sample_id} "
                f"speed={tokens_per_sec:.2f} tokens/s "
                f"peak_memory={peak_memory_mb} MB"
            )
            print("-" * 60)

# 濞ｅ洦绻傞悺銊︾▔?JSONL
with open(report_path, "w", encoding="utf-8") as file:
    for record in records:
        file.write(
            json.dumps(
                record,
                ensure_ascii=False,
            )
            + "\n"
        )

print(f"Saved {len(records)} samples to {report_path}")