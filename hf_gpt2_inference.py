"""Run fixed-prompt inference with Hugging Face GPT-2."""

import json
import os
import random
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "openai-community/gpt2"
SEED = 1337
TEMPERATURE = 0.8
TOP_K = 200
MAX_NEW_TOKENS = 100
NUM_SAMPLES = 5
OUTPUT_PATH = "reports/hf_gpt2_predictions.jsonl"

PROMPTS = [
    "Once upon a time",
    "In the middle of the night",
    "The most important lesson I learned was",
    "Scientists discovered a strange signal from",
    "At the edge of the old forest",
]

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main() -> None:
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device)
    model.eval()
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as output_file:
        for sample_id, prompt in enumerate(PROMPTS[:NUM_SAMPLES]):
            set_seed(SEED + sample_id)
            inputs = tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.synchronize(device)
            start_time = time.perf_counter()
            with torch.no_grad():
                generated = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    do_sample=True,
                    temperature=TEMPERATURE,
                    top_k=TOP_K,
                    max_new_tokens=MAX_NEW_TOKENS,
                    pad_token_id=tokenizer.pad_token_id,
                )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed_sec = time.perf_counter() - start_time
            prompt_length = input_ids.shape[1]
            generated_ids = generated[0, prompt_length:].tolist()
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            peak_memory_mb = None
            if device.type == "cuda":
                peak_memory_mb = torch.cuda.max_memory_allocated(device) / 1024 / 1024
            record = {
                "sample_id": sample_id,
                "model": MODEL_NAME,
                "seed": SEED + sample_id,
                "prompt": prompt,
                "prompt_token_ids": input_ids[0].tolist(),
                "generated_token_ids": generated_ids,
                "generated_text": generated_text,
                "temperature": TEMPERATURE,
                "top_k": TOP_K,
                "max_new_tokens": MAX_NEW_TOKENS,
                "generated_tokens": len(generated_ids),
                "elapsed_sec": elapsed_sec,
                "tokens_per_sec": len(generated_ids) / elapsed_sec,
                "peak_memory_mb": peak_memory_mb,
            }
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"sample={sample_id}: {prompt}{generated_text}")
            print(f"speed={record['tokens_per_sec']:.2f} tokens/s, peak_memory={peak_memory_mb} MB")
    print(f"Saved {NUM_SAMPLES} predictions to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
