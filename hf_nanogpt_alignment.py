"""Verify that nanoGPT GPT-2 weights align with Hugging Face GPT-2."""

import json
import os
import random
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from model import GPT


MODEL_NAME = "openai-community/gpt2"
NANOGPT_MODEL_NAME = "gpt2"
SEED = 1337
TEMPERATURE = 0.8
TOP_K = 200
MAX_NEW_TOKENS = 100
OUTPUT_PATH = "reports/hf_nanogpt_alignment.json"

PROMPTS = [
    "Once upon a time",
    "In the middle of the night",
    "The most important lesson I learned was",
    "Scientists discovered a strange signal from",
    "At the edge of the old forest",
]

TRANSPOSED_SUFFIXES = (
    "attn.c_attn.weight",
    "attn.c_proj.weight",
    "mlp.c_fc.weight",
    "mlp.c_proj.weight",
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def greedy_generate_nanogpt(model, input_ids, max_new_tokens):
    tokens = input_ids.clone()
    for _ in range(max_new_tokens):
        context = tokens[:, -model.config.block_size:]
        logits, _ = model(context)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        tokens = torch.cat((tokens, next_token), dim=1)
    return tokens


def parameter_report(hf_model, nano_model):
    hf_params = dict(hf_model.named_parameters())
    nano_params = dict(nano_model.named_parameters())
    names = sorted(set(hf_params) | set(nano_params))
    shape_rows = []
    exact_matches = 0
    transpose_matches = 0
    mismatches = []

    for name in names:
        hf_shape = tuple(hf_params[name].shape) if name in hf_params else None
        nano_shape = tuple(nano_params[name].shape) if name in nano_params else None
        exact = hf_shape == nano_shape and hf_shape is not None
        transposed = (
            name.endswith(TRANSPOSED_SUFFIXES)
            and hf_shape is not None
            and nano_shape is not None
            and hf_shape[::-1] == nano_shape
        )
        if exact:
            exact_matches += 1
        elif transposed:
            transpose_matches += 1
        else:
            mismatches.append({"name": name, "hf_shape": hf_shape, "nanogpt_shape": nano_shape})
        shape_rows.append({
            "name": name,
            "hf_shape": hf_shape,
            "nanogpt_shape": nano_shape,
            "exact_match": exact,
            "transpose_match": transposed,
        })

    return {
        "hf_parameter_count": sum(p.numel() for p in hf_model.parameters()),
        "nanogpt_parameter_count": sum(p.numel() for p in nano_model.parameters()),
        "parameter_count_equal": sum(p.numel() for p in hf_model.parameters()) == sum(p.numel() for p in nano_model.parameters()),
        "parameter_tensor_count_hf": len(hf_params),
        "parameter_tensor_count_nanogpt": len(nano_params),
        "exact_shape_matches": exact_matches,
        "transpose_shape_matches": transpose_matches,
        "shape_mismatches": mismatches,
        "shapes": shape_rows,
    }


def main() -> None:
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    hf_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device).eval()
    nanogpt_model = GPT.from_pretrained(NANOGPT_MODEL_NAME, {"dropout": 0.0}).to(device).eval()

    report = {
        "model": MODEL_NAME,
        "nanogpt_loader": "GPT.from_pretrained(gpt2)",
        "device": str(device),
        "seed": SEED,
        "generation": {
            "mode": "greedy",
            "temperature": TEMPERATURE,
            "top_k": TOP_K,
            "max_new_tokens": MAX_NEW_TOKENS,
        },
        "parameters": parameter_report(hf_model, nanogpt_model),
        "prompts": [],
    }

    for sample_id, prompt in enumerate(PROMPTS):
        set_seed(SEED + sample_id)
        encoded = tokenizer(prompt, return_tensors="pt")
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        with torch.no_grad():
            hf_logits = hf_model(input_ids=input_ids, attention_mask=attention_mask).logits[:, -1:, :]
            nano_logits, _ = nanogpt_model(input_ids)

        diff = (hf_logits.float() - nano_logits.float()).abs()
        start = time.perf_counter()
        with torch.no_grad():
            hf_generated = hf_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=False,
                max_new_tokens=MAX_NEW_TOKENS,
                pad_token_id=tokenizer.eos_token_id,
            )
        hf_time = time.perf_counter() - start

        start = time.perf_counter()
        nano_generated = greedy_generate_nanogpt(nanogpt_model, input_ids, MAX_NEW_TOKENS)
        nano_time = time.perf_counter() - start

        prompt_length = input_ids.shape[1]
        hf_new_ids = hf_generated[0, prompt_length:].tolist()
        nano_new_ids = nano_generated[0, prompt_length:].tolist()

        report["prompts"].append({
            "sample_id": sample_id,
            "prompt": prompt,
            "prompt_token_ids": input_ids[0].tolist(),
            "hf_generated_token_ids": hf_new_ids,
            "nanogpt_generated_token_ids": nano_new_ids,
            "generated_token_ids_equal": hf_new_ids == nano_new_ids,
            "hf_generated_text": tokenizer.decode(hf_new_ids, skip_special_tokens=True),
            "nanogpt_generated_text": tokenizer.decode(nano_new_ids, skip_special_tokens=True),
            "first_forward_logits_max_abs_error": diff.max().item(),
            "first_forward_logits_mean_abs_error": diff.mean().item(),
            "hf_generation_seconds": hf_time,
            "nanogpt_generation_seconds": nano_time,
        })

    report["summary"] = {
        "all_token_ids_equal": all(row["generated_token_ids_equal"] for row in report["prompts"]),
        "max_first_forward_logits_abs_error": max(row["first_forward_logits_max_abs_error"] for row in report["prompts"]),
        "mean_first_forward_logits_abs_error": sum(row["first_forward_logits_mean_abs_error"] for row in report["prompts"]) / len(report["prompts"]),
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as output_file:
        json.dump(report, output_file, ensure_ascii=False, indent=2)
    print(json.dumps(report["summary"], indent=2))
    print(f"Saved alignment report to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
