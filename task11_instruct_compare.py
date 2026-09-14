"""Compare a small instruct model with GPT-2 on the same Chinese questions."""

import json
import os
import random
import re
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


QWEN_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
GPT2_MODEL = "openai-community/gpt2"
SEED = 1337
TEMPERATURE = 0.7
TOP_K = 50
MAX_NEW_TOKENS = 120
OUTPUT_PATH = "reports/task11_instruct_comparison.json"
QWEN_OUTPUT_PATH = "reports/task11_qwen_results.jsonl"
GPT2_OUTPUT_PATH = "reports/task11_gpt2_results.jsonl"
SUMMARY_OUTPUT_PATH = "reports/task11_comparison.json"

QUESTIONS = [
    "请用一句话解释什么是机器学习。",
    "请列出学习 Python 的三个建议。",
    "为什么天空通常是蓝色的？",
    "请计算：23 + 19 等于多少？只给出计算结果和一句简短说明。",
    "请写一个不超过 50 个汉字的周末学习计划。",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_one(model, tokenizer, prompt_ids, attention_mask, device):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    start_time = time.perf_counter()

    with torch.no_grad():
        output = model.generate(
            input_ids=prompt_ids,
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
    prompt_length = prompt_ids.shape[1]
    generated_ids = output[0, prompt_length:].tolist()
    generated_text = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    ).strip()

    if device.type == "cuda":
        peak_memory_mb = (
            torch.cuda.max_memory_allocated(device) / 1024 / 1024
        )
    else:
        peak_memory_mb = None

    return {
        "prompt_token_ids": prompt_ids[0].tolist(),
        "generated_token_ids": generated_ids,
        "generated_text": generated_text,
        "generated_tokens": len(generated_ids),
        "elapsed_sec": elapsed_sec,
        "tokens_per_sec": len(generated_ids) / elapsed_sec if elapsed_sec else 0.0,
        "peak_memory_mb": peak_memory_mb,
    }


def score_answer(question_id, text):
    chinese_count = len(re.findall(r"[\u4e00-\u9fff]", text))
    useful_count = len(re.findall(r"[\u4e00-\u9fffA-Za-z0-9]", text))
    chinese_ratio = chinese_count / useful_count if useful_count else 0.0
    instruction = 0.0
    completeness = 0.0

    if question_id == 0:
        instruction = 1.0 if 0 < len(text) <= 45 else 0.0
        completeness = 1.0 if any(word in text for word in ["数据", "模型", "规律", "预测"]) else 0.0
    elif question_id == 1:
        items = len(re.findall(r"(?:^|\n)\s*(?:[-*]|[1-3][.)、])", text))
        instruction = 1.0 if items >= 3 else 0.0
        completeness = instruction
    elif question_id == 2:
        instruction = 1.0 if text else 0.0
        completeness = 1.0 if any(word in text for word in ["散射", "光", "大气", "波长"]) else 0.0
    elif question_id == 3:
        instruction = 1.0 if "42" in text else 0.0
        completeness = instruction
    elif question_id == 4:
        instruction = 1.0 if 0 < len(text) <= 50 else 0.0
        completeness = 1.0 if any(word in text for word in ["学习", "阅读", "复习", "计划"]) else 0.0

    return {
        "instruction_following": instruction,
        "chinese_expression": round(chinese_ratio, 4),
        "answer_completeness": completeness,
    }


def run_qwen(device):
    tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL)
    model = AutoModelForCausalLM.from_pretrained(QWEN_MODEL).to(device).eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    rows = []

    for question_id, question in enumerate(QUESTIONS):
        set_seed(SEED + question_id)
        messages = [
            {"role": "system", "content": "你是一个准确、简洁的中文助手。"},
            {"role": "user", "content": question},
        ]

        # apply_chat_template may return BatchEncoding in some transformers versions.
        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        prompt_ids = encoded["input_ids"].to(device)

        if "attention_mask" in encoded:
            attention_mask = encoded["attention_mask"].to(device)
        else:
            attention_mask = torch.ones_like(prompt_ids)

        row = generate_one(
            model,
            tokenizer,
            prompt_ids,
            attention_mask,
            device,
        )
        row.update({
            "question_id": question_id,
            "question": question,
            "chat_template": True,
        })
        row["scores"] = score_answer(question_id, row["generated_text"])
        rows.append(row)

    return rows


def run_gpt2(device):
    tokenizer = AutoTokenizer.from_pretrained(GPT2_MODEL)
    model = AutoModelForCausalLM.from_pretrained(GPT2_MODEL).to(device).eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    rows = []

    for question_id, question in enumerate(QUESTIONS):
        set_seed(SEED + question_id)
        encoded = tokenizer(question, return_tensors="pt")
        prompt_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        row = generate_one(
            model,
            tokenizer,
            prompt_ids,
            attention_mask,
            device,
        )
        row.update({
            "question_id": question_id,
            "question": question,
            "chat_template": False,
        })
        row["scores"] = score_answer(question_id, row["generated_text"])
        rows.append(row)

    return rows


def aggregate(rows):
    names = [
        "instruction_following",
        "chinese_expression",
        "answer_completeness",
    ]
    return {
        name: round(
            sum(row["scores"][name] for row in rows) / len(rows),
            4,
        )
        for name in names
    }


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    qwen_rows = run_qwen(device)
    gpt2_rows = run_gpt2(device)

    report = {
        "device": str(device),
        "seed": SEED,
        "generation": {
            "temperature": TEMPERATURE,
            "top_k": TOP_K,
            "max_new_tokens": MAX_NEW_TOKENS,
            "sampling": True,
        },
        "questions": QUESTIONS,
        "qwen": {
            "model": QWEN_MODEL,
            "uses_apply_chat_template": True,
            "results": qwen_rows,
            "average_scores": aggregate(qwen_rows),
        },
        "gpt2": {
            "model": GPT2_MODEL,
            "uses_apply_chat_template": False,
            "results": gpt2_rows,
            "average_scores": aggregate(gpt2_rows),
        },
        "scoring_note": "Scores are transparent heuristic behavior checks, not validation loss or human evaluation.",
    }

    os.makedirs("reports", exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    write_jsonl(QWEN_OUTPUT_PATH, qwen_rows)
    write_jsonl(GPT2_OUTPUT_PATH, gpt2_rows)

    with open(SUMMARY_OUTPUT_PATH, "w", encoding="utf-8") as file:
        json.dump({
            "qwen_average_scores": report["qwen"]["average_scores"],
            "gpt2_average_scores": report["gpt2"]["average_scores"],
            "conclusion": "Qwen is expected to outperform GPT-2 on Chinese instruction following because Qwen is instruction-tuned and uses apply_chat_template, while GPT-2 is a base causal language model.",
            "scoring_note": report["scoring_note"],
        }, file, ensure_ascii=False, indent=2)

    print(json.dumps({
        "qwen": report["qwen"]["average_scores"],
        "gpt2": report["gpt2"]["average_scores"],
    }, ensure_ascii=False, indent=2))
    print("Saved task 11 results to reports/")


if __name__ == "__main__":
    main()
