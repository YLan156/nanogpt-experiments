"""Finalize Task 4 after a training process that failed during W&B table upload."""
import json
import math
import os
import random
import tempfile
import time

# Keep W&B media files inside the project so external temp cleanup cannot remove them.
LOCAL_TMP = os.path.abspath(os.path.join('out-task4-finetuned', 'wandb-tmp'))
os.makedirs(LOCAL_TMP, exist_ok=True)
os.environ['TEMP'] = LOCAL_TMP
os.environ['TMP'] = LOCAL_TMP
tempfile.tempdir = LOCAL_TMP

import numpy as np
import torch
import wandb
from transformers import AutoTokenizer

from model import GPT

DATA_DIR = 'data/shakespeare'
CHECKPOINT_PATH = 'out-task4-finetuned/ckpt.pt'
REPORT_PATH = 'reports/task4_finetune.json'
RUN_ID = '1s5n392c'
RUN_URL = 'https://wandb.ai/ylan156-hong-kong-university-of-science-and-technology/shakespeare-char-task4/runs/1s5n392c'
DEVICE = 'cpu'
SEED = 1337
BATCH_SIZE = 1
BLOCK_SIZE = 1024
EVAL_ITERS = 20
MAX_NEW_TOKENS = 80
TEMPERATURE = 0.8
TOP_K = 40
PROMPTS = [
    'Once upon a time',
    'In the middle of the night',
    'The most important lesson I learned was',
    'Scientists discovered a strange signal from',
    'At the edge of the old forest',
]


def set_seed(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)


def get_batch(data, generator):
    ix = torch.randint(len(data) - BLOCK_SIZE - 1, (BATCH_SIZE,), generator=generator)
    x = torch.stack([torch.from_numpy(data[i:i + BLOCK_SIZE].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + BLOCK_SIZE].astype(np.int64)) for i in ix])
    return x.to(DEVICE), y.to(DEVICE)


@torch.no_grad()
def evaluate(model, data):
    model.eval()
    generator = torch.Generator().manual_seed(SEED + 1)
    start = time.perf_counter()
    losses = []
    tokens = 0
    for _ in range(EVAL_ITERS):
        x, y = get_batch(data, generator)
        _, loss = model(x, y)
        losses.append(loss.item())
        tokens += y.numel()
    elapsed = time.perf_counter() - start
    model.train()
    loss = float(np.mean(losses))
    return {
        'validation_loss': loss,
        'perplexity': float(math.exp(loss)),
        'inference_seconds': elapsed,
        'tokens_per_second': tokens / elapsed,
        'peak_memory_mb': 0.0,
    }


@torch.no_grad()
def generate(model, tokenizer):
    model.eval()
    rows = []
    for sample_id, prompt in enumerate(PROMPTS):
        set_seed(SEED + sample_id)
        ids = tokenizer(prompt, return_tensors='pt')['input_ids'].to(DEVICE)
        start = time.perf_counter()
        output = model.generate(ids, max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE, top_k=TOP_K)
        elapsed = time.perf_counter() - start
        new_ids = output[0, ids.shape[1]:].tolist()
        rows.append({
            'sample_id': sample_id,
            'prompt': prompt,
            'generated_text': tokenizer.decode(new_ids, skip_special_tokens=True),
            'generated_token_ids': new_ids,
            'tokens_per_second': len(new_ids) / elapsed,
        })
    model.train()
    return rows


def main():
    set_seed(SEED)
    val_data = np.memmap(os.path.join(DATA_DIR, 'val.bin'), dtype=np.uint16, mode='r')
    tokenizer = AutoTokenizer.from_pretrained('openai-community/gpt2')
    baseline = GPT.from_pretrained('gpt2', {'dropout': 0.0}).to(DEVICE).eval()
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    finetuned = GPT.from_pretrained('gpt2', {'dropout': 0.0}).to(DEVICE)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for key, value in list(state_dict.items()):
        if key.startswith(unwanted_prefix):
            state_dict[key[len(unwanted_prefix):]] = state_dict.pop(key)
    finetuned.load_state_dict(state_dict)
    finetuned.eval()

    run = wandb.init(
        project='shakespeare-char-task4',
        id=RUN_ID,
        resume='must',
        tags=['pretrained', 'gpt2', 'finetuned'],
        settings=wandb.Settings(root_dir=os.getcwd()),
    )
    before_metrics = evaluate(baseline, val_data)
    after_metrics = evaluate(finetuned, val_data)
    before_rows = generate(baseline, tokenizer)
    after_rows = generate(finetuned, tokenizer)
    table = wandb.Table(columns=[
        'sample_id', 'prompt', 'before_text', 'after_text',
        'before_token_ids', 'after_token_ids',
        'before_tokens_per_second', 'after_tokens_per_second',
    ])
    for before, after in zip(before_rows, after_rows):
        table.add_data(
            before['sample_id'], before['prompt'], before['generated_text'], after['generated_text'],
            json.dumps(before['generated_token_ids']), json.dumps(after['generated_token_ids']),
            before['tokens_per_second'], after['tokens_per_second'],
        )
    wandb.log({
        'before/validation_loss': before_metrics['validation_loss'],
        'before/perplexity': before_metrics['perplexity'],
        'before/tokens_per_second': before_metrics['tokens_per_second'],
        'after/validation_loss': after_metrics['validation_loss'],
        'after/perplexity': after_metrics['perplexity'],
        'after/tokens_per_second': after_metrics['tokens_per_second'],
        'tables/before_after_generation': table,
        'fine_tuning/best_checkpoint_iter': checkpoint['iter_num'],
        'fine_tuning/best_validation_loss': checkpoint['best_val_loss'],
    }, step=checkpoint['iter_num'] + 1)
    report = {
        'wandb_run_url': RUN_URL,
        'before': {'metrics': before_metrics, 'samples': before_rows},
        'after': {'metrics': after_metrics, 'samples': after_rows},
        'best_checkpoint_iter': checkpoint['iter_num'],
        'best_val_loss': checkpoint['best_val_loss'],
    }
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, 'w', encoding='utf-8') as output_file:
        json.dump(report, output_file, ensure_ascii=False, indent=2)
    print(f'W&B run: {RUN_URL}')
    print('W&B generation Table: tables/before_after_generation')
    print('W&B model Artifact: https://wandb.ai/ylan156-hong-kong-university-of-science-and-technology/shakespeare-char-task4/artifacts/model/task4-finetuned-gpt2-best-checkpoint/v3')
    print(f'Saved report to {REPORT_PATH}')
    wandb.finish()


if __name__ == '__main__':
    main()
