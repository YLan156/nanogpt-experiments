"""Task 3: evaluate nanoGPT GPT-2 against Hugging Face GPT-2 on BPE Shakespeare."""
import json
import math
import os
import random
import time
from ast import literal_eval
import numpy as np
import torch
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer
from model import GPT

dataset = 'shakespeare'
data_dir = os.path.join('data', dataset)
model_name = 'gpt2'
hf_model_name = 'openai-community/gpt2'
wandb_project = 'shakespeare-char-task3'
wandb_run_name = 'task3-pretrained-gpt2-eval'
wandb_tags = 'pretrained,gpt2,eval-only'
seed = 1337
device = 'cpu'
block_size = 1024
batch_size = 8
eval_iters = 20
max_new_tokens = 80
temperature = 0.8
top_k = 40
logits_tolerance = 1e-4
report_path = 'reports/task3_pretrained_eval.json'


def apply_overrides():
    import sys
    for arg in sys.argv[1:]:
        if '=' not in arg and not arg.startswith('--'):
            with open(arg, encoding='utf-8') as config_file:
                exec(config_file.read(), globals())
            continue
        if '=' not in arg:
            raise ValueError(f'Unexpected argument: {arg}')
        key, value = arg[2:].split('=', 1) if arg.startswith('--') else arg.split('=', 1)
        if key not in globals():
            raise ValueError(f'Unknown config key: {key}')
        try:
            value = literal_eval(value)
        except (SyntaxError, ValueError):
            pass
        globals()[key] = value


def set_seed(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def get_batch(data, generator):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,), generator=generator)
    x = torch.stack([torch.from_numpy(data[i:i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + block_size].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)


def peak_memory_mb():
    if device != 'cuda':
        return 0.0
    return torch.cuda.max_memory_allocated(device) / 1024**2


@torch.no_grad()
def evaluate_validation(nano_model, hf_model, val_data):
    result = {}
    for label, evaluator in [('nanoGPT', lambda x, y: nano_model(x, y)[1]), ('huggingface_gpt2', lambda x, y: hf_model(input_ids=x, labels=y).loss)]:
        generator = torch.Generator().manual_seed(seed + 1)
        if device == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        losses = []
        token_count = 0
        for _ in range(eval_iters):
            x, y = get_batch(val_data, generator)
            losses.append(evaluator(x, y).item())
            token_count += y.numel()
        if device == 'cuda':
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
        mean_loss = float(np.mean(losses))
        result[label] = {
            'validation_loss': mean_loss,
            'perplexity': float(math.exp(mean_loss)),
            'inference_seconds': elapsed,
            'tokens_per_second': token_count / elapsed,
            'peak_memory_mb': peak_memory_mb(),
        }
    return result


@torch.no_grad()
def build_model_comparison(nano_model, hf_model, tokenizer):
    prompts = [
        'Once upon a time',
        'In the middle of the night',
        'The most important lesson I learned was',
        'Scientists discovered a strange signal from',
        'At the edge of the old forest',
    ]
    table = wandb.Table(columns=['sample_id', 'prompt', 'nanoGPT_text', 'huggingface_text', 'nanoGPT_token_ids', 'huggingface_token_ids', 'token_ids_equal', 'logits_max_abs_error', 'logits_mean_abs_error'])
    rows = []
    for sample_id, prompt in enumerate(prompts):
        set_seed(seed + sample_id)
        inputs = tokenizer(prompt, return_tensors='pt')
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        nano_logits, _ = nano_model(input_ids)
        hf_logits = hf_model(input_ids=input_ids, attention_mask=attention_mask).logits[:, -1:, :]
        diff = (nano_logits.float() - hf_logits.float()).abs()
        nano_generated = nano_model.generate(input_ids.clone(), max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)
        set_seed(seed + sample_id)
        hf_generated = hf_model.generate(input_ids=input_ids, attention_mask=attention_mask, do_sample=True, temperature=temperature, top_k=top_k, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id)
        nano_ids = nano_generated[0, input_ids.shape[1]:].tolist()
        hf_ids = hf_generated[0, input_ids.shape[1]:].tolist()
        row = {
            'sample_id': sample_id,
            'prompt': prompt,
            'nanoGPT_text': tokenizer.decode(nano_ids, skip_special_tokens=True),
            'huggingface_text': tokenizer.decode(hf_ids, skip_special_tokens=True),
            'nanoGPT_token_ids': nano_ids,
            'huggingface_token_ids': hf_ids,
            'token_ids_equal': nano_ids == hf_ids,
            'logits_max_abs_error': float(diff.max().item()),
            'logits_mean_abs_error': float(diff.mean().item()),
        }
        rows.append(row)
        table.add_data(row['sample_id'], row['prompt'], row['nanoGPT_text'], row['huggingface_text'], json.dumps(nano_ids), json.dumps(hf_ids), row['token_ids_equal'], row['logits_max_abs_error'], row['logits_mean_abs_error'])
    return table, rows


def main():
    apply_overrides()
    set_seed(seed)
    val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
    hf_model = AutoModelForCausalLM.from_pretrained(hf_model_name).to(device).eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    hf_model.config.pad_token_id = tokenizer.pad_token_id
    nano_model = GPT.from_pretrained(model_name, {'dropout': 0.0}).to(device).eval()
    parameter_count = sum(p.numel() for p in nano_model.parameters())
    run = wandb.init(project=wandb_project, name=wandb_run_name, tags=wandb_tags.split(','), config={
        'dataset': dataset, 'tokenizer': 'GPT-2 BPE', 'model_name': model_name, 'hf_model_name': hf_model_name,
        'seed': seed, 'device': device, 'block_size': block_size, 'batch_size': batch_size,
        'eval_iters': eval_iters, 'parameter_count': parameter_count, 'hf_parameter_count': sum(p.numel() for p in hf_model.parameters()), 'eval_only': True,
    })
    print(f'W&B run: {run.url}')
    metrics = evaluate_validation(nano_model, hf_model, val_data)
    table, rows = build_model_comparison(nano_model, hf_model, tokenizer)
    alignment = {
        'all_token_ids_equal': all(row['token_ids_equal'] for row in rows),
        'max_logits_abs_error': max(row['logits_max_abs_error'] for row in rows),
        'mean_logits_abs_error': float(np.mean([row['logits_mean_abs_error'] for row in rows])),
        'logits_within_tolerance': all(row['logits_max_abs_error'] <= logits_tolerance for row in rows),
    }
    flat = {}
    for prefix, values in metrics.items():
        for key, value in values.items():
            flat[f'{prefix}/{key}'] = value
    flat.update({f'comparison/{key}': value for key, value in alignment.items()})
    flat['tables/model_comparison'] = table
    wandb.log(flat)
    report = {'metrics': metrics, 'alignment': alignment, 'prompts': rows, 'wandb_run_url': run.url}
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as output_file:
        json.dump(report, output_file, ensure_ascii=False, indent=2)
    print(json.dumps({'metrics': metrics, 'alignment': alignment}, indent=2))
    print(f'Saved report to {report_path}')
    wandb.finish()


if __name__ == '__main__':
    main()
# Review note: keep pretrained evaluation settings fixed for reproducible comparison.

