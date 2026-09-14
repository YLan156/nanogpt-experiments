"""Task 4: fine-tune nanoGPT GPT-2 on GPT-2 BPE Shakespeare."""
import json
import math
import os
import random
import time
from ast import literal_eval

import numpy as np
import torch
import wandb
from transformers import AutoTokenizer

from model import GPT


dataset = 'shakespeare'
data_dir = os.path.join('data', dataset)
model_name = 'gpt2'
hf_model_name = 'openai-community/gpt2'
wandb_project = 'shakespeare-char-task4'
wandb_run_name = 'task4-finetuned-gpt2'
wandb_tags = 'pretrained,gpt2,finetuned'
seed = 1337
device = 'cpu'
block_size = 1024
batch_size = 1
gradient_accumulation_steps = 8
max_iters = 20
eval_interval = 5
eval_iters = 20
learning_rate = 3e-5
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
max_new_tokens = 80
temperature = 0.8
top_k = 40
report_path = 'reports/task4_finetune.json'
out_dir = 'out-task4-finetuned'

PROMPTS = [
    'Once upon a time',
    'In the middle of the night',
    'The most important lesson I learned was',
    'Scientists discovered a strange signal from',
    'At the edge of the old forest',
]


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
def evaluate(model, val_data, generator_seed):
    model.eval()
    generator = torch.Generator().manual_seed(generator_seed)
    losses = []
    token_count = 0
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    for _ in range(eval_iters):
        x, y = get_batch(val_data, generator)
        _, loss = model(x, y)
        losses.append(loss.item())
        token_count += y.numel()
    if device == 'cuda':
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    model.train()
    mean_loss = float(np.mean(losses))
    return {
        'validation_loss': mean_loss,
        'perplexity': float(math.exp(mean_loss)),
        'inference_seconds': elapsed,
        'tokens_per_second': token_count / elapsed,
        'peak_memory_mb': peak_memory_mb(),
    }


@torch.no_grad()
def generate_samples(model, tokenizer):
    model.eval()
    rows = []
    for sample_id, prompt in enumerate(PROMPTS):
        set_seed(seed + sample_id)
        inputs = tokenizer(prompt, return_tensors='pt')
        input_ids = inputs['input_ids'].to(device)
        if device == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        output = model.generate(input_ids, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)
        if device == 'cuda':
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
        generated_ids = output[0, input_ids.shape[1]:].tolist()
        rows.append({
            'sample_id': sample_id,
            'prompt': prompt,
            'generated_text': tokenizer.decode(generated_ids, skip_special_tokens=True),
            'generated_token_ids': generated_ids,
            'tokens_per_second': len(generated_ids) / elapsed,
            'generation_seconds': elapsed,
            'peak_memory_mb': peak_memory_mb(),
        })
    model.train()
    return rows


def build_comparison_table(before_rows, after_rows):
    table = wandb.Table(columns=[
        'sample_id', 'prompt', 'before_text', 'after_text',
        'before_token_ids', 'after_token_ids',
        'before_tokens_per_second', 'after_tokens_per_second',
    ])
    for before, after in zip(before_rows, after_rows):
        table.add_data(
            before['sample_id'], before['prompt'],
            before['generated_text'], after['generated_text'],
            json.dumps(before['generated_token_ids']), json.dumps(after['generated_token_ids']),
            before['tokens_per_second'], after['tokens_per_second'],
        )
    return table


def save_checkpoint(model, optimizer, iter_num, best_val_loss, config, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'model_args': {
            'n_layer': model.config.n_layer,
            'n_head': model.config.n_head,
            'n_embd': model.config.n_embd,
            'block_size': model.config.block_size,
            'bias': model.config.bias,
            'vocab_size': model.config.vocab_size,
            'dropout': model.config.dropout,
        },
        'iter_num': iter_num,
        'best_val_loss': best_val_loss,
        'config': config,
    }, path)


def main():
    apply_overrides()
    set_seed(seed)
    train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
    model = GPT.from_pretrained(model_name, {'dropout': 0.0}).to(device)
    model.train()
    parameter_count = sum(p.numel() for p in model.parameters())
    experiment_config = {
        'dataset': dataset,
        'tokenizer': 'GPT-2 BPE',
        'model_name': model_name,
        'hf_model_name': hf_model_name,
        'seed': seed,
        'device': device,
        'batch_size': batch_size,
        'gradient_accumulation_steps': gradient_accumulation_steps,
        'block_size': block_size,
        'max_iters': max_iters,
        'eval_iters': eval_iters,
        'learning_rate': learning_rate,
        'parameter_count': parameter_count,
        'eval_only': False,
        'fine_tuning': True,
    }
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(beta1, beta2), weight_decay=weight_decay)
    run = wandb.init(project=wandb_project, name=wandb_run_name, tags=wandb_tags.split(','), config=experiment_config)
    print(f'W&B run: {run.url}')
    before_metrics = evaluate(model, val_data, seed + 1)
    before_rows = generate_samples(model, tokenizer)
    wandb.log({
        'before/validation_loss': before_metrics['validation_loss'],
        'before/perplexity': before_metrics['perplexity'],
        'before/tokens_per_second': before_metrics['tokens_per_second'],
        'before/peak_memory_mb': before_metrics['peak_memory_mb'],
    }, step=0)

    # Fine-tuned checkpoints are selected only from post-update evaluations.
    best_val_loss = float('inf')
    best_checkpoint_path = os.path.join(out_dir, 'ckpt.pt')
    batch_generator = torch.Generator().manual_seed(seed + 2)
    training_start = time.perf_counter()
    peak_training_memory = 0.0
    for iter_num in range(1, max_iters + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_value = 0.0
        grad_norm = 0.0
        for _ in range(gradient_accumulation_steps):
            x, y = get_batch(train_data, batch_generator)
            _, loss = model(x, y)
            loss_value += loss.item() / gradient_accumulation_steps
            (loss / gradient_accumulation_steps).backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip).item()
        optimizer.step()
        peak_training_memory = max(peak_training_memory, peak_memory_mb())
        wandb.log({
            'train/loss': loss_value,
            'train/learning_rate': learning_rate,
            'train/gradient_norm': grad_norm,
            'train/iteration': iter_num,
        }, step=iter_num)
        if iter_num % eval_interval == 0 or iter_num == max_iters:
            metrics = evaluate(model, val_data, seed + 1)
            wandb.log({
                'validation/loss': metrics['validation_loss'],
                'validation/perplexity': metrics['perplexity'],
                'validation/tokens_per_second': metrics['tokens_per_second'],
                'validation/peak_memory_mb': metrics['peak_memory_mb'],
            }, step=iter_num)
            print(f"iter {iter_num}: train loss {loss_value:.4f}, val loss {metrics['validation_loss']:.4f}, ppl {metrics['perplexity']:.2f}")
            if metrics['validation_loss'] < best_val_loss:
                best_val_loss = metrics['validation_loss']
                save_checkpoint(model, optimizer, iter_num, best_val_loss, config=experiment_config, path=best_checkpoint_path)
                artifact = wandb.Artifact(f'{wandb_run_name}-best-checkpoint', type='model', metadata={'iter': iter_num, 'best_val_loss': best_val_loss})
                artifact.add_file(best_checkpoint_path)
                logged = wandb.log_artifact(artifact, aliases=['best', f'iter-{iter_num}'])
                logged.wait()
                print(f'W&B model Artifact: {logged.url}')

    training_seconds = time.perf_counter() - training_start
    after_metrics = evaluate(model, val_data, seed + 1)
    after_rows = generate_samples(model, tokenizer)
    comparison_table = build_comparison_table(before_rows, after_rows)
    wandb.log({
        'after/validation_loss': after_metrics['validation_loss'],
        'after/perplexity': after_metrics['perplexity'],
        'after/tokens_per_second': after_metrics['tokens_per_second'],
        'after/peak_memory_mb': after_metrics['peak_memory_mb'],
        'training/total_seconds': training_seconds,
        'training/peak_memory_mb': peak_training_memory,
        'tables/before_after_generation': comparison_table,
    }, step=max_iters + 1)
    report = {
        'wandb_run_url': run.url,
        'before': {'metrics': before_metrics, 'samples': before_rows},
        'after': {'metrics': after_metrics, 'samples': after_rows},
        'training_seconds': training_seconds,
        'training_peak_memory_mb': peak_training_memory,
        'best_val_loss': best_val_loss,
    }
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as output_file:
        json.dump(report, output_file, ensure_ascii=False, indent=2)
    print(f'Saved report to {report_path}')
    wandb.finish()


if __name__ == '__main__':
    main()
