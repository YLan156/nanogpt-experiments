"""Task 8: LoRA fine-tuning for nanoGPT GPT-2 attention projections."""

import json
import math
import os
import random
import time
from ast import literal_eval
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import wandb
from transformers import AutoTokenizer

from model import GPT


ROOT = Path(__file__).resolve().parent
WANDB_TMP = ROOT / "reports" / "task8-wandb-tmp"
WANDB_TMP.mkdir(parents=True, exist_ok=True)
os.environ["TEMP"] = str(WANDB_TMP)
os.environ["TMP"] = str(WANDB_TMP)

DATASET = "shakespeare"
DATA_DIR = ROOT / "data" / DATASET
MODEL_NAME = "gpt2"
HF_MODEL_NAME = "openai-community/gpt2"
DEVICE = "cpu"
SEED = 1337
BLOCK_SIZE = 1024
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8
MAX_ITERS = 20
EVAL_INTERVAL = 5
EVAL_ITERS = 20
LEARNING_RATE = 3e-5
WEIGHT_DECAY = 0.1
BETA1 = 0.9
BETA2 = 0.95
GRAD_CLIP = 1.0
LORA_RANK = 8
LORA_ALPHA = 16.0
MAX_NEW_TOKENS = 80
TEMPERATURE = 0.8
TOP_K = 40
PROJECT = "shakespeare-char-task8"
GROUP = "task8-lora-comparison"
RUN_NAME = "task8-gpt2-lora"
REPORT_PATH = ROOT / "reports" / "task8_lora.json"
ADAPTER_PATH = ROOT / "out-task8-lora" / "adapter.pt"
PROMPTS = [
    "Once upon a time",
    "In the middle of the night",
    "The most important lesson I learned was",
    "Scientists discovered a strange signal from",
    "At the edge of the old forest",
]


class LoRALinear(nn.Module):
    """Frozen linear layer plus a trainable low-rank update."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        for parameter in self.base.parameters():
            parameter.requires_grad = False

    def forward(self, x):
        base_output = self.base(x)
        update = (x @ self.lora_A.t()) @ self.lora_B.t()
        return base_output + self.scaling * update


def apply_lora(model):
    replaced = []
    for layer_index, block in enumerate(model.transformer.h):
        for projection_name in ("c_attn", "c_proj"):
            projection = getattr(block.attn, projection_name)
            setattr(block.attn, projection_name, LoRALinear(projection, LORA_RANK, LORA_ALPHA))
            replaced.append(f"transformer.h.{layer_index}.attn.{projection_name}")
    return replaced


def apply_overrides():
    import sys
    for arg in sys.argv[1:]:
        if "=" not in arg and not arg.startswith("--"):
            with open(arg, encoding="utf-8") as config_file:
                exec(config_file.read(), globals())
            continue
        if "=" not in arg:
            raise ValueError(f"Unexpected argument: {arg}")
        key, value = arg[2:].split("=", 1) if arg.startswith("--") else arg.split("=", 1)
        if key not in globals():
            raise ValueError(f"Unknown config key: {key}")
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
    ix = torch.randint(len(data) - BLOCK_SIZE - 1, (BATCH_SIZE,), generator=generator)
    x = torch.stack([torch.from_numpy(data[i:i + BLOCK_SIZE].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + BLOCK_SIZE].astype(np.int64)) for i in ix])
    return x.to(DEVICE), y.to(DEVICE)


def peak_memory_mb():
    if DEVICE != "cuda":
        return 0.0
    return torch.cuda.max_memory_allocated(DEVICE) / 1024**2


@torch.no_grad()
def evaluate(model, val_data, generator_seed):
    model.eval()
    generator = torch.Generator().manual_seed(generator_seed)
    losses = []
    token_count = 0
    start = time.perf_counter()
    for _ in range(EVAL_ITERS):
        x, y = get_batch(val_data, generator)
        _, loss = model(x, y)
        losses.append(loss.item())
        token_count += y.numel()
    elapsed = time.perf_counter() - start
    model.train()
    mean_loss = float(np.mean(losses))
    return {
        "validation_loss": mean_loss,
        "perplexity": float(math.exp(mean_loss)),
        "inference_seconds": elapsed,
        "tokens_per_second": token_count / elapsed,
        "peak_memory_mb": peak_memory_mb(),
    }


@torch.no_grad()
def generate_samples(model, tokenizer):
    model.eval()
    rows = []
    for sample_id, prompt in enumerate(PROMPTS):
        set_seed(SEED + sample_id)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(DEVICE)
        start = time.perf_counter()
        output = model.generate(input_ids, max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE, top_k=TOP_K)
        elapsed = time.perf_counter() - start
        generated_ids = output[0, input_ids.shape[1]:].tolist()
        rows.append({
            "sample_id": sample_id,
            "prompt": prompt,
            "generated_text": tokenizer.decode(generated_ids, skip_special_tokens=True),
            "generated_token_ids": generated_ids,
            "tokens_per_second": len(generated_ids) / max(elapsed, 1e-9),
            "generation_seconds": elapsed,
            "peak_memory_mb": peak_memory_mb(),
        })
    model.train()
    return rows


def count_parameters(model):
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return total, trainable, trainable / total


def save_adapter(model, path):
    adapter_state = {
        name: parameter.detach().cpu()
        for name, parameter in model.state_dict().items()
        if "lora_A" in name or "lora_B" in name
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": adapter_state,
        "rank": LORA_RANK,
        "alpha": LORA_ALPHA,
        "target_modules": ["attn.c_attn", "attn.c_proj"],
        "base_model": MODEL_NAME,
    }, path)


def comparison_table(before_rows, after_rows):
    table = wandb.Table(columns=["prompt", "before_text", "lora_after_text", "before_tokens_per_second", "lora_tokens_per_second"])
    for before, after in zip(before_rows, after_rows):
        table.add_data(before["prompt"], before["generated_text"], after["generated_text"], before["tokens_per_second"], after["tokens_per_second"])
    return table


def main():
    apply_overrides()
    set_seed(SEED)
    train_data = np.memmap(DATA_DIR / "train.bin", dtype=np.uint16, mode="r")
    val_data = np.memmap(DATA_DIR / "val.bin", dtype=np.uint16, mode="r")
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)
    model = GPT.from_pretrained(MODEL_NAME, {"dropout": 0.0}).to(DEVICE)
    before_metrics = evaluate(model, val_data, SEED + 1)
    before_rows = generate_samples(model, tokenizer)
    full_report = json.loads((ROOT / "reports" / "task4_finetune.json").read_text(encoding="utf-8"))
    full_metrics = full_report["after"]["metrics"]
    full_training_seconds = float(full_report.get("training_seconds", 2626.0))
    for parameter in model.parameters():
        parameter.requires_grad = False
    replaced_modules = apply_lora(model)
    total_parameters, trainable_parameters, trainable_ratio = count_parameters(model)
    lora_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    experiment_config = {
        "task": "task8",
        "dataset": DATASET,
        "tokenizer": "GPT-2 BPE",
        "base_model": MODEL_NAME,
        "seed": SEED,
        "device": DEVICE,
        "batch_size": BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "block_size": BLOCK_SIZE,
        "max_iters": MAX_ITERS,
        "eval_iters": EVAL_ITERS,
        "learning_rate": LEARNING_RATE,
        "lora_rank": LORA_RANK,
        "lora_alpha": LORA_ALPHA,
        "lora_target_modules": replaced_modules,
        "total_parameter_count": total_parameters,
        "trainable_parameter_count": trainable_parameters,
        "trainable_parameter_ratio": trainable_ratio,
        "base_parameter_count": total_parameters - trainable_parameters,
        "full_finetune_validation_loss": full_metrics["validation_loss"],
        "full_finetune_perplexity": full_metrics["perplexity"],
    }
    optimizer = torch.optim.AdamW(lora_parameters, lr=LEARNING_RATE, betas=(BETA1, BETA2), weight_decay=WEIGHT_DECAY)
    run = wandb.init(entity="ylan156-hong-kong-university-of-science-and-technology", project=PROJECT, group=GROUP, name=RUN_NAME, tags=["task8", "lora", "gpt2"], config=experiment_config)
    print(f"W&B run: {run.url}")
    wandb.log({"before/validation_loss": before_metrics["validation_loss"], "before/perplexity": before_metrics["perplexity"]}, step=0)
    batch_generator = torch.Generator().manual_seed(SEED + 2)
    training_start = time.perf_counter()
    peak_training_memory = 0.0
    best_val_loss = float("inf")
    for iteration in range(1, MAX_ITERS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_value = 0.0
        for _ in range(GRADIENT_ACCUMULATION_STEPS):
            x, y = get_batch(train_data, batch_generator)
            _, loss = model(x, y)
            loss_value += loss.item() / GRADIENT_ACCUMULATION_STEPS
            (loss / GRADIENT_ACCUMULATION_STEPS).backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(lora_parameters, GRAD_CLIP).item()
        optimizer.step()
        peak_training_memory = max(peak_training_memory, peak_memory_mb())
        wandb.log({"train/loss": loss_value, "train/gradient_norm": grad_norm, "train/iteration": iteration}, step=iteration)
        if iteration % EVAL_INTERVAL == 0 or iteration == MAX_ITERS:
            metrics = evaluate(model, val_data, SEED + 1)
            wandb.log({"validation/loss": metrics["validation_loss"], "validation/perplexity": metrics["perplexity"], "validation/tokens_per_second": metrics["tokens_per_second"]}, step=iteration)
            print(f"iter {iteration}: train loss {loss_value:.4f}, val loss {metrics['validation_loss']:.4f}, ppl {metrics['perplexity']:.2f}")
            best_val_loss = min(best_val_loss, metrics["validation_loss"])
    training_seconds = time.perf_counter() - training_start
    after_metrics = evaluate(model, val_data, SEED + 1)
    after_rows = generate_samples(model, tokenizer)
    save_adapter(model, ADAPTER_PATH)
    artifact = wandb.Artifact("task8-gpt2-lora-adapter", type="model", metadata={"rank": LORA_RANK, "alpha": LORA_ALPHA, "base_model": MODEL_NAME, "best_val_loss": best_val_loss})
    artifact.add_file(str(ADAPTER_PATH), name="adapter.pt")
    logged_artifact = wandb.log_artifact(artifact, aliases=["best", "lora", f"iter-{MAX_ITERS}"])
    logged_artifact.wait()
    wandb.log({
        "after/validation_loss": after_metrics["validation_loss"],
        "after/perplexity": after_metrics["perplexity"],
        "after/tokens_per_second": after_metrics["tokens_per_second"],
        "training/seconds": training_seconds,
        "training/peak_memory_mb": peak_training_memory,
        "parameters/total": total_parameters,
        "parameters/trainable": trainable_parameters,
        "parameters/trainable_ratio": trainable_ratio,
        "full_finetune/validation_loss": full_metrics["validation_loss"],
        "full_finetune/perplexity": full_metrics["perplexity"],
        "full_finetune/training_seconds": full_training_seconds,
        "full_finetune/tokens_per_second": full_metrics["tokens_per_second"],
        "full_finetune/peak_memory_mb": full_metrics["peak_memory_mb"],
        "tables/before_after_generation": comparison_table(before_rows, after_rows),
    }, step=MAX_ITERS + 1)
    run.finish()
    comparison = wandb.init(
        entity="ylan156-hong-kong-university-of-science-and-technology",
        project=PROJECT,
        group=GROUP,
        name="task8-full-vs-lora-summary",
        job_type="comparison",
        tags=["task8", "comparison", "full-finetune", "lora"],
        config={
            "task": "task8",
            "full_finetune_source_run": "shakespeare-char-task4/1s5n392c",
            "lora_source_run": run.id,
            "base_model": MODEL_NAME,
            "lora_rank": LORA_RANK,
            "lora_alpha": LORA_ALPHA,
        },
    )
    comparison.log({
        "full_finetune/validation_loss": full_metrics["validation_loss"],
        "full_finetune/perplexity": full_metrics["perplexity"],
        "full_finetune/training_seconds": full_training_seconds,
        "full_finetune/tokens_per_second": full_metrics["tokens_per_second"],
        "full_finetune/peak_memory_mb": full_metrics["peak_memory_mb"],
        "lora/validation_loss": after_metrics["validation_loss"],
        "lora/perplexity": after_metrics["perplexity"],
        "lora/training_seconds": training_seconds,
        "lora/tokens_per_second": after_metrics["tokens_per_second"],
        "lora/peak_memory_mb": peak_training_memory,
        "lora/total_parameter_count": total_parameters,
        "lora/trainable_parameter_count": trainable_parameters,
        "lora/trainable_parameter_ratio": trainable_ratio,
    })
    comparison_url = comparison.url
    comparison.finish()
    report = {
        "wandb_run_url": run.url,
        "wandb_artifact_url": logged_artifact.url,
        "wandb_comparison_url": comparison_url,
        "group": GROUP,
        "config": experiment_config,
        "before": {"metrics": before_metrics, "samples": before_rows},
        "after": {"metrics": after_metrics, "samples": after_rows},
        "training_seconds": training_seconds,
        "training_peak_memory_mb": peak_training_memory,
        "best_val_loss": best_val_loss,
        "adapter_path": str(ADAPTER_PATH),
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"run_url": run.url, "artifact_url": logged_artifact.url, "report": str(REPORT_PATH)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
