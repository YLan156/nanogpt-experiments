"""Create a comparable W&B Group from the completed Tasks 2, 3 and 4 runs."""

import json
import math
import os
import pickle
import random
import time
from pathlib import Path

import numpy as np
import torch

# Keep W&B's temporary media files inside the project. This avoids Windows temp
# cleanup races while tables and artifacts are being uploaded.
PROJECT_ROOT = Path(__file__).resolve().parent
WANDB_TMP = PROJECT_ROOT / "reports" / "task5-wandb-tmp"
WANDB_TMP.mkdir(parents=True, exist_ok=True)
os.environ["TEMP"] = str(WANDB_TMP)
os.environ["TMP"] = str(WANDB_TMP)

import wandb

from model import GPT, GPTConfig


ENTITY = "ylan156-hong-kong-university-of-science-and-technology"
PROJECT = "shakespeare-char-task5"
GROUP = "task5-comparison"
SEED = 1337
DEVICE = "cpu"
MAX_NEW_TOKENS = 80
TEMPERATURE = 0.8
TOP_K = 40
PROMPTS = [
    "Once upon a time",
    "In the middle of the night",
    "The most important lesson I learned was",
    "Scientists discovered a strange signal from",
    "At the edge of the old forest",
]

TASK2_RUN_ID = "09b113i2"
TASK3_RUN_ID = "x8b56ad4"
TASK4_RUN_ID = "1s5n392c"


def load_task2_summary():
    candidates = sorted((PROJECT_ROOT / "wandb").glob(f"run-*-{TASK2_RUN_ID}/files/wandb-summary.json"))
    for path in candidates:
        data = json.loads(path.read_text(encoding="utf-8"))
        if "iter/parameter_count" in data and "val/loss" in data and "training/time_seconds" in data:
            return {
                "validation_loss": float(data["val/loss"]),
                "perplexity": float(math.exp(data["val/loss"])),
                "parameter_count": int(data["iter/parameter_count"]),
                "training_seconds": float(data["training/time_seconds"]),
                "peak_memory_mb": float(data.get("training/peak_memory_mb", 0.0)),
                "tokens_per_second": float(data["iter/tokens_per_second"]),
                "generation_tokens_per_second": None,
            }
    raise FileNotFoundError("Could not find the completed Task 2 W&B summary.")


def load_reports():
    task3 = json.loads((PROJECT_ROOT / "reports" / "task3_pretrained_eval.json").read_text(encoding="utf-8"))
    task4 = json.loads((PROJECT_ROOT / "reports" / "task4_finetune.json").read_text(encoding="utf-8"))
    task4_summaries = sorted((PROJECT_ROOT / "wandb").glob(f"run-*-{TASK4_RUN_ID}/files/wandb-summary.json"))
    task4_summary = json.loads(task4_summaries[-1].read_text(encoding="utf-8")) if task4_summaries else {}
    pretrained = task3["metrics"]["nanoGPT"]
    finetuned = task4["after"]["metrics"]
    return {
        "pretrained": {
            "validation_loss": float(pretrained["validation_loss"]),
            "perplexity": float(pretrained["perplexity"]),
            "parameter_count": 124439808,
            "training_seconds": 0.0,
            "peak_memory_mb": float(pretrained["peak_memory_mb"]),
            "tokens_per_second": float(pretrained["tokens_per_second"]),
            "generation_tokens_per_second": float(np.mean([row["tokens_per_second"] for row in task4["before"]["samples"]])),
            "samples": task3["prompts"],
        },
        "finetuned": {
            "validation_loss": float(finetuned["validation_loss"]),
            "perplexity": float(finetuned["perplexity"]),
            "parameter_count": 124439808,
            "training_seconds": float(task4.get("training_seconds", task4_summary.get("_wandb.runtime", task4_summary.get("_runtime", 0.0)))),
            "peak_memory_mb": float(finetuned["peak_memory_mb"]),
            "tokens_per_second": float(finetuned["tokens_per_second"]),
            "generation_tokens_per_second": float(np.mean([row["tokens_per_second"] for row in task4["after"]["samples"]])),
            "samples": task4["after"]["samples"],
        },
        "pretrained_samples": task4["before"]["samples"],
    }


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def load_scratch_outputs():
    checkpoint = torch.load(PROJECT_ROOT / "out-task2-scratch" / "ckpt.pt", map_location=DEVICE, weights_only=False)
    model = GPT(GPTConfig(**checkpoint["model_args"])).to(DEVICE)
    state_dict = checkpoint["model"]
    state_dict = {key.replace("_orig_mod.", ""): value for key, value in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    meta = pickle.loads((PROJECT_ROOT / "data" / "shakespeare_char" / "meta.pkl").read_bytes())
    stoi = meta["stoi"]
    itos = meta["itos"]
    decode = (lambda ids: "".join(itos[i] for i in ids)) if isinstance(itos, list) else (lambda ids: "".join(itos[str(i)] if str(i) in itos else itos[i] for i in ids))
    outputs = []
    timings = []
    for sample_id, prompt in enumerate(PROMPTS):
        set_seed(SEED + sample_id)
        ids = torch.tensor([[stoi[ch] for ch in prompt if ch in stoi]], dtype=torch.long, device=DEVICE)
        start = time.perf_counter()
        generated = model.generate(ids, MAX_NEW_TOKENS, temperature=TEMPERATURE, top_k=TOP_K)
        elapsed = time.perf_counter() - start
        new_ids = generated[0, ids.shape[1]:].tolist()
        timings.append(len(new_ids) / max(elapsed, 1e-9))
        outputs.append({"prompt": prompt, "generated_text": decode(new_ids), "tokens_per_second": timings[-1]})
    return outputs, float(np.mean(timings))


def metrics_for_run(run_metrics):
    return {
        "validation/loss": run_metrics["validation_loss"],
        "validation/perplexity": run_metrics["perplexity"],
        "model/parameter_count": run_metrics["parameter_count"],
        "training/seconds": run_metrics["training_seconds"],
        "system/peak_memory_mb": run_metrics["peak_memory_mb"],
        "performance/tokens_per_second": run_metrics["tokens_per_second"],
        "performance/generation_tokens_per_second": run_metrics["generation_tokens_per_second"],
    }


def log_mirror_run(name, job_type, tags, source_project, source_run_id, metrics, extra_config):
    run = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        group=GROUP,
        name=name,
        job_type=job_type,
        tags=tags,
        config={
            "task": "task5",
            "group": GROUP,
            "dataset": "Shakespeare",
            "tokenizer": "GPT-2 BPE" if job_type != "scratch" else "character",
            "seed": SEED,
            "device": DEVICE,
            "max_new_tokens": MAX_NEW_TOKENS,
            "temperature": TEMPERATURE,
            "top_k": TOP_K,
            "source_project": source_project,
            "source_run_id": source_run_id,
            **extra_config,
        },
    )
    run.log(metrics_for_run(metrics))
    run.summary["best_validation_checkpoint"] = "task4-finetuned-best-checkpoint:v3" if job_type == "finetuned" else "source run checkpoint"
    run.finish()
    return run


def main():
    scratch = load_task2_summary()
    scratch_outputs, scratch_generation_speed = load_scratch_outputs()
    scratch["generation_tokens_per_second"] = scratch_generation_speed
    reports = load_reports()
    pretrained = reports["pretrained"]
    finetuned = reports["finetuned"]

    runs = []
    runs.append(log_mirror_run("task5-scratch-character", "scratch", ["scratch", "character-tokenizer"], "shakespeare-char-task2", TASK2_RUN_ID, scratch, {"model": "small character GPT", "best_checkpoint": "out-task2-scratch/ckpt.pt"}))
    runs.append(log_mirror_run("task5-gpt2-pretrained", "pretrained-eval", ["pretrained", "gpt2", "eval-only"], "shakespeare-char-task3", TASK3_RUN_ID, pretrained, {"model": "GPT-2", "eval_only": True, "validation_data": "GPT-2 BPE Shakespeare"}))
    runs.append(log_mirror_run("task5-gpt2-finetuned", "finetuned", ["pretrained", "gpt2", "finetuned"], "shakespeare-char-task4", TASK4_RUN_ID, finetuned, {"model": "GPT-2", "init_from": "gpt2", "best_checkpoint": "out-task4-finetuned/ckpt.pt", "validation_data": "GPT-2 BPE Shakespeare"}))

    table = wandb.Table(columns=["prompt", "scratch_character", "pretrained_gpt2", "fine_tuned_gpt2", "observation"])
    for index, prompt in enumerate(PROMPTS):
        scratch_text = scratch_outputs[index]["generated_text"]
        pretrained_text = reports["pretrained_samples"][index]["generated_text"] if "generated_text" in reports["pretrained_samples"][index] else reports["pretrained_samples"][index]["nanoGPT_text"]
        finetuned_text = finetuned["samples"][index]["generated_text"]
        table.add_data(prompt, scratch_text, pretrained_text, finetuned_text, "GPT-2 BPE outputs are directly comparable; scratch uses a character tokenizer.")

    summary = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        group=GROUP,
        name="task5-comparison-summary",
        job_type="comparison",
        tags=["task5", "comparison"],
        config={"task": "task5", "group": GROUP, "fixed_prompts": PROMPTS, "best_validation_checkpoint": "task4-finetuned-best-checkpoint:v3"},
    )
    summary.log({
        "comparison/parameter_count/scratch": scratch["parameter_count"],
        "comparison/parameter_count/gpt2": finetuned["parameter_count"],
        "comparison/validation_loss/pretrained": pretrained["validation_loss"],
        "comparison/validation_loss/finetuned": finetuned["validation_loss"],
        "comparison/perplexity/pretrained": pretrained["perplexity"],
        "comparison/perplexity/finetuned": finetuned["perplexity"],
        "comparison/training_seconds/scratch": scratch["training_seconds"],
        "comparison/training_seconds/finetuned": finetuned["training_seconds"],
        "comparison/peak_memory_mb/scratch": scratch["peak_memory_mb"],
        "comparison/peak_memory_mb/gpt2": finetuned["peak_memory_mb"],
        "comparison/tokens_per_second/scratch": scratch["tokens_per_second"],
        "comparison/tokens_per_second/gpt2": finetuned["tokens_per_second"],
        "comparison/generation_tokens_per_second/scratch": scratch["generation_tokens_per_second"],
        "comparison/generation_tokens_per_second/pretrained": pretrained["generation_tokens_per_second"],
        "comparison/generation_tokens_per_second/finetuned": finetuned["generation_tokens_per_second"],
        "tables/fixed_prompt_outputs": table,
    })
    summary.summary["best_validation_checkpoint"] = "task4-finetuned-best-checkpoint:v3"
    artifact_url = None
    checkpoint_path = PROJECT_ROOT / "out-task4-finetuned" / "ckpt.pt"
    if checkpoint_path.exists():
        artifact = wandb.Artifact("task5-finetuned-best-checkpoint", type="model", description="Best GPT-2 fine-tuned checkpoint from Task 4")
        artifact.add_file(str(checkpoint_path), name="ckpt.pt")
        summary.log_artifact(artifact, aliases=["best", "task4", "iter-20"])
        artifact_url = f"https://wandb.ai/{ENTITY}/{PROJECT}/artifacts/model/task5-finetuned-best-checkpoint"
    summary_url = summary.url
    summary.finish()

    group_url = f"https://wandb.ai/{ENTITY}/{PROJECT}/runs?group={GROUP}"
    print(json.dumps({"group_url": group_url, "summary_run_url": summary_url, "artifact_url": artifact_url, "runs": [{"name": run.name, "id": run.id, "url": run.url} for run in runs]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
