"""Log the existing GPT-2 vs Qwen instruction comparison as a W&B Table."""

import json
import os
from pathlib import Path

import wandb


ROOT = Path(__file__).resolve().parent
REPORT_PATH = ROOT / "reports" / "task11_instruct_comparison.json"
ENTITY = "ylan156-hong-kong-university-of-science-and-technology"
PROJECT = "shakespeare-char-task6"
RUN_NAME = "task6-gpt2-vs-qwen-instruct"
TASK_TAGS = ["task6", "gpt2", "qwen", "instruct", "behavior-only"]


def load_report():
    if not REPORT_PATH.exists():
        raise FileNotFoundError(f"Missing previous comparison report: {REPORT_PATH}")
    return json.loads(REPORT_PATH.read_text(encoding="utf-8"))


def model_stats(report, model_key):
    rows = report[model_key]["results"]
    return {
        "parameter_count": 124439808 if model_key == "gpt2" else 494032768,
        "generation_seconds": float(sum(row["elapsed_sec"] for row in rows)),
        "peak_memory_mb": max((row.get("peak_memory_mb") or 0.0) for row in rows),
        "generation_tokens_per_second": float(sum(row["generated_tokens"] for row in rows) / max(sum(row["elapsed_sec"] for row in rows), 1e-9)),
        "average_scores": report[model_key]["average_scores"],
    }


def build_table(report):
    gpt2_rows = report["gpt2"]["results"]
    qwen_rows = report["qwen"]["results"]
    table = wandb.Table(columns=[
        "prompt",
        "gpt2_output",
        "qwen_output",
        "gpt2_instruction_following",
        "qwen_instruction_following",
        "gpt2_chinese_expression",
        "qwen_chinese_expression",
        "gpt2_answer_completeness",
        "qwen_answer_completeness",
        "observation",
    ])
    for gpt2, qwen in zip(gpt2_rows, qwen_rows):
        table.add_data(
            gpt2["question"],
            gpt2["generated_text"],
            qwen["generated_text"],
            gpt2["scores"]["instruction_following"],
            qwen["scores"]["instruction_following"],
            gpt2["scores"]["chinese_expression"],
            qwen["scores"]["chinese_expression"],
            gpt2["scores"]["answer_completeness"],
            qwen["scores"]["answer_completeness"],
            "Qwen uses an instruction/chat template; GPT-2 is a base causal LM without instruction tuning.",
        )
    return table


def main():
    report = load_report()
    gpt2 = model_stats(report, "gpt2")
    qwen = model_stats(report, "qwen")
    table = build_table(report)

    run = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        name=RUN_NAME,
        tags=TASK_TAGS,
        config={
            "task": "task6",
            "comparison_scope": "observable generation behavior only",
            "seed": report["seed"],
            "temperature": report["generation"]["temperature"],
            "top_k": report["generation"]["top_k"],
            "max_new_tokens": report["generation"]["max_new_tokens"],
            "questions": report["questions"],
            "gpt2_model": report["gpt2"]["model"],
            "qwen_model": report["qwen"]["model"],
            "gpt2_parameter_count": gpt2["parameter_count"],
            "qwen_parameter_count": qwen["parameter_count"],
            "gpt2_chat_template": report["gpt2"]["uses_apply_chat_template"],
            "qwen_chat_template": report["qwen"]["uses_apply_chat_template"],
        },
    )
    run.log({
        "table/qwen_vs_gpt2_answers": table,
        "gpt2/parameter_count": gpt2["parameter_count"],
        "qwen/parameter_count": qwen["parameter_count"],
        "gpt2/generation_seconds": gpt2["generation_seconds"],
        "qwen/generation_seconds": qwen["generation_seconds"],
        "gpt2/peak_memory_mb": gpt2["peak_memory_mb"],
        "qwen/peak_memory_mb": qwen["peak_memory_mb"],
        "gpt2/generation_tokens_per_second": gpt2["generation_tokens_per_second"],
        "qwen/generation_tokens_per_second": qwen["generation_tokens_per_second"],
        "gpt2/average_instruction_following": gpt2["average_scores"]["instruction_following"],
        "qwen/average_instruction_following": qwen["average_scores"]["instruction_following"],
        "gpt2/average_chinese_expression": gpt2["average_scores"]["chinese_expression"],
        "qwen/average_chinese_expression": qwen["average_scores"]["chinese_expression"],
        "gpt2/average_answer_completeness": gpt2["average_scores"]["answer_completeness"],
        "qwen/average_answer_completeness": qwen["average_scores"]["answer_completeness"],
    })
    run.summary["comparison_note"] = "Behavior-only comparison; not a validation-loss or human evaluation benchmark."
    run.summary["table_name"] = "table/qwen_vs_gpt2_answers"
    report["wandb_run_url"] = run.url
    report["wandb_table_name"] = "table/qwen_vs_gpt2_answers"
    report["wandb_project"] = PROJECT
    (ROOT / "reports" / "task6_qwen_instruct_wandb.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"run_url": run.url, "run_id": run.id, "table": "table/qwen_vs_gpt2_answers", "project": PROJECT}, ensure_ascii=False, indent=2))
    run.finish()


if __name__ == "__main__":
    main()
