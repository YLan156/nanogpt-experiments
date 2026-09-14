import ast
import importlib.util
import runpy
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


class ReproducibilityTests(unittest.TestCase):
    def test_character_tokenizer_round_trip(self):
        text = "First Citizen:\nBefore we proceed any further, hear me speak."
        chars = sorted(set(text))
        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for ch, i in stoi.items()}

        ids = [stoi[ch] for ch in text]
        decoded = "".join(itos[i] for i in ids)

        self.assertEqual(decoded, text)
        self.assertEqual(len(ids), len(text))
        self.assertEqual(set(ids), set(range(len(chars))))

    def test_gpt2_tiktoken_matches_huggingface_token_ids_when_available(self):
        if importlib.util.find_spec("tiktoken") is None:
            self.skipTest("tiktoken is not installed")
        if importlib.util.find_spec("transformers") is None:
            self.skipTest("transformers is not installed")

        import tiktoken
        from transformers import AutoTokenizer

        prompt = "ROMEO:\nBut soft, what light through yonder window breaks?"
        enc = tiktoken.get_encoding("gpt2")
        try:
            tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2", local_files_only=True)
        except Exception as exc:
            self.skipTest(f"Hugging Face GPT-2 tokenizer is not available locally: {exc}")

        self.assertEqual(enc.encode(prompt), tokenizer.encode(prompt, add_special_tokens=False))

    def test_small_gpt_forward_output_shape(self):
        from model import GPT, GPTConfig

        torch.manual_seed(1337)
        config = GPTConfig(
            block_size=8,
            vocab_size=16,
            n_layer=1,
            n_head=1,
            n_embd=8,
            dropout=0.0,
            bias=True,
        )
        model = GPT(config)
        model.eval()

        idx = torch.randint(0, config.vocab_size, (2, 5), dtype=torch.long)
        targets = torch.randint(0, config.vocab_size, (2, 5), dtype=torch.long)

        with torch.no_grad():
            logits, loss = model(idx, targets)

        self.assertEqual(tuple(logits.shape), (2, 5, config.vocab_size))
        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss))

    def test_experiment_configs_contain_required_fields(self):
        required = {
            "config/task3_pretrained_eval.py": {
                "dataset",
                "model_name",
                "hf_model_name",
                "wandb_project",
                "wandb_run_name",
                "wandb_tags",
                "seed",
                "device",
                "eval_iters",
            },
            "config/task4_finetune.py": {
                "dataset",
                "model_name",
                "hf_model_name",
                "wandb_project",
                "wandb_run_name",
                "wandb_tags",
                "seed",
                "device",
                "batch_size",
                "gradient_accumulation_steps",
                "max_iters",
                "eval_interval",
                "eval_iters",
                "learning_rate",
            },
            "config/task8_lora.py": {
                "DEVICE",
                "MAX_ITERS",
                "EVAL_INTERVAL",
                "EVAL_ITERS",
                "LEARNING_RATE",
                "LORA_RANK",
                "LORA_ALPHA",
            },
        }

        for rel_path, fields in required.items():
            with self.subTest(config=rel_path):
                values = runpy.run_path(str(ROOT / rel_path))
                missing = sorted(field for field in fields if field not in values)
                self.assertEqual(missing, [])

    def test_reproducibility_tests_do_not_launch_training(self):
        tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
        forbidden_calls = []

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue

            func = node.func
            if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
                if func.value.id == "subprocess":
                    forbidden_calls.append(f"subprocess.{func.attr}")
                if func.value.id == "os" and func.attr == "system":
                    forbidden_calls.append("os.system")

            if isinstance(func, ast.Attribute) and func.attr == "run_module":
                if node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value == "train":
                    forbidden_calls.append("run_module(train)")

        self.assertEqual(forbidden_calls, [])


if __name__ == "__main__":
    unittest.main()
