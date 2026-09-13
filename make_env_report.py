import json
import platform
import torch
import transformers
import tiktoken
import wandb
import os

out = {}
out["python_version"] = platform.python_version()
out["torch_version"] = torch.__version__
out["device"] = "cuda" if torch.cuda.is_available() else "cpu"
out["cuda_available"] = torch.cuda.is_available()
if torch.cuda.is_available():
    out["cuda_version"] = torch.version.cuda
    out["gpu_name"] = torch.cuda.get_device_name(0)
else:
    out["cuda_version"] = None
    out["gpu_name"] = None

out["transformers_version"] = transformers.__version__
out["tiktoken_version"] = tiktoken.__version__
out["wandb_version"] = wandb.__version__

os.makedirs("reports", exist_ok=True)
with open("reports/environment.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)

print("✅ 文件已生成：reports/environment.json")
