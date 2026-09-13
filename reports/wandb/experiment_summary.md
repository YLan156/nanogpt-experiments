# nanoGPT W&B 实验摘要

## 1. 实验范围

本摘要汇总任务一至任务六的 W&B 记录。所有实验在 Windows 10、CPU 上完成；机器检测到 NVIDIA GeForce RTX 5060 Laptop GPU，但当前 PyTorch wheel 不支持该 GPU 的 `sm_120` 架构，因此本项目实际使用 `device=cpu`。

任务七本身只生成本摘要文件，不创建新的训练 run。

## 2. W&B 入口与提交链接

| 内容 | 链接 |
|---|---|
| Task 1 scratch run | [task1-scratch](https://wandb.ai/ylan156-hong-kong-university-of-science-and-technology/shakespeare-char/runs/l2b614lk) |
| Task 1 checkpoint Artifact | [task1-scratch-best-checkpoint:v1](https://wandb.ai/ylan156-hong-kong-university-of-science-and-technology/shakespeare-char/artifacts/model/task1-scratch-best-checkpoint/v1) |
| Task 2 scratch run | [task2-scratch-character](https://wandb.ai/ylan156-hong-kong-university-of-science-and-technology/shakespeare-char-task2/runs/09b113i2) |
| Task 2 checkpoint Artifact | [task2-scratch-character-best-checkpoint:v5](https://wandb.ai/ylan156-hong-kong-university-of-science-and-technology/shakespeare-char-task2/artifacts/model/task2-scratch-character-best-checkpoint/v5) |
| Task 3 pretrained run | [task3-pretrained-gpt2-eval](https://wandb.ai/ylan156-hong-kong-university-of-science-and-technology/shakespeare-char-task3/runs/x8b56ad4) |
| Task 4 fine-tuning run | [task4-finetuned-gpt2](https://wandb.ai/ylan156-hong-kong-university-of-science-and-technology/shakespeare-char-task4/runs/1s5n392c) |
| Task 4 best checkpoint Artifact | [task4-finetuned-gpt2-best-checkpoint:v3](https://wandb.ai/ylan156-hong-kong-university-of-science-and-technology/shakespeare-char-task4/artifacts/model/task4-finetuned-gpt2-best-checkpoint/v3) |
| Task 5 comparison Group | [shakespeare-char-task5 / task5-comparison](https://wandb.ai/ylan156-hong-kong-university-of-science-and-technology/shakespeare-char-task5/runs?group=task5-comparison) |
| Task 5 comparison summary | [task5-comparison-summary](https://wandb.ai/ylan156-hong-kong-university-of-science-and-technology/shakespeare-char-task5/runs/uov5so7d) |
| Task 5 scratch run | [task5-scratch-character](https://wandb.ai/ylan156-hong-kong-university-of-science-and-technology/shakespeare-char-task5/runs/nu14xe7a) |
| Task 5 pretrained run | [task5-gpt2-pretrained](https://wandb.ai/ylan156-hong-kong-university-of-science-and-technology/shakespeare-char-task5/runs/a5umfxq4) |
| Task 5 fine-tuned run | [task5-gpt2-finetuned](https://wandb.ai/ylan156-hong-kong-university-of-science-and-technology/shakespeare-char-task5/runs/vw614wav) |
| Task 5 fine-tuned checkpoint Artifact | [task5-finetuned-best-checkpoint](https://wandb.ai/ylan156-hong-kong-university-of-science-and-technology/shakespeare-char-task5/artifacts/model/task5-finetuned-best-checkpoint) |
| Task 6 Qwen/GPT-2 run | [task6-gpt2-vs-qwen-instruct](https://wandb.ai/ylan156-hong-kong-university-of-science-and-technology/shakespeare-char-task6/runs/3vog9xvk) |

Run IDs:

```text
Task 1: l2b614lk
Task 2: 09b113i2
Task 3: x8b56ad4
Task 4: 1s5n392c
Task 5: nu14xe7a, a5umfxq4, vw614wav, uov5so7d
Task 6: 3vog9xvk
```

## 3. Run 配置与核心指标

### 3.1 Task 1：扩展 nanoGPT W&B 日志

Run: `shakespeare-char/l2b614lk`

| 配置项 | 值 |
|---|---|
| dataset | `shakespeare_char` |
| tokenizer | character |
| init_from | `scratch` |
| model | 6 layers, 6 heads, 384 embedding, dropout 0.2 |
| batch/block size | 64 / 256 |
| learning rate | `1e-3` |
| max iterations | 10 |
| device/dtype | CPU / float32 |
| parameter count | 10,745,088 |

最后一次记录：`train/loss=3.0888`，`val/loss=3.1167`，`iter/tokens_per_second=227.67`，`iter/peak_memory_mb=0`，`iter/grad_norm=1.9613`。

该 run 还记录了 `charts/token_frequency_table`、`charts/embedding_norm_table`、`tables/model_stats` 和固定 prompt generation Table。

### 3.2 Task 2：scratch 字符模型训练

Run: `shakespeare-char-task2/09b113i2`

| 配置项 | 值 |
|---|---|
| dataset/tokenizer | `shakespeare_char` / character |
| model | 4 layers, 4 heads, 128 embedding, dropout 0.2 |
| batch/block size | 32 / 128 |
| learning rate | `1e-3`，warmup 30 |
| max iterations | 300 |
| eval interval | 50 |
| device/dtype | CPU / float32 |
| parameter count | 812,288 |
| checkpoint | `out-task2-scratch/ckpt.pt` |

核心指标：

| 指标 | 值 |
|---|---:|
| final validation loss | 2.4206 |
| validation perplexity | 11.2527 |
| training time | 132.04 s |
| tokens/second | 404.23 |
| peak memory | 0 MB |
| final gradient norm | 0.3963 |

### 3.3 Scratch 训练曲线

下面的验证点来自 Task 2 的 scratch run；曲线含义是训练迭代增加时，训练/验证交叉熵下降。

| iteration | train loss | validation loss |
|---:|---:|---:|
| 0 | 4.1915 | 4.1901 |
| 50 | 2.8342 | 2.8453 |
| 100 | 2.5494 | 2.5465 |
| 150 | 2.4852 | 2.4912 |
| 200 | 2.4476 | 2.4602 |
| 250 | 2.4312 | 2.4293 |
| 300 | 2.4164 | 2.4265 |

W&B 中的完整 iteration 曲线和 summary 指标见 [Task 2 run](https://wandb.ai/ylan156-hong-kong-university-of-science-and-technology/shakespeare-char-task2/runs/09b113i2)。

### 3.4 Task 3：GPT-2 pretrained eval-only

Run: `shakespeare-char-task3/x8b56ad4`

| 配置项 | 值 |
|---|---|
| model | `gpt2` / `openai-community/gpt2` |
| tokenizer | GPT-2 BPE |
| dataset | `shakespeare` |
| eval only | `True` |
| eval iterations | 20 |
| block/batch size | 1024 / 8 |
| seed/device | 1337 / CPU |
| parameter count | 124,439,808 |

| 指标 | nanoGPT GPT-2 |
|---|---:|
| validation loss | 4.0265 |
| perplexity | 56.0669 |
| inference time | 82.57 s |
| tokens/second | 1,984.33 |
| peak memory | 0 MB |

与 Hugging Face GPT-2 的对齐结果：输入 token IDs 全部一致；最大 logits 绝对误差为 `0.03019`，平均绝对误差为 `0.01127`。由于最大误差超过设定容差 `1e-4`，`logits_within_tolerance=False`。

### 3.5 Task 4：GPT-2 fine-tuning

Run: `shakespeare-char-task4/1s5n392c`

| 配置项 | 值 |
|---|---|
| init model | GPT-2 pretrained |
| tokenizer/dataset | GPT-2 BPE / Shakespeare |
| batch size/gradient accumulation | 1 / 8 |
| learning rate | `3e-5` |
| max iterations | 20 |
| eval interval | 5 |
| seed/device | 1337 / CPU |
| parameter count | 124,439,808 |
| best checkpoint | iteration 20 |

| 指标 | fine-tuning 前 | fine-tuning 后 |
|---|---:|---:|
| validation loss | 3.9713 | **3.3447** |
| perplexity | 53.0557 | **28.3513** |
| tokens/second | 1,816.92 | 1,823.57 |
| peak memory | 0 MB | 0 MB |

微调后 validation loss 下降 `0.6267`，perplexity 从 `53.0557` 降到 `28.3513`。最佳 checkpoint 为 `out-task4-finetuned/ckpt.pt`，并已上传为 Artifact。

## 4. Token frequency 与 Embedding Norm

这两项图表在 Task 1 run 中生成：

- [Task 1 W&B run](https://wandb.ai/ylan156-hong-kong-university-of-science-and-technology/shakespeare-char/runs/l2b614lk)
- W&B Table key：`charts/token_frequency_table`
- W&B Table key：`charts/embedding_norm_table`
- 本地 token frequency 数据：[token_frequency_char.csv](../token_frequency_char.csv)
- 本地 token frequency 图：[token_frequency_char.png](../token_frequency_char.png)

Token frequency 的前几项是空格、`e`、`t`、`o`、`a`、`h`；其中空格占比约 `15.23%`。Embedding Norm 表记录了 50 个 token 的 embedding 范数，可在 Task 1 run 的 Charts/Media 中查看。

## 5. 固定 prompt 生成对比

Task 5 将 scratch character、GPT-2 pretrained 和 GPT-2 fine-tuned 放入同一个 project/group：

```text
Project: shakespeare-char-task5
Group: task5-comparison
```

固定 prompt 为：

```text
Once upon a time
In the middle of the night
The most important lesson I learned was
Scientists discovered a strange signal from
At the edge of the old forest
```

对比结果存放在 [Task 5 summary run](https://wandb.ai/ylan156-hong-kong-university-of-science-and-technology/shakespeare-char-task5/runs/uov5so7d) 的 Table：

```text
tables/fixed_prompt_outputs
```

GPT-2 pretrained 与 fine-tuned 使用相同 GPT-2 BPE tokenizer、prompt、seed 和生成参数；scratch character 模型单独作为字符级模型对照。

Task 5 的统一比较指标如下：

| run | validation loss | perplexity | parameter count | training seconds | tokens/second | generation tokens/second |
|---|---:|---:|---:|---:|---:|---:|
| scratch character | 2.4206 | 11.2527 | 812,288 | 132.04 | 404.23 | 749.03 |
| GPT-2 pretrained | 4.0265 | 56.0669 | 124,439,808 | eval-only | 1,984.33 | 32.64 |
| GPT-2 fine-tuned | 3.3447 | 28.3513 | 124,439,808 | 2,626 | 1,823.57 | 33.40 |

三条 mirror run 的 config 中保留了原始 project/run：`shakespeare-char-task2/09b113i2`、`shakespeare-char-task3/x8b56ad4`、`shakespeare-char-task4/1s5n392c`，因此比较结果可以追溯到原始实验。

## 6. Task 6：Qwen Instruct 问答对比

Run: `shakespeare-char-task6/3vog9xvk`

| 配置项 | GPT-2 | Qwen |
|---|---|---|
| model | `openai-community/gpt2` | `Qwen/Qwen2.5-0.5B-Instruct` |
| parameter count | 124,439,808 | 494,032,768 |
| chat template | false | true |
| generation time | 15.20 s | 8.91 s |
| generation tokens/second | 39.47 | 14.59 |
| peak memory | 0 MB | 0 MB |

行为评分来自 5 个中文问题，仅用于观察生成行为，不代表 validation loss 或人工评测：

| 指标 | GPT-2 | Qwen |
|---|---:|---:|
| instruction following | 0.20 | **1.00** |
| Chinese expression | 0.4293 | **0.7740** |
| answer completeness | 0.00 | **1.00** |

问答对比 Table：

```text
table/qwen_vs_gpt2_answers
```

入口：[Task 6 W&B run](https://wandb.ai/ylan156-hong-kong-university-of-science-and-technology/shakespeare-char-task6/runs/3vog9xvk)。

## 7. 复现实验命令

以下命令在项目根目录 `D:\git-demo\git-practice\nanoGPT` 执行。先激活环境并确保 W&B 已登录：

```powershell
cd D:\git-demo\git-practice\nanoGPT
.\.venv\Scripts\Activate.ps1
wandb login
```

### Task 1

```powershell
.\.venv\Scripts\python.exe train.py config/train_shakespeare_char.py `
  --wandb_log=True --wandb_project=shakespeare-char `
  --wandb_run_name=task1-scratch --max_iters=10 `
  --eval_interval=5 --eval_iters=5 --compile=False `
  --device=cpu --dtype=float32
```

### Task 2

```powershell
.\.venv\Scripts\python.exe train.py config/task2_scratch.py
```

### Task 3

```powershell
.\.venv\Scripts\python.exe task3_pretrained_eval.py config/task3_pretrained_eval.py
```

### Task 4

```powershell
.\.venv\Scripts\python.exe task4_finetune.py config/task4_finetune.py
```

如果需要重新补传 Task 4 的 Table/Artifact：

```powershell
.\.venv\Scripts\python.exe task4_finalize.py
```

### Task 5

```powershell
.\.venv\Scripts\python.exe task5_compare.py
```

### Task 6

```powershell
.\.venv\Scripts\python.exe task6_qwen_instruct.py
```

任务六也依赖本地 Hugging Face cache 中的 `Qwen/Qwen2.5-0.5B-Instruct` 和 `openai-community/gpt2`；若缓存不存在，Transformers 会尝试联网下载。

## 8. 结论

1. 字符级 scratch 模型在 300 iterations 内将 validation loss 从 `4.1901` 降至约 `2.4265`，说明训练曲线正常下降。
2. GPT-2 fine-tuning 后 validation loss 和 perplexity 明显下降，Shakespeare BPE 数据上的适应效果优于 pretrained eval-only 状态。
3. Task 1 的 W&B 记录包含 token frequency、Embedding Norm、model stats 和 generation 记录，可用于检查训练日志是否完整。
4. Qwen Instruct 在中文指令遵循、中文表达和答案完整性上明显优于未做指令微调的 GPT-2；该结论只针对本组生成行为观察，不应解释为通用模型能力排名。
