
# nanoGPT

![nanoGPT](assets/nanogpt.jpg)


---

**Update Nov 2025** nanoGPT has a new and improved cousin called [nanochat](https://github.com/karpathy/nanochat). It is very likely you meant to use/find nanochat instead. nanoGPT (this repo) is now very old and deprecated but I will leave it up for posterity.

---

The simplest, fastest repository for training/finetuning medium-sized GPTs. It is a rewrite of [minGPT](https://github.com/karpathy/minGPT) that prioritizes teeth over education. Still under active development, but currently the file `train.py` reproduces GPT-2 (124M) on OpenWebText, running on a single 8XA100 40GB node in about 4 days of training. The code itself is plain and readable: `train.py` is a ~300-line boilerplate training loop and `model.py` a ~300-line GPT model definition, which can optionally load the GPT-2 weights from OpenAI. That's it.

![repro124m](assets/gpt2_124M_loss.png)

Because the code is so simple, it is very easy to hack to your needs, train new models from scratch, or finetune pretrained checkpoints (e.g. biggest one currently available as a starting point would be the GPT-2 1.3B model from OpenAI).

## install

```
pip install torch numpy transformers datasets tiktoken wandb tqdm
```

Dependencies:

- [pytorch](https://pytorch.org) <3
- [numpy](https://numpy.org/install/) <3
-  `transformers` for huggingface transformers <3 (to load GPT-2 checkpoints)
-  `datasets` for huggingface datasets <3 (if you want to download + preprocess OpenWebText)
-  `tiktoken` for OpenAI's fast BPE code <3
-  `wandb` for optional logging <3
-  `tqdm` for progress bars <3

## quick start

If you are not a deep learning professional and you just want to feel the magic and get your feet wet, the fastest way to get started is to train a character-level GPT on the works of Shakespeare. First, we download it as a single (1MB) file and turn it from raw text into one large stream of integers:

```sh
python data/shakespeare_char/prepare.py
```

This creates a `train.bin` and `val.bin` in that data directory. Now it is time to train your GPT. The size of it very much depends on the computational resources of your system:

**I have a GPU**. Great, we can quickly train a baby GPT with the settings provided in the [config/train_shakespeare_char.py](config/train_shakespeare_char.py) config file:

```sh
python train.py config/train_shakespeare_char.py
```

If you peek inside it, you'll see that we're training a GPT with a context size of up to 256 characters, 384 feature channels, and it is a 6-layer Transformer with 6 heads in each layer. On one A100 GPU this training run takes about 3 minutes and the best validation loss is 1.4697. Based on the configuration, the model checkpoints are being written into the `--out_dir` directory `out-shakespeare-char`. So once the training finishes we can sample from the best model by pointing the sampling script at this directory:

```sh
python sample.py --out_dir=out-shakespeare-char
```

This generates a few samples, for example:

```
ANGELO:
And cowards it be strawn to my bed,
And thrust the gates of my threats,
Because he that ale away, and hang'd
An one with him.

DUKE VINCENTIO:
I thank your eyes against it.

DUKE VINCENTIO:
Then will answer him to save the malm:
And what have you tyrannous shall do this?

DUKE VINCENTIO:
If you have done evils of all disposition
To end his power, the day of thrust for a common men
That I leave, to fight with over-liking
Hasting in a roseman.
```

lol  `¯\_(ツ)_/¯`. Not bad for a character-level model after 3 minutes of training on a GPU. Better results are quite likely obtainable by instead finetuning a pretrained GPT-2 model on this dataset (see finetuning section later).

**I only have a macbook** (or other cheap computer). No worries, we can still train a GPT but we want to dial things down a notch. I recommend getting the bleeding edge PyTorch nightly ([select it here](https://pytorch.org/get-started/locally/) when installing) as it is currently quite likely to make your code more efficient. But even without it, a simple train run could look as follows:

```sh
python train.py config/train_shakespeare_char.py --device=cpu --compile=False --eval_iters=20 --log_interval=1 --block_size=64 --batch_size=12 --n_layer=4 --n_head=4 --n_embd=128 --max_iters=2000 --lr_decay_iters=2000 --dropout=0.0
```

Here, since we are running on CPU instead of GPU we must set both `--device=cpu` and also turn off PyTorch 2.0 compile with `--compile=False`. Then when we evaluate we get a bit more noisy but faster estimate (`--eval_iters=20`, down from 200), our context size is only 64 characters instead of 256, and the batch size only 12 examples per iteration, not 64. We'll also use a much smaller Transformer (4 layers, 4 heads, 128 embedding size), and decrease the number of iterations to 2000 (and correspondingly usually decay the learning rate to around max_iters with `--lr_decay_iters`). Because our network is so small we also ease down on regularization (`--dropout=0.0`). This still runs in about ~3 minutes, but gets us a loss of only 1.88 and therefore also worse samples, but it's still good fun:

```sh
python sample.py --out_dir=out-shakespeare-char --device=cpu
```
Generates samples like this:

```
GLEORKEN VINGHARD III:
Whell's the couse, the came light gacks,
And the for mought you in Aut fries the not high shee
bot thou the sought bechive in that to doth groan you,
No relving thee post mose the wear
```

Not bad for ~3 minutes on a CPU, for a hint of the right character gestalt. If you're willing to wait longer, feel free to tune the hyperparameters, increase the size of the network, the context length (`--block_size`), the length of training, etc.

Finally, on Apple Silicon Macbooks and with a recent PyTorch version make sure to add `--device=mps` (short for "Metal Performance Shaders"); PyTorch then uses the on-chip GPU that can *significantly* accelerate training (2-3X) and allow you to use larger networks. See [Issue 28](https://github.com/karpathy/nanoGPT/issues/28) for more.

## reproducing GPT-2

A more serious deep learning professional may be more interested in reproducing GPT-2 results. So here we go - we first tokenize the dataset, in this case the [OpenWebText](https://openwebtext2.readthedocs.io/en/latest/), an open reproduction of OpenAI's (private) WebText:

```sh
python data/openwebtext/prepare.py
```

This downloads and tokenizes the [OpenWebText](https://huggingface.co/datasets/openwebtext) dataset. It will create a `train.bin` and `val.bin` which holds the GPT2 BPE token ids in one sequence, stored as raw uint16 bytes. Then we're ready to kick off training. To reproduce GPT-2 (124M) you'll want at least an 8X A100 40GB node and run:

```sh
torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py
```

This will run for about 4 days using PyTorch Distributed Data Parallel (DDP) and go down to loss of ~2.85. Now, a GPT-2 model just evaluated on OWT gets a val loss of about 3.11, but if you finetune it it will come down to ~2.85 territory (due to an apparent domain gap), making the two models ~match.

If you're in a cluster environment and you are blessed with multiple GPU nodes you can make GPU go brrrr e.g. across 2 nodes like:

```sh
# Run on the first (master) node with example IP 123.456.123.456:
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
# Run on the worker node:
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
```

It is a good idea to benchmark your interconnect (e.g. iperf3). In particular, if you don't have Infiniband then also prepend `NCCL_IB_DISABLE=1` to the above launches. Your multinode training will work, but most likely _crawl_. By default checkpoints are periodically written to the `--out_dir`. We can sample from the model by simply `python sample.py`.

Finally, to train on a single GPU simply run the `python train.py` script. Have a look at all of its args, the script tries to be very readable, hackable and transparent. You'll most likely want to tune a number of those variables depending on your needs.

## baselines

OpenAI GPT-2 checkpoints allow us to get some baselines in place for openwebtext. We can get the numbers as follows:

```sh
$ python train.py config/eval_gpt2.py
$ python train.py config/eval_gpt2_medium.py
$ python train.py config/eval_gpt2_large.py
$ python train.py config/eval_gpt2_xl.py
```

and observe the following losses on train and val:

| model | params | train loss | val loss |
| ------| ------ | ---------- | -------- |
| gpt2 | 124M         | 3.11  | 3.12     |
| gpt2-medium | 350M  | 2.85  | 2.84     |
| gpt2-large | 774M   | 2.66  | 2.67     |
| gpt2-xl | 1558M     | 2.56  | 2.54     |

However, we have to note that GPT-2 was trained on (closed, never released) WebText, while OpenWebText is just a best-effort open reproduction of this dataset. This means there is a dataset domain gap. Indeed, taking the GPT-2 (124M) checkpoint and finetuning on OWT directly for a while reaches loss down to ~2.85. This then becomes the more appropriate baseline w.r.t. reproduction.

## finetuning

Finetuning is no different than training, we just make sure to initialize from a pretrained model and train with a smaller learning rate. For an example of how to finetune a GPT on new text go to `data/shakespeare` and run `prepare.py` to download the tiny shakespeare dataset and render it into a `train.bin` and `val.bin`, using the OpenAI BPE tokenizer from GPT-2. Unlike OpenWebText this will run in seconds. Finetuning can take very little time, e.g. on a single GPU just a few minutes. Run an example finetuning like:

```sh
python train.py config/finetune_shakespeare.py
```

This will load the config parameter overrides in `config/finetune_shakespeare.py` (I didn't tune them much though). Basically, we initialize from a GPT2 checkpoint with `init_from` and train as normal, except shorter and with a small learning rate. If you're running out of memory try decreasing the model size (they are `{'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}`) or possibly decreasing the `block_size` (context length). The best checkpoint (lowest validation loss) will be in the `out_dir` directory, e.g. in `out-shakespeare` by default, per the config file. You can then run the code in `sample.py --out_dir=out-shakespeare`:

```
THEODORE:
Thou shalt sell me to the highest bidder: if I die,
I sell thee to the first; if I go mad,
I sell thee to the second; if I
lie, I sell thee to the third; if I slay,
I sell thee to the fourth: so buy or sell,
I tell thee again, thou shalt not sell my
possession.

JULIET:
And if thou steal, thou shalt not sell thyself.

THEODORE:
I do not steal; I sell the stolen goods.

THEODORE:
Thou know'st not what thou sell'st; thou, a woman,
Thou art ever a victim, a thing of no worth:
Thou hast no right, no right, but to be sold.
```

Whoa there, GPT, entering some dark place over there. I didn't really tune the hyperparameters in the config too much, feel free to try!

## sampling / inference

Use the script `sample.py` to sample either from pre-trained GPT-2 models released by OpenAI, or from a model you trained yourself. For example, here is a way to sample from the largest available `gpt2-xl` model:

```sh
python sample.py \
    --init_from=gpt2-xl \
    --start="What is the answer to life, the universe, and everything?" \
    --num_samples=5 --max_new_tokens=100
```

If you'd like to sample from a model you trained, use the `--out_dir` to point the code appropriately. You can also prompt the model with some text from a file, e.g. ```python sample.py --start=FILE:prompt.txt```.

## efficiency notes

For simple model benchmarking and profiling, `bench.py` might be useful. It's identical to what happens in the meat of the training loop of `train.py`, but omits much of the other complexities.

Note that the code by default uses [PyTorch 2.0](https://pytorch.org/get-started/pytorch-2.0/). At the time of writing (Dec 29, 2022) this makes `torch.compile()` available in the nightly release. The improvement from the one line of code is noticeable, e.g. cutting down iteration time from ~250ms / iter to 135ms / iter. Nice work PyTorch team!

## todos

- Investigate and add FSDP instead of DDP
- Eval zero-shot perplexities on standard evals (e.g. LAMBADA? HELM? etc.)
- Finetune the finetuning script, I think the hyperparams are not great
- Schedule for linear batch size increase during training
- Incorporate other embeddings (rotary, alibi)
- Separate out the optim buffers from model params in checkpoints I think
- Additional logging around network health (e.g. gradient clip events, magnitudes)
- Few more investigations around better init etc.

## troubleshooting

Note that by default this repo uses PyTorch 2.0 (i.e. `torch.compile`). This is fairly new and experimental, and not yet available on all platforms (e.g. Windows). If you're running into related error messages try to disable this by adding `--compile=False` flag. This will slow down the code but at least it will run.

For some context on this repository, GPT, and language modeling it might be helpful to watch my [Zero To Hero series](https://karpathy.ai/zero-to-hero.html). Specifically, the [GPT video](https://www.youtube.com/watch?v=kCc8FmEb1nY) is popular if you have some prior language modeling context.

For more questions/discussions feel free to stop by **#nanoGPT** on Discord:

[![](https://dcbadge.vercel.app/api/server/3zy8kqD9Cp?compact=true&style=flat)](https://discord.gg/3zy8kqD9Cp)

## acknowledgements

All nanoGPT experiments are powered by GPUs on [Lambda labs](https://lambdalabs.com), my favorite Cloud GPU provider. Thank you Lambda labs for sponsoring nanoGPT!

## Results

### Hugging Face Evaluation

- GPT-2 pretrained evaluation was completed on the Shakespeare validation set.
- GPT-2 and nanoGPT logits and token IDs were compared.
- GPT-2 fine-tuning results were recorded before and after training.

### W&B Experiment Tracking

- Training metrics were logged to W&B.
- Results were recorded with W&B Tables.
- Model checkpoints and LoRA adapters were uploaded as W&B Artifacts.

## Course Experiment Workflow

This fork records a compact nanoGPT experiment workflow for Shakespeare data, GPT-2 evaluation, fine-tuning, W&B experiment tracking, and reproducibility checks. Generated datasets, local checkpoints, virtual environments, Hugging Face caches, `.env` files, and local `wandb/` cache directories are intentionally excluded from Git. Reproducible source code, configuration files, small reports, plots, and external run or Artifact links are tracked instead.

### Environment and Device Check

The environment is recorded with `make_env_report.py` and `reports/environment.json`. The report captures the Python package context, PyTorch availability, W&B version, and runtime device information. The small CPU-compatible training configuration is stored in `config/train_shakespeare_char_small.py`, which keeps the experiment reproducible on machines without a compatible CUDA build.

Run:

```sh
python make_env_report.py
```

### Shakespeare Data Preparation

Character-level Shakespeare data is prepared with `data/shakespeare_char/prepare.py`, while GPT-2 BPE Shakespeare data is prepared with `data/shakespeare/prepare.py`. The preparation scripts generate local binary files such as `train.bin`, `val.bin`, and `meta.pkl`; these files are ignored because they are deterministic outputs that can be regenerated from the scripts.

Run:

```sh
python data/shakespeare_char/prepare.py
python data/shakespeare/prepare.py
```

### Small Scratch Training and Inference

The small scratch experiment uses `config/task2_scratch.py` and `sample.py`. Fixed prompts are sampled before and after training to compare random-model behavior with the trained character model. The generated samples and training log are saved in `reports/samples_scratch_before.jsonl`, `reports/samples_scratch_after.jsonl`, and `reports/task2_training.log`.

Run:

```sh
python train.py config/task2_scratch.py --device=cpu --compile=False
python sample.py --out_dir=out-task2-scratch --device=cpu --compile=False
```

### Token Frequency and Embedding Norm Statistics

Token statistics are produced by the data preparation scripts and `task4_compare_tokenizer.py`. The tracked reports are `reports/token_frequency_char.csv`, `reports/token_frequency_char.png`, and `reports/tokenizer_comparison.csv`. These files document the character-token distribution and compare character tokenization with GPT-2 BPE tokenization.

Run:

```sh
python task4_compare_tokenizer.py
```

### Hugging Face GPT-2 Inference

Hugging Face GPT-2 inference is implemented in `hf_gpt2_inference.py` and `task3_pretrained_eval.py`, with configuration in `config/task3_pretrained_eval.py`. The evaluation records validation loss, perplexity, parameter count, inference speed, memory information, and fixed-prompt generations in `reports/task3_pretrained_eval.json` and `reports/hf_gpt2_predictions.jsonl`.

Model link: [openai-community/gpt2](https://huggingface.co/openai-community/gpt2)

Run:

```sh
python task3_pretrained_eval.py config/task3_pretrained_eval.py
```

### Hugging Face GPT-2 and nanoGPT GPT-2 Alignment

The alignment check is implemented in `hf_nanogpt_alignment.py`. It compares Hugging Face GPT-2 and nanoGPT GPT-2 token IDs, generated text, and logits differences under fixed prompts and seeds. The result is stored in `reports/hf_nanogpt_alignment.json`.

Run:

```sh
python hf_nanogpt_alignment.py
```

### GPT-2 Fine-Tuning and Inference

GPT-2 fine-tuning uses `config/finetune_shakespeare.py`, `config/task4_finetune.py`, `task4_finetune.py`, and `task4_finalize.py`. The experiment starts from GPT-2 weights and fine-tunes on GPT-2 BPE Shakespeare data. The before-and-after generation comparison is saved in `reports/gpt2_before_after_finetune.jsonl`, and evaluation metrics are saved in `reports/task4_finetune.json`.

Run:

```sh
python task4_finetune.py config/task4_finetune.py
python task4_finalize.py
```

### W&B Experiment Logging and Comparison

The training loop in `train.py` logs train loss, validation loss, learning rate, MFU, iteration loss, iteration time, tokens per second, gradient norm, parameter count, peak memory, token-frequency plots, embedding-norm plots, generation Tables, and checkpoint Artifacts. `task5_compare.py` builds a comparable W&B group for scratch, pretrained, and fine-tuned runs. `task6_qwen_instruct.py` records GPT-2 and Qwen instruction-answer comparisons, and `task8_lora.py` records LoRA results and adapter Artifacts.

W&B links:

- Task 1 run: [task1-scratch](https://wandb.ai/ylan156-hong-kong-university-of-science-and-technology/shakespeare-char/runs/l2b614lk)
- Task 1 model Artifact: [task1-scratch-best-checkpoint:v1](https://wandb.ai/ylan156-hong-kong-university-of-science-and-technology/shakespeare-char/artifacts/model/task1-scratch-best-checkpoint/v1)
- Task 2 scratch run: [task2-scratch-character](https://wandb.ai/ylan156-hong-kong-university-of-science-and-technology/shakespeare-char-task2/runs/09b113i2)
- Task 3 pretrained run: [task3-pretrained-gpt2-eval](https://wandb.ai/ylan156-hong-kong-university-of-science-and-technology/shakespeare-char-task3/runs/x8b56ad4)
- Task 4 fine-tuned run: [task4-finetuned-gpt2](https://wandb.ai/ylan156-hong-kong-university-of-science-and-technology/shakespeare-char-task4/runs/1s5n392c)
- Task 5 group: [task5-comparison](https://wandb.ai/ylan156-hong-kong-university-of-science-and-technology/shakespeare-char-task5/runs?group=task5-comparison)
- Task 6 Qwen Table run: [qwen-vs-gpt2](https://wandb.ai/ylan156-hong-kong-university-of-science-and-technology/shakespeare-char-task6/runs/3vog9xvk)
- Task 8 LoRA run: [task8-gpt2-lora](https://wandb.ai/ylan156-hong-kong-university-of-science-and-technology/shakespeare-char-task8/runs/fvlzh276)
- Task 8 comparison run: [task8-lora-comparison](https://wandb.ai/ylan156-hong-kong-university-of-science-and-technology/shakespeare-char-task8/runs/x125ry0n)
- Task 8 LoRA Artifact: [task8-gpt2-lora-adapter:v0](https://wandb.ai/ylan156-hong-kong-university-of-science-and-technology/shakespeare-char-task8/artifacts/model/task8-gpt2-lora-adapter/v0)

### Qwen Instruct Comparison

The instruction-following comparison uses `task6_qwen_instruct.py` and compares GPT-2 output with Qwen output on fixed prompts. The result table includes the prompt, GPT-2 output, Qwen output, instruction-following score, Chinese expression score, and answer-completeness score. The local summary is stored in `reports/task6_qwen_instruct_wandb.json`.

Model link: [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)

Run:

```sh
python task6_qwen_instruct.py
```

### Tests and Reproducibility Commands

Lightweight reproducibility tests are stored in `tests/test_reproducibility.py`. They check character encode/decode round trip, GPT-2 tokenizer ID consistency when local dependencies are available, small GPT forward output shape, required experiment configuration fields, and that tests do not launch formal training.

The Git/W&B reproducibility linkage is recorded in `reports/reproducibility_run.json`. It stores the GitHub repository URL, current branch, commit SHA, README path, submission manifest path, experiment-summary path, intended W&B project/group/run name, and the exact test command. In this checkout the live W&B reproducibility run is pending because the local Python virtual-environment launchers point to a missing Python executable and `wandb` is not available on `PATH`.

Clean-environment reproduction is recorded in `reports/clean_environment_reproduction.json`. The run used a fresh clone in `C:/Users/lenovo/Documents/Codex/2026-09-10/wa/nanogpt-clean-repro-task4-local2` at commit `bd1660e8c80d49407200c1af7392205f61890899`, installed dependencies into `.venv-repro`, ran syntax checks and all lightweight tests, prepared the character dataset, completed a 2-iteration CPU smoke training run, generated one scratch sample, ran Hugging Face GPT-2 inference, and created a local W&B offline reproducibility run.

Run:

```sh
python -m unittest discover -s tests -v
git status --short --branch
git log --oneline --graph --decorate --all -40
git ls-files | Select-String -Pattern '(^|/)(\.env|wandb|out[^/]*)/|\.pt$|\.bin$|\.pkl$|\.safetensors$'
```

After restoring Python or recreating the virtual environment, create the W&B reproducibility run with the same Git metadata:

```sh
python log_reproducibility_run.py
```

The offline W&B run from the clean reproduction can be synced after login:

```sh
wandb sync C:/Users/lenovo/Documents/Codex/2026-09-10/wa/nanogpt-clean-repro-task4-local2/wandb/offline-run-20260914_201542-822ms38o
```

The final command should produce no output. This confirms that local secrets, caches, checkpoints, and generated binary datasets are not tracked by Git.
