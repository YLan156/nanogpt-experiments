"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2' # 'run' + str(time.time())
wandb_tags = '' # comma-separated tags added to the automatic scratch/gpt2 tag
wandb_log_model_artifact = True
wandb_log_token_frequency = True
wandb_log_embedding_norm = True
wandb_log_model_stats = True
wandb_log_generations = True
wandb_generation_prompt = 'Once upon a time'
wandb_generation_max_new_tokens = 80
wandb_generation_temperature = 0.8
wandb_generation_top_k = 200
wandb_token_frequency_top_k = 50
wandb_comparison_prompts = '' # use | to separate fixed prompts for before/after comparison
wandb_comparison_max_new_tokens = 80
# data
dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'float32' if device == 'cpu' else ('bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16') # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster
seed = 1337
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(seed + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# poor man's data loader
data_dir = os.path.join('data', dataset)
def get_batch(split):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
meta = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

tokenizer_type = 'character' if meta is not None and 'stoi' in meta and 'itos' in meta else 'gpt2'

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout) # start with model_args from command line
if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.amp.GradScaler('cuda', enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

def get_base_model(model):
    base_model = model.module if hasattr(model, 'module') else model
    return base_model._orig_mod if hasattr(base_model, '_orig_mod') else base_model

def get_wandb_tags():
    init_tag = 'gpt2' if init_from.startswith('gpt2') else init_from
    extra_tags = [tag.strip() for tag in wandb_tags.split(',') if tag.strip()]
    return [init_tag] + extra_tags

def get_gpu_peak_memory_mb():
    if device_type != 'cuda':
        return 0.0
    return torch.cuda.max_memory_allocated(device) / 1024**2

def get_comparison_prompts():
    return [prompt for prompt in wandb_comparison_prompts.split('|') if prompt]

def get_total_grad_norm(parameters):
    grads = [p.grad.detach() for p in parameters if p.grad is not None]
    if len(grads) == 0:
        return 0.0
    norms = torch.stack([torch.linalg.vector_norm(g, 2) for g in grads])
    return torch.linalg.vector_norm(norms, 2).item()

def build_token_frequency_plot(wandb):
    train_bin = os.path.join(data_dir, 'train.bin')
    if not os.path.exists(train_bin):
        return None
    data = np.memmap(train_bin, dtype=np.uint16, mode='r')
    counts = np.bincount(np.asarray(data), minlength=model_args['vocab_size'])
    top_ids = np.argsort(counts)[-wandb_token_frequency_top_k:][::-1]
    table = wandb.Table(columns=['token_id', 'token', 'count'])
    for token_id in top_ids:
        if counts[token_id] == 0:
            continue
        token = decode_tokens([int(token_id)])
        table.add_data(int(token_id), token, int(counts[token_id]))
    return wandb.plot.bar(table, 'token', 'count', title='Token frequency')

def build_embedding_norm_plot(wandb):
    base_model = get_base_model(model)
    with torch.no_grad():
        norms = base_model.transformer.wte.weight.detach().float().norm(dim=1).cpu().numpy()
    top_ids = np.argsort(norms)[-wandb_token_frequency_top_k:][::-1]
    table = wandb.Table(columns=['token_id', 'token', 'embedding_norm'])
    for token_id in top_ids:
        table.add_data(int(token_id), decode_tokens([int(token_id)]), float(norms[token_id]))
    return wandb.plot.bar(table, 'token', 'embedding_norm', title='Embedding norm')

def build_model_stats_table(wandb):
    base_model = get_base_model(model)
    table = wandb.Table(columns=['name', 'value'])
    stats = {
        'parameters': base_model.get_num_params(non_embedding=False),
        'parameters_non_embedding': base_model.get_num_params(non_embedding=True),
        'n_layer': model_args['n_layer'],
        'n_head': model_args['n_head'],
        'n_embd': model_args['n_embd'],
        'block_size': model_args['block_size'],
        'vocab_size': model_args['vocab_size'],
    }
    for name, value in stats.items():
        table.add_data(name, value)
    return table

def encode_prompt(text):
    if tokenizer_type == 'character' and meta is not None:
        return [meta['stoi'][ch] for ch in text if ch in meta['stoi']]
    import tiktoken
    return tiktoken.get_encoding('gpt2').encode(text)

def decode_tokens(token_ids):
    if tokenizer_type == 'character' and meta is not None:
        return ''.join(meta['itos'].get(int(i), '') for i in token_ids)
    import tiktoken
    return tiktoken.get_encoding('gpt2').decode([int(i) for i in token_ids])

@torch.no_grad()
def build_generation_table(wandb):
    prompt_ids = encode_prompt(wandb_generation_prompt)
    if len(prompt_ids) == 0:
        return None
    base_model = get_base_model(model)
    was_training = base_model.training
    base_model.eval()
    idx = torch.tensor(prompt_ids, dtype=torch.long, device=device)[None, ...]
    output = base_model.generate(
        idx,
        max_new_tokens=wandb_generation_max_new_tokens,
        temperature=wandb_generation_temperature,
        top_k=wandb_generation_top_k,
    )
    if was_training:
        base_model.train()
    generated = decode_tokens(output[0].tolist())
    table = wandb.Table(columns=['iter', 'prompt', 'completion'])
    table.add_data(iter_num, wandb_generation_prompt, generated)
    return table

@torch.no_grad()
def generate_fixed_completions(prompts):
    base_model = get_base_model(model)
    was_training = base_model.training
    cpu_rng_state = torch.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state(device) if device_type == 'cuda' else None
    base_model.eval()
    completions = []
    for sample_id, prompt in enumerate(prompts):
        torch.manual_seed(seed + 10000 + sample_id)
        if device_type == 'cuda':
            torch.cuda.manual_seed_all(seed + 10000 + sample_id)
        prompt_ids = encode_prompt(prompt)
        if len(prompt_ids) == 0:
            completions.append('')
            continue
        idx = torch.tensor(prompt_ids, dtype=torch.long, device=device)[None, ...]
        output = base_model.generate(
            idx,
            max_new_tokens=wandb_comparison_max_new_tokens,
            temperature=wandb_generation_temperature,
            top_k=wandb_generation_top_k,
        )
        completions.append(decode_tokens(output[0].tolist()))
    torch.set_rng_state(cpu_rng_state)
    if device_type == 'cuda':
        torch.cuda.set_rng_state(cuda_rng_state, device)
    if was_training:
        base_model.train()
    return completions

def build_comparison_table(wandb, prompts, before, after):
    table = wandb.Table(columns=['sample_id', 'prompt', 'before_training', 'after_training'])
    for sample_id, (prompt, before_text, after_text) in enumerate(zip(prompts, before, after), start=1):
        table.add_data(sample_id, prompt, before_text, after_text)
    return table

def log_wandb_checkpoint_artifact(wandb, ckpt_path):
    if not wandb_log_model_artifact or not os.path.exists(ckpt_path):
        return
    artifact_name = ''.join(ch if ch.isalnum() or ch in '-_.' else '-' for ch in wandb_run_name)
    artifact = wandb.Artifact(
        name=f"{artifact_name}-best-checkpoint",
        type='model',
        metadata={'iter': iter_num, 'best_val_loss': float(best_val_loss), 'init_from': init_from},
    )
    artifact.add_file(ckpt_path)
    logged_artifact = wandb.log_artifact(artifact, aliases=['best', f'iter-{iter_num}'])
    logged_artifact.wait()
    print(f"W&B model Artifact: {logged_artifact.url}")

# logging
if wandb_log and master_process:
    import wandb
    wandb_config = dict(config)
    wandb_config.update({
        'model_args': dict(model_args),
        'tokenizer_type': tokenizer_type,
        'dataset': dataset,
        'seed': seed,
        'batch_size': batch_size,
        'gradient_accumulation_steps': gradient_accumulation_steps,
        'ddp_world_size': ddp_world_size,
        'tokens_per_iter': tokens_per_iter,
        'effective_batch_tokens': tokens_per_iter,
        'device': device,
        'device_type': device_type,
        'parameter_count': get_base_model(model).get_num_params(non_embedding=False),
        'parameter_count_non_embedding': get_base_model(model).get_num_params(non_embedding=True),
    })
    wandb.init(project=wandb_project, name=wandb_run_name, tags=get_wandb_tags(), config=wandb_config)
    print(f"W&B run: {wandb.run.url}")
    initial_logs = {}
    if wandb_log_token_frequency:
        token_frequency_plot = build_token_frequency_plot(wandb)
        if token_frequency_plot is not None:
            initial_logs['charts/token_frequency'] = token_frequency_plot
    if wandb_log_embedding_norm:
        initial_logs['charts/embedding_norm'] = build_embedding_norm_plot(wandb)
    if wandb_log_model_stats:
        initial_logs['tables/model_stats'] = build_model_stats_table(wandb)
    if initial_logs:
        wandb.log(initial_logs, step=iter_num)
    comparison_prompts = get_comparison_prompts()
    comparison_before = generate_fixed_completions(comparison_prompts) if comparison_prompts else []
    if device_type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)

# training loop
X, Y = get_batch('train') # fetch the very first batch
t0 = time.time()
training_start_time = t0
training_peak_memory_mb = 0.0
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = get_base_model(model) # unwrap DDP and torch.compile containers if needed
running_mfu = -1.0
while True:

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log:
            eval_logs = {
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu*100, # convert to percentage
            }
            if wandb_log_embedding_norm:
                eval_logs['charts/embedding_norm'] = build_embedding_norm_plot(wandb)
            if wandb_log_model_stats:
                eval_logs['tables/model_stats'] = build_model_stats_table(wandb)
            if wandb_log_generations:
                generation_table = build_generation_table(wandb)
                if generation_table is not None:
                    eval_logs['tables/generations'] = generation_table
            wandb.log(eval_logs, step=iter_num)
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                ckpt_path = os.path.join(out_dir, 'ckpt.pt')
                torch.save(checkpoint, ckpt_path)
                if wandb_log:
                    log_wandb_checkpoint_artifact(wandb, ckpt_path)
    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch('train')
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    grad_norm = None
    if wandb_log and master_process:
        scaler.unscale_(optimizer)
        grad_norm = get_total_grad_norm(model.parameters())
    # clip the gradient
    if grad_clip != 0.0:
        if grad_norm is None:
            scaler.unscale_(optimizer)
        clipped_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        if grad_norm is None:
            grad_norm = clipped_grad_norm.item()
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        tokens_per_second = tokens_per_iter / dt
        peak_memory_mb = get_gpu_peak_memory_mb()
        training_peak_memory_mb = max(training_peak_memory_mb, peak_memory_mb)
        if iter_num % log_interval == 0:
            print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
        if wandb_log:
            wandb.log({
                'iter': iter_num,
                'iter/loss': lossf,
                'iter/time_ms': dt * 1000,
                'iter/tokens_per_second': tokens_per_second,
                'iter/grad_norm': grad_norm if grad_norm is not None else 0.0,
                'iter/parameter_count': get_base_model(model).get_num_params(non_embedding=False),
                'iter/peak_memory_mb': peak_memory_mb,
                'lr': lr,
                'mfu': running_mfu * 100,
            }, step=iter_num)
        if device_type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

if wandb_log and master_process:
    training_time_seconds = time.time() - training_start_time
    comparison_prompts = get_comparison_prompts()
    if comparison_prompts:
        comparison_after = generate_fixed_completions(comparison_prompts)
        wandb.log({
            'tables/fixed_prompt_comparison': build_comparison_table(
                wandb, comparison_prompts, comparison_before, comparison_after
            ),
            'training/time_seconds': training_time_seconds,
            'training/peak_memory_mb': training_peak_memory_mb,
            'training/total_tokens': iter_num * tokens_per_iter,
        }, step=iter_num)
    else:
        wandb.log({
            'training/time_seconds': training_time_seconds,
            'training/peak_memory_mb': training_peak_memory_mb,
            'training/total_tokens': iter_num * tokens_per_iter,
        }, step=iter_num)

if ddp:
    destroy_process_group()
