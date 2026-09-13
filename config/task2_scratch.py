# Task 2: record a small character-level scratch training experiment.

out_dir = 'out-task2-scratch'
init_from = 'scratch'

wandb_log = True
wandb_project = 'shakespeare-char-task2'
wandb_run_name = 'task2-scratch-character'
wandb_tags = 'character-tokenizer'
wandb_log_model_artifact = True
wandb_log_token_frequency = False
wandb_log_embedding_norm = False
wandb_log_model_stats = True
wandb_log_generations = False

# Five deterministic prompts are generated before and after training and
# compared in one W&B Table.
wandb_comparison_prompts = 'ROMEO:|JULIET:|To be, or not to be:|Friends, Romans, countrymen,|Once more unto the breach,'
wandb_comparison_max_new_tokens = 80
wandb_generation_temperature = 0.8
wandb_generation_top_k = 40

dataset = 'shakespeare_char'
gradient_accumulation_steps = 1
batch_size = 32
block_size = 128

n_layer = 4
n_head = 4
n_embd = 128
dropout = 0.2

learning_rate = 1e-3
max_iters = 300
eval_interval = 50
eval_iters = 20
log_interval = 25
always_save_checkpoint = False
lr_decay_iters = 300
min_lr = 1e-4
warmup_iters = 30
beta2 = 0.99

device = 'cpu'
dtype = 'float32'
compile = False
