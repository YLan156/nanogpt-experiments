# Task 3: GPT-2 pretrained eval-only run.
dataset = 'shakespeare'
model_name = 'gpt2'
hf_model_name = 'openai-community/gpt2'
wandb_project = 'shakespeare-char-task3'
wandb_run_name = 'task3-pretrained-gpt2-eval'
wandb_tags = 'pretrained,gpt2,eval-only'
seed = 1337
device = 'cpu'
eval_iters = 20
