# Task 4: GPT-2 pretrained fine-tuning on GPT-2 BPE Shakespeare.
dataset = 'shakespeare'
model_name = 'gpt2'
hf_model_name = 'openai-community/gpt2'
wandb_project = 'shakespeare-char-task4'
wandb_run_name = 'task4-finetuned-gpt2'
wandb_tags = 'pretrained,gpt2,finetuned'
seed = 1337
device = 'cpu'
batch_size = 1
gradient_accumulation_steps = 8
max_iters = 20
eval_interval = 5
eval_iters = 20
learning_rate = 3e-5
