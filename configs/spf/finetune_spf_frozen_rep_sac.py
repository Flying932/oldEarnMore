_base_ = "../mask_sac_portfolio_management.py"

tag = "spf_finetune"

# Throughput runtime defaults for Stage-3
num_envs = 4
vector_env_type = "async"
buffer_device = "cpu"
batch_size = 64
repeat_times = 16
horizon_len = 64
buffer_size = 6000
if_use_tqdm = False

# Path to Stage-1 checkpoint (relative to root or absolute path)
pretrain_path = "workdir/spf_pretrain/pretrain_rep_best.pth"
pretrain_strict = False

# Adapter settings
adapter_rank = 16
adapter_dropout = 0.1
adapter_lr = 5e-5
adapter_actor_lr = 5e-5

# Actor/adapter gradient controls
actor_trainable = False
adapter_actor_grad = False
actor_on_adapter_weight = 0.0
