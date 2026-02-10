_base_ = "../mask_sac_portfolio_management.py"

tag = "spf_pretrain"

# Stage-1 training setup
pretrain_num_epochs = 1200
pretrain_min_epochs = 500
pretrain_patience = 50
pretrain_min_delta = 1e-4
pretrain_batch_size = 64
pretrain_num_workers = 8
pretrain_lr = 5e-5
pretrain_use_scheduler = False
pretrain_select_by = "total"  # total | recon | contrastive
pretrain_clip_grad_norm = 3.0

# Throughput / speed options
pretrain_use_amp = True
pretrain_amp_dtype = "fp16"  # fp16 | bf16
pretrain_pin_memory = True
pretrain_persistent_workers = True
pretrain_prefetch_factor = 2
pretrain_non_blocking = True
pretrain_compile = False
pretrain_compile_mode = "default"
pretrain_data_parallel = False
pretrain_deterministic = False
pretrain_cudnn_benchmark = True

# SPF pretraining objectives
lambda_recon = 1.0
lambda_contrastive = 0.7
contrastive_temperature = 0.2

# Masking and pair settings
mask_ratio = 0.25
pretrain_mask_ratio_delta = 0.01
corr_threshold = 0.7
corr_positive_weight = 0.5
pretrain_min_shift = 1
pretrain_max_shift = 5
