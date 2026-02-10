# SPF Running Guide (EarnMore)

This document describes how to run the new SPF pipeline:

- Stage 1: self-supervised pretraining (`tools/pretrain_masked_rep.py`)
- Stage 3 finetune: frozen rep + hierarchical adapters + SAC (`tools/finetune_sac_frozen_rep.py`)

## 1. Environment

Install dependencies first (same as project baseline):

```bash
pip install -r requirements.txt
```

If `gym` is not installed in your environment, install one compatible version:

```bash
pip install gym==0.21.0
```

## 2. Stage 1 Pretraining (run this first)

Default config:

```bash
python tools/pretrain_masked_rep.py \
  --config configs/spf/pretrain_spf_portfolio_management.py \
  --root /root/oldEarnMore \
  --workdir workdir \
  --tag spf_pretrain
```

Outputs:

- Best rep checkpoint: `workdir/spf_pretrain/pretrain_rep_best.pth`
- Logs: `workdir/spf_pretrain/pretrain_log.txt`
- TensorBoard logs under the same folder

Default key settings (already in config):

- `lambda_recon=1.0`
- `lambda_contrastive=0.7`
- `mask_ratio=0.25`
- `corr_threshold=0.7`
- `pretrain_select_by="total"` (uses `val_total_loss` to choose best checkpoint)
- `pretrain_num_epochs=1200`
- `pretrain_min_epochs=500`
- `pretrain_patience=50`
- `pretrain_min_delta=1e-4`
- no per-batch progress bar; only one concise line per epoch
- AMP enabled by default (`pretrain_use_amp=True`, `pretrain_amp_dtype="fp16"`)

### Quick override example

```bash
python tools/pretrain_masked_rep.py \
  --config configs/spf/pretrain_spf_portfolio_management.py \
  --cfg-options pretrain_num_epochs=20 pretrain_batch_size=32 lambda_contrastive=0.5
```

### Multi-GPU pretrain (DDP)

Use `torchrun` for multi-GPU scaling on a single node:

```bash
torchrun --nproc_per_node=4 tools/pretrain_masked_rep.py \
  --config configs/spf/pretrain_spf_portfolio_management.py \
  --root /root/oldEarnMore \
  --workdir workdir \
  --tag spf_pretrain_ddp
```

Optional speed tuning:
- increase `pretrain_batch_size` according to GPU memory
- tune `pretrain_num_workers` (e.g., 8/12/16)
- if your GPU supports it well, try `pretrain_amp_dtype=bf16`

## 3. Stage 3 Finetune (Frozen Rep + Adapter + SAC)

After Stage 1 finishes:

```bash
python tools/finetune_sac_frozen_rep.py \
  --config configs/spf/finetune_spf_frozen_rep_sac.py \
  --pretrain-path workdir/spf_pretrain/pretrain_rep_best.pth \
  --root /root/oldEarnMore \
  --workdir workdir \
  --tag spf_finetune
```

Outputs:

- Training checkpoints: `workdir/spf_finetune/checkpoint_XXXX.pth`
- Best checkpoint: `workdir/spf_finetune/best.pth`
- Logs: `workdir/spf_finetune/train_log.txt`, `train_infos.txt`
- Parameter statistics:
  - `workdir/spf_finetune/param_stats.txt`
  - `workdir/spf_finetune/param_stats.json`

## 4. Adapter/Actor gradient control

Default behavior:

- `actor_trainable=False`
- `adapter_actor_grad=False`
- `actor_on_adapter_weight=0.0`

This means only `Adapter + Critic + alpha_log` are updated.

Default Stage-3 throughput settings in `configs/spf/finetune_spf_frozen_rep_sac.py`:
- `num_envs=4` with `vector_env_type="async"` (parallel environment sampling)
- `buffer_device="cpu"` (avoid replay-buffer CUDA OOM)
- `batch_size=64`, `repeat_times=16` (balanced speed/stability)

### 24GB GPU presets (Stage-3)

Use these presets with `--cfg-options`:

1. Stable (Recommended)
```bash
python tools/finetune_sac_frozen_rep.py \
  --config configs/spf/finetune_spf_frozen_rep_sac.py \
  --pretrain-path workdir/spf_pretrain/pretrain_rep_best.pth \
  --root /root/oldEarnMore \
  --workdir workdir \
  --tag spf_finetune_stable \
  --cfg-options num_envs=2 batch_size=32 repeat_times=8 buffer_size=3000 buffer_device=cpu
```

2. Balanced
```bash
python tools/finetune_sac_frozen_rep.py \
  --config configs/spf/finetune_spf_frozen_rep_sac.py \
  --pretrain-path workdir/spf_pretrain/pretrain_rep_best.pth \
  --root /root/oldEarnMore \
  --workdir workdir \
  --tag spf_finetune_balanced \
  --cfg-options num_envs=4 batch_size=64 repeat_times=16 buffer_size=6000 buffer_device=cpu
```

3. Aggressive
```bash
python tools/finetune_sac_frozen_rep.py \
  --config configs/spf/finetune_spf_frozen_rep_sac.py \
  --pretrain-path workdir/spf_pretrain/pretrain_rep_best.pth \
  --root /root/oldEarnMore \
  --workdir workdir \
  --tag spf_finetune_aggressive \
  --cfg-options num_envs=8 batch_size=96 repeat_times=16 buffer_size=10000 buffer_device=cpu
```

If you still hit OOM, reduce in this order: `num_envs -> batch_size -> repeat_times`.

Enable actor training:

```bash
python tools/finetune_sac_frozen_rep.py \
  --config configs/spf/finetune_spf_frozen_rep_sac.py \
  --pretrain-path workdir/spf_pretrain/pretrain_rep_best.pth \
  --actor-trainable true
```

Allow actor gradient to also update adapter (only effective when weight > 0):

```bash
python tools/finetune_sac_frozen_rep.py \
  --config configs/spf/finetune_spf_frozen_rep_sac.py \
  --pretrain-path workdir/spf_pretrain/pretrain_rep_best.pth \
  --adapter-actor-grad true \
  --actor-on-adapter-weight 0.2
```

## 5. Notes

- Stage 1 validates and logs all three losses:
  - `val/loss_recon`
  - `val/loss_contrastive`
  - `val/loss_total`
- Best pretrain model is selected by `pretrain_select_by` (default: `total`).
- Finetune keeps baseline portfolio metrics (ARR, SR, MDD, CR, VOL, DD, SOR).

## 6. Test on DJ30 test split

Use SPF-specific test script (keeps frozen-rep + adapter inference path):

```bash
python tools/test_spf_dj30.py \
  --config configs/spf/finetune_spf_frozen_rep_sac.py \
  --root /root/oldEarnMore \
  --workdir workdir \
  --tag spf_finetune_balanced \
  --checkpoint workdir/spf_finetune_balanced/best.pth
```

Outputs:
- terminal prints full metrics and summary (`mean_ARR`, `mean_SR`, `mean_MDD`)
- saved file: `workdir/<tag>/test_metrics.json`
