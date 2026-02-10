import argparse
import json
import os
import random
import sys
from copy import deepcopy
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from mmengine.config import Config, DictAction
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

ROOT = str(Path(__file__).resolve().parents[1])
sys.path.append(ROOT)

from pm.registry import DATASET, ENVIRONMENT, NET, OPTIMIZER, SCHEDULER
from pm.utils import update_data_root
import pm.net  # noqa: F401


def init_runtime(seed=3407, deterministic=False, cudnn_benchmark=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = bool(cudnn_benchmark) and not bool(deterministic)
    torch.set_default_dtype(torch.float32)


def setup_distributed() -> Tuple[bool, int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return False, 0, 1, 0, device

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if not dist.is_initialized():
        dist.init_process_group(backend=backend, init_method="env://")
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    return True, rank, world_size, local_rank, device


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    if not dist.is_available() or not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def parse_amp_dtype(amp_dtype_name: str):
    name = str(amp_dtype_name).lower()
    if name in ("fp16", "float16", "half"):
        return torch.float16
    if name in ("bf16", "bfloat16"):
        return torch.bfloat16
    raise ValueError(f"unsupported amp dtype: {amp_dtype_name}, expected fp16 or bf16")


class WindowPairDataset(Dataset):
    def __init__(
        self,
        features: np.ndarray,
        days: int,
        min_shift: int = 1,
        max_shift: int = 5,
    ):
        super().__init__()
        self.features = features.astype(np.float32)  # [N, T, F]
        self.days = days
        self.min_shift = max(1, min_shift)
        self.max_shift = max(self.min_shift, max_shift)
        self.min_center = days - 1
        self.max_center = self.features.shape[1] - 1
        self.centers = np.arange(self.min_center, self.max_center + 1)

    def __len__(self):
        return len(self.centers)

    def _window(self, center: int):
        start = center - self.days + 1
        end = center + 1
        return self.features[:, start:end, :]

    def __getitem__(self, index):
        center_1 = int(self.centers[index])
        shift = random.randint(self.min_shift, self.max_shift)
        center_2 = min(center_1 + shift, self.max_center)
        if center_2 < self.min_center:
            center_2 = self.min_center
        x1 = self._window(center_1)
        x2 = self._window(center_2)
        return torch.from_numpy(x1), torch.from_numpy(x2)


def parse_args():
    parser = argparse.ArgumentParser(description="SPF Stage-1 pretraining")
    parser.add_argument(
        "--config",
        default=os.path.join(ROOT, "configs", "spf", "pretrain_spf_portfolio_management.py"),
        help="config file path",
    )
    parser.add_argument("--root", type=str, default=ROOT)
    parser.add_argument("--workdir", type=str, default="workdir")
    parser.add_argument("--tag", type=str, default="spf_pretrain")
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="override config in key=value format",
    )
    return parser.parse_args()


def build_corr_pos_index(labels: np.ndarray, corr_threshold: float):
    rets = labels[:, :, 0]  # ret1
    corr = np.corrcoef(rets)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    num_stocks = corr.shape[0]
    corr_pos_index = []
    for i in range(num_stocks):
        row = corr[i].copy()
        row[i] = -np.inf
        candidate = np.where(row > corr_threshold)[0]
        if len(candidate) > 0:
            best = candidate[np.argmax(row[candidate])]
            corr_pos_index.append(int(best))
        else:
            corr_pos_index.append(i)
    return np.asarray(corr_pos_index, dtype=np.int64)


def build_train_val_features(cfg):
    dataset = DATASET.build(cfg.dataset)

    train_env_cfg = deepcopy(cfg.environment)
    train_env_cfg.update(
        dict(
            mode="train",
            if_norm=True,
            dataset=dataset,
            start_date=cfg.train_start_date,
            end_date=cfg.val_start_date,
        )
    )
    train_env = ENVIRONMENT.build(train_env_cfg)

    val_env_cfg = deepcopy(cfg.environment)
    val_env_cfg.update(
        dict(
            mode="val",
            if_norm=True,
            dataset=dataset,
            scaler=train_env.scaler,
            start_date=cfg.val_start_date,
            end_date=getattr(cfg, "test_start_date", None),
        )
    )
    val_env = ENVIRONMENT.build(val_env_cfg)

    return train_env.features, val_env.features, train_env.labels


def build_pretrain_model(cfg, mask_ratio):
    rep_net = deepcopy(cfg.rep_net)
    delta = float(getattr(cfg, "pretrain_mask_ratio_delta", 0.01))
    rep_net.update(
        dict(
            mask_ratio_min=max(0.0, mask_ratio - delta),
            mask_ratio_max=min(1.0, mask_ratio + delta),
            mask_ratio_mu=mask_ratio,
            mask_ratio_std=max(1e-3, delta),
        )
    )
    return NET.build(rep_net)


def build_optimizer_scheduler(cfg, model, pretrain_lr, total_steps):
    optimizer_cfg = deepcopy(getattr(cfg, "optimizer", dict(type="AdamW", params=None, lr=pretrain_lr)))
    optimizer_cfg.update(dict(params=model.parameters(), lr=pretrain_lr))
    optimizer = OPTIMIZER.build(optimizer_cfg)

    use_scheduler = bool(getattr(cfg, "pretrain_use_scheduler", False))
    if not use_scheduler:
        return optimizer, None

    scheduler_cfg = deepcopy(getattr(cfg, "scheduler", None))
    if scheduler_cfg is None:
        return optimizer, None

    scheduler_cfg.update(dict(optimizer=optimizer))
    if "t_in_epochs" in scheduler_cfg:
        scheduler_cfg["t_in_epochs"] = False
    if "t_initial" in scheduler_cfg:
        scheduler_cfg["t_initial"] = max(total_steps, 1)
    if "multi_steps" in scheduler_cfg:
        scheduler_cfg["multi_steps"] = [int(total_steps * 0.6), int(total_steps * 0.8)]
    scheduler = SCHEDULER.build(scheduler_cfg)
    return optimizer, scheduler


def get_select_score(stats, select_by: str):
    if select_by == "recon":
        return stats["loss_recon"]
    if select_by == "contrastive":
        return stats["loss_contrastive"]
    return stats["loss_total"]


def run_epoch(
    model,
    dataloader,
    optimizer,
    scheduler,
    device,
    lambda_recon,
    lambda_contrastive,
    temperature,
    corr_pos_index,
    corr_positive_weight,
    global_step,
    clip_grad_norm=3.0,
    use_amp=False,
    amp_dtype=torch.float16,
    scaler=None,
    non_blocking=True,
    distributed=False,
    train=True,
):
    mode_ctx = torch.enable_grad if train else torch.no_grad
    model.train(train)
    meter = {
        "loss_total": 0.0,
        "loss_recon": 0.0,
        "loss_contrastive": 0.0,
    }
    count = 0

    with mode_ctx():
        for x1, x2 in dataloader:
            x1 = x1.to(device=device, dtype=torch.float32, non_blocking=non_blocking)
            x2 = x2.to(device=device, dtype=torch.float32, non_blocking=non_blocking)
            amp_enabled = bool(use_amp and device.type == "cuda")

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                output = model.forward_pretrain(
                    x1=x1,
                    x2=x2,
                    lambda_recon=lambda_recon,
                    lambda_contrastive=lambda_contrastive,
                    temperature=temperature,
                    corr_pos_index=corr_pos_index,
                    corr_positive_weight=corr_positive_weight,
                )
                loss = output["loss_total"]

            if train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    if clip_grad_norm is not None and clip_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if clip_grad_norm is not None and clip_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)
                    optimizer.step()

                if scheduler is not None:
                    scheduler.step_update(global_step)
                global_step += 1

            meter["loss_total"] += loss.detach().item()
            meter["loss_recon"] += output["loss_recon"].detach().item()
            meter["loss_contrastive"] += output["loss_contrastive"].detach().item()
            count += 1

    if distributed and dist.is_initialized():
        packed = torch.tensor(
            [
                meter["loss_total"],
                meter["loss_recon"],
                meter["loss_contrastive"],
                float(count),
            ],
            dtype=torch.float64,
            device=device,
        )
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)
        meter["loss_total"] = packed[0].item()
        meter["loss_recon"] = packed[1].item()
        meter["loss_contrastive"] = packed[2].item()
        count = int(packed[3].item())

    for k in meter:
        meter[k] = meter[k] / max(count, 1)
    return meter, global_step


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    if args.cfg_options is None:
        args.cfg_options = {}
    if args.root is not None:
        args.cfg_options["root"] = args.root
    if args.workdir is not None:
        args.cfg_options["workdir"] = args.workdir
    if args.tag is not None:
        args.cfg_options["tag"] = args.tag
    cfg.merge_from_dict(args.cfg_options)
    update_data_root(cfg, root=args.root)

    distributed, rank, world_size, local_rank, device = setup_distributed()
    writer: Optional[SummaryWriter] = None
    try:
        seed = int(getattr(cfg, "seed", 42))
        deterministic = bool(getattr(cfg, "pretrain_deterministic", False))
        cudnn_benchmark = bool(getattr(cfg, "pretrain_cudnn_benchmark", not deterministic))
        init_runtime(seed=seed + rank, deterministic=deterministic, cudnn_benchmark=cudnn_benchmark)

        exp_path = os.path.join(cfg.root, cfg.workdir, cfg.tag)
        if is_main_process():
            os.makedirs(exp_path, exist_ok=True)
            writer = SummaryWriter(exp_path)

        pretrain_num_epochs = int(getattr(cfg, "pretrain_num_epochs", 1200))
        pretrain_min_epochs = int(getattr(cfg, "pretrain_min_epochs", 500))
        pretrain_patience = int(getattr(cfg, "pretrain_patience", 50))
        pretrain_min_delta = float(getattr(cfg, "pretrain_min_delta", 1e-4))

        pretrain_batch_size = int(getattr(cfg, "pretrain_batch_size", getattr(cfg, "batch_size", 64)))
        pretrain_num_workers = int(getattr(cfg, "pretrain_num_workers", 0))
        pretrain_pin_memory = bool(getattr(cfg, "pretrain_pin_memory", True))
        pretrain_persistent_workers = bool(getattr(cfg, "pretrain_persistent_workers", True)) and pretrain_num_workers > 0
        pretrain_prefetch_factor = int(getattr(cfg, "pretrain_prefetch_factor", 2))
        pretrain_non_blocking = bool(getattr(cfg, "pretrain_non_blocking", True))

        pretrain_lr = float(getattr(cfg, "pretrain_lr", getattr(cfg, "rep_lr", 5e-5)))
        pretrain_clip_grad_norm = float(getattr(cfg, "pretrain_clip_grad_norm", 3.0))
        pretrain_min_shift = int(getattr(cfg, "pretrain_min_shift", 1))
        pretrain_max_shift = int(getattr(cfg, "pretrain_max_shift", 5))
        pretrain_select_by = str(getattr(cfg, "pretrain_select_by", "total")).lower()

        pretrain_use_amp = bool(getattr(cfg, "pretrain_use_amp", True))
        pretrain_amp_dtype = str(getattr(cfg, "pretrain_amp_dtype", "fp16"))
        amp_dtype = parse_amp_dtype(pretrain_amp_dtype) if pretrain_use_amp else torch.float16
        pretrain_data_parallel = bool(getattr(cfg, "pretrain_data_parallel", False))
        pretrain_compile = bool(getattr(cfg, "pretrain_compile", False))
        pretrain_compile_mode = str(getattr(cfg, "pretrain_compile_mode", "default"))

        lambda_recon = float(getattr(cfg, "lambda_recon", 1.0))
        lambda_contrastive = float(getattr(cfg, "lambda_contrastive", 0.7))
        mask_ratio = float(getattr(cfg, "mask_ratio", 0.25))
        corr_threshold = float(getattr(cfg, "corr_threshold", 0.7))
        corr_positive_weight = float(getattr(cfg, "corr_positive_weight", 0.5))
        temperature = float(getattr(cfg, "contrastive_temperature", 0.2))

        if is_main_process():
            print(
                f"[pretrain] distributed={distributed}, world_size={world_size}, local_rank={local_rank}, device={device}"
            )
            print(
                f"[pretrain] epochs={pretrain_num_epochs}, min_epochs={pretrain_min_epochs}, "
                f"patience={pretrain_patience}, min_delta={pretrain_min_delta:.2e}"
            )
            print(
                f"[pretrain] batch_size={pretrain_batch_size}, workers={pretrain_num_workers}, lr={pretrain_lr}, "
                f"amp={pretrain_use_amp}({pretrain_amp_dtype}), compile={pretrain_compile}"
            )
            print(
                f"[pretrain] lambda_recon={lambda_recon}, lambda_contrastive={lambda_contrastive}, "
                f"mask_ratio={mask_ratio}"
            )
            print(
                f"[pretrain] corr_threshold={corr_threshold}, corr_positive_weight={corr_positive_weight}, "
                f"temperature={temperature}"
            )

        train_features, val_features, train_labels = build_train_val_features(cfg)
        train_dataset = WindowPairDataset(
            features=train_features,
            days=cfg.days,
            min_shift=pretrain_min_shift,
            max_shift=pretrain_max_shift,
        )
        val_dataset = WindowPairDataset(
            features=val_features,
            days=cfg.days,
            min_shift=pretrain_min_shift,
            max_shift=pretrain_max_shift,
        )

        train_sampler = None
        val_sampler = None
        if distributed:
            train_sampler = DistributedSampler(
                train_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True
            )
            val_sampler = DistributedSampler(
                val_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False
            )

        train_loader_kwargs = dict(
            dataset=train_dataset,
            batch_size=pretrain_batch_size,
            num_workers=pretrain_num_workers,
            drop_last=True,
            pin_memory=pretrain_pin_memory,
            persistent_workers=pretrain_persistent_workers,
        )
        val_loader_kwargs = dict(
            dataset=val_dataset,
            batch_size=pretrain_batch_size,
            num_workers=pretrain_num_workers,
            drop_last=False,
            pin_memory=pretrain_pin_memory,
            persistent_workers=pretrain_persistent_workers,
        )
        if pretrain_num_workers > 0:
            train_loader_kwargs["prefetch_factor"] = pretrain_prefetch_factor
            val_loader_kwargs["prefetch_factor"] = pretrain_prefetch_factor

        if distributed:
            train_loader_kwargs["sampler"] = train_sampler
            train_loader_kwargs["shuffle"] = False
            val_loader_kwargs["sampler"] = val_sampler
            val_loader_kwargs["shuffle"] = False
        else:
            train_loader_kwargs["shuffle"] = True
            val_loader_kwargs["shuffle"] = False

        train_loader = DataLoader(**train_loader_kwargs)
        val_loader = DataLoader(**val_loader_kwargs)

        model = build_pretrain_model(cfg, mask_ratio=mask_ratio).to(device)
        if pretrain_compile and hasattr(torch, "compile"):
            model = torch.compile(model, mode=pretrain_compile_mode)

        if distributed:
            model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None)
        elif pretrain_data_parallel and torch.cuda.is_available() and torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model).to(device)
            if is_main_process():
                print(f"[pretrain] enabled DataParallel on {torch.cuda.device_count()} GPUs")

        total_steps = pretrain_num_epochs * max(len(train_loader), 1)
        optimizer, scheduler = build_optimizer_scheduler(cfg, model, pretrain_lr, total_steps)
        use_grad_scaler = pretrain_use_amp and device.type == "cuda" and amp_dtype == torch.float16
        scaler = torch.cuda.amp.GradScaler(enabled=use_grad_scaler)

        corr_pos_index = build_corr_pos_index(train_labels, corr_threshold).tolist()

        best_score = float("inf")
        best_epoch = -1
        global_step = 0
        no_improve_epochs = 0
        best_path = os.path.join(exp_path, "pretrain_rep_best.pth")

        for epoch in range(1, pretrain_num_epochs + 1):
            if distributed and train_sampler is not None:
                train_sampler.set_epoch(epoch)

            train_stats, global_step = run_epoch(
                model=model,
                dataloader=train_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                lambda_recon=lambda_recon,
                lambda_contrastive=lambda_contrastive,
                temperature=temperature,
                corr_pos_index=corr_pos_index,
                corr_positive_weight=corr_positive_weight,
                global_step=global_step,
                clip_grad_norm=pretrain_clip_grad_norm,
                use_amp=pretrain_use_amp,
                amp_dtype=amp_dtype,
                scaler=scaler,
                non_blocking=pretrain_non_blocking,
                distributed=distributed,
                train=True,
            )
            val_stats, _ = run_epoch(
                model=model,
                dataloader=val_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                lambda_recon=lambda_recon,
                lambda_contrastive=lambda_contrastive,
                temperature=temperature,
                corr_pos_index=corr_pos_index,
                corr_positive_weight=corr_positive_weight,
                global_step=global_step,
                clip_grad_norm=pretrain_clip_grad_norm,
                use_amp=pretrain_use_amp,
                amp_dtype=amp_dtype,
                scaler=scaler,
                non_blocking=pretrain_non_blocking,
                distributed=distributed,
                train=False,
            )

            select_score = get_select_score(val_stats, pretrain_select_by)
            improved = select_score < (best_score - pretrain_min_delta)
            if improved:
                best_score = select_score
                best_epoch = epoch
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1

            if is_main_process():
                assert writer is not None
                writer.add_scalar("train/loss_recon", train_stats["loss_recon"], epoch)
                writer.add_scalar("train/loss_contrastive", train_stats["loss_contrastive"], epoch)
                writer.add_scalar("train/loss_total", train_stats["loss_total"], epoch)
                writer.add_scalar("val/loss_recon", val_stats["loss_recon"], epoch)
                writer.add_scalar("val/loss_contrastive", val_stats["loss_contrastive"], epoch)
                writer.add_scalar("val/loss_total", val_stats["loss_total"], epoch)
                writer.add_scalar("val/select_score", select_score, epoch)
                writer.add_scalar("val/no_improve_epochs", no_improve_epochs, epoch)

                print(
                    f"[epoch {epoch:04d}/{pretrain_num_epochs:04d}] "
                    f"train_total={train_stats['loss_total']:.6f} "
                    f"train_recon={train_stats['loss_recon']:.6f} "
                    f"train_contrastive={train_stats['loss_contrastive']:.6f} "
                    f"val_total={val_stats['loss_total']:.6f} "
                    f"val_recon={val_stats['loss_recon']:.6f} "
                    f"val_contrastive={val_stats['loss_contrastive']:.6f} "
                    f"select_score={select_score:.6f} "
                    f"best={best_score:.6f} "
                    f"bad_epochs={no_improve_epochs}/{pretrain_patience}"
                )

                if improved:
                    raw_model = model.module if hasattr(model, "module") else model
                    save_obj = {
                        "rep": raw_model.state_dict(),
                        "epoch": epoch,
                        "best_val_recon_loss": val_stats["loss_recon"],
                        "best_val_contrastive_loss": val_stats["loss_contrastive"],
                        "best_val_total_loss": val_stats["loss_total"],
                        "best_val_select_score": select_score,
                        "pretrain_select_by": pretrain_select_by,
                        "lambda_recon": lambda_recon,
                        "lambda_contrastive": lambda_contrastive,
                        "mask_ratio": mask_ratio,
                        "corr_threshold": corr_threshold,
                        "corr_positive_weight": corr_positive_weight,
                        "temperature": temperature,
                    }
                    torch.save(save_obj, best_path)
                    print(f"[pretrain] saved best checkpoint to {best_path}")

                with open(os.path.join(exp_path, "pretrain_log.txt"), "a", encoding="utf-8") as op:
                    op.write(
                        json.dumps(
                            {
                                "epoch": epoch,
                                "train": train_stats,
                                "val": val_stats,
                                "select_score": select_score,
                                "best_score": best_score,
                                "best_epoch": best_epoch,
                                "improved": improved,
                                "no_improve_epochs": no_improve_epochs,
                            }
                        )
                        + "\n"
                    )

            should_stop = False
            if epoch >= pretrain_min_epochs and no_improve_epochs >= pretrain_patience:
                should_stop = True

            if distributed and dist.is_initialized():
                stop_flag = torch.tensor([int(should_stop)], dtype=torch.int32, device=device)
                dist.broadcast(stop_flag, src=0)
                should_stop = bool(stop_flag.item())

            if should_stop:
                if is_main_process():
                    print(
                        f"[pretrain] early stopping at epoch {epoch}: "
                        f"no improvement for {no_improve_epochs} epochs "
                        f"(min_epochs={pretrain_min_epochs}, patience={pretrain_patience})."
                    )
                break

        if is_main_process():
            print(f"[pretrain] finished, best_epoch={best_epoch}, best_score={best_score:.6f}")
            print(f"[pretrain] best checkpoint: {best_path}")
    finally:
        if writer is not None:
            writer.close()
        if distributed:
            cleanup_distributed()


if __name__ == "__main__":
    main()
