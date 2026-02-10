import argparse
import json
import os
import random
import sys
import warnings
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from typing import Dict, List

import gym
import numpy as np
import torch
from iopath.common.file_io import g_pathmgr as pathmgr
from mmengine.config import Config, DictAction
from torch.utils.tensorboard import SummaryWriter

warnings.filterwarnings("ignore")

ROOT = str(Path(__file__).resolve().parents[1])
sys.path.append(ROOT)

from pm.registry import AGENT, DATASET, ENVIRONMENT
from pm.utils import (
    ReplayBuffer,
    update_data_root,
    load_checkpoint,
    save_checkpoint,
    find_latest_checkpoint,
    print_table,
)
import pm.net  # noqa: F401
import pm.agent  # noqa: F401


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist()
    return obj


def align_transition_shape_with_num_envs(cfg):
    if not hasattr(cfg, "transition_shape"):
        return

    num_envs = int(cfg.num_envs)
    old_transition_shape = deepcopy(cfg.transition_shape)
    new_transition_shape = {}
    changed = {}

    for key, spec in old_transition_shape.items():
        item = dict(spec)
        shape = tuple(item.get("shape", ()))
        if len(shape) > 0 and shape[0] != num_envs:
            new_shape = (num_envs, *shape[1:])
            changed[key] = (shape, new_shape)
            item["shape"] = new_shape
        else:
            item["shape"] = shape
        new_transition_shape[key] = item

    cfg.transition_shape = new_transition_shape

    if len(changed) > 0:
        msg = ", ".join([f"{k}:{v[0]}->{v[1]}" for k, v in changed.items()])
        print(f"[runtime] align transition_shape with num_envs={num_envs}: {msg}")


def module_param_stats(module: torch.nn.Module, name: str) -> Dict:
    total = 0
    trainable = 0
    trainable_names: List[str] = []
    frozen_names: List[str] = []
    for n, p in module.named_parameters():
        num = int(p.numel())
        total += num
        if p.requires_grad:
            trainable += num
            trainable_names.append(f"{name}.{n}")
        else:
            frozen_names.append(f"{name}.{n}")
    return {
        "name": name,
        "total_params": total,
        "trainable_params": trainable,
        "frozen_params": total - trainable,
        "trainable_ratio": float(trainable / total) if total > 0 else 0.0,
        "trainable_names": trainable_names,
        "frozen_names": frozen_names,
    }


def dump_param_stats(agent, exp_path: str):
    module_stats = [
        module_param_stats(agent.rep, "rep"),
        module_param_stats(agent.adapter, "adapter"),
        module_param_stats(agent.act, "actor"),
        module_param_stats(agent.cri, "critic"),
    ]
    total_params = sum(item["total_params"] for item in module_stats)
    trainable_params = sum(item["trainable_params"] for item in module_stats)
    alpha_trainable = bool(getattr(agent.alpha_log, "requires_grad", False))
    total_with_alpha = total_params + 1
    trainable_with_alpha = trainable_params + (1 if alpha_trainable else 0)

    report = {
        "freeze_rep": bool(getattr(agent, "freeze_rep", True)),
        "actor_trainable": bool(getattr(agent, "actor_trainable", False)),
        "adapter_actor_grad": bool(getattr(agent, "adapter_actor_grad", False)),
        "actor_on_adapter_weight": float(getattr(agent, "actor_on_adapter_weight", 0.0)),
        "alpha_log_requires_grad": alpha_trainable,
        "total_params": total_with_alpha,
        "trainable_params": trainable_with_alpha,
        "frozen_params": total_with_alpha - trainable_with_alpha,
        "trainable_ratio": float(trainable_with_alpha / total_with_alpha) if total_with_alpha > 0 else 0.0,
        "modules": module_stats,
    }

    os.makedirs(exp_path, exist_ok=True)
    json_path = os.path.join(exp_path, "param_stats.json")
    txt_path = os.path.join(exp_path, "param_stats.txt")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(report), f, ensure_ascii=False, indent=2)

    lines = []
    lines.append("SPF Frozen-Rep Finetune Parameter Stats")
    lines.append(f"freeze_rep={report['freeze_rep']}")
    lines.append(f"actor_trainable={report['actor_trainable']}")
    lines.append(f"adapter_actor_grad={report['adapter_actor_grad']}")
    lines.append(f"actor_on_adapter_weight={report['actor_on_adapter_weight']}")
    lines.append(f"alpha_log_requires_grad={report['alpha_log_requires_grad']}")
    lines.append(
        f"overall: total={report['total_params']}, trainable={report['trainable_params']}, "
        f"frozen={report['frozen_params']}, ratio={report['trainable_ratio']:.6f}"
    )
    lines.append("")

    for module in module_stats:
        lines.append(
            f"[{module['name']}] total={module['total_params']}, trainable={module['trainable_params']}, "
            f"frozen={module['frozen_params']}, ratio={module['trainable_ratio']:.6f}"
        )
        lines.append(f"{module['name']}.trainable_names:")
        if len(module["trainable_names"]) == 0:
            lines.append("  - (none)")
        else:
            lines.extend([f"  - {n}" for n in module["trainable_names"]])
        lines.append(f"{module['name']}.frozen_names:")
        if len(module["frozen_names"]) == 0:
            lines.append("  - (none)")
        else:
            lines.extend([f"  - {n}" for n in module["frozen_names"]])
        lines.append("")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(
        f"[param_stats] total={report['total_params']}, trainable={report['trainable_params']}, "
        f"ratio={report['trainable_ratio']:.6f}"
    )
    print(f"[param_stats] saved: {txt_path}")
    print(f"[param_stats] saved: {json_path}")

    return report


def build_replay_buffer_with_fallback(cfg, transition, transition_shape, rb_device: torch.device):
    try:
        buffer = ReplayBuffer(
            buffer_size=cfg.buffer_size,
            transition=transition,
            transition_shape=transition_shape,
            if_use_per=bool(getattr(cfg, "if_use_per", False)),
            device=rb_device,
        )
        return buffer, rb_device
    except RuntimeError as err:
        msg = str(err).lower()
        if rb_device.type == "cuda" and ("out of memory" in msg or "cuda" in msg):
            print("[runtime] cuda replay buffer allocation failed, fallback to cpu buffer.")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            fallback_device = torch.device("cpu")
            buffer = ReplayBuffer(
                buffer_size=cfg.buffer_size,
                transition=transition,
                transition_shape=transition_shape,
                if_use_per=bool(getattr(cfg, "if_use_per", False)),
                device=fallback_device,
            )
            return buffer, fallback_device
        raise


def init_before_training(seed=3407):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_default_dtype(torch.float32)


def make_env(env_id, env_params):
    def thunk():
        env = gym.make(env_id, **env_params)
        return env

    return thunk


def build_vector_env(env_fns, vector_env_type: str):
    if vector_env_type == "async":
        return gym.vector.AsyncVectorEnv(env_fns)
    return gym.vector.SyncVectorEnv(env_fns)


def parse_args():
    def str2bool(value):
        if isinstance(value, bool):
            return value
        value = value.lower()
        if value in ("true", "1", "yes", "y"):
            return True
        if value in ("false", "0", "no", "n"):
            return False
        raise argparse.ArgumentTypeError(f"invalid bool value: {value}")

    parser = argparse.ArgumentParser(description="SPF Stage-3 finetune")
    parser.add_argument(
        "--config",
        default=os.path.join(ROOT, "configs", "spf", "finetune_spf_frozen_rep_sac.py"),
        help="config file path",
    )
    parser.add_argument("--root", type=str, default=ROOT)
    parser.add_argument("--workdir", type=str, default="workdir")
    parser.add_argument("--tag", type=str, default="spf_finetune")
    parser.add_argument("--pretrain-path", type=str, default=None)
    parser.add_argument("--actor-on-adapter-weight", type=float, default=None)
    parser.add_argument("--adapter-rank", type=int, default=None)
    parser.add_argument("--adapter-dropout", type=float, default=None)
    parser.add_argument(
        "--actor-trainable",
        type=str2bool,
        default=None,
        help="set actor trainable or frozen: true/false",
    )
    parser.add_argument(
        "--adapter-actor-grad",
        type=str2bool,
        default=None,
        help="allow adapter receiving actor gradients: true/false",
    )
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="override config in key=value format",
    )
    return parser.parse_args()


def resolve_pretrain_path(cfg, pretrain_path):
    if pretrain_path is None:
        pretrain_path = getattr(cfg, "pretrain_path", None)
    if pretrain_path is None:
        raise FileNotFoundError("pretrain checkpoint is required, pass --pretrain-path or set cfg.pretrain_path")
    if not os.path.isabs(pretrain_path):
        pretrain_path = os.path.join(cfg.root, pretrain_path)
    if not os.path.exists(pretrain_path):
        raise FileNotFoundError(f"pretrain checkpoint not found: {pretrain_path}")
    return pretrain_path


def build_frozen_agent_cfg(cfg, device, pretrain_path, args):
    state_shape = deepcopy(cfg.transition_shape["state"])
    action_shape = deepcopy(cfg.transition_shape["action"])
    reward_shape = deepcopy(cfg.transition_shape["reward"])
    done_shape = deepcopy(cfg.transition_shape["done"])
    next_state_shape = deepcopy(cfg.transition_shape["next_state"])
    transition_shape = dict(
        state=state_shape,
        action=action_shape,
        reward=reward_shape,
        done=done_shape,
        next_state=next_state_shape,
    )
    transition = ["state", "action", "reward", "done", "next_state"]

    rep_out_dim = int(getattr(cfg, "decoder_embed_dim", cfg.rep_net.get("decoder_embed_dim", 128)))
    adapter_rank = int(getattr(cfg, "adapter_rank", 16) if args.adapter_rank is None else args.adapter_rank)
    adapter_dropout = float(getattr(cfg, "adapter_dropout", 0.1) if args.adapter_dropout is None else args.adapter_dropout)
    adapter_net = dict(
        type="FrozenRepAdapterHierarchical",
        d_model=rep_out_dim,
        rank=adapter_rank,
        dropout=adapter_dropout,
    )

    actor_trainable = bool(getattr(cfg, "actor_trainable", False) if args.actor_trainable is None else args.actor_trainable)
    adapter_actor_grad = bool(
        getattr(cfg, "adapter_actor_grad", False) if args.adapter_actor_grad is None else args.adapter_actor_grad
    )
    actor_on_adapter_weight = float(
        getattr(cfg, "actor_on_adapter_weight", 0.0)
        if args.actor_on_adapter_weight is None
        else args.actor_on_adapter_weight
    )

    act_lr = float(getattr(cfg, "act_lr", getattr(cfg, "lr", 5e-5)))
    cri_lr = float(getattr(cfg, "cri_lr", getattr(cfg, "lr", 5e-5)))
    alpha_lr = float(getattr(cfg, "alpha_lr", getattr(cfg, "lr", 5e-5)))

    agent_cfg = dict(
        type="AgentSACFrozenRep",
        act_lr=act_lr,
        cri_lr=cri_lr,
        alpha_lr=alpha_lr,
        adapter_lr=getattr(cfg, "adapter_lr", None),
        adapter_actor_lr=getattr(cfg, "adapter_actor_lr", None),
        rep_net=deepcopy(cfg.rep_net),
        adapter_net=adapter_net,
        act_net=deepcopy(cfg.act_net),
        cri_net=deepcopy(cfg.cri_net),
        criterion=deepcopy(cfg.criterion),
        optimizer=deepcopy(cfg.optimizer),
        scheduler=deepcopy(cfg.scheduler),
        if_use_per=bool(getattr(cfg, "if_use_per", False)),
        num_envs=int(cfg.num_envs),
        max_step=1e4,
        transition_shape=transition_shape,
        gamma=float(getattr(cfg, "gamma", 0.99)),
        reward_scale=float(getattr(cfg, "reward_scale", 2 ** 0)),
        repeat_times=float(cfg.repeat_times),
        batch_size=int(cfg.batch_size),
        clip_grad_norm=float(getattr(cfg, "clip_grad_norm", 3.0)),
        soft_update_tau=float(getattr(cfg, "soft_update_tau", 5e-3)),
        state_value_tau=float(getattr(cfg, "state_value_tau", 0.0)),
        device=device,
        action_wrapper_method=str(getattr(cfg, "action_wrapper_method", "reweight")),
        T=float(getattr(cfg, "T", 1.0)),
        pretrain_path=pretrain_path,
        pretrain_strict=bool(getattr(cfg, "pretrain_strict", False)),
        freeze_rep=True,
        actor_trainable=actor_trainable,
        adapter_actor_grad=adapter_actor_grad,
        actor_on_adapter_weight=actor_on_adapter_weight,
        if_use_tqdm=bool(getattr(cfg, "if_use_tqdm", True)),
    )
    return agent_cfg, transition, transition_shape


def train_one_episode(environment, buffer, agent, horizon_len):
    infos = dict()
    stats = {"episode_stats": {}, "horizon_stats": {}}

    environment.reset()
    while True:
        buffer_items = agent.explore_env(environment, horizon_len)
        buffer.update(buffer_items)

        torch.set_grad_enabled(True)
        logging_tuple = agent.update_net(buffer)
        torch.set_grad_enabled(False)

        positive_indices = torch.nonzero(buffer_items[-2] > 0)
        if positive_indices.numel() == 0:
            min_row_index = horizon_len - 1
        else:
            min_row_index = torch.min(positive_indices[:, 0]).item()

        for k, v in logging_tuple.items():
            stats["horizon_stats"].setdefault(k, []).append(v)

        if min_row_index < horizon_len - 1:
            break

    for k, v in stats["horizon_stats"].items():
        stats["episode_stats"][k] = np.mean(v)
    return stats, infos


def validate(environment, agent):
    stats = {"episode_stats": {}}
    logging_tuple, infos = agent.validate_net(environment)
    for k, v in logging_tuple.items():
        stats["episode_stats"][k] = v
    return stats, infos


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
    align_transition_shape_with_num_envs(cfg)
    update_data_root(cfg, root=args.root)

    init_before_training(int(cfg.seed))
    exp_path = os.path.join(cfg.root, cfg.workdir, cfg.tag)
    os.makedirs(exp_path, exist_ok=True)
    writer = SummaryWriter(exp_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pretrain_path = resolve_pretrain_path(cfg, args.pretrain_path)
    print(f"[finetune] using pretrain checkpoint: {pretrain_path}")

    dataset = DATASET.build(cfg.dataset)

    vector_env_type = str(getattr(cfg, "vector_env_type", "async")).lower()
    if vector_env_type not in ("sync", "async"):
        vector_env_type = "async"

    cfg.environment.update(
        dict(
            mode="train",
            if_norm=True,
            dataset=dataset,
            start_date=cfg.train_start_date,
            end_date=cfg.val_start_date,
        )
    )
    train_environment = ENVIRONMENT.build(cfg.environment)
    train_envs = build_vector_env(
        [
            make_env(
                "PortfolioManagement-v0",
                env_params=dict(env=deepcopy(train_environment), transition_shape=cfg.transition_shape, seed=cfg.seed + i),
            )
            for i in range(cfg.num_envs)
        ],
        vector_env_type=vector_env_type,
    )

    cfg.environment.update(
        dict(
            mode="val",
            if_norm=True,
            dataset=dataset,
            scaler=train_environment.scaler,
            start_date=cfg.val_start_date,
            end_date=cfg.test_start_date,
        )
    )
    val_environment = ENVIRONMENT.build(cfg.environment)
    val_envs = build_vector_env(
        [
            make_env(
                "PortfolioManagement-v0",
                env_params=dict(env=deepcopy(val_environment), transition_shape=cfg.transition_shape),
            )
            for _ in range(len(val_environment.aux_stocks))
        ],
        vector_env_type=vector_env_type,
    )

    agent_cfg, transition, transition_shape = build_frozen_agent_cfg(cfg, device, pretrain_path, args)
    agent = AGENT.build(agent_cfg)
    dump_param_stats(agent, exp_path)

    state = train_envs.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    agent.last_state = state

    buffer_device = str(getattr(cfg, "buffer_device", "cpu")).lower()
    if buffer_device == "cuda":
        rb_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    else:
        rb_device = torch.device("cpu")

    buffer, rb_device = build_replay_buffer_with_fallback(
        cfg=cfg,
        transition=transition,
        transition_shape=transition_shape,
        rb_device=rb_device,
    )
    print(
        f"[runtime] buffer_device={rb_device}, num_envs={cfg.num_envs}, batch_size={cfg.batch_size}, "
        f"repeat_times={cfg.repeat_times}, horizon_len={cfg.horizon_len}, vector_env_type={vector_env_type}"
    )

    buffer_items = agent.explore_env(train_envs, cfg.horizon_len)
    buffer.update(buffer_items)

    latest_path = find_latest_checkpoint(exp_path, suffix="pth")
    if latest_path:
        start_episode = load_checkpoint(agent, latest_path)
    else:
        start_episode = 0
    print(f"start episode {start_episode + 1}, end episode {cfg.num_episodes}")

    max_metrics = -np.inf
    horizon_step = 0
    for episode in range(start_episode + 1, cfg.num_episodes + 1):
        infos = {"episode": [episode]}
        episode_stats_log = {"episode": [episode]}

        print(f"Train Episode: [{episode}/{cfg.num_episodes}]")
        train_stats, train_infos = train_one_episode(train_envs, buffer, agent, cfg.horizon_len)
        horizon_stats = train_stats["horizon_stats"]
        episode_stats = train_stats["episode_stats"]

        for k, v in horizon_stats.items():
            for item in v:
                writer.add_scalar(f"train/horizon_{k}", item, horizon_step)
                horizon_step += 1
        for k, v in episode_stats.items():
            writer.add_scalar(f"train/episode_{k}", v, episode)

        train_episode_stats_log = OrderedDict(
            {"episode": [episode], **{f"train_{k}": [f"{v:04f}"] for k, v in episode_stats.items()}}
        )
        episode_stats_log.update(train_episode_stats_log)
        infos.update(train_infos)
        print(print_table(train_episode_stats_log))

        if episode % cfg.save_freq == 0:
            save_checkpoint(episode, agent, exp_path, if_best=False)

        print(f"Validate Episode: [{episode}/{cfg.num_episodes}]")
        val_stats, val_infos = validate(val_envs, agent)
        metric = np.mean([val_stats["episode_stats"]["ARR%_env0"]])
        if metric > max_metrics:
            max_metrics = metric
            save_checkpoint(episode, agent, exp_path, if_best=True)

        episode_stats = val_stats["episode_stats"]
        for k, v in episode_stats.items():
            writer.add_scalar(f"val/episode_{k}", v, episode)

        val_episode_log_stats = OrderedDict(
            {"episode": [episode], **{f"val_{k}": [f"{v:04f}"] for k, v in episode_stats.items()}}
        )
        episode_stats_log.update(val_episode_log_stats)
        infos.update(val_infos)
        print(print_table(val_episode_log_stats))

        with pathmgr.open(os.path.join(exp_path, "train_log.txt"), "a") as op:
            op.write(json.dumps(to_jsonable(episode_stats_log)) + "\n")
        with pathmgr.open(os.path.join(exp_path, "train_infos.txt"), "a") as op:
            op.write(json.dumps(to_jsonable(infos)) + "\n")


if __name__ == "__main__":
    main()
