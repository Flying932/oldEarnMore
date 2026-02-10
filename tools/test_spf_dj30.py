import argparse
import json
import os
import sys
from copy import deepcopy
from pathlib import Path

import gym
import numpy as np
import torch
from mmengine.config import Config, DictAction

ROOT = str(Path(__file__).resolve().parents[1])
sys.path.append(ROOT)

from pm.registry import AGENT, DATASET, ENVIRONMENT
from pm.utils import update_data_root, load_checkpoint, find_latest_checkpoint
import pm.net  # noqa: F401
import pm.agent  # noqa: F401


def make_env(env_id, env_params):
    def thunk():
        return gym.make(env_id, disable_env_checker=True, **env_params)

    return thunk


class LegacySyncVectorEnv:
    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        self.num_envs = len(self.envs)

    def reset(self):
        states = []
        for env in self.envs:
            out = env.reset()
            if isinstance(out, tuple) and len(out) == 2:
                out = out[0]
            states.append(out)
        return np.stack(states, axis=0)

    def step(self, actions):
        next_states, rewards, dones, infos = [], [], [], []
        for env, action in zip(self.envs, actions):
            out = env.step(action)
            if isinstance(out, tuple) and len(out) == 5:
                state, reward, terminated, truncated, info = out
                done = bool(terminated or truncated)
            else:
                state, reward, done, info = out
            next_states.append(state)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)
        return (
            np.stack(next_states, axis=0),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(dones, dtype=np.bool_),
            infos,
        )

    def close(self):
        for env in self.envs:
            env.close()


def parse_args():
    parser = argparse.ArgumentParser(description="SPF test on DJ30 test split")
    parser.add_argument(
        "--config",
        default=os.path.join(ROOT, "configs", "spf", "finetune_spf_frozen_rep_sac.py"),
        help="config file path",
    )
    parser.add_argument("--root", type=str, default=ROOT)
    parser.add_argument("--workdir", type=str, default="workdir")
    parser.add_argument("--tag", type=str, default="spf_finetune_balanced")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="checkpoint path; if omitted, try <root>/<workdir>/<tag>/best.pth then latest checkpoint",
    )
    parser.add_argument(
        "--include-env0",
        action="store_true",
        help="include env0 in summary aggregation",
    )
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="override config in key=value format",
    )
    return parser.parse_args()


def align_transition_shape_with_num_envs(cfg):
    if not hasattr(cfg, "transition_shape"):
        return

    num_envs = int(cfg.num_envs)
    old_transition_shape = deepcopy(cfg.transition_shape)
    new_transition_shape = {}
    for key, spec in old_transition_shape.items():
        item = dict(spec)
        shape = tuple(item.get("shape", ()))
        if len(shape) > 0 and shape[0] != num_envs:
            item["shape"] = (num_envs, *shape[1:])
        else:
            item["shape"] = shape
        new_transition_shape[key] = item
    cfg.transition_shape = new_transition_shape


def resolve_pretrain_path(cfg):
    pretrain_path = getattr(cfg, "pretrain_path", None)
    if pretrain_path is None:
        raise FileNotFoundError("pretrain_path is required in config for SPF test")
    if not os.path.isabs(pretrain_path):
        pretrain_path = os.path.join(cfg.root, pretrain_path)
    if not os.path.exists(pretrain_path):
        raise FileNotFoundError(f"pretrain checkpoint not found: {pretrain_path}")
    return pretrain_path


def resolve_checkpoint(cfg, ckpt_arg):
    if ckpt_arg:
        ckpt = ckpt_arg
        if not os.path.isabs(ckpt):
            ckpt = os.path.join(cfg.root, ckpt)
        if not os.path.exists(ckpt):
            raise FileNotFoundError(f"checkpoint not found: {ckpt}")
        return ckpt

    exp_path = os.path.join(cfg.root, cfg.workdir, cfg.tag)
    best_path = os.path.join(exp_path, "best.pth")
    if os.path.exists(best_path):
        return best_path

    latest_path = find_latest_checkpoint(exp_path, suffix="pth")
    if latest_path:
        return latest_path

    raise FileNotFoundError(f"no checkpoint found under {exp_path}")


def build_frozen_agent_cfg(cfg, device, pretrain_path):
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

    rep_out_dim = int(getattr(cfg, "decoder_embed_dim", cfg.rep_net.get("decoder_embed_dim", 128)))
    adapter_rank = int(getattr(cfg, "adapter_rank", 16))
    adapter_dropout = float(getattr(cfg, "adapter_dropout", 0.1))
    adapter_net = dict(
        type="FrozenRepAdapterHierarchical",
        d_model=rep_out_dim,
        rank=adapter_rank,
        dropout=adapter_dropout,
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
        actor_trainable=bool(getattr(cfg, "actor_trainable", False)),
        adapter_actor_grad=bool(getattr(cfg, "adapter_actor_grad", False)),
        actor_on_adapter_weight=float(getattr(cfg, "actor_on_adapter_weight", 0.0)),
        if_use_tqdm=False,
    )
    return agent_cfg


def summarize_metrics(metrics: dict, include_env0: bool = False):
    def _collect(prefix: str):
        items = []
        for k, v in metrics.items():
            if not k.startswith(prefix):
                continue
            try:
                env_idx = int(k.split("env")[-1])
            except (ValueError, IndexError):
                continue
            if include_env0 or env_idx > 0:
                items.append((env_idx, v))
        items.sort(key=lambda x: x[0])
        return [v for _, v in items]

    arr_vals = _collect("ARR%_env")
    sr_vals = _collect("SR_env")
    mdd_vals = _collect("MDD%_env")
    mean_arr = sum(arr_vals) / len(arr_vals) if arr_vals else float("nan")
    mean_sr = sum(sr_vals) / len(sr_vals) if sr_vals else float("nan")
    mean_mdd = sum(mdd_vals) / len(mdd_vals) if mdd_vals else float("nan")
    return mean_arr, mean_sr, mean_mdd


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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrain_path = resolve_pretrain_path(cfg)
    checkpoint_path = resolve_checkpoint(cfg, args.checkpoint)

    print(50 * "-" + "build dataset" + "-" * 50)
    dataset = DATASET.build(cfg.dataset)

    print(50 * "-" + "build train env for scaler" + "-" * 50)
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

    print(50 * "-" + "build test env" + "-" * 50)
    cfg.environment.update(
        dict(
            mode="test",
            if_norm=True,
            dataset=dataset,
            scaler=train_environment.scaler,
            start_date=cfg.test_start_date,
            end_date=getattr(cfg, "test_end_date", None),
        )
    )
    test_environment = ENVIRONMENT.build(cfg.environment)
    test_envs = LegacySyncVectorEnv(
        [
            make_env(
                "PortfolioManagement-v0",
                env_params=dict(env=deepcopy(test_environment), transition_shape=cfg.transition_shape),
            )
            for _ in range(len(test_environment.aux_stocks))
        ]
    )

    print(50 * "-" + "build frozen-rep agent" + "-" * 50)
    agent_cfg = build_frozen_agent_cfg(cfg, device, pretrain_path)
    agent = AGENT.build(agent_cfg)

    print(f"load checkpoint: {checkpoint_path}")
    _ = load_checkpoint(agent, checkpoint_path)

    metrics, infos = agent.validate_net(test_envs)
    mean_arr, mean_sr, mean_mdd = summarize_metrics(metrics, include_env0=args.include_env0)

    print("test metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v}")
    print(f"summary: mean_ARR={mean_arr:.6f}, mean_SR={mean_sr:.6f}, mean_MDD={mean_mdd:.6f}")

    exp_path = os.path.join(cfg.root, cfg.workdir, cfg.tag)
    os.makedirs(exp_path, exist_ok=True)
    save_path = os.path.join(exp_path, "test_metrics.json")
    out = {
        "checkpoint": checkpoint_path,
        "summary": {
            "mean_ARR": mean_arr,
            "mean_SR": mean_sr,
            "mean_MDD": mean_mdd,
            "include_env0": bool(args.include_env0),
        },
        "metrics": metrics,
        "num_test_envs": int(test_envs.num_envs),
        "num_steps": len(infos.get("portfolios", [])),
    }
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"saved test metrics to {save_path}")


if __name__ == "__main__":
    main()
