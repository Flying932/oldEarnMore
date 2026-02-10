import argparse
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
import pm.net  # noqa: F401
import pm.agent  # noqa: F401
from pm.utils import update_data_root, load_checkpoint, find_latest_checkpoint

# DEFAULT_TAG = "mask_sac_nepx2000_daysx10_bsx128_bufsx100000_hlx128_edx64_depx1_dedx64_dedepx1_rtx128_lrx1e-05_sdx42_nvx16_actlrx1e-05_crilrx1e-05_replrx1e-05_betlrx1e-05_repwx1.0_betwx0.01_awmxsoftmax_Tx0.1_dj30_dgx0"
DEFAULT_TAG = "mask_sac_nepx2000_daysx10_bsx128_bufsx10000_hlx64_edx128_depx1_dedx128_dedepx1_rtx128_lrx1e-05_sdx42_nvx10_actlrx1e-05_crilrx1e-05_replrx1e-05_betlrx1e-05_repwx1.0_betwx0.01_awmxsoftmax_Tx0.1_dj30_dgx0"
DEFAULT_TAG = "mask_sac_nepx2000_daysx10_bsx32_bufsx50000_hlx32_edx64_depx1_dedx64_dedepx1_rtx8_lrx1e-05_sdx42_nvx1_vecxsync_bufdevxcpu_actlrx1e-05_crilrx1e-05_replrx1e-05_betlrx1e-05_repwx1.0_betwx0.01_awmxreweight_Tx0.1_dj30_dgx0"
DEFAULT_TAG = "mask_sac_nepx2000_daysx10_bsx128_bufsx12500_hlx128_edx64_depx1_dedx64_dedepx1_rtx128_lrx1e-05_sdx42_nvx8_vecxasync_bufdevxcpu_actlrx1e-05_crilrx1e-05_replrx1e-05_betlrx1e-05_repwx1.0_betwx0.01_awmxsoftmax_Tx0.1_dj30_dgx0"
DEFAULT_CONFIG = f"configs/mask_sac/{DEFAULT_TAG}.py"
DEFAULT_CKPT = (
    f"workdir/{DEFAULT_TAG}/best.pth"
)


def make_env(env_id, env_params):
    def thunk():
        return gym.make(env_id, disable_env_checker=True, **env_params)

    return thunk


class LegacySyncVectorEnv:
    """A minimal sync vec env with old-style reset/step outputs."""

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
    parser = argparse.ArgumentParser(description="Standalone DJ30 test script")
    parser.add_argument(
        "--config",
        default=os.path.join(ROOT, DEFAULT_CONFIG),
        help="config file path",
    )
    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CKPT,
        help="checkpoint path. If empty, try <root>/<workdir>/<tag>/best.pth then latest checkpoint.",
    )
    parser.add_argument("--root", type=str, default=ROOT)
    parser.add_argument("--workdir", type=str, default="workdir")
    parser.add_argument("--tag", type=str, default=DEFAULT_TAG)
    parser.add_argument(
        "--include-env0",
        action="store_true",
        help="include env0 (All/GSP) in summary aggregation; default excludes env0 to focus on CSPs",
    )
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="override config in key=value format",
    )
    return parser.parse_args()


def resolve_checkpoint(cfg, ckpt_arg):
    if ckpt_arg:
        if not os.path.isabs(ckpt_arg):
            ckpt_arg = os.path.join(cfg.root, ckpt_arg)
        if not os.path.exists(ckpt_arg):
            raise FileNotFoundError(f"checkpoint not found: {ckpt_arg}")
        return ckpt_arg

    exp_path = os.path.join(cfg.root, cfg.workdir, cfg.tag)
    best_path = os.path.join(exp_path, "best.pth")
    if os.path.exists(best_path):
        return best_path

    latest_path = find_latest_checkpoint(exp_path, suffix="pth")
    if latest_path:
        return latest_path

    raise FileNotFoundError(
        f"no checkpoint found under {exp_path}. "
        f"please pass --checkpoint explicitly."
    )


def collect_candidate_checkpoints(cfg, ckpt_arg):
    exp_path = os.path.join(cfg.root, cfg.workdir, cfg.tag)
    candidates = []

    # user-specified/default checkpoint first
    if ckpt_arg:
        ckpt = ckpt_arg
        if not os.path.isabs(ckpt):
            ckpt = os.path.join(cfg.root, ckpt)
        if os.path.exists(ckpt):
            candidates.append(ckpt)

    best_path = os.path.join(exp_path, "best.pth")
    if os.path.exists(best_path):
        candidates.append(best_path)

    latest_path = find_latest_checkpoint(exp_path, suffix="pth")
    if latest_path and os.path.exists(latest_path):
        candidates.append(latest_path)

    # de-duplicate while keeping order
    uniq = []
    seen = set()
    for p in candidates:
        if p not in seen:
            uniq.append(p)
            seen.add(p)
    return uniq


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
    update_data_root(cfg, root=args.root)

    if not hasattr(cfg, "root"):
        cfg.root = args.root
    if not hasattr(cfg, "workdir"):
        cfg.workdir = args.workdir
    if not hasattr(cfg, "tag"):
        raise ValueError("cfg.tag is required. pass --tag or set tag in config.")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

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

    print(50 * "-" + "build agent" + "-" * 50)
    cfg.agent.update(dict(device=device))
    agent = AGENT.build(cfg.agent)

    candidates = collect_candidate_checkpoints(cfg, args.checkpoint)
    if not candidates:
        checkpoint_path = resolve_checkpoint(cfg, args.checkpoint)
        candidates = [checkpoint_path]

    print("checkpoint candidates:")
    for p in candidates:
        print(p)

    results = []
    for checkpoint_path in candidates:
        _ = load_checkpoint(agent, checkpoint_path)
        print(f"\nloaded checkpoint: {checkpoint_path}")

        metrics, _ = agent.validate_net(test_envs)
        mean_arr, mean_sr, mean_mdd = summarize_metrics(metrics, include_env0=args.include_env0)
        results.append((checkpoint_path, mean_arr, mean_sr, mean_mdd, metrics))

        print("test metrics:")
        for k, v in metrics.items():
            print(f"{k}: {v}")
        print(f"summary: mean_ARR={mean_arr:.6f}, mean_SR={mean_sr:.6f}, mean_MDD={mean_mdd:.6f}")

    results.sort(key=lambda x: x[1], reverse=True)
    print("\nranked by mean_ARR (high -> low):")
    for i, (ckpt, mean_arr, mean_sr, mean_mdd, _) in enumerate(results, start=1):
        print(f"{i}. {ckpt}")
        print(f"   mean_ARR={mean_arr:.6f}, mean_SR={mean_sr:.6f}, mean_MDD={mean_mdd:.6f}")


if __name__ == "__main__":
    main()
