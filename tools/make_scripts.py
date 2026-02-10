import warnings
warnings.filterwarnings("ignore")
import os
import sys
import argparse
from pathlib import Path
from mmengine.config import Config, DictAction

ROOT = str(Path(__file__).resolve().parents[1])
sys.path.append(ROOT)

# -----------------------------------------------------------------------------
# Paper-aligned defaults for Mask-SAC on DJ30.
# CLI args still override these defaults.
# -----------------------------------------------------------------------------
LOCAL_DEFAULTS = {
    # dataset
    "config": os.path.join(ROOT, "configs", "mask_sac_portfolio_management.py"),
    "dataset": "dj30",   # dj30 / sp500
    "num_stocks": 28,    # dj30=28, sp500 usually 420
    "gpu_id": 0,
    "mask": True,

    # training
    "num_episodes": 2000,  # example uses 2000; use 5~20 for quick validation
    "days": 10,
    "batch_size": 128,
    "buffer_size": 100000,
    "horizon_len": 128,
    "embed_dim": 64,
    "depth": 1,
    "decoder_embed_dim": 64,
    "decoder_depth": 1,
    "repeat_times": 128,

    # lr
    "lr": 1e-5,
    "act_lr": 1e-5,
    "cri_lr": 1e-5,
    "rep_lr": 1e-5,
    "beta_lr": 1e-5,
    "rep_loss_weight": 1.0,
    "beta_loss_weight": 0.01,

    # runtime
    "seed": 42,
    "num_envs": 1,
    "vector_env_type": "sync",  # "sync" | "async"
    "buffer_device": "cpu",      # "cpu" | "cuda"
    "data_group": "0",  # 0 / 1 / 2
    "action_wrapper_method": "softmax",
    "T": 0.1,
}

def update_keys(cfg, name, rename):
    if isinstance(cfg, Config):
        setattr(cfg, name, rename)
        for key, value in cfg.items():
            update_keys(value, name, rename)
    elif isinstance(cfg, dict):
        for key, value in cfg.items():
            if key == name:
                cfg[name] = rename
            elif isinstance(value, dict):
                update_keys(value, name, rename)

def parse_args():
    parser = argparse.ArgumentParser(description='PM train script')
    parser.add_argument("--config", default=LOCAL_DEFAULTS["config"], help="config file path")

    parser.add_argument("--gpu_id", type = int, default=LOCAL_DEFAULTS["gpu_id"])
    parser.add_argument("--mask", action="store_true", default=LOCAL_DEFAULTS["mask"])

    #dataset
    # parser.add_argument("--dataset", type=str, default="sg1")
    # parser.add_argument("--num_stocks", type=int, default=1)

    parser.add_argument("--dataset", type=str, default=LOCAL_DEFAULTS["dataset"])
    parser.add_argument("--num_stocks", type=int, default=LOCAL_DEFAULTS["num_stocks"])

    # parser.add_argument("--dataset", type=str, default="sp500")
    # parser.add_argument("--num_stocks", type=int, default=420)

    # train parameters
    parser.add_argument("--num_episodes", type=int, default=LOCAL_DEFAULTS["num_episodes"])
    parser.add_argument("--days", type=int, default=LOCAL_DEFAULTS["days"])
    parser.add_argument("--batch_size", type=int, default=LOCAL_DEFAULTS["batch_size"])
    parser.add_argument("--buffer_size", type=int, default=LOCAL_DEFAULTS["buffer_size"])
    parser.add_argument("--horizon_len", type=int, default=LOCAL_DEFAULTS["horizon_len"])
    parser.add_argument("--embed_dim", type=int, default=LOCAL_DEFAULTS["embed_dim"])
    parser.add_argument("--depth", type=int, default=LOCAL_DEFAULTS["depth"])
    parser.add_argument("--decoder_embed_dim", type=int, default=LOCAL_DEFAULTS["decoder_embed_dim"])
    parser.add_argument("--decoder_depth", type=int, default=LOCAL_DEFAULTS["decoder_depth"])
    parser.add_argument("--repeat_times", type=int, default=LOCAL_DEFAULTS["repeat_times"])

    parser.add_argument("--lr", type=float, default=LOCAL_DEFAULTS["lr"])
    parser.add_argument("--act_lr", type=float, default=LOCAL_DEFAULTS["act_lr"])
    parser.add_argument("--cri_lr", type=float, default=LOCAL_DEFAULTS["cri_lr"])
    parser.add_argument("--rep_lr", type=float, default=LOCAL_DEFAULTS["rep_lr"])
    parser.add_argument("--beta_lr", type=float, default=LOCAL_DEFAULTS["beta_lr"])

    parser.add_argument("--rep_loss_weight", type=float, default=LOCAL_DEFAULTS["rep_loss_weight"])
    parser.add_argument("--beta_loss_weight", type=float, default=LOCAL_DEFAULTS["beta_loss_weight"])

    parser.add_argument("--seed", type=int, default=LOCAL_DEFAULTS["seed"])
    parser.add_argument("--num_envs", type=int, default=LOCAL_DEFAULTS["num_envs"])
    parser.add_argument("--vector_env_type", type=str, default=LOCAL_DEFAULTS["vector_env_type"])
    parser.add_argument("--buffer_device", type=str, default=LOCAL_DEFAULTS["buffer_device"])

    #date
    parser.add_argument("--data_group", type=str, default=LOCAL_DEFAULTS["data_group"]) # data_group = 0,1,2

    parser.add_argument("--action_wrapper_method", type=str, default=LOCAL_DEFAULTS["action_wrapper_method"])
    parser.add_argument("--T", type=float, default=LOCAL_DEFAULTS["T"])

    args = parser.parse_args()
    return args
def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)

    tag_name = cfg.tag
    tag = cfg.tag

    save_config_path = os.path.join(ROOT, "configs", tag_name)
    save_script_path = os.path.join(ROOT, "scripts", tag_name)

    os.makedirs(save_config_path, exist_ok=True)
    os.makedirs(save_script_path, exist_ok=True)

    gpu_id = args.gpu_id

    train_parameters = {
        "num_episodes": args.num_episodes,
        "days": args.days,
        "batch_size": args.batch_size,
        "buffer_size": args.buffer_size,
        "horizon_len": args.horizon_len,
        "embed_dim": args.embed_dim,
        "depth": args.depth,
        "decoder_embed_dim": args.decoder_embed_dim,
        "decoder_depth": args.decoder_depth,
        "repeat_times": args.repeat_times,
        "lr": args.lr,
        "seed": args.seed,
        "num_envs": args.num_envs,
        "vector_env_type": args.vector_env_type,
        "buffer_device": args.buffer_device,
        "act_lr": args.act_lr,
        "cri_lr": args.cri_lr,
        "rep_lr": args.rep_lr,
        "beta_lr": args.beta_lr,
        "rep_loss_weight": args.rep_loss_weight,
        "beta_loss_weight": args.beta_loss_weight,
        "action_wrapper_method": args.action_wrapper_method,
        "T": args.T,
    }
    tag_rename = {
        "num_episodes": "nep",
        "days": "days",
        "batch_size": "bs",
        "buffer_size": "bufs",
        "horizon_len": "hl",
        "embed_dim": "ed",
        "depth": "dep",
        "decoder_embed_dim": "ded",
        "decoder_depth": "dedep",
        "repeat_times": "rt",
        "lr": "lr",
        "seed":"sd",
        "num_envs": "nv",
        "vector_env_type": "vec",
        "buffer_device": "bufdev",
        "act_lr": "actlr",
        "cri_lr": "crilr",
        "rep_lr": "replr",
        "beta_lr": "betlr",
        "rep_loss_weight": "repw",
        "beta_loss_weight": "betw",
        "action_wrapper_method": "awm",
        "T": "T",
    }

    for param, param_value in train_parameters.items():
        update_keys(cfg, param, param_value)
        tag += "_{}x{}".format(tag_rename[param], param_value)

    # update dataset config
    data_path = f"datasets/{args.dataset}/features"
    stocks_path = f"datasets/{args.dataset}/stocks.txt"
    aux_stocks_path = f"datasets/{args.dataset}/aux_stocks_files"
    update_keys(cfg, "data_path", data_path)
    update_keys(cfg, "stocks_path", stocks_path)
    update_keys(cfg, "aux_stocks_path", aux_stocks_path)
    update_keys(cfg, "num_stocks", args.num_stocks)
    update_keys(cfg, "pred_num_stocks", args.num_stocks)
    tag += f"_{args.dataset}"

    """
    2018-01-26 2019-07-22
    2019-07-22 2021-01-08
    2021-01-08 2022-06-26

    """

    if args.data_group == "0":
        train_start_date = "2007-09-26"
        val_start_date = "2018-01-26"
        test_start_date = "2019-07-22"
        test_end_date = "2021-01-08"
        n_steps_per_episode = 1024

    elif args.data_group == "1":
        train_start_date = "2007-09-26"
        val_start_date = "2019-07-22"
        test_start_date = "2021-01-08"
        test_end_date = "2022-06-26"
        n_steps_per_episode = 1280

    elif args.data_group == "2":
        train_start_date = "2007-09-26"
        val_start_date = "2021-01-08"
        test_start_date = "2022-06-26"
        test_end_date = None
        n_steps_per_episode = 1536
    else:
        raise ValueError("data_group must be 0, 1, 2")
    update_keys(cfg, "train_start_date", train_start_date)
    update_keys(cfg, "val_start_date", val_start_date)
    update_keys(cfg, "test_start_date", test_start_date)
    update_keys(cfg, "test_end_date", test_end_date)
    update_keys(cfg, "n_steps_per_episode", n_steps_per_episode)

    # update scheduler config
    if args.num_episodes in (1000, 2000):
        # Follow paper Appendix C: warmup 300 episodes, milestones at 600/1000/1400.
        cfg.scheduler["multi_steps"] = [
            600 * n_steps_per_episode,
            1000 * n_steps_per_episode,
            1400 * n_steps_per_episode,
        ]
        cfg.scheduler["warmup_t"] = 300 * n_steps_per_episode

    tag += f"_dgx{args.data_group}"

    # update mask config
    if not args.mask:
        transition_shape = dict(
            state=dict(shape=(args.num_envs, args.num_stocks, cfg.days, cfg.num_features), type="float32"),
            action=dict(shape=(args.num_envs, args.num_stocks + 1), type="float32"),
            reward=dict(shape=(args.num_envs,), type="float32"),
            done=dict(shape=(args.num_envs,), type="float32"),
            next_state=dict(shape=(args.num_envs, args.num_stocks, cfg.days, cfg.num_features), type="float32")
        )

        if "ppo" in cfg.tag:
            transition_shape = dict(
                state=dict(shape=(args.num_envs, args.num_stocks, cfg.days, cfg.num_features), type="float32"),
                action=dict(shape=(args.num_envs, args.num_stocks + 1), type="float32"),
                logprob=dict(shape=(args.num_envs,), type="float32"),
                reward=dict(shape=(args.num_envs,), type="float32"),
                done=dict(shape=(args.num_envs,), type="float32"),
                next_state=dict(shape=(args.num_envs, args.num_stocks, cfg.days, cfg.num_features), type="float32")
            )

        update_keys(cfg, "transition_shape", transition_shape)
    else:
        transition_shape = dict(
            state = dict(shape=(args.num_envs, args.num_stocks, cfg.days, cfg.num_features), type="float32"),
            action = dict(shape=(args.num_envs, args.num_stocks + 1), type="float32"),
            mask = dict(shape=(args.num_envs, args.num_stocks), type="int32"),
            ids_restore = dict(shape=(args.num_envs, args.num_stocks), type="int64"),
            reward = dict(shape=(args.num_envs,), type="float32"),
            done = dict(shape=(args.num_envs,), type="float32"),
            next_state = dict(shape=(args.num_envs, args.num_stocks, cfg.days, cfg.num_features), type="float32")
        )
        if "ppo" in cfg.tag:
            transition_shape = dict(
                state=dict(shape=(args.num_envs, args.num_stocks, cfg.days, cfg.num_features), type="float32"),
                action=dict(shape=(args.num_envs, args.num_stocks + 1), type="float32"),
                logprob=dict(shape=(args.num_envs,), type="float32"),
                mask=dict(shape=(args.num_envs, args.num_stocks), type="int32"),
                ids_restore=dict(shape=(args.num_envs, args.num_stocks), type="int64"),
                reward=dict(shape=(args.num_envs,), type="float32"),
                done=dict(shape=(args.num_envs,), type="float32"),
                next_state=dict(shape=(args.num_envs, args.num_stocks, cfg.days, cfg.num_features), type="float32")
            )

        update_keys(cfg, "transition_shape", transition_shape)

    feature_size = (cfg.days, cfg.num_features)
    update_keys(cfg, "feature_size", feature_size)
    patch_size = (cfg.days, cfg.num_features)
    update_keys(cfg, "patch_size", patch_size)


    cfg.tag = tag
    cfg.dump(os.path.join(save_config_path, "{}.py".format(tag)))

    cmd = f"""CUDA_VISIBLE_DEVICES={gpu_id} python tools/train.py --config configs/{tag_name}/{tag}.py"""
    print(cmd)
    os.makedirs(os.path.join(ROOT, "scripts", tag_name), exist_ok=True)
    with open(os.path.join(ROOT, "scripts", tag_name,"{}.sh".format(tag)), "w") as op:
        op.write(cmd)
    # keep stdout clean to avoid copying tag text as a shell command by mistake

if __name__ == '__main__':
    main()
