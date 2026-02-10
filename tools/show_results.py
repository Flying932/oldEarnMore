import json
import os
import re
import argparse
from collections import defaultdict

import matplotlib.pyplot as plt
import pandas as pd


def to_float(v):
    if isinstance(v, (list, tuple)) and len(v) > 0:
        return to_float(v[0])
    if isinstance(v, str):
        return float(v)
    return float(v)


def load_train_log(path: str) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            row = {}
            for k, v in obj.items():
                try:
                    row[k] = to_float(v)
                except Exception:
                    continue
            if row:
                rows.append(row)
    if not rows:
        raise ValueError(f"No valid rows parsed from: {path}")
    df = pd.DataFrame(rows)
    if "episode" not in df.columns:
        raise ValueError(f"'episode' column missing in: {path}")
    df = df.sort_values("episode").drop_duplicates(subset=["episode"], keep="last").reset_index(drop=True)
    return df


def collect_env_metrics(columns):
    # Match keys like val_ARR%_env0, test_SR_env3
    pattern = re.compile(r"^(?P<split>[A-Za-z]+)_(?P<metric>.+)_env(?P<env>\d+)$")
    groups = defaultdict(list)  # (split, metric) -> [(env_id, col_name)]
    for c in columns:
        m = pattern.match(c)
        if not m:
            continue
        split = m.group("split")
        metric = m.group("metric")
        env_id = int(m.group("env"))
        groups[(split, metric)].append((env_id, c))
    for key in groups:
        groups[key].sort(key=lambda x: x[0])
    return groups


def ema(series: pd.Series, alpha: float = 0.12) -> pd.Series:
    return series.ewm(alpha=alpha, adjust=False).mean()


def plot_env_subplots(df: pd.DataFrame, out_dir: str, smooth_alpha: float = 0.12):
    os.makedirs(out_dir, exist_ok=True)
    groups = collect_env_metrics(df.columns)
    saved = []
    for (split, metric), items in sorted(groups.items()):
        n = len(items)
        cols = 2
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(12, 3.8 * rows), squeeze=False)
        axes_flat = axes.flatten()

        for idx, (env_id, col) in enumerate(items):
            ax = axes_flat[idx]
            ax.plot(df["episode"], df[col], linewidth=1.0, color="#b0b8c2", alpha=0.9, label="raw")
            ax.plot(df["episode"], ema(df[col], alpha=smooth_alpha), linewidth=2.0, color="#1f77b4", label="ema")
            ax.set_title(f"env{env_id}")
            ax.set_xlabel("episode")
            ax.set_ylabel(f"{split}_{metric}")
            ax.grid(alpha=0.3, linestyle="--")
            ax.legend(loc="best", fontsize=8)

        for idx in range(len(items), len(axes_flat)):
            fig.delaxes(axes_flat[idx])

        fig.suptitle(f"{split}_{metric} vs episode", fontsize=12)
        safe_metric = metric.replace("%", "pct").replace("/", "_")
        out_path = os.path.join(out_dir, f"{split}_{safe_metric}.png")
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        fig.savefig(out_path, dpi=180)
        plt.close(fig)
        saved.append(out_path)
    return saved


def _plot_single(ax, x, y, title, y_label, smooth_alpha=0.12):
    ax.plot(x, y, linewidth=1.0, color="#b0b8c2", alpha=0.9, label="raw")
    ax.plot(x, ema(y, alpha=smooth_alpha), linewidth=2.0, color="#1f77b4", label="ema")
    ax.set_title(title)
    ax.set_xlabel("episode")
    ax.set_ylabel(y_label)
    ax.grid(alpha=0.3, linestyle="--")
    ax.legend(loc="best", fontsize=8)


def plot_learning_summary(df: pd.DataFrame, out_dir: str, smooth_alpha: float = 0.12):
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), squeeze=False)
    axes_flat = axes.flatten()

    use_cols = [
        ("val_select_mean_arr", "Selection ARR", "ARR%"),
        ("val_select_mean_sr", "Selection SR", "SR"),
        ("val_select_mean_mdd", "Selection MDD", "MDD%"),
        ("train_obj_critics", "Train Critic Loss", "loss"),
    ]

    for idx, (col, title, y_label) in enumerate(use_cols):
        ax = axes_flat[idx]
        if col in df.columns:
            _plot_single(ax, df["episode"], df[col], title, y_label, smooth_alpha=smooth_alpha)
        else:
            ax.set_title(f"{title} (missing)")
            ax.axis("off")

    fig.suptitle("Learning Summary (Raw + EMA)", fontsize=12)
    out_path = os.path.join(out_dir, "summary_learning.png")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def parse_args():
    parser = argparse.ArgumentParser(description="Plot training curves from train_log.jsonl")
    parser.add_argument("--log-path", default="./train_log.txt")
    parser.add_argument("--out-dir", default="./figures")
    parser.add_argument("--smooth-alpha", type=float, default=0.12)
    return parser.parse_args()


def main():
    args = parse_args()
    log_path = args.log_path
    out_dir = args.out_dir
    df = load_train_log(log_path)
    saved = plot_env_subplots(df, out_dir, smooth_alpha=args.smooth_alpha)
    summary_path = plot_learning_summary(df, out_dir, smooth_alpha=args.smooth_alpha)
    saved.append(summary_path)
    print(f"parsed episodes: {len(df)}")
    print(f"saved figures: {len(saved)}")
    for p in saved:
        print(p)


if __name__ == "__main__":
    main()
