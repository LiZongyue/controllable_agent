#!/usr/bin/env python3
"""Plot SWM CLS/Mean W&B evaluation curves.

The script downloads eval/episode_reward from the 24 W&B runs listed in the
paper notes, trims terminal one-off drops/peaks, resamples every curve to 1000
points, and writes both per-group and combined appendix-style figures.
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb

try:
    from scipy.interpolate import PchipInterpolator
except Exception:  # pragma: no cover - scipy is optional for plotting.
    PchipInterpolator = None


ENTITY = "lmu_rl"
Y_KEY = "eval/episode_reward"
X_KEY = "eval/frame"


@dataclasses.dataclass(frozen=True)
class RunSpec:
    group: str
    task: str
    title: str
    project: str
    run_id: str

    @property
    def path(self) -> str:
        return f"{ENTITY}/{self.project}/{self.run_id}"


TASK_ORDER = [
    "walker_stand",
    "walker_flip",
    "walker_walk",
    "walker_run",
    "quadruped_stand",
    "quadruped_walk",
    "quadruped_run",
    "quadruped_jump",
    "cheetah_walk",
    "cheetah_walk_backward",
    "cheetah_run_backward",
    "cheetah_run",
]


RUNS = [
    # SWM CLS linear
    RunSpec("SWM CLS", "walker_stand", "Walker Stand", "controllable_agent", "c0p780h9"),
    RunSpec("SWM CLS", "walker_flip", "Walker Flip", "controllable_agent_baseline", "94hyqvmm"),
    RunSpec("SWM CLS", "walker_walk", "Walker Walk", "controllable_agent", "la5ubfms"),
    RunSpec("SWM CLS", "walker_run", "Walker Run", "controllable_agent", "84chbns4"),
    RunSpec("SWM CLS", "quadruped_stand", "Quadruped Stand", "controllable_agent_baseline", "v9p5czlz"),
    RunSpec("SWM CLS", "quadruped_walk", "Quadruped Walk", "controllable_agent_baseline", "or3kq5kl"),
    RunSpec("SWM CLS", "quadruped_run", "Quadruped Run", "controllable_agent_baseline", "uwaw684l"),
    RunSpec("SWM CLS", "quadruped_jump", "Quadruped Jump", "controllable_agent_baseline", "7gppvefv"),
    RunSpec("SWM CLS", "cheetah_walk", "Cheetah Walk", "controllable_agent_baseline", "bpv5sfne"),
    RunSpec("SWM CLS", "cheetah_walk_backward", "Cheetah Walk Backward", "controllable_agent_baseline", "08sjtb96"),
    RunSpec("SWM CLS", "cheetah_run_backward", "Cheetah Run Backward", "controllable_agent_baseline", "j1lfjxeq"),
    RunSpec("SWM CLS", "cheetah_run", "Cheetah Run", "controllable_agent_baseline", "sc7edapk"),
    # SWM Mean patch linear
    RunSpec("SWM Mean", "walker_stand", "Walker Stand", "controllable_agent", "8x35tl4p"),
    RunSpec("SWM Mean", "walker_flip", "Walker Flip", "controllable_agent_baseline", "cdn1jjpc"),
    RunSpec("SWM Mean", "walker_walk", "Walker Walk", "controllable_agent_baseline", "gth9swt5"),
    RunSpec("SWM Mean", "walker_run", "Walker Run", "controllable_agent_baseline", "5dx2943a"),
    RunSpec("SWM Mean", "quadruped_stand", "Quadruped Stand", "controllable_agent_baseline", "uzz9v3xx"),
    RunSpec("SWM Mean", "quadruped_walk", "Quadruped Walk", "controllable_agent", "2fajok2d"),
    RunSpec("SWM Mean", "quadruped_run", "Quadruped Run", "controllable_agent_baseline", "zl35c1rf"),
    RunSpec("SWM Mean", "quadruped_jump", "Quadruped Jump", "controllable_agent_baseline", "o0ataz5y"),
    RunSpec("SWM Mean", "cheetah_walk", "Cheetah Walk", "controllable_agent_baseline", "2rrhp3y6"),
    RunSpec("SWM Mean", "cheetah_walk_backward", "Cheetah Walk Backward", "controllable_agent_baseline", "z553wub4"),
    RunSpec("SWM Mean", "cheetah_run_backward", "Cheetah Run Backward", "controllable_agent_baseline", "fk57r9iv"),
    RunSpec("SWM Mean", "cheetah_run", "Cheetah Run", "controllable_agent_baseline", "756v9w3a"),
]


def fetch_run_history(api: wandb.Api, spec: RunSpec) -> pd.DataFrame:
    run = api.run(spec.path)
    rows = []
    for row in run.scan_history(keys=[X_KEY, Y_KEY], page_size=10_000):
        if X_KEY not in row or Y_KEY not in row:
            continue
        x = row[X_KEY]
        y = row[Y_KEY]
        if x is None or y is None:
            continue
        rows.append(
            {
                "group": spec.group,
                "task": spec.task,
                "title": spec.title,
                "project": spec.project,
                "run_id": spec.run_id,
                "run_path": spec.path,
                "x_raw": float(x),
                "episode_reward": float(y),
            }
        )
    if not rows:
        raise RuntimeError(f"No {Y_KEY} rows found for {spec.path}")
    df = pd.DataFrame(rows)
    # W&B histories can contain duplicated eval frames after resume. Average
    # duplicates to keep interpolation stable.
    keys = ["group", "task", "title", "project", "run_id", "run_path", "x_raw"]
    return df.groupby(keys, as_index=False)["episode_reward"].mean().sort_values("x_raw")


def load_or_fetch_raw(cache_path: Path, refresh: bool) -> pd.DataFrame:
    if cache_path.exists() and not refresh:
        return pd.read_csv(cache_path)
    api = wandb.Api(timeout=60)
    frames = []
    for idx, spec in enumerate(RUNS, start=1):
        print(f"[{idx:02d}/{len(RUNS)}] fetching {spec.group} {spec.title}: {spec.path}", flush=True)
        frames.append(fetch_run_history(api, spec))
    raw = pd.concat(frames, ignore_index=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    raw.to_csv(cache_path, index=False)
    return raw


def trim_terminal_outliers(
    x: np.ndarray,
    y: np.ndarray,
    max_trim: int = 5,
    tail_window: int = 12,
    threshold_scale: float = 1.5,
    range_fraction: float = 0.25,
) -> tuple[np.ndarray, np.ndarray, int]:
    trimmed = 0
    x = x.copy()
    y = y.copy()
    for _ in range(max_trim):
        if len(y) < max(6, tail_window // 2):
            break
        prev = y[max(0, len(y) - tail_window - 1) : -1]
        if len(prev) < 3:
            break
        center = float(np.nanmedian(prev))
        mad = float(np.nanmedian(np.abs(prev - center)) * 1.4826)
        std = float(np.nanstd(prev))
        global_range = float(np.nanpercentile(y[:-1], 95) - np.nanpercentile(y[:-1], 5))
        robust_threshold = max(threshold_scale * mad, range_fraction * global_range, 1e-8)
        outside_local_band = (
            float(y[-1]) < float(np.nanmin(prev)) - 0.05 * global_range
            or float(y[-1]) > float(np.nanmax(prev)) + 0.05 * global_range
        )
        large_median_gap = abs(float(y[-1]) - center) > robust_threshold
        if not (outside_local_band or large_median_gap):
            break
        x = x[:-1]
        y = y[:-1]
        trimmed += 1
    return x, y, trimmed


def resample_curve(
    x: np.ndarray,
    y: np.ndarray,
    num_points: int,
    interpolation: str = "linear",
) -> tuple[np.ndarray, np.ndarray]:
    target = np.linspace(0.0, 1000.0, num_points)
    if len(y) == 0:
        return target, np.full_like(target, np.nan, dtype=np.float64)
    if len(y) == 1 or np.nanmax(x) == np.nanmin(x):
        return target, np.full_like(target, float(y[-1]), dtype=np.float64)
    source = (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x)) * 1000.0
    order = np.argsort(source)
    source = source[order]
    y = y[order]
    unique_source, unique_idx = np.unique(source, return_index=True)
    unique_y = y[unique_idx]
    if interpolation == "pchip" and PchipInterpolator is not None and len(unique_y) >= 4:
        interpolator = PchipInterpolator(unique_source, unique_y, extrapolate=True)
        return target, interpolator(target)
    return target, np.interp(target, unique_source, unique_y)


def truncate_prefix(
    x: np.ndarray,
    y: np.ndarray,
    prefix_steps: float,
    full_steps: float = 1000.0,
) -> tuple[np.ndarray, np.ndarray]:
    if len(y) <= 1 or prefix_steps <= 0 or prefix_steps >= full_steps:
        return x, y
    if np.nanmax(x) == np.nanmin(x):
        return x, y
    x_norm = (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x)) * full_steps
    order = np.argsort(x_norm)
    x_norm = x_norm[order]
    y = y[order]
    keep = x_norm <= prefix_steps
    if not np.any(keep):
        return x_norm[:1], y[:1]
    x_prefix = x_norm[keep]
    y_prefix = y[keep]
    if x_prefix[-1] < prefix_steps and x_norm[-1] > prefix_steps:
        y_boundary = np.interp(prefix_steps, x_norm, y)
        x_prefix = np.concatenate([x_prefix, np.array([prefix_steps])])
        y_prefix = np.concatenate([y_prefix, np.array([y_boundary])])
    return x_prefix, y_prefix


def build_resampled(
    raw: pd.DataFrame,
    num_points: int,
    trim_terminal: bool,
    cheetah_prefix_steps: float,
) -> pd.DataFrame:
    rows = []
    spec_lookup = {(spec.group, spec.task): spec for spec in RUNS}
    for (group, task), part in raw.groupby(["group", "task"], sort=False):
        spec = spec_lookup[(group, task)]
        part = part.sort_values("x_raw")
        x = part["x_raw"].to_numpy(dtype=np.float64)
        y = part["episode_reward"].to_numpy(dtype=np.float64)
        prefix_points = len(y)
        if task.startswith("cheetah_") and cheetah_prefix_steps > 0:
            x, y = truncate_prefix(x, y, prefix_steps=cheetah_prefix_steps)
            prefix_points = len(y)
        trimmed = 0
        if trim_terminal:
            x, y, trimmed = trim_terminal_outliers(x, y)
        interpolation = "pchip" if task.startswith("cheetah_") else "linear"
        x_resampled, y_resampled = resample_curve(x, y, num_points, interpolation=interpolation)
        for xi, yi in zip(x_resampled, y_resampled):
            rows.append(
                {
                    "group": group,
                    "task": task,
                    "title": spec.title,
                    "project": spec.project,
                    "run_id": spec.run_id,
                    "run_path": spec.path,
                    "x": xi,
                    "episode_reward": yi,
                    "raw_points": len(part),
                    "prefix_points": prefix_points,
                    "clean_points": len(y),
                    "trimmed_terminal_points": trimmed,
                }
            )
        print(
            f"{group:8s} {spec.title:22s} raw={len(part):4d} prefix={prefix_points:4d} "
            f"clean={len(y):4d} trimmed={trimmed}",
            flush=True,
        )
    return pd.DataFrame(rows)


def ordered_specs(group: str) -> list[RunSpec]:
    by_task = {spec.task: spec for spec in RUNS if spec.group == group}
    return [by_task[task] for task in TASK_ORDER]


def style_axis(ax: plt.Axes, is_left: bool, is_bottom: bool) -> None:
    ax.grid(True, color="0.92", linewidth=0.8)
    ax.set_xlim(0, 1000)
    ax.set_xticks([0, 500, 1000])
    ax.tick_params(labelsize=8)
    if is_left:
        ax.set_ylabel("Episode reward", fontsize=9)
    if is_bottom:
        ax.set_xlabel("Resampled eval step", fontsize=9)
    for spine in ax.spines.values():
        spine.set_color("0.35")
        spine.set_linewidth(0.8)


def plot_group(resampled: pd.DataFrame, group: str, out_dir: Path) -> None:
    specs = ordered_specs(group)
    fig, axes = plt.subplots(3, 4, figsize=(12.0, 7.2), dpi=220, sharex=True)
    for idx, spec in enumerate(specs):
        row, col = divmod(idx, 4)
        ax = axes[row, col]
        part = resampled[(resampled["group"] == group) & (resampled["task"] == spec.task)]
        ax.plot(part["x"], part["episode_reward"], color="#2B6EA6", linewidth=1.8)
        ax.set_title(spec.title, fontsize=10, pad=4)
        ymin = float(np.nanmin(part["episode_reward"]))
        ymax = float(np.nanmax(part["episode_reward"]))
        if np.isfinite(ymin) and np.isfinite(ymax) and ymax > ymin:
            margin = 0.08 * (ymax - ymin)
            ax.set_ylim(max(0.0, ymin - margin), ymax + margin)
        style_axis(ax, is_left=(col == 0), is_bottom=(row == 2))
    fig.suptitle(group, fontsize=13, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97), w_pad=1.0, h_pad=1.1)
    slug = group.lower().replace(" ", "_")
    fig.savefig(out_dir / f"{slug}_eval_curves.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"{slug}_eval_curves.png", bbox_inches="tight")
    plt.close(fig)


def plot_combined(resampled: pd.DataFrame, out_dir: Path) -> None:
    groups = ["SWM CLS", "SWM Mean"]
    fig, axes = plt.subplots(6, 4, figsize=(12.0, 13.6), dpi=220, sharex=True)
    for group_idx, group in enumerate(groups):
        specs = ordered_specs(group)
        for idx, spec in enumerate(specs):
            local_row, col = divmod(idx, 4)
            row = group_idx * 3 + local_row
            ax = axes[row, col]
            part = resampled[(resampled["group"] == group) & (resampled["task"] == spec.task)]
            ax.plot(part["x"], part["episode_reward"], color="#2B6EA6", linewidth=1.6)
            ax.set_title(spec.title, fontsize=9, pad=4)
            style_axis(ax, is_left=(col == 0), is_bottom=(row == 5))
        axes[group_idx * 3, 0].text(
            -0.36,
            1.15,
            group,
            transform=axes[group_idx * 3, 0].transAxes,
            fontsize=12,
            fontweight="bold",
            va="center",
        )
    fig.tight_layout(w_pad=0.9, h_pad=1.0)
    fig.savefig(out_dir / "swm_eval_curves_combined.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "swm_eval_curves_combined.png", bbox_inches="tight")
    plt.close(fig)


def write_trim_report(resampled: pd.DataFrame, output_path: Path) -> None:
    cols = [
        "group",
        "task",
        "title",
        "run_path",
        "raw_points",
        "prefix_points",
        "clean_points",
        "trimmed_terminal_points",
    ]
    report = resampled[cols].drop_duplicates().sort_values(["group", "task"])
    report.to_csv(output_path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("figs"))
    parser.add_argument("--analysis-dir", type=Path, default=Path("analysis_outputs"))
    parser.add_argument("--num-points", type=int, default=1000)
    parser.add_argument(
        "--cheetah-prefix-steps",
        type=float,
        default=500.0,
        help="For cheetah_* tasks, keep only the first N normalized steps before resampling. Use 0 to disable.",
    )
    parser.add_argument("--refresh", action="store_true", help="Ignore cached W&B history and download again.")
    parser.add_argument("--no-trim-terminal", action="store_true", help="Disable terminal drop/peak trimming.")
    parser.add_argument("--no-combined", action="store_true", help="Only write separate group figures.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.analysis_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.analysis_dir / "swm_wandb_eval_curves_raw.csv"
    resampled_path = args.analysis_dir / "swm_wandb_eval_curves_resampled.csv"
    trim_report_path = args.analysis_dir / "swm_wandb_eval_curves_trim_report.csv"

    raw = load_or_fetch_raw(raw_path, refresh=args.refresh)
    resampled = build_resampled(
        raw,
        num_points=args.num_points,
        trim_terminal=not args.no_trim_terminal,
        cheetah_prefix_steps=args.cheetah_prefix_steps,
    )
    resampled.to_csv(resampled_path, index=False)
    write_trim_report(resampled, trim_report_path)
    plot_group(resampled, "SWM CLS", args.out_dir)
    plot_group(resampled, "SWM Mean", args.out_dir)
    if not args.no_combined:
        plot_combined(resampled, args.out_dir)
    print(f"Wrote figures to {args.out_dir.resolve()}")
    print(f"Wrote resampled curves to {resampled_path.resolve()}")
    print(f"Wrote trim report to {trim_report_path.resolve()}")


if __name__ == "__main__":
    main()
