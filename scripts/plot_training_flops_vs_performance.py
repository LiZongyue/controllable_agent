#!/usr/bin/env python3
"""Plot training FLOPs vs aggregate performance for DMC visual control.

This script generates one vertical figure with Cheetah, Quadruped, and Walker
stacked from top to bottom. Performance values are fixed from the prompt.

FLOPs are static architecture estimates, not hardware profiler numbers:
  - one multiply-add is counted as 2 FLOPs;
  - trainable Linear/Conv forward+backward is approximated as 3x forward;
  - forward-only or frozen feature extraction is counted as 1x forward;
  - environment simulation, rendering, data movement, layernorm/activation,
    optimizer elementwise ops, logging, checkpointing, and evaluation are not
    included.

TD-MPC2 visual FLOPs use the official visual setup: 64x64 RGB observations,
a 4-layer convolutional encoder, and the default 5M online RL architecture.
The 2502.03550 TD-M(PC)^2 paper describes a policy-constraint modification on
top of TD-MPC2 with no additional computational budget, so this script uses the
same FLOPs for that point.

Dreamer defaults to a size1m visual configuration for a fairer single-task DMC
baseline comparison. The local Dreamer config's bare dmc_vision preset expands
to a much larger model, which is useful to reproduce that exact command but is
not the fairest architecture-size comparison against the FB runs here.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np


METHOD_ORDER = ["Dreamer", "TD-MPC2", "CNN-FB", "SWM"]
DOMAIN_ORDER = ["Walker", "Quadruped", "Cheetah"]

PERFORMANCE = {
    "Walker": {
        "Dreamer": 114.3,
        "TD-MPC2": 266.95,
        "CNN-FB": 338.83,
        "SWM": 324.16,
    },
    "Quadruped": {
        "Dreamer": 56.23,
        "TD-MPC2": 459.09,
        "CNN-FB": 516.59,
        "SWM": 540.52,
    },
    "Cheetah": {
        "Dreamer": 44.58,
        "TD-MPC2": 108.79,
        "CNN-FB": 144.69,
        "SWM": 139.28,
    },
}


COLORS = {
    "Dreamer": "#4C78A8",
    "TD-MPC2": "#F58518",
    "CNN-FB": "#54A24B",
    "SWM": "#E45756",
}

MARKERS = {
    "Dreamer": "o",
    "TD-MPC2": "s",
    "CNN-FB": "^",
    "SWM": "P",
}

POINT_LABELS = {
    "Dreamer": "Dreamer",
    "TD-MPC2": "TD-MPC",
    "CNN-FB": "CNN-FB",
    "SWM": "SWM",
}


@dataclass(frozen=True)
class DomainConfig:
    action_dim: int
    goal_dim: int
    visual_goal_space: bool


@dataclass(frozen=True)
class DreamerSize:
    deter: int
    hidden: int
    classes: int
    depth: int
    units: int


DOMAINS = {
    "Cheetah": DomainConfig(action_dim=6, goal_dim=512, visual_goal_space=True),
    "Walker": DomainConfig(action_dim=6, goal_dim=3, visual_goal_space=False),
    "Quadruped": DomainConfig(action_dim=12, goal_dim=2, visual_goal_space=False),
}

DREAMER_SIZES = {
    "size1m": DreamerSize(deter=512, hidden=64, classes=4, depth=4, units=64),
    "size12m": DreamerSize(deter=2048, hidden=256, classes=16, depth=16, units=256),
    "size25m": DreamerSize(deter=3072, hidden=384, classes=24, depth=24, units=384),
    "size50m": DreamerSize(deter=4096, hidden=512, classes=32, depth=32, units=512),
    "size100m": DreamerSize(deter=6144, hidden=768, classes=48, depth=48, units=768),
    "size200m": DreamerSize(deter=8192, hidden=1024, classes=64, depth=64, units=1024),
}


def conv_out(size: int, kernel: int, stride: int = 1, padding: int = 0) -> int:
    return (size + 2 * padding - kernel) // stride + 1


def linear_flops(in_dim: int, out_dim: int, batch: int = 1) -> float:
    return float(2 * batch * in_dim * out_dim)


def mlp_flops(dims: list[int], batch: int = 1) -> float:
    return sum(linear_flops(a, b, batch) for a, b in zip(dims[:-1], dims[1:]))


def conv2d_flops(
    cin: int,
    cout: int,
    h: int,
    w: int,
    kernel: int,
    stride: int = 1,
    padding: int = 0,
    batch: int = 1,
) -> tuple[float, int, int]:
    oh = conv_out(h, kernel, stride, padding)
    ow = conv_out(w, kernel, stride, padding)
    flops = 2 * batch * oh * ow * cout * cin * kernel * kernel
    return float(flops), oh, ow


def train_factor(forward_flops: float) -> float:
    return 3.0 * forward_flops


def input_grad_factor(forward_flops: float) -> float:
    return 2.0 * forward_flops


def fb_agent_steps(num_train_frames: int, action_repeat: int) -> int:
    return math.ceil(num_train_frames / action_repeat)


def fb_update_count(
    num_train_frames: int,
    action_repeat: int,
    num_seed_frames: int,
    update_every_steps: int,
) -> int:
    steps = fb_agent_steps(num_train_frames, action_repeat)
    seed_steps = math.ceil(num_seed_frames / action_repeat)
    first = seed_steps
    while first % update_every_steps:
        first += 1
    if first >= steps:
        return 0
    return (steps - first + update_every_steps - 1) // update_every_steps


def cnn_fb_encoder_forward(batch: int = 1, frame_stack: int = 3, size: int = 84) -> float:
    c, h, w = 3 * frame_stack, size, size
    total = 0.0
    for cout, kernel, stride in [(32, 3, 2), (32, 3, 1), (32, 3, 1), (32, 3, 1)]:
        flops, h, w = conv2d_flops(c, cout, h, w, kernel, stride, batch=batch)
        total += flops
        c = cout
    return total


def dino_vitb14_forward(image_size: int = 224) -> float:
    patch = 14
    hidden = 768
    layers = 12
    mlp_ratio = 4
    patches = (image_size // patch) ** 2
    tokens = patches + 1
    patch_embed = 2 * patches * 3 * patch * patch * hidden
    qkv = 2 * tokens * hidden * (3 * hidden)
    attn_scores = 2 * tokens * tokens * hidden
    attn_apply = 2 * tokens * tokens * hidden
    proj = 2 * tokens * hidden * hidden
    mlp = (
        2 * tokens * hidden * (mlp_ratio * hidden)
        + 2 * tokens * (mlp_ratio * hidden) * hidden
    )
    return float(patch_embed + layers * (qkv + attn_scores + attn_apply + proj + mlp))


def dino_adapter_forward(batch: int = 1, in_dim: int = 768, out_dim: int = 512) -> float:
    return linear_flops(in_dim, out_dim, batch)


def fb_actor_forward(obs_dim: int, action_dim: int, batch: int) -> float:
    hidden = 1024
    feature = 512
    z_dim = 50
    return (
        mlp_flops([obs_dim, hidden, feature], batch)
        + mlp_flops([obs_dim + z_dim, hidden, feature], batch)
        + mlp_flops([2 * feature, hidden, action_dim], batch)
    )


def fb_forward_map_forward(obs_dim: int, action_dim: int, batch: int) -> float:
    hidden = 1024
    feature = 512
    z_dim = 50
    return (
        mlp_flops([obs_dim + action_dim, hidden, feature], batch)
        + mlp_flops([obs_dim + z_dim, hidden, feature], batch)
        + 2 * mlp_flops([2 * feature, hidden, z_dim], batch)
    )


def fb_backward_forward(goal_dim: int, batch: int) -> float:
    return mlp_flops([goal_dim, 526, 50], batch)


def fb_network_update_forward(obs_dim: int, goal_dim: int, action_dim: int, batch: int) -> float:
    actor = fb_actor_forward(obs_dim, action_dim, batch)
    fmap = fb_forward_map_forward(obs_dim, action_dim, batch)
    bmap = fb_backward_forward(goal_dim, batch)

    # update_fb: target actor/F/B are no-grad; current F/B receive gradients.
    update_fb = actor + fmap + bmap + train_factor(fmap + bmap)

    # update_actor: actor receives gradients, forward map supplies gradients to
    # the sampled action but is not optimized by actor_opt.
    update_actor = train_factor(actor) + input_grad_factor(fmap)
    return update_fb + update_actor


def estimate_fb_flops(
    domain: str,
    encoder_forward: Callable[[int], float],
    obs_dim: int,
    include_dino_backbone: bool,
    num_train_frames: int,
    action_repeat: int,
    seed_frames: int,
    update_every_steps: int,
    batch_size: int,
) -> float:
    cfg = DOMAINS[domain]
    agent_steps = fb_agent_steps(num_train_frames, action_repeat)
    updates = fb_update_count(num_train_frames, action_repeat, seed_frames, update_every_steps)

    train_encodes_per_update = 2 if cfg.visual_goal_space else 1
    forward_encodes_per_update = 2 if cfg.visual_goal_space else 1

    encoder_updates = updates * (
        train_encodes_per_update * train_factor(encoder_forward(batch_size))
        + forward_encodes_per_update * encoder_forward(batch_size)
    )
    action_selection = agent_steps * (encoder_forward(1) + fb_actor_forward(obs_dim, cfg.action_dim, 1))
    fb_updates = updates * fb_network_update_forward(obs_dim, cfg.goal_dim, cfg.action_dim, batch_size)
    dino_env = agent_steps * dino_vitb14_forward() if include_dino_backbone else 0.0
    return dino_env + action_selection + encoder_updates + fb_updates


def tdmpc2_visual_encoder_forward(batch: int = 1) -> float:
    # Official TD-MPC2 rgb observations stack 3 RGB frames at 64x64.
    c, h, w = 9, 64, 64
    total = 0.0
    for cout, kernel, stride in [(32, 7, 2), (32, 5, 2), (32, 3, 2), (32, 3, 1)]:
        flops, h, w = conv2d_flops(c, cout, h, w, kernel, stride, batch=batch)
        total += flops
        c = cout
    return total


def tdmpc2_dynamics_forward(action_dim: int, batch: int) -> float:
    return mlp_flops([512 + action_dim, 512, 512, 512], batch)


def tdmpc2_reward_forward(action_dim: int, batch: int) -> float:
    return mlp_flops([512 + action_dim, 512, 512, 101], batch)


def tdmpc2_policy_forward(action_dim: int, batch: int) -> float:
    return mlp_flops([512, 512, 512, 2 * action_dim], batch)


def tdmpc2_q_forward(action_dim: int, batch: int, num_q: int = 5) -> float:
    return num_q * mlp_flops([512 + action_dim, 512, 512, 101], batch)


def estimate_tdmpc2_visual_flops(domain: str, steps: int, batch_size: int = 256) -> float:
    action_dim = DOMAINS[domain].action_dim
    horizon = 3
    iterations = 6 + 2 * int(action_dim >= 20)
    population = 512
    pi_trajs = 24
    seed_steps = 2500
    # OnlineTrainer pretrains for seed_steps gradient updates once seed data is
    # collected, then does one update per decision step; this is approximately
    # steps total updates.
    updates = steps

    target_next_z = horizon * tdmpc2_visual_encoder_forward(batch_size)
    target_td = (
        tdmpc2_policy_forward(action_dim, horizon * batch_size)
        + tdmpc2_q_forward(action_dim, horizon * batch_size)
    )
    train_model = train_factor(
        tdmpc2_visual_encoder_forward(batch_size)
        + horizon * tdmpc2_dynamics_forward(action_dim, batch_size)
        + tdmpc2_reward_forward(action_dim, horizon * batch_size)
        + tdmpc2_q_forward(action_dim, horizon * batch_size)
    )
    pi_update = train_factor(tdmpc2_policy_forward(action_dim, (horizon + 1) * batch_size))
    pi_update += input_grad_factor(tdmpc2_q_forward(action_dim, (horizon + 1) * batch_size))
    update_flops = target_next_z + target_td + train_model + pi_update

    policy_traj = (
        horizon * tdmpc2_policy_forward(action_dim, pi_trajs)
        + (horizon - 1) * tdmpc2_dynamics_forward(action_dim, pi_trajs)
    )
    rollout = horizon * (
        tdmpc2_reward_forward(action_dim, population)
        + tdmpc2_dynamics_forward(action_dim, population)
    )
    terminal = tdmpc2_policy_forward(action_dim, population) + tdmpc2_q_forward(action_dim, population)
    plan_flops = tdmpc2_visual_encoder_forward(1) + policy_traj + iterations * (rollout + terminal)

    planned_steps = max(0, steps - seed_steps)
    return updates * update_flops + planned_steps * plan_flops


def dreamer_encoder_forward(batch: int = 1, image_size: int = 64, depth: int = 4) -> float:
    channels = 3
    h = w = image_size
    total = 0.0
    for channels_out in [2 * depth, 3 * depth, 4 * depth, 4 * depth]:
        # DreamerV3 dmc_vision uses same-conv followed by 2x2 spatial pooling.
        total += 2 * batch * h * w * channels * channels_out * 5 * 5
        h //= 2
        w //= 2
        channels = channels_out
    return float(total)


def dreamer_rssm_step_forward(
    action_dim: int,
    batch: int = 1,
    deter: int = 512,
    hidden: int = 64,
    classes: int = 4,
    depth: int = 4,
) -> float:
    stoch = 32 * classes
    blocks = 8
    tokens = 4 * 4 * (4 * depth)
    core = (
        linear_flops(deter, hidden, batch)
        + linear_flops(stoch, hidden, batch)
        + linear_flops(action_dim, hidden, batch)
    )
    per_group_in = deter // blocks + 3 * hidden
    core += 2 * batch * blocks * per_group_in * (deter // blocks)
    core += 2 * batch * blocks * (deter // blocks) * (3 * deter // blocks)
    obs = linear_flops(deter + tokens, hidden, batch) + linear_flops(hidden, stoch, batch)
    prior = (
        linear_flops(deter, hidden, batch)
        + linear_flops(hidden, hidden, batch)
        + linear_flops(hidden, stoch, batch)
    )
    return float(core + obs + prior)


def dreamer_policy_forward(action_dim: int, batch: int, feature_dim: int, units: int) -> float:
    return mlp_flops([feature_dim, units, units, units, 2 * action_dim], batch)


def dreamer_value_forward(batch: int, feature_dim: int, units: int) -> float:
    return mlp_flops([feature_dim, units, units, units, 255], batch)


def estimate_dreamer_dmc_vision_flops(
    domain: str,
    steps: int,
    size_name: str,
    train_ratio: float,
    batch_size: int,
    batch_length: int,
) -> float:
    action_dim = DOMAINS[domain].action_dim
    size = DREAMER_SIZES[size_name]
    imag_length = 15
    transitions_per_update = batch_size * batch_length
    updates = math.ceil(steps * train_ratio / transitions_per_update)
    feature_dim = size.deter + 32 * size.classes

    encoder = dreamer_encoder_forward(transitions_per_update, depth=size.depth)
    rssm = dreamer_rssm_step_forward(
        action_dim,
        transitions_per_update,
        deter=size.deter,
        hidden=size.hidden,
        classes=size.classes,
        depth=size.depth,
    )
    decoder = dreamer_encoder_forward(transitions_per_update, depth=size.depth)  # symmetric decoder approximation
    heads = (
        mlp_flops([feature_dim, size.units, 255], transitions_per_update)
        + mlp_flops([feature_dim, size.units, 1], transitions_per_update)
        + dreamer_value_forward(transitions_per_update, feature_dim, size.units)
    )
    imagined_batch = batch_size * batch_length * imag_length
    imagined = (
        dreamer_rssm_step_forward(
            action_dim,
            imagined_batch,
            deter=size.deter,
            hidden=size.hidden,
            classes=size.classes,
            depth=size.depth,
        )
        + dreamer_policy_forward(action_dim, imagined_batch, feature_dim, size.units)
        + dreamer_value_forward(imagined_batch, feature_dim, size.units)
    )
    # World-model/reconstruction losses backprop through encoder/RSSM/decoder/heads.
    # DreamerV3 stops gradients through imagined features by default, so imagined
    # RSSM is counted forward-only; policy/value heads on imagined features train.
    update_flops = train_factor(encoder + rssm + decoder + heads)
    update_flops += imagined + train_factor(
        dreamer_policy_forward(action_dim, imagined_batch, feature_dim, size.units)
        + dreamer_value_forward(imagined_batch, feature_dim, size.units)
    )
    action_flops = steps * (
        dreamer_encoder_forward(1, depth=size.depth)
        + dreamer_rssm_step_forward(
            action_dim,
            1,
            deter=size.deter,
            hidden=size.hidden,
            classes=size.classes,
            depth=size.depth,
        )
        + dreamer_policy_forward(action_dim, 1, feature_dim, size.units)
    )
    return updates * update_flops + action_flops


def build_rows(args: argparse.Namespace) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for domain in DOMAIN_ORDER:
        flops = {
            "Dreamer": estimate_dreamer_dmc_vision_flops(
                domain,
                args.dreamer_steps,
                args.dreamer_size,
                args.dreamer_train_ratio,
                args.dreamer_batch_size,
                args.dreamer_batch_length,
            ),
            "TD-MPC2": estimate_tdmpc2_visual_flops(domain, args.tdmpc2_steps),
            "CNN-FB": estimate_fb_flops(
                domain,
                encoder_forward=cnn_fb_encoder_forward,
                obs_dim=32 * 35 * 35,
                include_dino_backbone=False,
                num_train_frames=args.fb_train_frames,
                action_repeat=args.fb_action_repeat,
                seed_frames=args.fb_seed_frames,
                update_every_steps=args.fb_update_every,
                batch_size=args.fb_batch_size,
            ),
            "SWM": estimate_fb_flops(
                domain,
                encoder_forward=dino_adapter_forward,
                obs_dim=512,
                include_dino_backbone=True,
                num_train_frames=args.fb_train_frames,
                action_repeat=args.fb_action_repeat,
                seed_frames=args.fb_seed_frames,
                update_every_steps=args.fb_update_every,
                batch_size=args.fb_batch_size,
            ),
        }
        for method in METHOD_ORDER:
            rows.append(
                {
                    "domain": domain,
                    "method": method,
                    "performance": PERFORMANCE[domain][method],
                    "training_flops": flops[method],
                    "training_pflops": flops[method] / 1e15,
                }
            )
    return rows


def write_csv(rows: list[dict[str, float | str]], path: Path) -> None:
    fieldnames = ["domain", "method", "performance", "training_flops", "training_pflops"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot(rows: list[dict[str, float | str]], out_png: Path, out_pdf: Path) -> None:
    by_domain = {
        domain: [row for row in rows if row["domain"] == domain]
        for domain in DOMAIN_ORDER
    }
    all_x = np.array([float(row["training_pflops"]) for row in rows])
    xmin, xmax = all_x.min() * 0.65, all_x.max() * 1.6

    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.titlesize": 18,
            "axes.labelsize": 16,
            "legend.fontsize": 16,
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "font.family": "DejaVu Sans",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig, axes = plt.subplots(3, 1, figsize=(3.55, 7.6), sharex=True)
    fig.subplots_adjust(hspace=0.5, left=0.17, right=0.96, top=0.96, bottom=0.06)

    label_offsets = {
        "Dreamer": (9, 7),
        "TD-MPC2": (24, -15),
        "CNN-FB": (8, 8),
        "SWM": (8, 8),
    }

    for ax, domain in zip(axes, DOMAIN_ORDER):
        domain_rows = by_domain[domain]
        ys = np.array([float(row["performance"]) for row in domain_rows])
        ypad = max(16.0, 0.22 * (ys.max() - ys.min()))
        for row in domain_rows:
            method = str(row["method"])
            x = float(row["training_pflops"])
            y = float(row["performance"])
            ax.scatter(
                x,
                y,
                s=220,
                color=COLORS[method],
                marker=MARKERS[method],
                edgecolor="white",
                linewidth=1.2,
                zorder=3,
                label=method if domain == DOMAIN_ORDER[0] else None,
            )
            dx, dy = label_offsets[method]
            ax.annotate(
                POINT_LABELS[method],
                xy=(x, y),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=16,
                color="#222222",
            )
        ax.set_xscale("log")
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(float(ys.min() - ypad), float(ys.max() + ypad))
        ax.set_title(domain, loc="left", fontweight="bold")
        ax.set_ylabel("")
        ax.grid(True, which="major", color="#D9D9D9", linewidth=0.8)
        ax.grid(True, which="minor", axis="x", color="#EEEEEE", linewidth=0.5)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    axes[-1].set_xlabel("")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_pdf.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=Path("analysis_outputs/flops_vs_performance"))
    parser.add_argument("--fb-train-frames", type=int, default=2_000_010)
    parser.add_argument("--fb-action-repeat", type=int, default=2)
    parser.add_argument("--fb-seed-frames", type=int, default=4_000)
    parser.add_argument("--fb-update-every", type=int, default=2)
    parser.add_argument("--fb-batch-size", type=int, default=1024)
    parser.add_argument(
        "--tdmpc2-steps",
        type=int,
        default=1_000_000,
        help="TD-MPC2 decision steps. Its DMControl wrapper repeats each action twice, so 1M decisions align with 2M raw DMC frames.",
    )
    parser.add_argument("--dreamer-steps", type=int, default=2_000_000)
    parser.add_argument("--dreamer-size", choices=sorted(DREAMER_SIZES), default="size1m")
    parser.add_argument("--dreamer-train-ratio", type=float, default=256.0)
    parser.add_argument("--dreamer-batch-size", type=int, default=16)
    parser.add_argument("--dreamer-batch-length", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = build_rows(args)
    csv_path = args.outdir / "training_flops_vs_performance.csv"
    png_path = args.outdir / "training_flops_vs_performance.png"
    pdf_path = args.outdir / "training_flops_vs_performance.pdf"
    svg_path = args.outdir / "training_flops_vs_performance.svg"
    write_csv(rows, csv_path)
    plot(rows, png_path, pdf_path)
    print(f"Wrote {csv_path}")
    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")
    print(f"Wrote {svg_path}")


if __name__ == "__main__":
    main()
