#!/usr/bin/env python3
"""Generate DMC demo figures used by the paper appendix."""

import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "osmesa")

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np

from url_benchmark import dmc


OUT_DIR = REPO_ROOT / "figs"
RENDER_SHAPE = (224, 224)
DOMAINS = [
    ("Cheetah", "cheetah_walk"),
    ("Quadruped", "quadruped_walk"),
    ("Walker", "walker_walk"),
]
CONDITIONS = [
    ("Clean", None),
    ("Color-E", "color_easy"),
    ("Color-H", "color_hard"),
    ("Bg-E", "background_easy"),
    ("Bg-H", "background_hard"),
    ("Cam-E", "camera_easy"),
    ("Cam-H", "camera_hard"),
    ("Combined", "combined_easy"),
]


def render_frame(task: str, condition: str | None = None, seed: int = 7) -> np.ndarray:
    env = dmc.make(
        task,
        obs_type="pixels",
        frame_stack=1,
        action_repeat=1,
        seed=seed,
        render_shape=RENDER_SHAPE,
        visual_perturbation=condition,
        visual_perturb_seed=seed + 100,
    )
    time_step = env.reset()
    # Step a few times with zero actions so the pose is not always the reset pose.
    action = np.zeros(env.action_spec().shape, dtype=np.float32)
    for _ in range(6):
        time_step = env.step(action)
    obs = time_step.observation
    if obs.ndim != 3:
        raise RuntimeError(f"Expected CHW pixels, got {obs.shape}")
    return obs[:3].transpose(1, 2, 0)


def save_domain_demo() -> None:
    fig, axes = plt.subplots(1, len(DOMAINS), figsize=(8.4, 2.8), dpi=220)
    for ax, (name, task) in zip(axes, DOMAINS):
        ax.imshow(render_frame(task))
        ax.set_title(name, fontsize=10, pad=5)
        ax.axis("off")
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT_DIR / "appendix_dmc_domains.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "appendix_dmc_domains.png", bbox_inches="tight")
    plt.close(fig)


def save_robustness_demo() -> None:
    fig, axes = plt.subplots(
        len(DOMAINS),
        len(CONDITIONS),
        figsize=(13.6, 5.3),
        dpi=220,
    )
    for row, (domain_name, task) in enumerate(DOMAINS):
        for col, (condition_name, condition) in enumerate(CONDITIONS):
            ax = axes[row, col]
            ax.imshow(render_frame(task, condition=condition, seed=11 + row * 37 + col))
            if row == 0:
                ax.set_title(condition_name, fontsize=8, pad=4)
            if col == 0:
                ax.set_ylabel(domain_name, fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.4)
                spine.set_color("0.75")
    fig.tight_layout(w_pad=0.2, h_pad=0.25)
    fig.savefig(OUT_DIR / "appendix_visual_perturbations.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "appendix_visual_perturbations.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    save_domain_demo()
    save_robustness_demo()
    print(f"Wrote figures to {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
