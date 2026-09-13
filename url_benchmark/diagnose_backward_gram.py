#!/usr/bin/env python3
"""Diagnose how far E[B(y) B(y)^T] is from I_{z_dim} for a trained FB checkpoint.

This answers the question the training loop does not log directly: the
orthonormality regularizer in ``FBDDPGAgent.update_fb`` (see
``url_benchmark/agent/fb_ddpg.py``) is estimated from batch-pairwise Gram
entries, and its per-step diagnostics (``orth_l2``, ``orth_linf``) are only
ever printed to the training CSV/wandb log for the batches seen during
training. This script recomputes the same z_dim x z_dim second-moment matrix
offline, from a fixed bank of states drawn from a trained checkpoint, and
reports its full eigenvalue spectrum instead of just a scalar norm.

Run it once per checkpoint. To compare the goal_space=null (B reads the
visual/DINO latent) and goal_space=simplified_* (B reads low-dim privileged
features) configurations, run it against one checkpoint of each and compare
the printed condition numbers and eigenvalue ranges directly -- the script
does not need to know which configuration it is looking at, because that is
recorded in the checkpoint's own config.yaml (``cfg.goal_space``).

Usage:
    python -m url_benchmark.diagnose_backward_gram \
        --config /path/to/run/.hydra/config.yaml \
        --checkpoint /path/to/run/snapshot_300000.pt \
        --num-states 20480
"""
# REDUNDANCY REVIEW: manual, run-it-yourself diagnostic with no test file and no caller
# anywhere in pretrain.py or launch_*.sh -- it is invoked by hand against a checkpoint after
# the fact. Does not affect the training pipeline; candidate for moving to scripts/ or
# deleting once its findings have been used.
import argparse
from pathlib import Path
from typing import Any, Dict

import numpy as np

from url_benchmark.reward_label_sensitivity import (
    DinoCache,
    collect_bank,
    load_agent,
    load_cfg,
    make_env,
    stable_seed,
)


def gram_diagnostics(backward: np.ndarray) -> Dict[str, Any]:
    """Compute E[B B^T] (z_dim x z_dim) and its deviation from identity."""
    b64 = np.asarray(backward, dtype=np.float64)
    n, z_dim = b64.shape
    second_moment = b64.T @ b64 / n
    diff = second_moment - np.eye(z_dim)
    eigenvalues = np.linalg.eigvalsh(second_moment)  # ascending
    return {
        "n_states": n,
        "z_dim": z_dim,
        "trace_over_zdim": float(np.trace(second_moment) / z_dim),
        "frobenius_diff": float(np.linalg.norm(diff)),
        "max_abs_diff": float(np.max(np.abs(diff))),
        "eig_min": float(eigenvalues[0]),
        "eig_max": float(eigenvalues[-1]),
        "condition_number": float(eigenvalues[-1] / max(eigenvalues[0], 1e-12)),
        "eigenvalues": eigenvalues,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True,
                        help="Path to the run's .hydra/config.yaml")
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Path to a snapshot_*.pt checkpoint")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-states", type=int, default=20480,
                        help="Bank size used to estimate E[B B^T]")
    parser.add_argument("--bank-seed", type=int, default=1101)
    parser.add_argument("--z-resample-steps", type=int, default=200,
                        help="Env steps between behavior z resamples during rollout")
    parser.add_argument("--encode-batch-size", type=int, default=1024)
    parser.add_argument("--collector", default="random_z_mean",
                        choices=["random_z_mean", "random_z_sample"])
    parser.add_argument("--dino-model", default="facebook/dinov2-base",
                        help="Only used if the checkpoint's obs_type == 'dino'")
    args = parser.parse_args()

    cfg = load_cfg(args.config, device=args.device)
    dino_cache = DinoCache(args.dino_model, args.device)
    env = make_env(
        cfg, dino_cache, seed=args.bank_seed,
        visual_seed=stable_seed("diagnose_backward_gram", args.bank_seed),
    )
    agent, global_step = load_agent(cfg, env, args.checkpoint)

    bank = collect_bank(
        agent, cfg, dino_cache,
        size=args.num_states,
        bank_seed=args.bank_seed,
        z_resample_steps=args.z_resample_steps,
        encode_batch_size=args.encode_batch_size,
        checkpoint_fingerprint="diagnose_backward_gram",
        dino_model_name=args.dino_model,
        collector=args.collector,
    )

    stats = gram_diagnostics(bank.backward)
    print(f"checkpoint        : {args.checkpoint}")
    print(f"global_step       : {global_step}")
    print(f"task              : {cfg.task}")
    print(f"goal_space        : {cfg.goal_space}")
    print(f"obs_type          : {cfg.obs_type}")
    print(f"n_states          : {stats['n_states']}")
    print(f"z_dim             : {stats['z_dim']}")
    print(f"tr(E[BB^T])/z_dim : {stats['trace_over_zdim']:.6f}  "
          f"(architecturally pinned to ~1.0 whenever agent.norm_z=True)")
    print(f"||E[BB^T]-I||_F   : {stats['frobenius_diff']:.4f}")
    print(f"max|E[BB^T]-I|    : {stats['max_abs_diff']:.4f}")
    print(f"eigenvalue range  : [{stats['eig_min']:.4f}, {stats['eig_max']:.4f}]  "
          f"(1.0 each = perfectly orthonormal)")
    print(f"condition number  : {stats['condition_number']:.2f}  (1.0 = perfectly orthonormal)")


if __name__ == "__main__":
    main()
