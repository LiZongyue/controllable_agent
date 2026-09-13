#!/usr/bin/env python3
"""Read-only optimizer/topology preflight for the 16 bs=128 Walker/Quadruped runs.

This reuses ``audit_agent`` from ``audit_walker_quadruped_cls3_lr`` (unchanged) and
adds only a ``batch_size == 128`` assertion. It does not sample, forward,
backpropagate, step, or change the random-number state, and never creates an
environment, W&B run, or training process.
"""

from __future__ import annotations

import gc
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from audit_walker_quadruped_cls3_lr import audit_agent  # noqa: E402

# (task, group, separated, lr, source) — source selects which bs=1024 campaign
# each row's actual config and code version is cloned from:
#   "shared" -> 20260904_walker_quadruped_cls3_visualB_alllr5e5_s1 (shared adapter, lr=5e-5)
#   "sep"    -> wq_cls3_lr_20260907T033919Z_retry1 (separated F/B adapters, lr=5e-5 or 1e-5)
ROWS_SPEC = (
    ("walker_stand", "shared_lr5e5", False, 5e-5, "shared"),
    ("walker_stand", "separated_lr5e5", True, 5e-5, "sep"),
    ("walker_walk", "shared_lr5e5", False, 5e-5, "shared"),
    ("walker_walk", "separated_lr5e5", True, 5e-5, "sep"),
    ("walker_run", "shared_lr5e5", False, 5e-5, "shared"),
    ("walker_run", "separated_lr5e5", True, 5e-5, "sep"),
    ("walker_flip", "shared_lr5e5", False, 5e-5, "shared"),
    ("walker_flip", "separated_lr5e5", True, 5e-5, "sep"),
    ("quadruped_stand", "shared_lr5e5", False, 5e-5, "shared"),
    ("quadruped_stand", "separated_lr1e5", True, 1e-5, "sep"),
    ("quadruped_walk", "shared_lr5e5", False, 5e-5, "shared"),
    ("quadruped_walk", "separated_lr1e5", True, 1e-5, "sep"),
    ("quadruped_jump", "shared_lr5e5", False, 5e-5, "shared"),
    ("quadruped_jump", "separated_lr1e5", True, 1e-5, "sep"),
    ("quadruped_run", "separated_lr1e5", True, 1e-5, "sep"),
    ("quadruped_run", "separated_lr5e5", True, 5e-5, "sep"),
)
EXPECTED_BATCH_SIZE = 128
# Three tasks per GPU max (bs=128 is much lighter than the bs=1024 source runs,
# which packed four per GPU); index-aligned with ROWS_SPEC.
GPU_ASSIGNMENT = (2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 2, 3, 4, 5)
RUN_CPUS = tuple((16 + 2 * index, 17 + 2 * index, 80 + 2 * index, 81 + 2 * index)
                  for index in range(len(ROWS_SPEC)))


def assigned_cpus(row: dict, index: int) -> list[int]:
    """Read list assignments, retaining compatibility with older manifests."""
    assert 0 <= index < len(RUN_CPUS), index
    values = row.get("cpus")
    if values is None:
        values = [row["cpu"]] if "cpu" in row else RUN_CPUS[index]
    cpus = [int(cpu) for cpu in values]
    assert cpus and len(cpus) == len(set(cpus)) and min(cpus) >= 0, cpus
    return sorted(cpus)


def audit_agent_bs128(agent: Any, expected_lr: float, separated: bool) -> dict[str, Any]:
    """Assert actual optimizer ownership, LR, DINO CLS3 topology, and batch_size=128."""
    report = audit_agent(agent, expected_lr, separated)
    cfg = agent.cfg
    assert int(cfg.batch_size) == EXPECTED_BATCH_SIZE, \
        f"agent.batch_size must equal {EXPECTED_BATCH_SIZE}, got {cfg.batch_size}"
    report["batch_size"] = int(cfg.batch_size)
    return report


def preflight_manifest(manifest_path: Path, output_path: Path) -> dict[str, Any]:
    """Compose every requested launch config and audit full-size CPU optimizers."""
    import hydra
    import numpy as np
    import torch
    from dm_env import specs
    from omegaconf import OmegaConf
    from url_benchmark import pretrain  # registers the production Hydra schema

    manifest = json.loads(manifest_path.read_text())
    runs = manifest["runs"]
    assert len(runs) == len(ROWS_SPEC), f"expected {len(ROWS_SPEC)} runs, got {len(runs)}"
    keys = [(run["task"], run["group"]) for run in runs]
    expected = {(task, group) for task, group, *_ in ROWS_SPEC}
    assert len(set(keys)) == len(ROWS_SPEC) and set(keys) == expected, \
        "missing, duplicate, or unexpected run"
    assert len({run["run_dir"] for run in runs}) == len(ROWS_SPEC), "duplicate run directories"
    report = {"status": "RUNNING", "manifest": str(manifest_path), "runs": []}
    torch.set_num_threads(1)
    with hydra.initialize_config_dir(config_dir=str(REPO_ROOT / "url_benchmark"), version_base="1.1"):
        for index, run in enumerate(runs):
            cfg = hydra.compose(config_name="base_config", overrides=run["overrides"])
            assert cfg.task == run["task"], "manifest/config task mismatch"
            assert int(cfg.agent.batch_size) == EXPECTED_BATCH_SIZE, "batch_size must be 128"
            assert cfg.dino_model_name == "facebook/dinov2-base", "expected frozen DINOv2 base"
            assert not cfg.auto_resume and cfg.load_model is None and cfg.load_replay_buffer is None, \
                "training must start fresh"
            assert not cfg.append_goal_to_observation and cfg.goal_space is None, \
                "privileged observation/goal must be disabled"
            assert cfg.use_wandb, "W&B must be enabled in launch config"
            pretrain._validate_idm_training_config(cfg)
            obs_spec = specs.Array(shape=(2304,), dtype=np.float32, name="observation")
            action_dim = 6 if cfg.task.startswith("walker_") else 12
            action_spec = specs.Array(shape=(action_dim,), dtype=np.float32, name="action")
            cfg.agent.obs_type = cfg.obs_type
            cfg.agent.obs_shape = obs_spec.shape
            cfg.agent.action_shape = action_spec.shape
            cfg.agent.num_expl_steps = cfg.num_seed_frames // cfg.action_repeat
            resolved = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
            # This process performs no environment setup or GPU allocation.
            cfg.device = "cpu"
            cfg.agent.device = "cpu"
            torch.manual_seed(int(cfg.seed))
            agent = pretrain.make_agent(cfg.obs_type, obs_spec, action_spec,
                                        cfg.num_seed_frames // cfg.action_repeat, cfg.agent)
            audit = audit_agent_bs128(agent, float(run["lr"]), run["separated"])
            report["runs"].append({"task": run["task"], "group": run["group"],
                                   "run_dir": run["run_dir"], "resolved_config": resolved,
                                   "audit": audit})
            print(f"[{index + 1}/{len(ROWS_SPEC)}] PASS {run['task']} {run['group']} "
                  f"lr={run['lr']} batch_size={audit['batch_size']}", flush=True)
            del agent, cfg
            gc.collect()
    report["status"] = "PASS"
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    manifest_path = args.manifest.resolve()
    output_path = args.output or manifest_path.with_name("preflight.json")
    preflight_manifest(manifest_path, output_path)
    print(f"PASS: all {len(ROWS_SPEC)} full-size CPU agents audited; {output_path}", flush=True)


if __name__ == "__main__":
    main()
