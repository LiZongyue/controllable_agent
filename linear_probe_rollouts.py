#!/usr/bin/env python3
"""Roll out trained DMC agents and probe representations against env state.

Default use:
  python linear_probe_rollouts.py --dry-run
  python linear_probe_rollouts.py --limit 4 --episodes 3

This script is intentionally independent from Hydra.  It discovers checkpoint
runs, rolls each checkpoint in its own task, extracts frozen representations,
then trains linear/MLP probes to regress the DMC physics state and evaluates
the trained probes on held-out rollout samples.

Evaluation trains linear and non-linear probes from latent embeddings to DMC
physics state and reports MSE plus Pearson r.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import math
import os
import re
import signal
import sys
import time
import traceback
import typing as tp
import warnings
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ["MUJOCO_GL"] = os.environ.get("MUJOCO_GL", "egl")

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from torch import nn

from url_benchmark import dmc

warnings.filterwarnings("ignore", category=DeprecationWarning)
signal.signal(signal.SIGPIPE, signal.SIG_DFL)


TASK_GOAL_SPACES = {
    "walker_stand": "simplified_walker",
    "walker_walk": "simplified_walker",
    "walker_run": "simplified_walker",
    "walker_flip": "simplified_walker",
    "quadruped_stand": "simplified_quadruped",
    "quadruped_walk": "simplified_quadruped",
    "quadruped_run": "simplified_quadruped",
    "quadruped_jump": "simplified_quadruped",
    "cheetah_walk": None,
    "cheetah_run": None,
    "cheetah_walk_backward": None,
    "cheetah_run_backward": None,
}
TASKS = sorted(TASK_GOAL_SPACES, key=len, reverse=True)


@dataclasses.dataclass(frozen=True)
class CheckpointJob:
    family: str
    run_name: str
    task: str
    checkpoint: Path
    checkpoint_step: tp.Optional[int]
    seed: tp.Optional[int] = None


@dataclasses.dataclass
class RolloutBatch:
    features: tp.Dict[str, np.ndarray]
    targets: tp.Dict[str, np.ndarray]
    rewards: tp.List[float]
    steps: int


@dataclasses.dataclass
class ProbeResult:
    summary: tp.Dict[str, tp.Any]
    epochs: tp.List[tp.Dict[str, tp.Any]]


class DinoBackboneCache:
    def __init__(self, device: str) -> None:
        self.device = device
        self.processor = None
        self.model = None

    def get(self) -> tp.Tuple[tp.Any, torch.nn.Module]:
        if self.processor is None or self.model is None:
            from transformers import AutoImageProcessor, AutoModel

            print("loading facebook/dinov2-base for DINO CLS env embedding", flush=True)
            self.processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base", use_fast=True)
            self.model = AutoModel.from_pretrained("facebook/dinov2-base")
            self.model.to(self.device)
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad_(False)
        return self.processor, self.model


def infer_task_from_name(name: str) -> tp.Optional[str]:
    padded = f"_{name}_"
    for task in TASKS:
        if f"_{task}_" in padded:
            return task
    return None


def infer_seed_from_name(name: str) -> tp.Optional[int]:
    match = re.search(r"(?:^|_)seed(\d+)(?:_|$)", name)
    if match:
        return int(match.group(1))
    return None


def snapshot_step(path: Path) -> tp.Optional[int]:
    match = re.match(r"snapshot_(\d+)\.pt$", path.name)
    if match:
        return int(match.group(1))
    return None


def parse_steps(value: tp.Optional[str]) -> tp.Optional[tp.Set[tp.Optional[int]]]:
    if not value:
        return None
    out: tp.Set[tp.Optional[int]] = set()
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if item == "latest":
            out.add(None)
        else:
            out.add(int(item))
    return out


def parse_int_set(value: tp.Optional[str]) -> tp.Optional[tp.Set[int]]:
    if value is None:
        return None
    value = value.strip()
    if not value or value.lower() == "all":
        return None
    return {int(item.strip()) for item in value.split(",") if item.strip()}


def parse_probe_types(value: str) -> tp.List[str]:
    probes = [item.strip().lower() for item in value.split(",") if item.strip()]
    allowed = {"linear", "mlp"}
    unknown = [probe for probe in probes if probe not in allowed]
    if unknown:
        raise ValueError(f"Unsupported probe type(s): {unknown}. Allowed: {sorted(allowed)}")
    if not probes:
        raise ValueError("At least one probe type is required")
    return probes


def parse_targets(value: str) -> tp.List[str]:
    targets = [item.strip().lower() for item in value.split(",") if item.strip()]
    allowed = {"physics", "observation", "visible_pose", "motion"}
    unknown = [target for target in targets if target not in allowed]
    if unknown:
        raise ValueError(f"Unsupported target(s): {unknown}. Allowed: {sorted(allowed)}")
    if not targets:
        raise ValueError("At least one target is required")
    return list(dict.fromkeys(targets))


def discover_jobs(
    checkpoint_root: Path,
    family: str,
    run_glob: str,
    include_latest: bool,
    allowed_tasks: tp.Optional[tp.Set[str]],
    allowed_steps: tp.Optional[tp.Set[tp.Optional[int]]],
    allowed_seeds: tp.Optional[tp.Set[int]],
    max_ckpts_per_run: int,
) -> tp.List[CheckpointJob]:
    jobs: tp.List[CheckpointJob] = []
    for run_dir in sorted(checkpoint_root.glob(run_glob)):
        if not run_dir.is_dir():
            continue
        task = infer_task_from_name(run_dir.name)
        if task is None:
            continue
        if allowed_tasks is not None and task not in allowed_tasks:
            continue
        run_seed = infer_seed_from_name(run_dir.name)
        if allowed_seeds is not None and run_seed not in allowed_seeds:
            continue
        ckpts = sorted(run_dir.glob("snapshot_*.pt"), key=lambda p: snapshot_step(p) or -1)
        if include_latest and (run_dir / "latest.pt").exists():
            ckpts.append(run_dir / "latest.pt")
        if allowed_steps is not None:
            ckpts = [p for p in ckpts if snapshot_step(p) in allowed_steps or (p.name == "latest.pt" and None in allowed_steps)]
        if max_ckpts_per_run > 0:
            ckpts = ckpts[:max_ckpts_per_run]
        for ckpt in ckpts:
            jobs.append(
                CheckpointJob(
                    family=family,
                    run_name=run_dir.name,
                    task=task,
                    checkpoint=ckpt,
                    checkpoint_step=snapshot_step(ckpt),
                    seed=run_seed,
                )
            )
    return jobs


def load_agent(checkpoint: Path, device: str) -> tp.Tuple[tp.Any, tp.Dict[str, tp.Any]]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    info: tp.Dict[str, tp.Any] = {}
    if isinstance(payload, dict) and "agent" in payload:
        agent = payload["agent"]
        info["global_step"] = payload.get("global_step")
        info["global_episode"] = payload.get("global_episode")
    else:
        agent = payload
    prepare_agent(agent, device)
    return agent, info


def prepare_agent(agent: tp.Any, device: str) -> None:
    cfg = getattr(agent, "cfg", None)
    if cfg is not None and hasattr(cfg, "device"):
        cfg.device = device
    # Checkpoints created before the optional visual-B adapter was introduced
    # do not have these attributes, while the current ``train(False)`` method
    # expects them to exist.
    for name in ("backward_encoder", "backward_encoder_target"):
        if not hasattr(agent, name):
            setattr(agent, name, None)
    for value in vars(agent).values():
        if isinstance(value, nn.Module):
            value.to(device)
            value.eval()
    if hasattr(agent, "train"):
        try:
            agent.train(False)
        except TypeError:
            pass


def agent_cfg_value(agent: tp.Any, name: str, default: tp.Any = None) -> tp.Any:
    cfg = getattr(agent, "cfg", None)
    if cfg is not None and hasattr(cfg, name):
        return getattr(cfg, name)
    return getattr(agent, name, default)


def infer_render_shape(agent: tp.Any, family: str) -> tp.Tuple[int, int]:
    obs_shape = tuple(agent_cfg_value(agent, "obs_shape", ()))
    obs_type = agent_cfg_value(agent, "obs_type", None)
    if obs_type == "pixels" and len(obs_shape) == 3:
        return int(obs_shape[1]), int(obs_shape[2])
    if family == "cnn":
        return (84, 84)
    return (224, 224)


def infer_dino_frame_stack(agent: tp.Any) -> int:
    """Recover the embedding history length retained by a DINO checkpoint.

    Workspace-level ``dino_frame_stack`` is not serialized in the agent cfg,
    but CLS checkpoints do retain their flattened input shape.  DINOv2-base
    emits a 768-dimensional CLS token, so 768 and 2304 correspond to one and
    three frames, respectively.
    """
    if agent_cfg_value(agent, "obs_type", None) != "dino":
        return 1
    configured = agent_cfg_value(agent, "dino_frame_stack", None)
    if configured is not None:
        stack = int(configured)
        if stack < 1:
            raise ValueError(f"Invalid dino_frame_stack={stack}")
        return stack
    obs_shape = tuple(agent_cfg_value(agent, "obs_shape", ()))
    if bool(agent_cfg_value(agent, "use_cls", True)) and len(obs_shape) == 1:
        obs_dim = int(obs_shape[0])
        if obs_dim % 768 != 0:
            raise ValueError(f"Cannot infer DINO CLS frame stack from obs_shape={obs_shape}")
        stack = obs_dim // 768
        if stack >= 1:
            return stack
    return 1


def make_env_for_agent(
    agent: tp.Any,
    task: str,
    family: str,
    seed: int,
    action_repeat: int,
    dino_cache: DinoBackboneCache,
) -> dmc.EnvWrapper:
    obs_type = agent_cfg_value(agent, "obs_type", None)
    if family == "dreamer" and obs_type not in {"states", "pixels", "dino", "vit"}:
        obs_type = "pixels"
    if obs_type not in {"states", "pixels", "dino", "vit"}:
        raise ValueError(f"Unsupported obs_type={obs_type!r} for {family}")

    frame_stack = 3 if obs_type == "pixels" else 1
    dino_frame_stack = infer_dino_frame_stack(agent)
    use_cls = bool(agent_cfg_value(agent, "use_cls", True))
    goal_space = agent_cfg_value(agent, "goal_space", TASK_GOAL_SPACES.get(task))
    if goal_space == "null":
        goal_space = None
    render_shape = infer_render_shape(agent, family)
    dino_model = None
    dino_processor = None
    if obs_type == "dino":
        dino_processor, dino_model = dino_cache.get()
    return dmc.make(
        task,
        obs_type=obs_type,
        frame_stack=frame_stack,
        action_repeat=action_repeat,
        seed=seed,
        goal_space=goal_space,
        append_goal_to_observation=False,
        use_cls=use_cls,
        dino_model=dino_model,
        dino_processor=dino_processor,
        render_shape=render_shape,
        dino_frame_stack=dino_frame_stack,
    )


def init_meta(agent: tp.Any) -> tp.Mapping[str, np.ndarray]:
    if hasattr(agent, "init_meta"):
        meta = agent.init_meta()
        return meta if meta is not None else {}
    return {}


def act(agent: tp.Any, obs: np.ndarray, meta: tp.Mapping[str, np.ndarray], step: int) -> np.ndarray:
    if hasattr(agent, "act"):
        with torch.no_grad():
            return agent.act(obs, meta, step, eval_mode=True)
    raise AttributeError(f"{type(agent).__name__} does not expose act(obs, meta, step, eval_mode)")


@torch.no_grad()
def module_feature(module: nn.Module, obs: np.ndarray, device: str) -> np.ndarray:
    tensor = torch.as_tensor(obs, device=device, dtype=torch.float32).unsqueeze(0)
    out = module(tensor)
    if isinstance(out, (tuple, list)):
        out = out[0]
    return out.reshape(out.shape[0], -1).squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)


def raw_feature(obs: np.ndarray) -> np.ndarray:
    return np.asarray(obs, dtype=np.float32).reshape(-1)


def dreamer_encoder(agent: tp.Any) -> tp.Optional[nn.Module]:
    candidates = [
        getattr(agent, "encoder", None),
        getattr(getattr(agent, "world_model", None), "encoder", None),
        getattr(getattr(agent, "wm", None), "encoder", None),
        getattr(getattr(agent, "model", None), "encoder", None),
    ]
    for candidate in candidates:
        if isinstance(candidate, nn.Module):
            return candidate
    return None


def build_extractors(
    agent: tp.Any,
    family: str,
    device: str,
    include_dino_raw: bool,
    include_dino_adapter: bool,
) -> tp.Dict[str, tp.Callable[[np.ndarray], np.ndarray]]:
    obs_type = agent_cfg_value(agent, "obs_type", None)
    extractors: tp.Dict[str, tp.Callable[[np.ndarray], np.ndarray]] = {}

    if family == "dino" or obs_type == "dino":
        dino_frame_stack = infer_dino_frame_stack(agent)
        stack_suffix = f"_stack{dino_frame_stack}" if dino_frame_stack > 1 else ""
        if include_dino_raw:
            extractors[f"dino_cls_raw{stack_suffix}"] = raw_feature
        encoder = getattr(agent, "encoder", None)
        if include_dino_adapter and isinstance(encoder, nn.Module):
            if encoder.__class__.__name__ == "Identity":
                extractors[f"dino_cls_identity{stack_suffix}"] = raw_feature
            else:
                extractors[f"dino_cls_adapter{stack_suffix}"] = lambda obs, enc=encoder: module_feature(enc, obs, device)
        return extractors

    if family == "cnn" or obs_type == "pixels":
        encoder = getattr(agent, "encoder", None)
        if isinstance(encoder, nn.Module):
            extractors["cnn_encoder"] = lambda obs, enc=encoder: module_feature(enc, obs, device)
        else:
            extractors["pixels_raw"] = raw_feature
        return extractors

    if family == "dreamer":
        encoder = dreamer_encoder(agent)
        if encoder is not None:
            extractors["dreamer_encoder"] = lambda obs, enc=encoder: module_feature(enc, obs, device)
        else:
            extractors["dreamer_raw_obs"] = raw_feature
        return extractors

    extractors[f"{family}_raw_obs"] = raw_feature
    return extractors


def filter_extractors_for_job(
    job: CheckpointJob,
    extractors: tp.Dict[str, tp.Callable[[np.ndarray], np.ndarray]],
    args: argparse.Namespace,
) -> tp.Dict[str, tp.Callable[[np.ndarray], np.ndarray]]:
    if job.family != "dino" or args.dino_raw_seed is None:
        return extractors
    filtered = {}
    for name, extractor in extractors.items():
        if name.startswith("dino_cls_raw") and job.seed != args.dino_raw_seed:
            continue
        filtered[name] = extractor
    return filtered


def _as_1d(value: tp.Any) -> np.ndarray:
    return np.asarray(value, dtype=np.float32).reshape(-1)


def _named_joint_values(physics: tp.Any, field: str, exclude: tp.Set[str]) -> np.ndarray:
    values = getattr(physics.named.data, field)
    names = [name for name in values.axes.row.names if name not in exclude]
    if not names:
        return np.empty((0,), dtype=np.float32)
    return _as_1d(values[names])


def _optional_physics_call(physics: tp.Any, name: str) -> np.ndarray:
    if not hasattr(physics, name):
        return np.empty((0,), dtype=np.float32)
    return _as_1d(getattr(physics, name)())


def visible_pose_target(env: dmc.EnvWrapper, task: str) -> np.ndarray:
    """Single-frame visible geometry target, excluding velocities/global translation."""
    physics = env.physics
    if task.startswith("cheetah_"):
        return _as_1d(physics.data.qpos[1:])
    if task.startswith("walker_"):
        return np.concatenate(
            [
                _as_1d(physics.orientations()),
                _as_1d(physics.torso_height()),
                _as_1d(physics.torso_upright()),
            ]
        )
    if task.startswith("quadruped_"):
        hinge_qpos = _named_joint_values(physics, "qpos", {"root", "ball_root"})
        return np.concatenate(
            [
                hinge_qpos,
                _optional_physics_call(physics, "torso_upright"),
                _optional_physics_call(physics, "com_height"),
                _optional_physics_call(physics, "toe_positions"),
            ]
        )
    raise ValueError(f"Unsupported visible_pose target for task={task}")


def motion_target(env: dmc.EnvWrapper, task: str) -> np.ndarray:
    """Motion target for testing frame-stack access to velocities."""
    physics = env.physics
    if task.startswith("cheetah_"):
        return np.concatenate(
            [
                _as_1d(physics.velocity()),
                _optional_physics_call(physics, "speed"),
                _optional_physics_call(physics, "angmomentum"),
            ]
        )
    if task.startswith("walker_"):
        return np.concatenate(
            [
                _as_1d(physics.velocity()),
                _optional_physics_call(physics, "horizontal_velocity"),
                _optional_physics_call(physics, "angmomentum"),
            ]
        )
    if task.startswith("quadruped_"):
        hinge_qvel = _named_joint_values(physics, "qvel", {"root", "ball_root"})
        torso_velocity = _as_1d(physics.torso_velocity())
        return np.concatenate([hinge_qvel, torso_velocity, _as_1d(np.linalg.norm(torso_velocity))])
    raise ValueError(f"Unsupported motion target for task={task}")


def target_from_timestep(env: dmc.EnvWrapper, time_step: dmc.TimeStep, target: str, task: str) -> np.ndarray:
    if target == "physics":
        value = time_step.physics
    elif target == "observation":
        value = time_step.observation
    elif target == "visible_pose":
        return visible_pose_target(env, task)
    elif target == "motion":
        return motion_target(env, task)
    else:
        raise ValueError(f"Unsupported target={target}")
    return _as_1d(value)


def rollout(
    agent: tp.Any,
    env: dmc.EnvWrapper,
    extractors: tp.Dict[str, tp.Callable[[np.ndarray], np.ndarray]],
    task: str,
    episodes: int,
    max_steps: int,
    max_samples: int,
    target_names: tp.Sequence[str],
    update_meta: bool,
) -> RolloutBatch:
    feature_lists: tp.Dict[str, tp.List[np.ndarray]] = {name: [] for name in extractors}
    target_lists: tp.Dict[str, tp.List[np.ndarray]] = {name: [] for name in target_names}
    rewards: tp.List[float] = []
    total_steps = 0

    for _episode in range(episodes):
        time_step = env.reset()
        meta = init_meta(agent)
        episode_reward = 0.0
        episode_steps = 0
        while not time_step.last():
            obs = np.asarray(time_step.observation)
            for target_name in target_names:
                target_lists[target_name].append(target_from_timestep(env, time_step, target_name, task))
            for name, extractor in extractors.items():
                feature_lists[name].append(extractor(obs))

            action = act(agent, obs, meta, total_steps)
            next_time_step = env.step(action)
            episode_reward += float(next_time_step.reward)
            episode_steps += 1
            total_steps += 1
            if update_meta and hasattr(agent, "update_meta"):
                try:
                    meta = agent.update_meta(meta, total_steps, next_time_step, finetune=False, replay_loader=None)
                except TypeError:
                    meta = agent.update_meta(meta, total_steps, next_time_step)
            time_step = next_time_step

            if max_steps > 0 and episode_steps >= max_steps:
                break
            sample_count = len(next(iter(target_lists.values())))
            if max_samples > 0 and sample_count >= max_samples:
                break
        rewards.append(episode_reward)
        sample_count = len(next(iter(target_lists.values())))
        if max_samples > 0 and sample_count >= max_samples:
            break

    features = {name: np.stack(values, axis=0) for name, values in feature_lists.items()}
    targets = {name: np.stack(values, axis=0) for name, values in target_lists.items()}
    return RolloutBatch(features=features, targets=targets, rewards=rewards, steps=total_steps)


def finite_rows(x: np.ndarray, y: np.ndarray) -> tp.Tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(y).all(axis=1) & np.isfinite(x).all(axis=1)
    return x[mask], y[mask]


def train_probe(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    probe_type: str,
    seed: int,
    train_fraction: float,
    probe_epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    mlp_hidden_dim: int,
    mlp_depth: int,
    device: str,
) -> ProbeResult:
    features, targets = finite_rows(features.astype(np.float32), targets.astype(np.float32))
    if features.shape[0] < 4:
        raise ValueError(f"Need at least 4 finite samples, got {features.shape[0]}")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(features.shape[0])
    train_n = int(features.shape[0] * train_fraction)
    train_n = min(max(train_n, 2), features.shape[0] - 1)
    train_idx = perm[:train_n]
    test_idx = perm[train_n:]

    x_train = features[train_idx]
    y_train = targets[train_idx]
    x_test = features[test_idx]
    y_test = targets[test_idx]

    x_mean = x_train.mean(axis=0, keepdims=True)
    x_std = x_train.std(axis=0, keepdims=True)
    x_std[x_std < 1e-6] = 1.0
    y_mean = y_train.mean(axis=0, keepdims=True)
    y_std = y_train.std(axis=0, keepdims=True)
    y_std[y_std < 1e-6] = 1.0

    x_train = (x_train - x_mean) / x_std
    x_test = (x_test - x_mean) / x_std
    y_train_std = (y_train - y_mean) / y_std

    torch.manual_seed(seed)
    if probe_type == "linear":
        probe = nn.Linear(x_train.shape[1], y_train.shape[1]).to(device)
    elif probe_type == "mlp":
        layers: tp.List[nn.Module] = []
        in_dim = x_train.shape[1]
        for _ in range(mlp_depth):
            layers.extend([nn.Linear(in_dim, mlp_hidden_dim), nn.ReLU(inplace=True)])
            in_dim = mlp_hidden_dim
        layers.append(nn.Linear(in_dim, y_train.shape[1]))
        probe = nn.Sequential(*layers).to(device)
    else:
        raise ValueError(f"Unsupported probe_type={probe_type}")
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
    x_train_t = torch.from_numpy(x_train)
    y_train_t = torch.from_numpy(y_train_std)
    x_test_t = torch.from_numpy(x_test)

    best_metrics: tp.Optional[tp.Dict[str, tp.Any]] = None
    last_metrics: tp.Optional[tp.Dict[str, tp.Any]] = None
    epoch_metrics: tp.List[tp.Dict[str, tp.Any]] = []

    def evaluate_probe(epoch: int, train_loss: float) -> tp.Dict[str, tp.Any]:
        preds: tp.List[np.ndarray] = []
        probe.eval()
        with torch.no_grad():
            for start in range(0, x_test.shape[0], batch_size):
                xb = x_test_t[start : start + batch_size].to(device)
                pred_std = probe(xb).cpu().numpy()
                preds.append(pred_std * y_std + y_mean)
        y_pred = np.concatenate(preds, axis=0)
        return compute_probe_metrics(
            y_test=y_test,
            y_pred=y_pred,
            features=features,
            targets=targets,
            train_n=train_n,
            probe_type=probe_type,
            epoch=epoch,
            train_loss=train_loss,
        )

    for epoch in range(1, probe_epochs + 1):
        probe.train()
        epoch_perm = torch.randperm(train_n)
        losses = []
        for start in range(0, train_n, batch_size):
            idx = epoch_perm[start : start + batch_size]
            xb = x_train_t[idx].to(device)
            yb = y_train_t[idx].to(device)
            pred = probe(xb)
            loss = torch.mean((pred - yb) ** 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        train_loss = float(np.mean(losses)) if losses else float("nan")
        last_metrics = evaluate_probe(epoch, train_loss)
        epoch_metrics.append(dict(last_metrics))
        if best_metrics is None or last_metrics["mse"] < best_metrics["mse"]:
            best_metrics = dict(last_metrics)

    assert best_metrics is not None and last_metrics is not None
    out = dict(best_metrics)
    out["selected_epoch"] = best_metrics["epoch"]
    out["selected_by"] = "test_mse"
    out["final_epoch"] = last_metrics["epoch"]
    out["final_mse"] = last_metrics["mse"]
    out["final_r"] = last_metrics["r"]
    out["final_train_loss"] = last_metrics["train_loss"]
    return ProbeResult(summary=out, epochs=epoch_metrics)


def compute_probe_metrics(
    *,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    features: np.ndarray,
    targets: np.ndarray,
    train_n: int,
    probe_type: str,
    epoch: int,
    train_loss: float,
) -> tp.Dict[str, tp.Any]:

    err = y_pred - y_test
    per_dim_mse = np.mean(err ** 2, axis=0)
    mse = float(np.mean(err ** 2))
    rmse = float(math.sqrt(mse))
    target_var = float(np.mean((y_test - y_test.mean(axis=0, keepdims=True)) ** 2))
    normalized_mse = float(mse / max(target_var, 1e-12))

    y_flat = y_test.reshape(-1)
    pred_flat = y_pred.reshape(-1)
    y_centered = y_flat - y_flat.mean()
    pred_centered = pred_flat - pred_flat.mean()
    r_denom = float(np.sqrt(np.sum(y_centered ** 2) * np.sum(pred_centered ** 2)))
    pearson_r = float(np.sum(y_centered * pred_centered) / r_denom) if r_denom > 1e-12 else float("nan")

    y_dim_centered = y_test - y_test.mean(axis=0, keepdims=True)
    pred_dim_centered = y_pred - y_pred.mean(axis=0, keepdims=True)
    r_num_dim = np.sum(y_dim_centered * pred_dim_centered, axis=0)
    r_denom_dim = np.sqrt(np.sum(y_dim_centered ** 2, axis=0) * np.sum(pred_dim_centered ** 2, axis=0))
    valid_r_dim = r_denom_dim > 1e-12
    per_dim_r = np.full(y_test.shape[1], np.nan, dtype=np.float64)
    per_dim_r[valid_r_dim] = r_num_dim[valid_r_dim] / r_denom_dim[valid_r_dim]

    sse = float(np.sum(err ** 2))
    sst = float(np.sum((y_test - y_test.mean(axis=0, keepdims=True)) ** 2))
    r2 = float(1.0 - sse / max(sst, 1e-12))
    per_dim_sse = np.sum(err ** 2, axis=0)
    per_dim_sst = np.sum((y_test - y_test.mean(axis=0, keepdims=True)) ** 2, axis=0)
    per_dim_r2 = 1.0 - per_dim_sse / np.maximum(per_dim_sst, 1e-12)

    if np.isfinite(per_dim_r).any():
        r_mean_dim = float(np.nanmean(per_dim_r))
        r_median_dim = float(np.nanmedian(per_dim_r))
    else:
        r_mean_dim = float("nan")
        r_median_dim = float("nan")

    return {
        "samples": int(features.shape[0]),
        "train_samples": int(train_n),
        "test_samples": int(y_test.shape[0]),
        "feature_dim": int(features.shape[1]),
        "target_dim": int(targets.shape[1]),
        "probe_type": probe_type,
        "epoch": int(epoch),
        "train_loss": float(train_loss),
        "mse": mse,
        "mse_std_dim": float(np.std(per_dim_mse)),
        "rmse": rmse,
        "r": pearson_r,
        "r_mean_dim": r_mean_dim,
        "r_median_dim": r_median_dim,
        "target_var": target_var,
        "normalized_mse": normalized_mse,
        "r2": r2,
        "r2_mean_dim": float(np.mean(per_dim_r2)),
        "r2_median_dim": float(np.median(per_dim_r2)),
    }


def append_jsonl(path: Path, row: tp.Mapping[str, tp.Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def load_done_keys(path: Path) -> tp.Set[tp.Tuple[str, str, str, str]]:
    done: tp.Set[tp.Tuple[str, str, str, str]] = set()
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("status") == "ok":
                done.add((row["checkpoint"], row["feature"], row.get("probe_type", "linear"), row.get("target", "physics")))
    return done


def load_done_keys_many(paths: tp.Iterable[Path]) -> tp.Set[tp.Tuple[str, str, str, str]]:
    done: tp.Set[tp.Tuple[str, str, str, str]] = set()
    for path in paths:
        done.update(load_done_keys(path))
    return done


def write_summary_csv(jsonl_path: Path, csv_path: Path) -> None:
    rows = []
    if not jsonl_path.exists():
        return
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl_rows(path: Path, rows: tp.Iterable[tp.Mapping[str, tp.Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def read_jsonl_rows(path: Path) -> tp.List[tp.Dict[str, tp.Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_table_csv(jsonl_path: Path, csv_path: Path) -> None:
    rows = read_jsonl_rows(jsonl_path)
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def excel_col_name(index: int) -> str:
    name = ""
    index += 1
    while index:
        index, rem = divmod(index - 1, 26)
        name = chr(ord("A") + rem) + name
    return name


def xlsx_cell(value: tp.Any, row_idx: int, col_idx: int) -> str:
    cell_ref = f"{excel_col_name(col_idx)}{row_idx}"
    if value is None:
        return f'<c r="{cell_ref}" t="inlineStr"><is><t></t></is></c>'
    if isinstance(value, bool):
        return f'<c r="{cell_ref}" t="b"><v>{1 if value else 0}</v></c>'
    if isinstance(value, (int, float)) and np.isfinite(value):
        return f'<c r="{cell_ref}"><v>{value}</v></c>'
    text = escape(str(value), {'"': "&quot;"})
    return f'<c r="{cell_ref}" t="inlineStr"><is><t>{text}</t></is></c>'


def xlsx_sheet_xml(rows: tp.List[tp.Dict[str, tp.Any]]) -> str:
    if not rows:
        headers: tp.List[str] = []
    else:
        headers = sorted({key for row in rows for key in row})
    xml_rows = []
    header_cells = [xlsx_cell(header, 1, col_idx) for col_idx, header in enumerate(headers)]
    xml_rows.append(f'<row r="1">{"".join(header_cells)}</row>')
    for row_number, row in enumerate(rows, start=2):
        cells = [xlsx_cell(row.get(header), row_number, col_idx) for col_idx, header in enumerate(headers)]
        xml_rows.append(f'<row r="{row_number}">{"".join(cells)}</row>')
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<sheetData>'
        + "".join(xml_rows)
        + "</sheetData></worksheet>"
    )


def write_minimal_xlsx(sheets: tp.Mapping[str, tp.List[tp.Dict[str, tp.Any]]], xlsx_path: Path) -> None:
    sheet_items = [(name[:31], rows) for name, rows in sheets.items() if rows]
    if not sheet_items:
        return
    workbook_sheets = []
    workbook_rels = []
    overrides = []
    for idx, (sheet_name, _rows) in enumerate(sheet_items, start=1):
        workbook_sheets.append(f'<sheet name="{escape(sheet_name)}" sheetId="{idx}" r:id="rId{idx}"/>')
        workbook_rels.append(
            f'<Relationship Id="rId{idx}" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
            f'Target="worksheets/sheet{idx}.xml"/>'
        )
        overrides.append(
            f'<Override PartName="/xl/worksheets/sheet{idx}.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        )

    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        + "".join(overrides)
        + '<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>'
        + '<Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>'
        + "</Types>"
    )
    root_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="xl/workbook.xml"/>'
        '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" '
        'Target="docProps/core.xml"/>'
        '<Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" '
        'Target="docProps/app.xml"/>'
        "</Relationships>"
    )
    workbook = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        "<sheets>"
        + "".join(workbook_sheets)
        + "</sheets></workbook>"
    )
    workbook_rels_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        + "".join(workbook_rels)
        + "</Relationships>"
    )
    core = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
        'xmlns:dc="http://purl.org/dc/elements/1.1/">'
        "<dc:creator>linear_probe_rollouts.py</dc:creator>"
        "</cp:coreProperties>"
    )
    app = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"/>'
    )

    xlsx_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(xlsx_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types)
        zf.writestr("_rels/.rels", root_rels)
        zf.writestr("xl/workbook.xml", workbook)
        zf.writestr("xl/_rels/workbook.xml.rels", workbook_rels_xml)
        zf.writestr("docProps/core.xml", core)
        zf.writestr("docProps/app.xml", app)
        for idx, (_sheet_name, rows) in enumerate(sheet_items, start=1):
            zf.writestr(f"xl/worksheets/sheet{idx}.xml", xlsx_sheet_xml(rows))


def write_excel(summary_jsonl: Path, epochs_jsonl: Path, xlsx_path: Path) -> None:
    summary_rows = read_jsonl_rows(summary_jsonl)
    epoch_rows = read_jsonl_rows(epochs_jsonl)
    if not summary_rows and not epoch_rows:
        return
    try:
        import importlib.util
        import pandas as pd

        if importlib.util.find_spec("openpyxl") is not None:
            engine = "openpyxl"
        elif importlib.util.find_spec("xlsxwriter") is not None:
            engine = "xlsxwriter"
        else:
            raise ImportError("Neither openpyxl nor xlsxwriter is installed")
        xlsx_path.parent.mkdir(parents=True, exist_ok=True)
        with pd.ExcelWriter(xlsx_path, engine=engine) as writer:
            if summary_rows:
                pd.DataFrame(summary_rows).to_excel(writer, sheet_name="summary", index=False)
            if epoch_rows:
                pd.DataFrame(epoch_rows).to_excel(writer, sheet_name="epochs", index=False)
    except ImportError as exc:
        print(f"{exc}; writing Excel with stdlib fallback", flush=True)
        write_minimal_xlsx({"summary": summary_rows, "epochs": epoch_rows}, xlsx_path)


def run_job(
    job: CheckpointJob,
    args: argparse.Namespace,
    dino_cache: DinoBackboneCache,
    done: tp.Optional[tp.Set[tp.Tuple[str, str, str, str]]] = None,
) -> tp.List[tp.Dict[str, tp.Any]]:
    started = time.time()
    agent, ckpt_info = load_agent(job.checkpoint, args.device)
    extractors = build_extractors(
        agent,
        job.family,
        args.device,
        include_dino_raw=not args.no_dino_raw,
        include_dino_adapter=not args.no_dino_adapter,
    )
    if not extractors:
        raise RuntimeError(f"No feature extractor available for {job.family}: {job.checkpoint}")
    extractors = filter_extractors_for_job(job, extractors, args)
    if not extractors:
        return []
    probe_types = parse_probe_types(args.probe_types)
    target_names = parse_targets(args.target)
    if done is not None and all(
        (str(job.checkpoint), name, probe_type, target_name) in done
        for name in extractors
        for probe_type in probe_types
        for target_name in target_names
    ):
        return []

    env = make_env_for_agent(agent, job.task, job.family, args.seed, args.action_repeat, dino_cache)
    batch = rollout(
        agent,
        env,
        extractors,
        task=job.task,
        episodes=args.episodes,
        max_steps=args.max_steps,
        max_samples=args.max_samples,
        target_names=target_names,
        update_meta=args.update_meta,
    )

    rows: tp.List[tp.Dict[str, tp.Any]] = []
    for target_name, targets in batch.targets.items():
        for feature_name, features in batch.features.items():
            for probe_type in probe_types:
                result = train_probe(
                    features,
                    targets,
                    probe_type=probe_type,
                    seed=args.seed,
                    train_fraction=args.train_fraction,
                    probe_epochs=args.probe_epochs,
                    batch_size=args.probe_batch_size,
                    lr=args.probe_lr,
                    weight_decay=args.probe_weight_decay,
                    mlp_hidden_dim=args.mlp_hidden_dim,
                    mlp_depth=args.mlp_depth,
                    device=args.probe_device or args.device,
                )
                base_row = {
                    "status": "ok",
                    "family": job.family,
                    "feature": feature_name,
                    "task": job.task,
                    "run_name": job.run_name,
                    "seed": job.seed,
                    "checkpoint": str(job.checkpoint),
                    "checkpoint_step": job.checkpoint_step if job.checkpoint_step is not None else "latest",
                    "global_step": ckpt_info.get("global_step"),
                    "global_episode": ckpt_info.get("global_episode"),
                    "obs_type": agent_cfg_value(agent, "obs_type"),
                    "use_cls": agent_cfg_value(agent, "use_cls"),
                    "dino_use_adapter": agent_cfg_value(agent, "dino_use_adapter"),
                    "dino_frame_stack": infer_dino_frame_stack(agent),
                    "target": target_name,
                    "episodes": len(batch.rewards),
                    "rollout_steps": batch.steps,
                    "probe_trained": True,
                    "episode_reward_mean": float(np.mean(batch.rewards)) if batch.rewards else float("nan"),
                    "episode_reward_std": float(np.std(batch.rewards)) if batch.rewards else float("nan"),
                    "elapsed_sec": float(time.time() - started),
                }
                row = {**base_row, **result.summary}
                epoch_rows = [{**base_row, **epoch_metrics} for epoch_metrics in result.epochs]
                row["_epoch_rows"] = epoch_rows
                rows.append(row)
    return rows


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint-root", type=Path, default=Path("/mnt/data_7tb/fanfeng/controallable_agent_ckpt"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/dmc_physics_probe_eval"))
    parser.add_argument("--dino-run-glob", default="20260426_123938_seed*_dino_cls")
    parser.add_argument("--cnn-run-glob", default="*_cnn*")
    parser.add_argument("--dreamer-run-glob", default="*dreamer*")
    parser.add_argument("--dino-seeds", default="1992,2009,7532,8164", help="Comma list of DINO seeds to include, or all.")
    parser.add_argument("--dino-raw-seed", type=int, default=1992, help="Only emit raw DINO CLS rows for this seed. Use -1 to emit all.")
    parser.add_argument("--no-dino", action="store_true", help="Do not include DINO checkpoints.")
    parser.add_argument("--no-cnn", action="store_true", help="Do not include CNN checkpoints.")
    parser.add_argument("--include-dreamer", action="store_true", help="Try generic Dreamer checkpoints matching --dreamer-run-glob.")
    parser.add_argument("--include-latest", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--steps", default=None, help="Comma list of snapshot frame counts, plus optional latest. Example: 100000,500000,latest")
    parser.add_argument("--tasks", default=None, help="Comma list of tasks. Default: all tasks found in run names.")
    parser.add_argument("--limit", type=int, default=0, help="Limit total checkpoint jobs after discovery. 0 means no limit.")
    parser.add_argument("--max-ckpts-per-run", type=int, default=0, help="Limit checkpoints per run. 0 means no limit.")
    parser.add_argument("--num-shards", type=int, default=1, help="Split discovered jobs into this many deterministic shards.")
    parser.add_argument("--shard-index", type=int, default=0, help="Run only jobs where job_index %% num_shards == shard_index.")
    parser.add_argument("--skip-results", type=Path, action="append", default=[], help="Additional results.jsonl files used only for skip-existing.")
    parser.add_argument("--no-finalize", action="store_true", help="Do not write CSV/XLSX at the end. Useful for parallel shard workers.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--probe-device", default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--action-repeat", type=int, default=2)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=0, help="Max env steps per episode. 0 uses env episode termination.")
    parser.add_argument("--max-samples", type=int, default=4096, help="Max state/feature samples per checkpoint rollout.")
    parser.add_argument(
        "--target",
        default="physics",
        help="Comma-separated target(s): physics, observation, visible_pose, motion.",
    )
    parser.add_argument("--update-meta", action="store_true", help="Call agent.update_meta during rollout.")

    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--probe-types", default="linear,mlp")
    parser.add_argument("--probe-epochs", type=int, default=10)
    parser.add_argument("--probe-batch-size", type=int, default=1024)
    parser.add_argument("--probe-lr", type=float, default=1e-3)
    parser.add_argument("--probe-weight-decay", type=float, default=1e-4)
    parser.add_argument("--mlp-hidden-dim", type=int, default=512)
    parser.add_argument("--mlp-depth", type=int, default=2)
    parser.add_argument("--no-dino-raw", action="store_true")
    parser.add_argument("--no-dino-adapter", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    allowed_tasks = set(args.tasks.split(",")) if args.tasks else None
    allowed_steps = parse_steps(args.steps)
    dino_seeds = parse_int_set(args.dino_seeds)
    if args.dino_raw_seed < 0:
        args.dino_raw_seed = None

    jobs: tp.List[CheckpointJob] = []
    if not args.no_dino:
        jobs.extend(
            discover_jobs(
                args.checkpoint_root,
                family="dino",
                run_glob=args.dino_run_glob,
                include_latest=args.include_latest,
                allowed_tasks=allowed_tasks,
                allowed_steps=allowed_steps,
                allowed_seeds=dino_seeds,
                max_ckpts_per_run=args.max_ckpts_per_run,
            )
        )
    if not args.no_cnn:
        jobs.extend(
            discover_jobs(
                args.checkpoint_root,
                family="cnn",
                run_glob=args.cnn_run_glob,
                include_latest=args.include_latest,
                allowed_tasks=allowed_tasks,
                allowed_steps=allowed_steps,
                allowed_seeds=None,
                max_ckpts_per_run=args.max_ckpts_per_run,
            )
        )
    if args.include_dreamer:
        jobs.extend(
            discover_jobs(
                args.checkpoint_root,
                family="dreamer",
                run_glob=args.dreamer_run_glob,
                include_latest=args.include_latest,
                allowed_tasks=allowed_tasks,
                allowed_steps=allowed_steps,
                allowed_seeds=None,
                max_ckpts_per_run=args.max_ckpts_per_run,
            )
        )

    jobs.sort(key=lambda j: (j.family, j.task, j.seed or -1, j.run_name, j.checkpoint_step or 10**18, j.checkpoint.name))
    if args.limit > 0:
        jobs = jobs[: args.limit]
    total_jobs = len(jobs)
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must satisfy 0 <= shard_index < num_shards")
    if args.num_shards > 1:
        jobs = [job for idx, job in enumerate(jobs) if idx % args.num_shards == args.shard_index]

    if args.num_shards > 1:
        print(f"discovered {total_jobs} checkpoint jobs; shard {args.shard_index}/{args.num_shards} has {len(jobs)} jobs", flush=True)
    else:
        print(f"discovered {len(jobs)} checkpoint jobs", flush=True)
    for job in jobs[:20]:
        step = job.checkpoint_step if job.checkpoint_step is not None else "latest"
        seed = "" if job.seed is None else f" seed={job.seed}"
        print(f"{job.family:7s} {job.task:24s} {step!s:>8s}{seed:10s} {job.checkpoint}", flush=True)
    if len(jobs) > 20:
        print(f"... {len(jobs) - 20} more", flush=True)
    if args.dry_run:
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "results.jsonl"
    summary_path = args.output_dir / "summary.csv"
    epoch_results_path = args.output_dir / "epoch_results.jsonl"
    epoch_summary_path = args.output_dir / "epoch_results.csv"
    excel_path = args.output_dir / "probe_results.xlsx"
    done_paths = [results_path, *args.skip_results]
    done = load_done_keys_many(done_paths) if args.skip_existing else set()
    dino_cache = DinoBackboneCache(args.device)

    for index, job in enumerate(jobs, start=1):
        print(f"[{index}/{len(jobs)}] {job.family} {job.task} {job.checkpoint.name}", flush=True)
        try:
            rows = run_job(job, args, dino_cache, done if args.skip_existing else None)
            if not rows:
                print("  skip existing checkpoint/features", flush=True)
                continue
            for row in rows:
                key = (row["checkpoint"], row["feature"], row["probe_type"], row["target"])
                if key in done:
                    print(f"  skip existing {row['feature']} {row['probe_type']}", flush=True)
                    continue
                epoch_rows = row.pop("_epoch_rows", [])
                append_jsonl(results_path, row)
                write_jsonl_rows(epoch_results_path, epoch_rows)
                done.add(key)
                print(
                    f"  {row['target']} {row['feature']} {row['probe_type']}: "
                    f"trained_probe=True mse={row['mse']:.6g} r={row['r']:.4f} "
                    f"epoch={row['selected_epoch']} train={row['train_samples']} test={row['test_samples']}",
                    flush=True,
                )
        except Exception as exc:  # pylint: disable=broad-except
            error_row = {
                "status": "error",
                "family": job.family,
                "feature": "error",
                "task": job.task,
                "run_name": job.run_name,
                "checkpoint": str(job.checkpoint),
                "checkpoint_step": job.checkpoint_step if job.checkpoint_step is not None else "latest",
                "error": repr(exc),
                "traceback": traceback.format_exc(limit=8),
            }
            append_jsonl(results_path, error_row)
            print(f"  ERROR {exc!r}", flush=True)
    if not args.no_finalize:
        write_summary_csv(results_path, summary_path)
        write_table_csv(epoch_results_path, epoch_summary_path)
        write_excel(results_path, epoch_results_path, excel_path)
    print(f"wrote {results_path}", flush=True)
    if not args.no_finalize:
        print(f"wrote {summary_path}", flush=True)
    print(f"wrote {epoch_results_path}", flush=True)
    if not args.no_finalize:
        print(f"wrote {epoch_summary_path}", flush=True)
        if excel_path.exists():
            print(f"wrote {excel_path}", flush=True)


if __name__ == "__main__":
    main()
