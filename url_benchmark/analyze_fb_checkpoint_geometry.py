#!/usr/bin/env python3
"""Offline geometry diagnostics for a sequence of FB checkpoints.

The script never trains or mutates an agent.  It reconstructs one fixed
transition batch from cached ExORL DINO-CLS embeddings, fixes one isotropic
latent ``z`` per transition, and evaluates exactly the same observations,
actions, next observations, and latents at every checkpoint.

The historical Cheetah checkpoints used by this analysis did not serialize
their training replay buffer.  Consequently the fixed batch produced here is
an explicit ExORL-RND proxy, not a recovered training minibatch.  Its raw
arrays, source indices, metadata, and SHA-256 digest are saved with the output.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import hashlib
import inspect
import json
import math
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

# These must be set before importing checkpoint classes or matplotlib.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/controllable_agent_mpl_cache")

import matplotlib.pyplot as plt
import numpy as np
import omegaconf as omgcf
import torch
from torch import nn


DEFAULT_RUN_DIR = Path(
    "/mnt/data_7tb/fanfeng/controllable_agent_runs/"
    "20260817_cheetah_fb_stability_paramgrad/dino_cls3_cheetah_walk_seed1"
)
DEFAULT_CHECKPOINT_DIR = Path(
    "/mnt/data_7tb/fanfeng/controallable_agent_ckpt/cheetah_fb_stability/"
    "dino_cls3_cheetah_walk_seed1"
)
DEFAULT_EXORL_DIR = Path(
    "/mnt/data_7tb/fanfeng/exoRL_datasets/cheetah/rnd/buffer_dino_cls"
)
DEFAULT_INDEX_BANK = Path(
    "analysis_outputs/"
    "reward_label_sensitivity_dinov2_best12_exorl_rnd_anchor_20260727/"
    "cheetah_walk/banks/bank_cd511a318fc46f35eee2.npz"
)
DEFAULT_OUTPUT_DIR = Path(
    "analysis_outputs/"
    "cheetah_fb_dino_cls3_pixel_b_checkpoint_geometry_20260827"
)
DEFAULT_FRAMES = (100_000, 200_000, 500_000, 800_000, 1_000_000, 1_500_000, 2_000_000)
DEFAULT_WEAK_ALPHAS = (0.03, 0.10, 0.30)
DEFAULT_TOP_KS = (5, 10, 20)
EPISODE_RE = re.compile(r"^episode_(\d+)_(\d+)\.npz$")
SCHEMA_VERSION = 1


@dataclass
class FixedBatch:
    obs: np.ndarray
    action: np.ndarray
    next_obs: np.ndarray
    z: np.ndarray
    episode_id: np.ndarray
    step_in_episode: np.ndarray
    index_bank_row: np.ndarray
    metadata: Dict[str, Any]

    def validate(self) -> None:
        size = self.obs.shape[0]
        expected_first_dims = {
            "action": self.action,
            "next_obs": self.next_obs,
            "z": self.z,
            "episode_id": self.episode_id,
            "step_in_episode": self.step_in_episode,
            "index_bank_row": self.index_bank_row,
        }
        if self.obs.ndim != 2 or self.next_obs.shape != self.obs.shape:
            raise ValueError(
                f"Expected obs/next_obs [N,D] with equal shapes, got "
                f"{self.obs.shape} and {self.next_obs.shape}"
            )
        if self.action.ndim != 2 or self.z.ndim != 2:
            raise ValueError("Expected action and z to be rank-two arrays")
        for name, value in expected_first_dims.items():
            if value.shape[0] != size:
                raise ValueError(f"{name} has {value.shape[0]} rows, expected {size}")
        for name in ("episode_id", "step_in_episode", "index_bank_row"):
            if getattr(self, name).shape != (size,):
                raise ValueError(f"{name} must have shape {(size,)}")
        for name in ("obs", "action", "next_obs", "z"):
            if not np.isfinite(getattr(self, name)).all():
                raise ValueError(f"Fixed batch {name} contains a non-finite value")
        z_norms = np.linalg.norm(self.z.astype(np.float64), axis=1)
        target = math.sqrt(self.z.shape[1])
        if not np.allclose(z_norms, target, rtol=2e-6, atol=2e-6):
            raise ValueError("Fixed z rows are not normalized to sqrt(z_dim)")


def parse_ints(value: str) -> Tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def parse_floats(value: str) -> Tuple[float, ...]:
    return tuple(float(item.strip()) for item in value.split(",") if item.strip())


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def hash_arrays(items: Iterable[Tuple[str, np.ndarray]]) -> str:
    digest = hashlib.sha256()
    for name, value in items:
        array = np.ascontiguousarray(value)
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with tmp.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with tmp.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def fixed_batch_digest(batch: FixedBatch) -> str:
    return hash_arrays(
        (
            ("obs", batch.obs),
            ("action", batch.action),
            ("next_obs", batch.next_obs),
            ("z", batch.z),
            ("episode_id", batch.episode_id),
            ("step_in_episode", batch.step_in_episode),
            ("index_bank_row", batch.index_bank_row),
        )
    )


def save_fixed_batch(path: Path, batch: FixedBatch) -> None:
    batch.validate()
    metadata = dict(batch.metadata)
    metadata["batch_sha256"] = fixed_batch_digest(batch)
    batch.metadata = metadata
    atomic_npz(
        path,
        obs=batch.obs.astype(np.float32, copy=False),
        action=batch.action.astype(np.float32, copy=False),
        next_obs=batch.next_obs.astype(np.float32, copy=False),
        z=batch.z.astype(np.float32, copy=False),
        episode_id=batch.episode_id.astype(np.int32, copy=False),
        step_in_episode=batch.step_in_episode.astype(np.int32, copy=False),
        index_bank_row=batch.index_bank_row.astype(np.int32, copy=False),
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
    )


def load_fixed_batch(path: Path) -> FixedBatch:
    with np.load(path, allow_pickle=False) as payload:
        batch = FixedBatch(
            obs=np.asarray(payload["obs"], dtype=np.float32),
            action=np.asarray(payload["action"], dtype=np.float32),
            next_obs=np.asarray(payload["next_obs"], dtype=np.float32),
            z=np.asarray(payload["z"], dtype=np.float32),
            episode_id=np.asarray(payload["episode_id"], dtype=np.int32),
            step_in_episode=np.asarray(payload["step_in_episode"], dtype=np.int32),
            index_bank_row=np.asarray(payload["index_bank_row"], dtype=np.int32),
            metadata=json.loads(str(payload["metadata"].item())),
        )
    batch.validate()
    observed = fixed_batch_digest(batch)
    expected = batch.metadata.get("batch_sha256")
    if expected != observed:
        raise ValueError(f"Fixed-batch digest mismatch: {expected!r} != {observed!r}")
    return batch


def episode_file_map(source_dir: Path) -> Dict[int, Tuple[Path, int]]:
    output: Dict[int, Tuple[Path, int]] = {}
    for path in sorted(source_dir.glob("episode_*.npz")):
        match = EPISODE_RE.fullmatch(path.name)
        if match is None:
            continue
        episode_id, length = (int(item) for item in match.groups())
        if episode_id in output:
            raise ValueError(f"Duplicate ExORL episode ID {episode_id}")
        output[episode_id] = (path, length)
    if not output:
        raise FileNotFoundError(f"No ExORL episodes in {source_dir}")
    return output


def frame_stack_indices(state_index: int, stack: int) -> np.ndarray:
    return np.asarray(
        [max(0, state_index - stack + 1 + offset) for offset in range(stack)],
        dtype=np.int64,
    )


def build_fixed_batch(
    index_bank: Path,
    exorl_dir: Path,
    size: int,
    frame_stack: int,
    batch_seed: int,
    z_seed: int,
    expected_z_dim: int,
) -> FixedBatch:
    if size <= 0:
        raise ValueError("batch-size must be positive")
    if frame_stack <= 0:
        raise ValueError("frame-stack must be positive")
    with np.load(index_bank, allow_pickle=False) as payload:
        episode_ids_all = np.asarray(payload["episode_id"], dtype=np.int32)
        steps_all = np.asarray(payload["step_in_episode"], dtype=np.int32)
        source_metadata = json.loads(str(payload["metadata"].item()))
    if episode_ids_all.shape != steps_all.shape or episode_ids_all.ndim != 1:
        raise ValueError("Index bank episode/step arrays must be matching vectors")
    if size > episode_ids_all.size:
        raise ValueError(f"Requested {size} rows from index bank of size {episode_ids_all.size}")

    selector = np.random.RandomState(batch_seed)
    selected_rows = np.sort(
        selector.choice(episode_ids_all.size, size=size, replace=False).astype(np.int32)
    )
    episode_ids = episode_ids_all[selected_rows]
    steps = steps_all[selected_rows]
    files = episode_file_map(exorl_dir)

    obs: Optional[np.ndarray] = None
    next_obs: Optional[np.ndarray] = None
    action: Optional[np.ndarray] = None
    source_embedding_dim: Optional[int] = None
    unique_episodes = np.unique(episode_ids)
    for position, episode_id_value in enumerate(unique_episodes, start=1):
        episode_id = int(episode_id_value)
        if episode_id not in files:
            raise FileNotFoundError(f"Missing source episode {episode_id} in {exorl_dir}")
        path, filename_length = files[episode_id]
        output_rows = np.flatnonzero(episode_ids == episode_id)
        episode_steps = steps[output_rows].astype(np.int64)
        with np.load(path, allow_pickle=False) as payload:
            if "dino_emb" not in payload.files or "action" not in payload.files:
                raise KeyError(f"{path} lacks dino_emb or action")
            embeddings = np.asarray(payload["dino_emb"], dtype=np.float32)
            actions = np.asarray(payload["action"], dtype=np.float32)
            token = str(np.asarray(payload["dino_token"]).item())
            model = str(np.asarray(payload["dino_model"]).item())
        if token != "cls" or model != "facebook/dinov2-base":
            raise ValueError(f"Unexpected DINO metadata in {path}: model={model}, token={token}")
        if embeddings.ndim != 2 or embeddings.shape[0] < filename_length + 1:
            raise ValueError(f"Malformed embedding array in {path}: {embeddings.shape}")
        if actions.ndim != 2 or actions.shape[0] < filename_length + 1:
            raise ValueError(f"Malformed action array in {path}: {actions.shape}")
        if np.any(episode_steps < 0) or np.any(episode_steps >= filename_length):
            raise IndexError(f"Transition index outside episode {episode_id}")

        if obs is None:
            source_embedding_dim = int(embeddings.shape[1])
            obs_dim = source_embedding_dim * frame_stack
            obs = np.empty((size, obs_dim), dtype=np.float32)
            next_obs = np.empty_like(obs)
            action = np.empty((size, actions.shape[1]), dtype=np.float32)
        assert obs is not None and next_obs is not None and action is not None
        assert source_embedding_dim is not None
        if embeddings.shape[1] != source_embedding_dim or actions.shape[1] != action.shape[1]:
            raise ValueError(f"Feature/action dimension changed at {path}")

        for destination, step in zip(output_rows, episode_steps):
            # Replay alignment: row zero is reset.  Transition s consumes
            # obs[s], action[s+1], next_obs[s+1].  EmbedStackWrapper orders
            # frames oldest -> newest and repeats state zero at episode start.
            obs_indices = frame_stack_indices(int(step), frame_stack)
            next_indices = frame_stack_indices(int(step) + 1, frame_stack)
            obs[destination] = embeddings[obs_indices].reshape(-1)
            next_obs[destination] = embeddings[next_indices].reshape(-1)
            action[destination] = actions[int(step) + 1]
        if position % 50 == 0 or position == len(unique_episodes):
            print(
                f"fixed batch: loaded {position}/{len(unique_episodes)} source episodes",
                flush=True,
            )

    assert obs is not None and next_obs is not None and action is not None
    assert source_embedding_dim is not None
    z_rng = np.random.RandomState(z_seed)
    z = z_rng.standard_normal((size, expected_z_dim)).astype(np.float32)
    z /= np.linalg.norm(z.astype(np.float64), axis=1, keepdims=True).astype(np.float32)
    z *= np.float32(math.sqrt(expected_z_dim))

    metadata: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "batch_role": "fixed_offline_replay_proxy",
        "historical_training_replay_available": False,
        "source_policy": "exorl_rnd",
        "source_dir": str(exorl_dir.resolve()),
        "source_index_bank": str(index_bank.resolve()),
        "source_index_bank_sha256": sha256_file(index_bank),
        "source_index_bank_metadata": source_metadata,
        "source_index_bank_rows": int(episode_ids_all.size),
        "selected_episode_count": int(unique_episodes.size),
        "size": int(size),
        "batch_seed": int(batch_seed),
        "z_seed": int(z_seed),
        "z_distribution": "row-normalized N(0,I), norm=sqrt(z_dim)",
        "z_dim": int(expected_z_dim),
        "frame_stack": int(frame_stack),
        "single_frame_embedding_dim": int(source_embedding_dim),
        "stack_order": "oldest_to_newest_with_reset_frame_left_padding",
        "transition_alignment": "obs[s], action[s+1], next_obs[s+1]",
    }
    batch = FixedBatch(
        obs=obs,
        action=action,
        next_obs=next_obs,
        z=z,
        episode_id=episode_ids,
        step_in_episode=steps,
        index_bank_row=selected_rows,
        metadata=metadata,
    )
    batch.validate()
    return batch


def load_or_build_fixed_batch(args: argparse.Namespace, expected_z_dim: int) -> FixedBatch:
    path = args.output_dir / "fixed_replay_batch.npz"
    if path.exists():
        batch = load_fixed_batch(path)
        checks = {
            "size": args.batch_size,
            "batch_seed": args.batch_seed,
            "z_seed": args.z_seed,
            "z_dim": expected_z_dim,
            "frame_stack": args.frame_stack,
            "source_dir": str(args.exorl_dir.resolve()),
            "source_index_bank": str(args.index_bank.resolve()),
        }
        mismatches = {
            key: (batch.metadata.get(key), expected)
            for key, expected in checks.items()
            if batch.metadata.get(key) != expected
        }
        if mismatches:
            raise ValueError(
                f"Existing fixed batch is incompatible with requested settings: {mismatches}. "
                "Choose a new output directory."
            )
        print(f"fixed batch: reused {path} ({batch.metadata['batch_sha256'][:16]})", flush=True)
        return batch
    batch = build_fixed_batch(
        index_bank=args.index_bank,
        exorl_dir=args.exorl_dir,
        size=args.batch_size,
        frame_stack=args.frame_stack,
        batch_seed=args.batch_seed,
        z_seed=args.z_seed,
        expected_z_dim=expected_z_dim,
    )
    save_fixed_batch(path, batch)
    print(f"fixed batch: saved {path} ({batch.metadata['batch_sha256'][:16]})", flush=True)
    return batch


def prepare_agent(agent: Any, device: str) -> None:
    cfg = getattr(agent, "cfg", None)
    if cfg is not None and hasattr(cfg, "device"):
        cfg.device = device
    for name in ("backward_encoder", "backward_encoder_target"):
        if not hasattr(agent, name):
            setattr(agent, name, None)
    for value in vars(agent).values():
        if isinstance(value, nn.Module):
            value.to(device)
            value.eval()
    if hasattr(agent, "train"):
        with contextlib.suppress(TypeError):
            agent.train(False)


def load_checkpoint_agent(path: Path, device: str) -> Tuple[Any, Dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "agent" not in payload:
        raise KeyError(f"Checkpoint {path} has no serialized agent")
    agent = payload["agent"]
    prepare_agent(agent, device)
    info = {
        "global_step": int(payload.get("global_step", -1)),
        "global_episode": int(payload.get("global_episode", -1)),
    }
    return agent, info


def infer_agent_contract(checkpoint: Path) -> Dict[str, Any]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "agent" not in payload:
        raise KeyError(f"Checkpoint {checkpoint} has no serialized agent")
    agent = payload["agent"]
    cfg = agent.cfg
    contract = {
        "obs_type": str(cfg.obs_type),
        "obs_dim": int(cfg.obs_shape[0]),
        "action_dim": int(cfg.action_shape[0]),
        "z_dim": int(cfg.z_dim),
        "goal_space": None if cfg.goal_space is None else str(cfg.goal_space),
        "norm_z": bool(cfg.norm_z),
        "dino_separate_backward_adapter": bool(
            getattr(cfg, "dino_separate_backward_adapter", False)
        ),
        "training_batch_size": int(cfg.batch_size),
    }
    del agent, payload
    return contract


def encode_fixed_batch(
    agent: Any,
    batch: FixedBatch,
    device: str,
    inference_batch_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    f1_parts: List[np.ndarray] = []
    f2_parts: List[np.ndarray] = []
    b_parts: List[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, batch.obs.shape[0], inference_batch_size):
            stop = min(batch.obs.shape[0], start + inference_batch_size)
            obs_raw = torch.as_tensor(batch.obs[start:stop], device=device, dtype=torch.float32)
            next_raw = torch.as_tensor(
                batch.next_obs[start:stop], device=device, dtype=torch.float32
            )
            action = torch.as_tensor(
                batch.action[start:stop], device=device, dtype=torch.float32
            )
            z = torch.as_tensor(batch.z[start:stop], device=device, dtype=torch.float32)
            obs = agent.aug_and_encode(obs_raw)
            next_goal = agent.backward_aug_and_encode(next_raw)
            f1, f2 = agent.forward_net(obs, z, action)
            backward = agent.backward_net(next_goal)
            f1_parts.append(f1.detach().to(device="cpu", dtype=torch.float64).numpy())
            f2_parts.append(f2.detach().to(device="cpu", dtype=torch.float64).numpy())
            b_parts.append(backward.detach().to(device="cpu", dtype=torch.float64).numpy())
    return (
        np.concatenate(f1_parts, axis=0),
        np.concatenate(f2_parts, axis=0),
        np.concatenate(b_parts, axis=0),
    )


def vector_stats(values: np.ndarray, prefix: str) -> Dict[str, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_std": float(np.std(values)),
        f"{prefix}_rms": float(np.sqrt(np.mean(np.square(values)))),
        f"{prefix}_min": float(np.min(values)),
        f"{prefix}_max": float(np.max(values)),
        f"{prefix}_abs_p95": float(np.quantile(np.abs(values), 0.95)),
        f"{prefix}_abs_p99": float(np.quantile(np.abs(values), 0.99)),
        f"{prefix}_abs_max": float(np.max(np.abs(values))),
    }


def matrix_norm_stats(value: np.ndarray, prefix: str) -> Dict[str, float]:
    row_norms = np.linalg.norm(value, axis=1)
    return {
        f"{prefix}_norm_rms": float(np.sqrt(np.mean(np.square(row_norms)))),
        f"{prefix}_row_norm_min": float(np.min(row_norms)),
        f"{prefix}_row_norm_mean": float(np.mean(row_norms)),
        f"{prefix}_row_norm_std": float(np.std(row_norms)),
        f"{prefix}_row_norm_p95": float(np.quantile(row_norms, 0.95)),
        f"{prefix}_row_norm_p99": float(np.quantile(row_norms, 0.99)),
        f"{prefix}_row_norm_max": float(np.max(row_norms)),
        f"{prefix}_element_abs_max": float(np.max(np.abs(value))),
    }


def alpha_label(alpha: float) -> str:
    return f"a{int(round(alpha * 100)):03d}"


def projection_energy_ratio(value: np.ndarray, basis: np.ndarray, denominator: float) -> float:
    if denominator <= np.finfo(np.float64).tiny:
        return float("nan")
    if basis.shape[1] == 0:
        return 0.0
    coordinates = value @ basis
    return float(np.square(coordinates).sum() / denominator)


def forward_geometry(
    value: np.ndarray,
    z: np.ndarray,
    eigvecs: np.ndarray,
    eigvals: np.ndarray,
    weak_alphas: Sequence[float],
    prefix: str,
) -> Tuple[Dict[str, float], np.ndarray]:
    z64 = np.asarray(z, dtype=np.float64)
    q = np.einsum("nd,nd->n", value, z64)
    z_energy = np.einsum("nd,nd->n", z64, z64)
    parallel = (q / z_energy)[:, None] * z64
    perpendicular = value - parallel
    total_energy = float(np.square(value).sum())
    parallel_energy = float(np.square(parallel).sum())
    perpendicular_energy = float(np.square(perpendicular).sum())
    output: Dict[str, float] = {}
    output.update(matrix_norm_stats(value, prefix))
    output.update(matrix_norm_stats(parallel, f"{prefix}_parallel"))
    output.update(matrix_norm_stats(perpendicular, f"{prefix}_perp"))
    output[f"{prefix}_perp_energy_ratio"] = perpendicular_energy / total_energy
    output[f"{prefix}_parallel_energy_ratio"] = parallel_energy / total_energy
    output[f"{prefix}_pythagoras_relative_error"] = abs(
        total_energy - parallel_energy - perpendicular_energy
    ) / max(total_energy, np.finfo(np.float64).tiny)
    output[f"{prefix}_perp_z_dot_abs_max"] = float(
        np.max(np.abs(np.einsum("nd,nd->n", perpendicular, z64)))
    )
    output.update(vector_stats(q, f"{prefix}_q"))

    mean_eigenvalue = float(np.mean(eigvals))
    for alpha in weak_alphas:
        label = alpha_label(alpha)
        weak_mask = eigvals <= alpha * mean_eigenvalue
        weak_basis = eigvecs[:, weak_mask]
        output[f"b_weak_dim_{label}"] = int(weak_mask.sum())
        output[f"b_weak_threshold_{label}"] = float(alpha * mean_eigenvalue)
        output[f"b_weak_trace_fraction_{label}"] = float(
            eigvals[weak_mask].sum() / max(eigvals.sum(), np.finfo(np.float64).tiny)
        )
        output[f"{prefix}_weak_energy_ratio_{label}"] = projection_energy_ratio(
            value, weak_basis, total_energy
        )
        output[f"{prefix}_rho_dangerous_{label}"] = projection_energy_ratio(
            perpendicular, weak_basis, total_energy
        )
    main_label = alpha_label(0.10)
    output[f"{prefix}_weak_energy_ratio"] = output[
        f"{prefix}_weak_energy_ratio_{main_label}"
    ]
    output[f"{prefix}_rho_dangerous"] = output[
        f"{prefix}_rho_dangerous_{main_label}"
    ]
    for bottom_k in (5, 10):
        use_k = min(bottom_k, eigvecs.shape[1])
        bottom_basis = eigvecs[:, :use_k]
        output[f"{prefix}_bottom{bottom_k}_energy_ratio"] = projection_energy_ratio(
            value, bottom_basis, total_energy
        )
        output[f"{prefix}_bottom{bottom_k}_rho_dangerous"] = projection_energy_ratio(
            perpendicular, bottom_basis, total_energy
        )
    return output, perpendicular


def covariance_geometry(backward: np.ndarray) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    size, dimension = backward.shape
    covariance = backward.T @ backward / size
    covariance = 0.5 * (covariance + covariance.T)
    eigvals, eigvecs = np.linalg.eigh(covariance)
    largest = float(eigvals[-1])
    tolerance = max(size, dimension) * np.finfo(np.float64).eps * max(largest, 1.0)
    numerical_rank = int(np.sum(eigvals > tolerance))
    condition = float("inf") if eigvals[0] <= tolerance else largest / float(eigvals[0])
    effective_condition = largest / max(float(eigvals[0]), tolerance)
    trace = float(eigvals.sum())
    participation = trace * trace / max(float(np.square(eigvals).sum()), np.finfo(np.float64).tiny)
    output: Dict[str, float] = {
        "b_cov_trace": trace,
        "b_eigenvalue_min": float(eigvals[0]),
        "b_eigenvalue_max": largest,
        "b_eigenvalue_mean": float(eigvals.mean()),
        "b_condition_number": condition,
        "b_effective_condition_number": effective_condition,
        "b_log10_effective_condition": float(math.log10(effective_condition)),
        "b_numerical_rank": numerical_rank,
        "b_rank_tolerance": tolerance,
        "b_participation_ratio": participation,
        "b_participation_ratio_fraction": participation / dimension,
        "b_mean_vector_norm": float(np.linalg.norm(backward.mean(axis=0))),
    }
    output.update(matrix_norm_stats(backward, "b"))
    for index, eigenvalue in enumerate(eigvals):
        output[f"b_eigenvalue_{index:02d}"] = float(eigenvalue)
    return output, eigvals, eigvecs


def online_m_stats(forward: np.ndarray, backward: np.ndarray, prefix: str) -> Dict[str, float]:
    # N=1024 by default, matching training.  Computing the full N x N matrix
    # gives exact online-M moments and quantiles rather than a pair subsample.
    online_m = forward @ backward.T
    output = vector_stats(online_m, f"{prefix}_m_online")
    diagonal = np.diag(online_m)
    output.update(vector_stats(diagonal, f"{prefix}_m_online_diag"))
    return output


def principal_angles(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    singular_values = np.linalg.svd(left.T @ right, compute_uv=False)
    return np.arccos(np.clip(singular_values, 0.0, 1.0))


def relative_top_eigengap(eigvals: np.ndarray, top_k: int) -> float:
    dimension = eigvals.size
    if top_k >= dimension:
        return float("nan")
    included = float(eigvals[dimension - top_k])
    excluded = float(eigvals[dimension - top_k - 1])
    return (included - excluded) / max(abs(included), np.finfo(np.float64).tiny)


def angle_stats(angles: np.ndarray, prefix: str) -> Dict[str, Any]:
    degrees = np.rad2deg(angles)
    return {
        f"{prefix}_angles_deg_json": json.dumps([round(float(item), 8) for item in degrees]),
        f"{prefix}_mean_deg": float(np.mean(degrees)),
        f"{prefix}_rms_deg": float(np.sqrt(np.mean(np.square(degrees)))),
        f"{prefix}_max_deg": float(np.max(degrees)),
        f"{prefix}_chordal_rms": float(np.sqrt(np.mean(np.square(np.sin(angles))))),
    }


def drift_metrics(
    previous_backward: Optional[np.ndarray],
    previous_eigvecs: Optional[np.ndarray],
    current_backward: np.ndarray,
    current_eigvals: np.ndarray,
    current_eigvecs: np.ndarray,
    top_ks: Sequence[int],
) -> Dict[str, Any]:
    output: Dict[str, Any] = {}
    dimension = current_eigvals.size
    for top_k in top_ks:
        if not 0 < top_k < dimension:
            raise ValueError(f"top-k must lie in [1,{dimension - 1}], got {top_k}")
        output[f"b_top{top_k}_relative_eigengap"] = relative_top_eigengap(
            current_eigvals, top_k
        )
    if previous_backward is None or previous_eigvecs is None:
        output["b_procrustes_relative_residual"] = float("nan")
        for top_k in top_ks:
            for aligned in (False, True):
                label = f"b_top{top_k}_drift" + ("_aligned" if aligned else "")
                output.update(
                    {
                        f"{label}_angles_deg_json": "",
                        f"{label}_mean_deg": float("nan"),
                        f"{label}_rms_deg": float("nan"),
                        f"{label}_max_deg": float("nan"),
                        f"{label}_chordal_rms": float("nan"),
                    }
                )
        return output

    cross = current_backward.T @ previous_backward
    left, _, right_t = np.linalg.svd(cross, full_matrices=False)
    rotation = left @ right_t
    aligned_backward = current_backward @ rotation
    output["b_procrustes_relative_residual"] = float(
        np.linalg.norm(aligned_backward - previous_backward)
        / max(np.linalg.norm(previous_backward), np.finfo(np.float64).tiny)
    )
    for top_k in top_ks:
        previous_top = previous_eigvecs[:, -top_k:]
        current_top = current_eigvecs[:, -top_k:]
        raw_angles = principal_angles(previous_top, current_top)
        # B_current R is expressed in the previous checkpoint's coordinates,
        # so its eigenvectors are R^T U_current.
        aligned_angles = principal_angles(previous_top, rotation.T @ current_top)
        output.update(angle_stats(raw_angles, f"b_top{top_k}_drift"))
        output.update(angle_stats(aligned_angles, f"b_top{top_k}_drift_aligned"))
    return output


def exact_evaluation(eval_rows: Sequence[Mapping[str, float]], frame: int) -> Mapping[str, float]:
    matches = [row for row in eval_rows if row["frame"] == frame]
    values = {(row["reward"], row["reward_std"]) for row in matches}
    if len(values) == 1 and all(np.isfinite(value) for value in next(iter(values))):
        return matches[0]
    # Missing or conflicting resumed evaluations are never nearest-matched.
    return {"frame": float("nan"), "reward": float("nan"), "reward_std": float("nan")}


def read_evaluations(path: Path) -> List[Dict[str, float]]:
    output: List[Dict[str, float]] = []
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            output.append(
                {
                    "frame": float(row["frame"]),
                    "reward": float(row["episode_reward"]),
                    "reward_std": float(row["episode_reward#std"]),
                }
            )
    if not output:
        raise ValueError(f"No evaluation rows in {path}")
    return output


def analyze_checkpoint(
    checkpoint: Path,
    expected_frame: int,
    agent_contract: Mapping[str, Any],
    batch: FixedBatch,
    eval_rows: Sequence[Mapping[str, float]],
    device: str,
    inference_batch_size: int,
    weak_alphas: Sequence[float],
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    agent, info = load_checkpoint_agent(checkpoint, device)
    cfg = agent.cfg
    observed_contract = {
        "obs_type": str(cfg.obs_type),
        "obs_dim": int(cfg.obs_shape[0]),
        "action_dim": int(cfg.action_shape[0]),
        "z_dim": int(cfg.z_dim),
        "goal_space": None if cfg.goal_space is None else str(cfg.goal_space),
        "norm_z": bool(cfg.norm_z),
        "dino_separate_backward_adapter": bool(
            getattr(cfg, "dino_separate_backward_adapter", False)
        ),
        "training_batch_size": int(cfg.batch_size),
    }
    if observed_contract != dict(agent_contract):
        raise ValueError(f"Agent contract changed at {checkpoint}: {observed_contract}")
    if batch.obs.shape[1] != observed_contract["obs_dim"]:
        raise ValueError("Fixed observation dimension does not match checkpoint")
    if batch.action.shape[1] != observed_contract["action_dim"]:
        raise ValueError("Fixed action dimension does not match checkpoint")
    if batch.z.shape[1] != observed_contract["z_dim"]:
        raise ValueError("Fixed z dimension does not match checkpoint")

    f1, f2, backward = encode_fixed_batch(
        agent, batch, device=device, inference_batch_size=inference_batch_size
    )
    if not all(np.isfinite(item).all() for item in (f1, f2, backward)):
        raise FloatingPointError(f"Non-finite network output at {checkpoint}")
    covariance_output, eigvals, eigvecs = covariance_geometry(backward)
    f1_output, _ = forward_geometry(
        f1, batch.z, eigvecs, eigvals, weak_alphas, "f1"
    )
    f2_output, _ = forward_geometry(
        f2, batch.z, eigvecs, eigvals, weak_alphas, "f2"
    )
    q1 = np.einsum("nd,nd->n", f1, batch.z.astype(np.float64))
    q2 = np.einsum("nd,nd->n", f2, batch.z.astype(np.float64))
    evaluation = exact_evaluation(eval_rows, expected_frame)
    stat = checkpoint.stat()
    row: Dict[str, Any] = {
        "run_id": "dino_cls3_cheetah_walk_seed1-paramgrad",
        "checkpoint_frame": expected_frame,
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_size_bytes": stat.st_size,
        "checkpoint_mtime_ns": stat.st_mtime_ns,
        "global_step": info["global_step"],
        "global_episode": info["global_episode"],
        "derived_frame_from_global_step": 2 * info["global_step"],
        "eval_frame": evaluation["frame"],
        "eval_frame_delta": evaluation["frame"] - expected_frame,
        "eval_reward": evaluation["reward"],
        "eval_reward_std": evaluation["reward_std"],
        "fixed_batch_size": batch.obs.shape[0],
        "fixed_batch_sha256": batch.metadata["batch_sha256"],
        "fixed_batch_source": "exorl_rnd_proxy",
        "fixed_z_seed": batch.metadata["z_seed"],
    }
    if row["derived_frame_from_global_step"] != expected_frame:
        raise ValueError(
            f"Frame mismatch for {checkpoint}: global_step*2="
            f"{row['derived_frame_from_global_step']} != {expected_frame}"
        )
    row.update(covariance_output)
    row.update(f1_output)
    row.update(f2_output)
    row.update(online_m_stats(f1, backward, "f1"))
    row.update(online_m_stats(f2, backward, "f2"))
    row.update(vector_stats(np.minimum(q1, q2), "q_twin_min"))
    row.update(vector_stats(np.abs(q1 - q2), "q_twin_abs_gap"))

    # Both branches must agree on the B-only weak-subspace columns.
    for alpha in weak_alphas:
        label = alpha_label(alpha)
        for suffix in ("dim", "threshold", "trace_fraction"):
            key = f"b_weak_{suffix}_{label}"
            if key in f2_output and not np.isclose(row[key], f2_output[key]):
                raise AssertionError(f"Inconsistent shared-B diagnostic {key}")

    del agent
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return row, backward, eigvals, eigvecs


def atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty CSV")
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with tmp.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def plot_geometry(rows: Sequence[Mapping[str, Any]], output: Path) -> None:
    frames = np.asarray([row["checkpoint_frame"] for row in rows], dtype=float) / 1e6
    rewards = np.asarray([row["eval_reward"] for row in rows], dtype=float)
    reward_std = np.asarray([row["eval_reward_std"] for row in rows], dtype=float)
    figure, axes = plt.subplots(3, 2, figsize=(12, 12))

    ax = axes[0, 0]
    ax.errorbar(frames, rewards, yerr=reward_std, marker="o", capsize=3)
    ax.set(title="Exact-frame evaluation reward", ylabel="episode reward")

    ax = axes[0, 1]
    ax.semilogy(
        frames,
        [row["b_effective_condition_number"] for row in rows],
        marker="o",
        color="tab:red",
        label="condition number",
    )
    ax.set(title="B second-moment conditioning", ylabel="condition number")
    twin = ax.twinx()
    twin.plot(
        frames,
        [row["b_participation_ratio"] for row in rows],
        marker="s",
        color="tab:blue",
        label="participation ratio",
    )
    twin.set_ylabel("participation ratio", color="tab:blue")

    ax = axes[1, 0]
    dimension = sum(key.startswith("b_eigenvalue_") and key[-2:].isdigit() for key in rows[0])
    for row in rows:
        spectrum = np.asarray([row[f"b_eigenvalue_{index:02d}"] for index in range(dimension)])
        ax.semilogy(
            np.arange(1, dimension + 1),
            np.maximum(spectrum[::-1], np.finfo(float).tiny),
            label=f"{row['checkpoint_frame'] / 1e6:g}M",
        )
    ax.set(title="C_B eigenvalue spectra", xlabel="rank (largest first)", ylabel="eigenvalue")
    ax.legend(ncol=2, fontsize=8)

    ax = axes[1, 1]
    for branch, color in (("f1", "tab:blue"), ("f2", "tab:orange")):
        ax.plot(
            frames,
            [row[f"{branch}_perp_energy_ratio"] for row in rows],
            marker="o",
            color=color,
            label=f"{branch.upper()} perp / total",
        )
    ax.set(title="F energy orthogonal to z", ylabel="energy ratio", ylim=(0, 1.02))
    ax.legend(fontsize=8)

    ax = axes[2, 0]
    for branch, color in (("f1", "tab:blue"), ("f2", "tab:orange")):
        ax.plot(
            frames,
            [row[f"{branch}_rho_dangerous"] for row in rows],
            marker="o",
            color=color,
            label=f"{branch.upper()} dangerous",
        )
        ax.plot(
            frames,
            [row[f"{branch}_weak_energy_ratio"] for row in rows],
            linestyle="--",
            color=color,
            alpha=0.65,
            label=f"{branch.upper()} all weak",
        )
    ax.set(title="Energy in B-weak directions (alpha=0.10)", ylabel="fraction of total F energy")
    ax.legend(fontsize=8, ncol=2)

    ax = axes[2, 1]
    ax.plot(
        frames,
        [row["b_top10_drift_mean_deg"] for row in rows],
        marker="o",
        label="raw mean angle",
    )
    ax.plot(
        frames,
        [row["b_top10_drift_max_deg"] for row in rows],
        marker="s",
        label="raw max angle",
    )
    ax.plot(
        frames,
        [row["b_top10_drift_aligned_mean_deg"] for row in rows],
        marker="^",
        label="Procrustes-aligned mean",
    )
    ax.set(title="Consecutive B top-10 principal angles", ylabel="degrees")
    ax.legend(fontsize=8)

    for row_index, column_index in np.ndindex(axes.shape):
        ax = axes[row_index, column_index]
        if (row_index, column_index) != (1, 0):
            ax.set_xlabel("training frames (millions)")
        ax.grid(alpha=0.25)
    figure.suptitle("FB checkpoint geometry: fixed ExORL-RND batch", fontsize=14)
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_dynamics(rows: Sequence[Mapping[str, Any]], output: Path) -> None:
    frames = np.asarray([row["checkpoint_frame"] for row in rows], dtype=float) / 1e6
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))

    ax = axes[0, 0]
    for branch, color in (("f1", "tab:blue"), ("f2", "tab:orange")):
        ax.plot(frames, [row[f"{branch}_norm_rms"] for row in rows], marker="o", color=color, label=f"{branch.upper()} total")
        ax.plot(frames, [row[f"{branch}_perp_norm_rms"] for row in rows], linestyle="--", color=color, label=f"{branch.upper()} perp")
    ax.set(title="F magnitude", ylabel="batch Frobenius RMS")
    ax.legend(fontsize=8, ncol=2)

    ax = axes[0, 1]
    for branch, color in (("f1", "tab:blue"), ("f2", "tab:orange")):
        ax.semilogy(frames, [row[f"{branch}_m_online_rms"] for row in rows], marker="o", color=color, label=f"{branch.upper()} RMS")
        ax.semilogy(frames, [row[f"{branch}_m_online_abs_p99"] for row in rows], linestyle="--", color=color, label=f"{branch.upper()} |M| p99")
    ax.set(title="Online M = F B^T magnitude", ylabel="magnitude")
    ax.legend(fontsize=8, ncol=2)

    ax = axes[1, 0]
    for branch, color in (("f1", "tab:blue"), ("f2", "tab:orange")):
        ax.plot(frames, [row[f"{branch}_q_rms"] for row in rows], marker="o", color=color, label=f"{branch.upper()} Q RMS")
        ax.plot(frames, [row[f"{branch}_q_std"] for row in rows], linestyle="--", color=color, label=f"{branch.upper()} Q std")
    ax.set(title="Q = F dot z", ylabel="magnitude")
    ax.legend(fontsize=8, ncol=2)

    ax = axes[1, 1]
    target_b_norm = math.sqrt(50.0)
    b_mean = np.asarray([row["b_row_norm_mean"] for row in rows], dtype=float)
    b_p99 = np.asarray([row["b_row_norm_p99"] for row in rows], dtype=float)
    b_max = np.asarray([row["b_row_norm_max"] for row in rows], dtype=float)
    ax.plot(frames, b_mean, marker="o", label="B row norm mean")
    ax.plot(frames, b_p99, marker="s", label="B row norm p99")
    ax.plot(frames, b_max, marker="^", label="B row norm max")
    ax.axhline(target_b_norm, color="black", linewidth=1, alpha=0.4, label="sqrt(z_dim)")
    ax.set(
        title="B magnitude (architecturally normalized)",
        ylabel="row L2 norm",
        ylim=(target_b_norm - 0.001, target_b_norm + 0.001),
    )
    ax.ticklabel_format(style="plain", axis="y", useOffset=False)
    ax.legend(fontsize=8)

    for ax in axes.flat:
        ax.set_xlabel("training frames (millions)")
        ax.grid(alpha=0.25)
    figure.suptitle("FB checkpoint magnitudes: fixed ExORL-RND batch", fontsize=14)
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks


def correlation(left: Sequence[float], right: Sequence[float], spearman: bool = False) -> float:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if x.size < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    if spearman:
        x, y = rankdata(x), rankdata(y)
    return float(np.corrcoef(x, y)[0, 1])


def fold_change(final: float, initial: float) -> float:
    return final / initial if initial != 0 else float("inf")


def diagnosis_text(
    rows: Sequence[Mapping[str, Any]],
    eval_rows: Sequence[Mapping[str, float]],
    batch: FixedBatch,
) -> str:
    selected_peak_index = int(np.argmax([row["eval_reward"] for row in rows]))
    selected_peak = rows[selected_peak_index]
    final = rows[-1]
    full_peak = max(eval_rows, key=lambda row: row["reward"])
    reward_drop = 1.0 - float(final["eval_reward"]) / float(full_peak["reward"])

    cond_fold = fold_change(
        float(final["b_effective_condition_number"]),
        float(selected_peak["b_effective_condition_number"]),
    )
    pr_fold = fold_change(
        float(final["b_participation_ratio"]),
        float(selected_peak["b_participation_ratio"]),
    )
    minimum_eigenvalue_fold = fold_change(
        float(final["b_eigenvalue_min"]), float(selected_peak["b_eigenvalue_min"])
    )
    danger_peak = np.mean(
        [selected_peak["f1_rho_dangerous"], selected_peak["f2_rho_dangerous"]]
    )
    danger_final = np.mean([final["f1_rho_dangerous"], final["f2_rho_dangerous"]])
    danger_fold = fold_change(float(danger_final), float(danger_peak))
    danger_sensitivity_folds: Dict[str, float] = {}
    for label in ("a003", "a010", "a030"):
        start_value = np.mean(
            [
                selected_peak[f"f1_rho_dangerous_{label}"],
                selected_peak[f"f2_rho_dangerous_{label}"],
            ]
        )
        end_value = np.mean(
            [final[f"f1_rho_dangerous_{label}"], final[f"f2_rho_dangerous_{label}"]]
        )
        danger_sensitivity_folds[label] = fold_change(float(end_value), float(start_value))
    bottom10_peak = np.mean(
        [
            selected_peak["f1_bottom10_rho_dangerous"],
            selected_peak["f2_bottom10_rho_dangerous"],
        ]
    )
    bottom10_final = np.mean(
        [final["f1_bottom10_rho_dangerous"], final["f2_bottom10_rho_dangerous"]]
    )
    weak_peak = np.mean(
        [selected_peak["f1_weak_energy_ratio"], selected_peak["f2_weak_energy_ratio"]]
    )
    weak_final = np.mean([final["f1_weak_energy_ratio"], final["f2_weak_energy_ratio"]])
    m_peak = np.mean(
        [selected_peak["f1_m_online_rms"], selected_peak["f2_m_online_rms"]]
    )
    m_post = [
        np.mean([row["f1_m_online_rms"], row["f2_m_online_rms"]])
        for row in rows[selected_peak_index:]
    ]
    q_peak = np.mean([selected_peak["f1_q_rms"], selected_peak["f2_q_rms"]])
    q_post = [
        np.mean([row["f1_q_rms"], row["f2_q_rms"]])
        for row in rows[selected_peak_index:]
    ]
    q_mean_peak = np.mean([selected_peak["f1_q_mean"], selected_peak["f2_q_mean"]])
    q_mean_final = np.mean([final["f1_q_mean"], final["f2_q_mean"]])
    q_std_peak = np.mean([selected_peak["f1_q_std"], selected_peak["f2_q_std"]])
    q_std_final = np.mean([final["f1_q_std"], final["f2_q_std"]])
    q_gap_peak = float(selected_peak["q_twin_abs_gap_mean"])
    q_gap_final = float(final["q_twin_abs_gap_mean"])
    late_raw_angles = [
        float(row["b_top10_drift_mean_deg"])
        for row in rows[selected_peak_index + 1 :]
        if np.isfinite(float(row["b_top10_drift_mean_deg"]))
    ]
    late_aligned_angles = [
        float(row["b_top10_drift_aligned_mean_deg"])
        for row in rows[selected_peak_index + 1 :]
        if np.isfinite(float(row["b_top10_drift_aligned_mean_deg"]))
    ]

    b_support = cond_fold > 1.5 and minimum_eigenvalue_fold < 0.67
    f_support = danger_fold > 1.5 and danger_final > danger_peak + 0.005
    drift_support = bool(late_aligned_angles) and max(late_aligned_angles) > 10.0
    dynamics_explosion = max(m_post) > 1.5 * m_peak or max(q_post) > 1.5 * q_peak
    dynamics_distribution_shift = (
        q_std_final > 1.10 * q_std_peak and q_gap_final > 1.25 * q_gap_peak
    )
    if b_support and f_support and drift_support and dynamics_distribution_shift:
        verdict = (
            "证据强烈支持前三个几何环节，并显示 Q 分布退化；但 M/Q 没有幅值爆炸，"
            "因此只部分支持完整链条，且不能据此推断因果。"
        )
    elif sum((b_support, f_support, drift_support)) >= 2:
        verdict = "证据支持部分几何环节，但没有形成完整的 M/Q→reward 证据链。"
    else:
        verdict = "这组离线证据不支持完整链条，reward collapse 与大多数中间指标并未按预测共同变化。"

    rewards = [float(row["eval_reward"]) for row in rows]
    log_condition = [float(row["b_log10_effective_condition"]) for row in rows]
    dangerous = [
        np.mean([row["f1_rho_dangerous"], row["f2_rho_dangerous"]]) for row in rows
    ]
    m_rms = [
        np.mean([row["f1_m_online_rms"], row["f2_m_online_rms"]]) for row in rows
    ]
    q_means = [np.mean([row["f1_q_mean"], row["f2_q_mean"]]) for row in rows]
    q_stds = [np.mean([row["f1_q_std"], row["f2_q_std"]]) for row in rows]
    q_gaps = [float(row["q_twin_abs_gap_mean"]) for row in rows]
    decline_rows = rows[selected_peak_index:]
    decline_rewards = rewards[selected_peak_index:]

    def metric_values(source_rows: Sequence[Mapping[str, Any]], name: str) -> List[float]:
        if name == "log_condition":
            return [float(row["b_log10_effective_condition"]) for row in source_rows]
        if name == "dangerous":
            return [
                np.mean([row["f1_rho_dangerous"], row["f2_rho_dangerous"]])
                for row in source_rows
            ]
        if name == "m_rms":
            return [
                np.mean([row["f1_m_online_rms"], row["f2_m_online_rms"]])
                for row in source_rows
            ]
        if name == "q_mean":
            return [np.mean([row["f1_q_mean"], row["f2_q_mean"]]) for row in source_rows]
        if name == "q_std":
            return [np.mean([row["f1_q_std"], row["f2_q_std"]]) for row in source_rows]
        if name == "q_gap":
            return [float(row["q_twin_abs_gap_mean"]) for row in source_rows]
        raise KeyError(name)

    def mark(value: bool) -> str:
        return "支持" if value else "不支持/较弱"

    lines = [
        "# Short diagnosis",
        "",
        f"**结论：{verdict}**",
        "",
        "固定批次说明：训练 checkpoint 没有保存 replay buffer，因此无法恢复历史训练 minibatch。"
        f"本分析对所有 checkpoint 使用同一个 ExORL-RND Cheetah 代理批次（N={batch.obs.shape[0]}，"
        f"SHA-256 `{batch.metadata['batch_sha256']}`），并固定 raw rows、actions、三帧堆叠和 z。",
        "",
        "| 链条环节 | 观测 | 判定 |",
        "|---|---|---|",
        (
            "| B ill-conditioning | 从选定 checkpoint reward 峰值 "
            f"{selected_peak['checkpoint_frame']/1e6:g}M 到 "
            f"{final['checkpoint_frame']/1e6:g}M，condition ×{cond_fold:.3g}，"
            f"lambda_min ×{minimum_eigenvalue_fold:.3g}，PR ×{pr_fold:.3g}。"
            f"PR 上升说明顶部谱较分散，但不抵消底部近零尾恶化。 | {mark(b_support)} |"
        ),
        (
            "| F near-nullspace energy | twin 平均 weak-F 比例 "
            f"{weak_peak:.4f}→{weak_final:.4f}；rho_dangerous "
            f"{danger_peak:.4f}→{danger_final:.4f}（×{danger_fold:.3g}）。"
            f"alpha=.03/.10/.30 的 fold 为 ×{danger_sensitivity_folds['a003']:.2f}/"
            f"×{danger_sensitivity_folds['a010']:.2f}/×{danger_sensitivity_folds['a030']:.2f}，"
            f"fixed bottom-10 为 ×{bottom10_final/bottom10_peak:.2f}。 | {mark(f_support)} |"
        ),
        (
            "| B-subspace drift | collapse 区间 top-10 mean principal angle 的最大值："
            f"raw {max(late_raw_angles) if late_raw_angles else float('nan'):.2f}°，"
            f"Procrustes-aligned {max(late_aligned_angles) if late_aligned_angles else float('nan'):.2f}°。 "
            "| " + mark(drift_support) + " |"
        ),
        (
            "| M/Q instability | M RMS 末值/峰值 checkpoint 为 "
            f"×{m_post[-1]/m_peak:.3g}（收缩而非爆炸）；Q mean ×{q_mean_final/q_mean_peak:.3g}，"
            f"Q std ×{q_std_final/q_std_peak:.3g}，twin |Q1-Q2| mean ×{q_gap_final/q_gap_peak:.3g}。 "
            "| "
            + (
                "支持幅值爆炸" if dynamics_explosion else
                "部分支持：Q 分布退化，无幅值爆炸" if dynamics_distribution_shift else
                "不支持/较弱"
            )
            + " |"
        ),
        (
            "| Reward collapse | 完整 eval 曲线峰值 "
            f"{full_peak['reward']:.2f}@{full_peak['frame']/1e6:g}M；"
            f"{final['checkpoint_frame']/1e6:g}M 为 {final['eval_reward']:.2f}，"
            f"下降 {100*reward_drop:.1f}%。 | 明确存在 |"
        ),
        "",
        "## Descriptive correlations across the seven checkpoints",
        "",
        f"这些相关性只有 n={len(rows)}，仅用于描述共变方向。",
        "",
        "| Pair | Pearson all | Spearman all | Pearson decline | Spearman decline |",
        "|---|---:|---:|---:|---:|",
        f"| reward vs log10(cond(C_B)) | {correlation(rewards, log_condition):.3f} | {correlation(rewards, log_condition, True):.3f} | {correlation(decline_rewards, metric_values(decline_rows, 'log_condition')):.3f} | {correlation(decline_rewards, metric_values(decline_rows, 'log_condition'), True):.3f} |",
        f"| reward vs twin-mean rho_dangerous | {correlation(rewards, dangerous):.3f} | {correlation(rewards, dangerous, True):.3f} | {correlation(decline_rewards, metric_values(decline_rows, 'dangerous')):.3f} | {correlation(decline_rewards, metric_values(decline_rows, 'dangerous'), True):.3f} |",
        f"| reward vs twin-mean M RMS | {correlation(rewards, m_rms):.3f} | {correlation(rewards, m_rms, True):.3f} | {correlation(decline_rewards, metric_values(decline_rows, 'm_rms')):.3f} | {correlation(decline_rewards, metric_values(decline_rows, 'm_rms'), True):.3f} |",
        f"| reward vs twin-mean Q mean | {correlation(rewards, q_means):.3f} | {correlation(rewards, q_means, True):.3f} | {correlation(decline_rewards, metric_values(decline_rows, 'q_mean')):.3f} | {correlation(decline_rewards, metric_values(decline_rows, 'q_mean'), True):.3f} |",
        f"| reward vs twin-mean Q std | {correlation(rewards, q_stds):.3f} | {correlation(rewards, q_stds, True):.3f} | {correlation(decline_rewards, metric_values(decline_rows, 'q_std')):.3f} | {correlation(decline_rewards, metric_values(decline_rows, 'q_std'), True):.3f} |",
        f"| reward vs twin |Q1-Q2| mean | {correlation(rewards, q_gaps):.3f} | {correlation(rewards, q_gaps, True):.3f} | {correlation(decline_rewards, metric_values(decline_rows, 'q_gap')):.3f} | {correlation(decline_rewards, metric_values(decline_rows, 'q_gap'), True):.3f} |",
        "",
        "## Interpretation limits",
        "",
        "- `C_B = E[B B^T]` 是未中心化 second moment；weak 主口径为 eigenvalue <= 0.10 × mean eigenvalue，CSV 同时给出 0.03/0.30 和 fixed-bottom-k 敏感性。",
        "- `rho_dangerous` 表示同时对当前 z/Q 正交、又落入 B 弱方向的 F 能量；它上升本身不等价于 Q 必然变大。",
        "- raw principal angle 可能包含 FB 的整体 latent-rotation gauge；Procrustes-aligned angle 更接近结构性 drift，并需结合 eigengap 解读。",
        "- 固定 ExORL-RND batch 提供严格的 checkpoint 间可比性，但它不是各时点的训练分布，不能单独建立 `B→F→drift→M/Q→reward` 的因果顺序。",
        "",
    ]
    return "\n".join(lines)


CHEETAH_TASKS = (
    "cheetah_walk", "cheetah_run", "cheetah_walk_backward", "cheetah_run_backward",
)


def finite_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def discover_flare_pairs(manifest: Path, output_dir: Path) -> List[Dict[str, Any]]:
    """Use the approved run identities, but re-discover files on every invocation."""
    source = json.loads(manifest.read_text())
    pairs = []
    evidence = []
    for entry in source["pairs"]:
        task = entry["task"]
        if task not in CHEETAH_TASKS:
            raise ValueError(f"Unsupported task: {task}")
        pair: Dict[str, Any] = {"task": task}
        for variant, original in (("flare", entry["flare"]), ("baseline", entry["baseline_original"])):
            run_dir = Path(original["run_dir"])
            config_path = run_dir / ".hydra/config.yaml"
            cfg = omgcf.OmegaConf.load(config_path)
            if (str(cfg.task) != task or str(cfg.obs_type) != "dino" or not cfg.use_cls
                    or int(cfg.dino_frame_stack) != 3 or cfg.goal_space is not None
                    or cfg.custom_reward is not None):
                raise ValueError(f"Not an ordinary pixel-B CLS3 run: {config_path}")
            if bool(getattr(cfg.agent, "dino_flare_b", False)) != (variant == "flare"):
                raise ValueError(f"Variant flag mismatch: {config_path}")
            checkpoint_dir = Path(cfg.checkpoint_root) / run_dir.name
            checkpoints = {}
            for path in checkpoint_dir.glob("snapshot_*.pt"):
                match = re.fullmatch(r"snapshot_(\d+)\.pt", path.name)
                if match and path.is_file():
                    checkpoints[int(match.group(1))] = path
            if not checkpoints:
                raise FileNotFoundError(f"No numbered checkpoints: {checkpoint_dir}")
            segments = [original] if variant == "flare" else entry["baseline_segments"]
            eval_rows = []
            eval_sources = []
            for segment in segments:
                path = Path(segment["run_dir"]) / "eval.csv"
                if path.is_file():
                    rows = read_evaluations(path)
                    eval_rows.extend(rows)
                    eval_sources.append({"path": str(path), "sha256": sha256_file(path), "rows": rows})
            pair[variant] = {
                "cfg": cfg, "run_dir": run_dir, "checkpoints": checkpoints,
                "config_sha256": sha256_file(config_path), "eval_rows": eval_rows,
            }
            evidence.append({
                "variant": variant, "task": task, "run_dir": str(run_dir),
                "config_path": str(config_path), "config_sha256": sha256_file(config_path),
                "config_yaml": config_path.read_text(), "eval_sources": eval_sources,
                "checkpoint_every_frames": int(cfg.checkpoint_every),
                "eval_every_frames": int(cfg.eval_every_frames),
                "snapshot_at": list(cfg.snapshot_at),
                "numbered_checkpoints": {str(frame): str(path) for frame, path in sorted(checkpoints.items())},
            })
        pair["frames"] = sorted(set(pair["flare"]["checkpoints"]) & set(pair["baseline"]["checkpoints"]))
        pairs.append(pair)
    if {pair["task"] for pair in pairs} != set(CHEETAH_TASKS):
        raise ValueError("Manifest must contain each of the four Cheetah tasks exactly once")
    if len(pairs) != len(CHEETAH_TASKS):
        raise ValueError("Duplicate task in manifest")
    atomic_json(output_dir / "discovery.json", finite_json({
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "runs": evidence,
        "intersections": {pair["task"]: pair["frames"] for pair in pairs},
    }))
    return pairs


def select_noise_checkpoints(pair: Mapping[str, Any]) -> Dict[str, Any]:
    frames = sorted(pair["flare"]["checkpoints"])
    if len(frames) < 3:
        raise ValueError(f"Need three existing checkpoints for {pair['task']}: {frames}")
    evaluations = sorted(pair["flare"]["eval_rows"], key=lambda row: row["frame"])
    exact = {frame: exact_evaluation(evaluations, frame) for frame in frames}
    valid = [frame for frame in frames if np.isfinite(exact[frame]["reward"])]
    if len(valid) < 3:
        raise ValueError(f"Need three checkpoints with exact reward AND std: {pair['task']}")
    local_peaks = [row for before, row, after in zip(evaluations, evaluations[1:], evaluations[2:])
                   if row["reward"] > before["reward"] and row["reward"] > after["reward"]]
    saved_peaks = [row for row in local_peaks if int(row["frame"]) in valid[:-1]]
    if pair["task"].endswith("_backward"):
        peak, adjacent = valid[0], valid[1]
        selected = [valid[0], valid[1], valid[-1]]
        roles = ["early_level", "adjacent_level", "latest_level"]
    else:
        peak = (int(max(saved_peaks, key=lambda row: row["reward"])["frame"])
                if saved_peaks else max(valid[:-1], key=lambda frame: exact[frame]["reward"]))
        adjacent = valid[valid.index(peak) + 1]
        early = next(frame for frame in valid if frame not in (peak, adjacent))
        selected = sorted([early, peak, adjacent])
        roles = ["peak_probe" if frame == peak else "adjacent_checkpoint" if frame == adjacent
                 else "early" if frame < peak else "additional_checkpoint"
                 for frame in selected]
    drops = [(a, b) for a, b in zip(evaluations, evaluations[1:]) if a["reward"] > b["reward"]]
    largest = max(drops, key=lambda ab: ab[0]["reward"] - ab[1]["reward"]) if drops else None
    return finite_json({
        "task": pair["task"], "frames": selected, "roles": dict(zip(map(str, selected), roles)),
        "peak_frame": peak, "adjacent_checkpoint_frame": adjacent,
        "logged_peak_reward": exact[peak]["reward"], "logged_peak_std": exact[peak]["reward_std"],
        "logged_adjacent_reward": exact[adjacent]["reward"], "logged_adjacent_std": exact[adjacent]["reward_std"],
        "probe_interval_frames": adjacent - peak,
        "peak_is_true_local_eval_peak": peak in {int(row["frame"]) for row in local_peaks},
        "immediate_eval_after_peak": next((row for row in evaluations if row["frame"] > peak), None),
        "probe_is_immediate_eval_pair": adjacent - peak == int(pair["flare"]["cfg"].eval_every_frames),
        "largest_logged_drop": None if largest is None else {
            "from": largest[0], "to": largest[1],
            "magnitude": largest[0]["reward"] - largest[1]["reward"],
            "both_weights_available": all(int(row["frame"]) in frames for row in largest),
        },
        "limitation": "Unsaved 10k eval peaks cannot be replaced by nearest saved weights; coarse probes only.",
    })


def load_or_build_reward_labels(batch: FixedBatch, exorl_dir: Path, path: Path) -> Tuple[np.ndarray, str]:
    """Four fixed task labels on the SAME next-state physics rows; no DINO."""
    label_sources = {str(p.relative_to(Path(__file__).parent)): sha256_file(p) for p in (
        Path(__file__).with_name("goals.py"), Path(__file__).parent / "custom_dmc_tasks/cheetah.py",
    )}
    if path.exists():
        with np.load(path, allow_pickle=False) as payload:
            rewards = np.asarray(payload["rewards"])
            metadata = json.loads(str(payload["metadata"].item()))
        digest = hash_arrays((("rewards", rewards),))
        if (metadata["sha256"] != digest or metadata["batch_sha256"] != batch.metadata["batch_sha256"]
                or metadata.get("label_sources") != label_sources
                or metadata["tasks"] != list(CHEETAH_TASKS)
                or rewards.shape != (batch.obs.shape[0], len(CHEETAH_TASKS))
                or not np.isfinite(rewards).all()):
            raise ValueError("Fixed reward labels digest/contract mismatch")
        return rewards, digest
    from url_benchmark.goals import DmcReward

    rewards = np.empty((batch.obs.shape[0], len(CHEETAH_TASKS)), dtype=np.float32)
    files = episode_file_map(exorl_dir)
    rewarders = [DmcReward(task) for task in CHEETAH_TASKS]
    try:
        for episode in np.unique(batch.episode_id):
            rows = np.flatnonzero(batch.episode_id == episode)
            with np.load(files[int(episode)][0], allow_pickle=False) as payload:
                physics = payload["physics"][batch.step_in_episode[rows] + 1]
            for destination, state in zip(rows, physics):
                rewards[destination] = [rewarder.from_physics(state) for rewarder in rewarders]
    finally:
        for rewarder in rewarders:
            rewarder._env.close()
    if not np.isfinite(rewards).all():
        raise FloatingPointError("Non-finite reward bank labels")
    digest = hash_arrays((("rewards", rewards),))
    atomic_npz(path, rewards=rewards, metadata=np.asarray(json.dumps({
        "sha256": digest, "batch_sha256": batch.metadata["batch_sha256"], "tasks": list(CHEETAH_TASKS),
        "label_sources": label_sources,
        "definition": "DmcReward(task).from_physics(episode.physics[step_in_episode+1]); single next-state reward, not action-repeat sum",
    })))
    return rewards, digest


def infer_fixed_reward_vectors(agent: Any, batch: FixedBatch, rewards: np.ndarray,
                               device: str, chunk_size: int) -> np.ndarray:
    """One deterministic fixed-bank reward vector per task and checkpoint."""
    parts = []
    with torch.inference_mode():
        for start in range(0, len(batch.next_obs), chunk_size):
            obs = torch.as_tensor(batch.next_obs[start:start + chunk_size], device=device)
            backward = agent.backward_net(agent.backward_aug_and_encode(obs))
            parts.append(backward.cpu())
        backward = torch.cat(parts)
        reward_t = torch.as_tensor(rewards, dtype=backward.dtype)
        vectors = reward_t.T @ backward / len(backward)
        norms = torch.linalg.vector_norm(vectors, dim=1, keepdim=True)
        if not torch.isfinite(vectors).all() or torch.any(norms <= 1e-12):
            raise ValueError("Zero/non-finite reward vector: direction is undefined")
        if bool(agent.cfg.norm_z):
            vectors = math.sqrt(int(agent.cfg.z_dim)) * vectors / norms
        return vectors.numpy().astype(np.float32)


def load_frozen_rollout_dino(model_name: str, device: str,
                             cuda_runtime_discovery: bool = False) -> Tuple[Any, Any, Dict[str, Any]]:
    """Load the frontend once, only for fresh visual rollouts, from local cache."""
    import importlib.metadata
    from transformers import AutoImageProcessor, AutoModel

    if cuda_runtime_discovery:
        # PyTorch 2.4's NVML discovery can wait on other physical GPUs. Use the
        # actual CUDART visible count for this process only; no driver changes.
        if not device.startswith("cuda") or not hasattr(torch.cuda, "_cached_device_count"):
            raise ValueError("Runtime discovery requires the supported CUDA/PyTorch runtime")
        count = torch._C._cuda_getDeviceCount()
        if count < 1:
            raise RuntimeError("CUDA runtime sees no available devices")
        torch.cuda._cached_device_count = count
        print(f"CUDA discovery: CUDART reports {count} visible device(s); NVML skipped", flush=True)
    processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True, local_files_only=True)
    model = AutoModel.from_pretrained(model_name, local_files_only=True).eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    provenance = {
        "model_name": model_name, "revision": getattr(model.config, "_commit_hash", None),
        "processor_class": type(processor).__name__, "processor_config": processor.to_dict(),
        "versions": {package: importlib.metadata.version(package)
                     for package in ("torch", "transformers", "dm-control", "mujoco", "numpy")},
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "mujoco_gl": os.environ.get("MUJOCO_GL"),
        "mujoco_egl_device_id": os.environ.get("MUJOCO_EGL_DEVICE_ID"), "device": device,
        "local_files_only": True,
        "cuda_device_discovery": "CUDART visible-count cache" if cuda_runtime_discovery else "PyTorch default",
    }
    return model, processor, provenance


def seeded_fixed_z_rollout(agent: Any, cfg: Any, seed: int, z: np.ndarray, frame: int,
                           device: str, model: Any, processor: Any) -> Dict[str, Any]:
    from url_benchmark import dmc

    # The existing dmc.make DINO wrapper defaults to cuda; use isolated GPU visibility.
    if device not in ("cuda", "cuda:0"):
        raise ValueError("DINO rollouts require logical cuda:0 with CUDA_VISIBLE_DEVICES set")
    fixed_z = np.asarray(z, dtype=np.float32).copy()
    if fixed_z.shape != (int(agent.cfg.z_dim),) or not np.isfinite(fixed_z).all():
        raise ValueError("Invalid fixed task vector")
    z_sha = hashlib.sha256(fixed_z.tobytes()).hexdigest()
    meta = {"z": fixed_z}
    env = None
    try:
        env = dmc.make(
            str(cfg.task), obs_type="dino", frame_stack=int(cfg.frame_stack),
            action_repeat=int(cfg.action_repeat), seed=int(seed), goal_space=None,
            append_goal_to_observation=False, use_cls=True, dino_model=model,
            dino_processor=processor, render_shape=tuple(cfg.render_shape),
            dino_frame_stack=int(cfg.dino_frame_stack),
        )
        reward, steps = 0.0, 0
        with torch.inference_mode():
            time_step = env.reset()
            while not time_step.last():
                action = agent.act(time_step.observation, meta, frame // int(cfg.action_repeat), eval_mode=True)
                time_step = env.step(action)
                reward += float(time_step.reward)
                steps += 1
        if hashlib.sha256(meta["z"].tobytes()).hexdigest() != z_sha or not np.isfinite(reward):
            raise ValueError("Fixed z changed or rollout reward is non-finite")
        return {"task": str(cfg.task), "checkpoint_frame": frame, "environment_seed": seed,
                "episode_reward": reward, "episode_steps": steps,
                "episode_frames": steps * int(cfg.action_repeat), "z_sha256": z_sha}
    finally:
        if env is not None:
            env.close()


def flare_provenance(args: argparse.Namespace, pairs: Sequence[Mapping[str, Any]],
                     batch: FixedBatch, labels_sha: str, dino: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "command_line": [sys.executable, *sys.argv], "analyzer_sha256": sha256_file(Path(__file__)),
        "config_hashes": {pair["task"]: {v: pair[v]["config_sha256"] for v in ("flare", "baseline")} for pair in pairs},
        "fixed_batch_sha256": batch.metadata["batch_sha256"], "fixed_reward_labels_sha256": labels_sha,
        "environment_seeds": list(args.eval_seeds), "reward_bank_size": len(batch.obs),
        "dinov2": dict(dino), "inferred_z_bootstrap": False,
        "lr_matched": False, "phase0_manifest_sha256": sha256_file(args.pair_manifest),
    }


def evaluation_code_identity() -> Dict[str, Any]:
    return {
        "evaluator_sha256": hashlib.sha256("\n".join(inspect.getsource(func) for func in (
            prepare_agent, load_checkpoint_agent, infer_fixed_reward_vectors,
            seeded_fixed_z_rollout, load_frozen_rollout_dino,
        )).encode()).hexdigest(),
        "source_sha256": {name: sha256_file(Path(__file__).parent / name) for name in (
            "agent/fb_ddpg.py", "dmc.py", "goals.py", "utils.py", "custom_dmc_tasks/cheetah.py",
        )},
    }


def evaluate_fixed_checkpoint(args: argparse.Namespace, pair: Mapping[str, Any], variant: str,
                              frame: int, batch: FixedBatch, labels: np.ndarray, labels_sha: str,
                              model: Any, processor: Any, dino: Mapping[str, Any]) -> Dict[str, Any]:
    run = pair[variant]
    path = run["checkpoints"][frame]
    output = args.output_dir / "evaluations" / pair["task"] / variant / str(frame)
    output.mkdir(parents=True, exist_ok=True)
    contract = {
        "checkpoint_path": str(path), "checkpoint_sha256": sha256_file(path),
        "config_sha256": run["config_sha256"], "batch_sha256": batch.metadata["batch_sha256"],
        "labels_sha256": labels_sha, "dinov2": dict(dino), "schema_version": 1,
        **evaluation_code_identity(),
        "inference": "float32 mean(reward*B), normalize to sqrt(z_dim), fixed bank; no bootstrap",
    }
    contract_path = output / "contract.json"
    if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
        raise ValueError(f"Evaluation provenance mismatch: {output}")
    agent = None
    z_path = output / "fixed_z.npz"
    try:
        if z_path.exists():
            if not contract_path.exists():
                raise ValueError(f"Missing provenance for fixed z: {output}")
            with np.load(z_path, allow_pickle=False) as saved:
                vectors = np.asarray(saved["vectors"])
                expected = str(saved["sha256"].item())
            if hash_arrays((("vectors", vectors),)) != expected:
                raise ValueError(f"Fixed z SHA mismatch: {z_path}")
        else:
            atomic_json(contract_path, contract)
            agent, info = load_checkpoint_agent(path, args.device)
            if info["global_step"] * int(run["cfg"].action_repeat) != frame:
                raise ValueError(f"Checkpoint frame mismatch: {path}")
            vectors = infer_fixed_reward_vectors(agent, batch, labels, args.device, args.inference_batch_size)
            atomic_npz(z_path, vectors=vectors, sha256=np.asarray(hash_arrays((("vectors", vectors),))))
        z = vectors[CHEETAH_TASKS.index(pair["task"])]
        z_sha = hashlib.sha256(z.tobytes()).hexdigest()
        seeds = []
        for seed in args.eval_seeds:
            seed_path = output / f"seed_{seed}.json"
            if seed_path.exists():
                result = json.loads(seed_path.read_text())
                if (result["z_sha256"] != z_sha or result["environment_seed"] != seed
                        or result["checkpoint_frame"] != frame or result["task"] != pair["task"]):
                    raise ValueError(f"Saved seed result mismatch: {seed_path}")
            else:
                if agent is None:
                    agent, info = load_checkpoint_agent(path, args.device)
                    if info["global_step"] * int(run["cfg"].action_repeat) != frame:
                        raise ValueError(f"Checkpoint frame mismatch: {path}")
                result = seeded_fixed_z_rollout(agent, run["cfg"], seed, z, frame,
                                                args.device, model, processor)
                atomic_json(seed_path, result)
            seeds.append(result)
            print(f"eval {variant} {pair['task']} {frame} seed={seed}: {result['episode_reward']:.6g}", flush=True)
        returns = [row["episode_reward"] for row in seeds]
        summary = {
            "variant": variant, "task": pair["task"], "checkpoint_frame": frame,
            "reward_mean": float(np.mean(returns)), "reward_std": float(np.std(returns, ddof=1)),
            "reward_std_definition": "sample std across independently seeded episodes, ddof=1; not SEM",
            "environment_seeds": list(args.eval_seeds), "z_sha256": z_sha, "checkpoint_sha256": contract["checkpoint_sha256"],
            "seed_results": seeds,
        }
        atomic_json(output / "summary.json", summary)
        return summary
    finally:
        del agent
        gc.collect()
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()


def run_flare_noise(args: argparse.Namespace) -> None:
    if len(set(args.eval_seeds)) != len(args.eval_seeds) or len(args.eval_seeds) < 5:
        raise ValueError("At least five distinct environment seeds are required")
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir = args.output_dir / "workers" / args.noise_task if args.noise_task else args.output_dir
    pairs = discover_flare_pairs(args.pair_manifest, metadata_dir)
    selections = {pair["task"]: select_noise_checkpoints(pair) for pair in pairs}
    atomic_json(metadata_dir / "phase1_selection.json", selections)
    bank_args = argparse.Namespace(**vars(args))
    bank_args.output_dir = args.output_dir / "reward_bank"
    bank_args.batch_size = args.reward_bank_size
    batch = load_or_build_fixed_batch(bank_args, expected_z_dim=50)
    # Initialize CUDA before constructing/closing DMC physics for new labels.
    model, processor, dino = load_frozen_rollout_dino(
        str(pairs[0]["flare"]["cfg"].dino_model_name), args.device, args.cuda_runtime_discovery)
    labels, labels_sha = load_or_build_reward_labels(batch, args.exorl_dir, bank_args.output_dir / "reward_labels.npz")
    provenance = flare_provenance(args, pairs, batch, labels_sha, dino)
    atomic_json(metadata_dir / "phase1_provenance.json", provenance)
    rows = []
    for pair in pairs:
        if args.noise_task and pair["task"] != args.noise_task:
            continue
        for frame in selections[pair["task"]]["frames"]:
            rows.append(evaluate_fixed_checkpoint(args, pair, "flare", frame, batch, labels, labels_sha,
                                                  model, processor, dino))
            atomic_json(metadata_dir / "phase1_progress.json", {"completed": rows, "provenance": provenance})
    if args.noise_task:
        print(f"Phase 1 worker complete: {args.noise_task}; global noise gate awaits all four tasks", flush=True)
        return
    write_noise_summary(args, pairs, selections, rows, provenance)


def summarize_saved_noise(args: argparse.Namespace) -> None:
    """Validate all four completed workers and report without loading DINO/GPU."""
    args.output_dir = args.output_dir.resolve()
    rows, selections, workers = [], {}, {}
    common: Dict[str, Any] = {}
    identity = evaluation_code_identity()
    batch = load_fixed_batch(args.output_dir / "reward_bank/fixed_replay_batch.npz")
    if not (args.output_dir / "reward_bank/reward_labels.npz").is_file():
        raise FileNotFoundError("Cannot summarize without the original persisted reward labels")
    _, labels_sha = load_or_build_reward_labels(
        batch, args.exorl_dir, args.output_dir / "reward_bank/reward_labels.npz")
    for task in CHEETAH_TASKS:
        worker = args.output_dir / "workers" / task
        progress = json.loads((worker / "phase1_progress.json").read_text())
        provenance = progress["provenance"]
        workers[task] = provenance
        conditions = {key: provenance[key] for key in (
            "git_sha", "analyzer_sha256", "config_hashes", "fixed_batch_sha256",
            "fixed_reward_labels_sha256", "environment_seeds", "dinov2", "reward_bank_size",
        )}
        if common and conditions != common:
            raise ValueError(f"Worker measurement conditions differ: {task}")
        common = conditions
        if (common["fixed_batch_sha256"] != batch.metadata["batch_sha256"]
                or common["fixed_reward_labels_sha256"] != labels_sha
                or common["environment_seeds"] != list(args.eval_seeds)):
            raise ValueError("Worker fixed-bank/seed provenance mismatch")
        selections[task] = json.loads((worker / "phase1_selection.json").read_text())[task]
        completed = progress["completed"]
        if sorted(row["checkpoint_frame"] for row in completed) != sorted(selections[task]["frames"]):
            raise ValueError(f"Worker incomplete: {task}")
        for row in completed:
            frame = row["checkpoint_frame"]
            directory = args.output_dir / "evaluations" / task / "flare" / str(frame)
            contract = json.loads((directory / "contract.json").read_text())
            if (any(contract[key] != value for key, value in identity.items())
                    or contract["checkpoint_sha256"] != sha256_file(Path(contract["checkpoint_path"]))
                    or contract["batch_sha256"] != common["fixed_batch_sha256"]
                    or contract["labels_sha256"] != labels_sha
                    or contract["dinov2"] != common["dinov2"]):
                raise ValueError(f"Saved checkpoint evaluation contract changed: {directory}")
            with np.load(directory / "fixed_z.npz", allow_pickle=False) as saved:
                vectors = saved["vectors"]
                if hash_arrays((("vectors", vectors),)) != str(saved["sha256"].item()):
                    raise ValueError(f"Fixed z digest mismatch: {directory}")
            z_sha = hashlib.sha256(vectors[CHEETAH_TASKS.index(task)].tobytes()).hexdigest()
            seed_rows = [json.loads((directory / f"seed_{seed}.json").read_text()) for seed in args.eval_seeds]
            if (seed_rows != row["seed_results"] or row["task"] != task or row["z_sha256"] != z_sha
                    or any(r["z_sha256"] != z_sha or r["environment_seed"] != seed
                           or r["checkpoint_frame"] != frame or r["task"] != task
                           for seed, r in zip(args.eval_seeds, seed_rows))):
                raise ValueError(f"Seed results changed: {directory}")
            returns = [r["episode_reward"] for r in seed_rows]
            if (not np.isfinite(returns).all() or len(set(args.eval_seeds)) < 5
                    or not np.isclose(row["reward_mean"], np.mean(returns), rtol=0, atol=1e-12)
                    or not np.isclose(row["reward_std"], np.std(returns, ddof=1), rtol=0, atol=1e-12)):
                raise ValueError(f"Invalid reward statistics: {directory}")
            rows.append(row)
    provenance = {**common, "worker_provenance": workers,
                  "aggregation_command_line": [sys.executable, *sys.argv],
                  "aggregation_analyzer_sha256": sha256_file(Path(__file__)),
                  "lr_matched": False, "inferred_z_bootstrap": False}
    atomic_json(args.output_dir / "phase1_selection.json", selections)
    write_noise_summary(args, [{"task": task} for task in CHEETAH_TASKS], selections, rows, provenance)


def write_noise_summary(args: argparse.Namespace, pairs: Sequence[Mapping[str, Any]],
                        selections: Mapping[str, Any], rows: Sequence[Mapping[str, Any]],
                        provenance: Mapping[str, Any]) -> None:
    decisions = []
    for pair in pairs:
        task = pair["task"]
        selected = selections[task]
        task_rows = {row["checkpoint_frame"]: row for row in rows if row["task"] == task}
        peak, adjacent = (task_rows[selected[key]] for key in ("peak_frame", "adjacent_checkpoint_frame"))
        drop = peak["reward_mean"] - adjacent["reward_mean"]
        noise = max(peak["reward_std"], adjacent["reward_std"])
        decisions.append({
            "task": task, "task_group": "direction_recovery" if task.endswith("_backward") else "spike_diagnosis",
            "peak_frame": peak["checkpoint_frame"], "adjacent_frame": adjacent["checkpoint_frame"],
            "fixed_bank_reward_drop": drop, "max_endpoint_seed_std": noise,
            "drop_exceeds_seed_std": bool(drop > noise), "drop_exceeds_two_seed_std": bool(drop > 2 * noise),
            "fine_spike_resolvable": selected["probe_is_immediate_eval_pair"],
            "logged_drop": selected["logged_peak_reward"] - selected["logged_adjacent_reward"],
            "limitation": selected["limitation"],
        })
    proceed = any(row["drop_exceeds_seed_std"] for row in decisions if row["task_group"] == "spike_diagnosis")
    summary = {"proceed_to_geometry": proceed, "decisions": decisions, "checkpoints": rows, "provenance": provenance,
               "note": "Gate concerns measured coarse checkpoint drops only. Unsaved fine spikes remain unresolved; inferred-z sampling noise has not been tested."}
    atomic_json(args.output_dir / "phase1_summary.json", summary)
    atomic_csv(args.output_dir / "checkpoint_diagnostics.csv", [
        {key: row[key] for key in ("variant", "task", "checkpoint_frame", "reward_mean", "reward_std", "z_sha256")}
        for row in rows
    ])
    text = ["# Phase 1: fixed-bank z, independent environment seeds", "",
            "| Task | Frame | Mean | Seed std (ddof=1) |", "|---|---:|---:|---:|"]
    text += [f"| {r['task']} | {r['checkpoint_frame']} | {r['reward_mean']:.6g} | {r['reward_std']:.6g} |" for r in rows]
    text += ["", "Fixed z is reused across every seed at a checkpoint. No inferred-z bootstrap.", ""]
    for row in decisions:
        if row["task_group"] == "direction_recovery":
            levels = sorted((item for item in rows if item["task"] == row["task"]), key=lambda item: item["checkpoint_frame"])
            first, last = levels[0], levels[-1]
            text.append(f"- {row['task']} (direction recovery): early {first['reward_mean']:.6g} ± {first['reward_std']:.6g}, "
                        f"latest {last['reward_mean']:.6g} ± {last['reward_std']:.6g}. Assess absolute reward/off-zero performance; "
                        "this is not a spike classification and baseline is not re-evaluated in Phase 1.")
            continue
        text.append(f"- {row['task']}: fixed-bank drop {row['fixed_bank_reward_drop']:.6g}, "
                    f"max endpoint seed std {row['max_endpoint_seed_std']:.6g}; exceeds one std: "
                    f"{row['drop_exceeds_seed_std']}. Fine spike resolvable: {row['fine_spike_resolvable']}.")
    text += ["", f"Proceed to coarse geometry: {proceed}.", summary["note"], "",
             f"Rollout renderer: {provenance['dinov2']['mujoco_gl']}. Pixel equivalence to training EGL is unverified."]
    (args.output_dir / "PHASE1_REPORT.md").write_text("\n".join(text) + "\n")
    print(f"Phase 1 complete; proceed_to_geometry={proceed}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("legacy", "flare-noise"), default="legacy")
    parser.add_argument("--pair-manifest", type=Path, default=Path(
        "analysis_outputs/cheetah_flare_b_phase0_20260905/run_discovery.json"))
    parser.add_argument("--eval-seeds", type=parse_ints, default=(101, 202, 303, 404, 505))
    parser.add_argument("--reward-bank-size", type=int, default=20480)
    parser.add_argument("--noise-task", choices=CHEETAH_TASKS,
                        help="Evaluate one task in an isolated worker; never decide the global geometry gate")
    parser.add_argument("--summarize-noise-only", action="store_true",
                        help="Verify and summarize all saved workers without model loading or rollouts")
    parser.add_argument("--cuda-runtime-discovery", action="store_true",
                        help="Use CUDART discovery when host NVML initialization stalls (PyTorch 2.4)")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument(
        "--frames", type=parse_ints, default=DEFAULT_FRAMES, help="comma-separated frame counts"
    )
    parser.add_argument("--exorl-dir", type=Path, default=DEFAULT_EXORL_DIR)
    parser.add_argument("--index-bank", type=Path, default=DEFAULT_INDEX_BANK)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--batch-seed", type=int, default=20260827)
    parser.add_argument("--z-seed", type=int, default=20260828)
    parser.add_argument("--frame-stack", type=int, default=3)
    parser.add_argument("--inference-batch-size", type=int, default=256)
    parser.add_argument(
        "--weak-alphas", type=parse_floats, default=DEFAULT_WEAK_ALPHAS,
        help="comma-separated weak eigenvalue thresholds relative to mean eigenvalue",
    )
    parser.add_argument(
        "--top-ks", type=parse_ints, default=DEFAULT_TOP_KS,
        help="comma-separated fixed top-eigenspace ranks for principal angles",
    )
    parser.add_argument(
        "--device", default=None
    )
    return parser


def validate_inputs(args: argparse.Namespace) -> Tuple[List[Path], omgcf.DictConfig]:
    config_path = args.run_dir / ".hydra" / "config.yaml"
    eval_path = args.run_dir / "eval.csv"
    for path in (config_path, eval_path, args.index_bank, args.exorl_dir):
        if not path.exists():
            raise FileNotFoundError(path)
    cfg = omgcf.OmegaConf.load(config_path)
    if str(cfg.task) != "cheetah_walk" or str(cfg.obs_type) != "dino" or not bool(cfg.use_cls):
        raise ValueError("This analysis expects the DINO-CLS cheetah_walk run")
    if int(cfg.dino_frame_stack) != args.frame_stack:
        raise ValueError(
            f"Run has dino_frame_stack={cfg.dino_frame_stack}, requested {args.frame_stack}"
        )
    if bool(cfg.save_replay_buffer_in_checkpoint):
        raise ValueError("Expected the historical run with replay saving disabled")
    if 0.10 not in args.weak_alphas:
        raise ValueError("weak-alphas must include the main alpha=0.10")
    if 10 not in args.top_ks:
        raise ValueError("top-ks must include the main top-k=10")
    checkpoints = [args.checkpoint_dir / f"snapshot_{frame}.pt" for frame in args.frames]
    missing = [path for path in checkpoints if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing checkpoints: {missing}")
    return checkpoints, cfg


def main() -> None:
    args = build_parser().parse_args()
    if args.device is None:
        args.device = "cpu" if args.summarize_noise_only else "cuda:0" if torch.cuda.is_available() else "cpu"
    if args.mode == "flare-noise":
        if args.summarize_noise_only:
            summarize_saved_noise(args)
        else:
            run_flare_noise(args)
        return
    args.run_dir = args.run_dir.resolve()
    args.checkpoint_dir = args.checkpoint_dir.resolve()
    args.exorl_dir = args.exorl_dir.resolve()
    args.index_bank = args.index_bank.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.inference_batch_size <= 0:
        raise ValueError("inference-batch-size must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is unavailable")
    checkpoints, run_cfg = validate_inputs(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    contract = infer_agent_contract(checkpoints[0])
    expected_obs_dim = args.frame_stack * 768
    if contract != {
        **contract,
        "obs_type": "dino",
        "obs_dim": expected_obs_dim,
        "goal_space": None,
        "norm_z": True,
        "dino_separate_backward_adapter": False,
    }:
        raise ValueError(f"Unexpected target checkpoint contract: {contract}")
    if args.batch_size != contract["training_batch_size"]:
        print(
            f"warning: fixed batch N={args.batch_size}, training N={contract['training_batch_size']}",
            flush=True,
        )
    batch = load_or_build_fixed_batch(args, expected_z_dim=contract["z_dim"])
    if batch.obs.shape[1] != contract["obs_dim"]:
        raise ValueError("Constructed fixed batch does not match checkpoint obs_dim")

    eval_rows = read_evaluations(args.run_dir / "eval.csv")
    rows: List[Dict[str, Any]] = []
    previous_backward: Optional[np.ndarray] = None
    previous_eigvecs: Optional[np.ndarray] = None
    for index, (frame, checkpoint) in enumerate(zip(args.frames, checkpoints), start=1):
        print(f"checkpoint {index}/{len(checkpoints)}: {frame:,} frames", flush=True)
        row, backward, eigvals, eigvecs = analyze_checkpoint(
            checkpoint=checkpoint,
            expected_frame=frame,
            agent_contract=contract,
            batch=batch,
            eval_rows=eval_rows,
            device=args.device,
            inference_batch_size=args.inference_batch_size,
            weak_alphas=args.weak_alphas,
        )
        row.update(
            drift_metrics(
                previous_backward=previous_backward,
                previous_eigvecs=previous_eigvecs,
                current_backward=backward,
                current_eigvals=eigvals,
                current_eigvecs=eigvecs,
                top_ks=args.top_ks,
            )
        )
        rows.append(row)
        previous_backward = backward
        previous_eigvecs = eigvecs
        print(
            f"  reward={row['eval_reward']:.2f}, cond={row['b_effective_condition_number']:.3g}, "
            f"rho=(F1 {row['f1_rho_dangerous']:.4f}, F2 {row['f2_rho_dangerous']:.4f})",
            flush=True,
        )

    csv_path = args.output_dir / "checkpoint_diagnostics.csv"
    atomic_csv(csv_path, rows)
    plot_geometry(rows, args.output_dir / "geometry_over_training.png")
    plot_dynamics(rows, args.output_dir / "dynamics_over_training.png")
    diagnosis = diagnosis_text(rows, eval_rows, batch)
    diagnosis_path = args.output_dir / "DIAGNOSIS.md"
    diagnosis_path.write_text(diagnosis, encoding="utf-8")

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": "dino_cls3_cheetah_walk_seed1-paramgrad",
        "run_dir": str(args.run_dir),
        "checkpoint_dir": str(args.checkpoint_dir),
        "frames": list(args.frames),
        "device": args.device,
        "agent_contract": contract,
        "run_config": {
            "task": str(run_cfg.task),
            "obs_type": str(run_cfg.obs_type),
            "use_cls": bool(run_cfg.use_cls),
            "dino_frame_stack": int(run_cfg.dino_frame_stack),
            "action_repeat": int(run_cfg.action_repeat),
            "save_replay_buffer_in_checkpoint": bool(run_cfg.save_replay_buffer_in_checkpoint),
        },
        "fixed_batch": batch.metadata,
        "metric_definitions": {
            "C_B": "B.T @ B / N (uncentered second moment)",
            "reported_matrix_norm": "Frobenius norm / sqrt(N), labeled norm_rms",
            "F_parallel": "row projection of F onto its fixed z row",
            "F_perp": "F - F_parallel",
            "weak_subspace": "eigenvalue <= alpha * trace(C_B)/z_dim",
            "main_weak_alpha": 0.10,
            "weak_alpha_sensitivity": list(args.weak_alphas),
            "rho_dangerous": "||F_perp @ V_weak||_F^2 / ||F||_F^2",
            "top_eigenspace_ranks": list(args.top_ks),
            "main_top_k": 10,
            "M_online": "full fixed-batch F @ B.T, exact N^2 statistics",
            "Q": "row dot product F_i dot z_i",
        },
        "outputs": {
            "csv": str(csv_path),
            "geometry_plot": str(args.output_dir / "geometry_over_training.png"),
            "dynamics_plot": str(args.output_dir / "dynamics_over_training.png"),
            "diagnosis": str(diagnosis_path),
            "fixed_batch": str(args.output_dir / "fixed_replay_batch.npz"),
        },
    }
    atomic_json(args.output_dir / "analysis_metadata.json", metadata)
    print(f"wrote {csv_path}", flush=True)
    print(f"wrote {diagnosis_path}", flush=True)


if __name__ == "__main__":
    main()
