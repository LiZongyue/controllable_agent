#!/usr/bin/env python3
"""Reward-label count and quality sensitivity for FB policies.

This runner deliberately bypasses registered-goal inference.  It builds one
calibration bank from fresh rollouts or a local ExORL-RND source, caches only
backward features and scalar rewards, constructs nested subsets for several
observation/label quality conditions, and evaluates the resulting
reward-projected task vectors with paired environment seeds.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import re
import time
import zlib
from collections import OrderedDict, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

# dm_control selects its rendering backend while its modules are imported.
# Set EGL before importing url_benchmark.dmc so headless visual evaluation does
# not accidentally initialize GLFW first.
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import omegaconf as omgcf
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel

from url_benchmark import dmc, goals as goal_utils, utils
from url_benchmark.pretrain import make_agent


SCHEMA_VERSION = 1
DEFAULT_DINO_MODEL = "facebook/dinov2-base"
DEFAULT_EXORL_ROOT = Path("/mnt/data_7tb/fanfeng/exoRL_datasets")
EXORL_TARGET_EPISODE_CLUSTERS = 256
DEFAULT_KS = (1, 4, 16, 64, 256, 1024, 5120, 20480)
DEFAULT_QUALITIES = (
    "iid_clean",
    "stratified_clean",
    "correlated_clean",
    "iid_corrupt20",
)


def stable_seed(*parts: Any) -> int:
    text = "::".join(str(part) for part in parts)
    return zlib.crc32(text.encode("utf-8")) & 0x7FFFFFFF


def parse_int_csv(value: str) -> List[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def parse_str_csv(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def fingerprint_file(path: Path) -> str:
    stat = path.stat()
    payload = f"{path.resolve()}::{stat.st_size}::{stat.st_mtime_ns}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def hash_payload(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:20]


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write(json.dumps(dict(row), sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r") as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    return rows


class DinoCache:
    def __init__(self, model_name: str, device: str, local_files_only: bool = True) -> None:
        self.model_name = model_name
        self.device = device
        self.local_files_only = local_files_only
        self.processor: Any = None
        self.model: Any = None

    def get(self) -> Tuple[Any, Any]:
        if self.processor is None or self.model is None:
            kwargs = {"local_files_only": self.local_files_only}
            self.processor = AutoImageProcessor.from_pretrained(
                self.model_name,
                use_fast=True,
                **kwargs,
            )
            self.model = AutoModel.from_pretrained(self.model_name, **kwargs)
            self.model.to(self.device)
            self.model.eval()
            for parameter in self.model.parameters():
                parameter.requires_grad_(False)
        return self.processor, self.model


def load_cfg(config_path: Path, device: str, seed: Optional[int] = None) -> omgcf.DictConfig:
    cfg = omgcf.OmegaConf.load(config_path)
    omgcf.OmegaConf.set_struct(cfg, False)
    cfg.device = device
    cfg.use_tb = False
    cfg.use_wandb = False
    cfg.use_hiplog = False
    cfg.save_video = False
    cfg.save_train_video = False
    cfg.load_model = None
    cfg.checkpoint_root = None
    cfg.agent.device = device
    cfg.agent.use_tb = False
    cfg.agent.use_wandb = False
    cfg.agent.use_hiplog = False
    if seed is not None:
        cfg.seed = int(seed)
    return cfg


def clone_cfg(cfg: omgcf.DictConfig, seed: int) -> omgcf.DictConfig:
    cloned = omgcf.OmegaConf.create(
        omgcf.OmegaConf.to_container(cfg, resolve=False, throw_on_missing=False)
    )
    omgcf.OmegaConf.set_struct(cloned, False)
    cloned.seed = int(seed)
    cloned.device = cfg.device
    cloned.agent.device = cfg.device
    return cloned


def make_env(
    cfg: omgcf.DictConfig,
    dino_cache: DinoCache,
    seed: int,
    visual_seed: int,
    condition: Optional[str] = None,
) -> dmc.EnvWrapper:
    env_cfg = clone_cfg(cfg, seed)
    processor = None
    dino_model = None
    if env_cfg.obs_type == "dino":
        processor, dino_model = dino_cache.get()
    return dmc.make(
        env_cfg.task,
        env_cfg.obs_type,
        env_cfg.frame_stack,
        env_cfg.action_repeat,
        env_cfg.seed,
        goal_space=env_cfg.goal_space,
        append_goal_to_observation=env_cfg.append_goal_to_observation,
        dino_model=dino_model,
        dino_processor=processor,
        use_cls=env_cfg.use_cls,
        render_shape=tuple(env_cfg.render_shape),
        visual_perturbation=condition,
        visual_perturb_seed=visual_seed,
        dino_frame_stack=int(getattr(env_cfg, "dino_frame_stack", 1)),
    )


def load_agent(
    cfg: omgcf.DictConfig,
    env: dmc.EnvWrapper,
    checkpoint_path: Path,
) -> Tuple[Any, int]:
    agent = make_agent(
        cfg.obs_type,
        env.observation_spec(),
        env.action_spec(),
        cfg.num_seed_frames // cfg.action_repeat,
        cfg.agent,
    )
    payload = torch.load(checkpoint_path, map_location=cfg.device, weights_only=False)
    if "agent" not in payload:
        raise KeyError(f"Checkpoint {checkpoint_path} does not contain an agent")
    agent.init_from(payload["agent"])
    return agent, int(payload.get("global_step", 0))


@dataclass
class CalibrationBank:
    backward: np.ndarray
    reward: np.ndarray
    episode_id: np.ndarray
    step_in_episode: np.ndarray
    behavior_id: np.ndarray
    metadata: Dict[str, Any]

    def __len__(self) -> int:
        return int(self.reward.shape[0])

    def validate(self) -> None:
        size = len(self)
        if self.backward.ndim != 2:
            raise ValueError(f"Expected backward features [N,D], got {self.backward.shape}")
        for name in ("episode_id", "step_in_episode", "behavior_id"):
            value = getattr(self, name)
            if value.shape != (size,):
                raise ValueError(f"Expected {name} shape {(size,)}, got {value.shape}")
        if self.reward.shape != (size,):
            raise ValueError(f"Expected reward shape {(size,)}, got {self.reward.shape}")
        arrays = (self.backward, self.reward)
        if not all(np.isfinite(array).all() for array in arrays):
            raise ValueError("Calibration bank contains non-finite values")

    def save(self, path: Path) -> None:
        self.validate()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}.npz")
        try:
            np.savez_compressed(
                tmp_path,
                backward=self.backward.astype(np.float32, copy=False),
                reward=self.reward.astype(np.float32, copy=False),
                episode_id=self.episode_id.astype(np.int32, copy=False),
                step_in_episode=self.step_in_episode.astype(np.int32, copy=False),
                behavior_id=self.behavior_id.astype(np.int32, copy=False),
                metadata=np.asarray(json.dumps(self.metadata, sort_keys=True)),
            )
            tmp_path.replace(path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    @classmethod
    def load(cls, path: Path) -> "CalibrationBank":
        with np.load(path, allow_pickle=False) as payload:
            bank = cls(
                backward=np.asarray(payload["backward"], dtype=np.float32),
                reward=np.asarray(payload["reward"], dtype=np.float32),
                episode_id=np.asarray(payload["episode_id"], dtype=np.int32),
                step_in_episode=np.asarray(payload["step_in_episode"], dtype=np.int32),
                behavior_id=np.asarray(payload["behavior_id"], dtype=np.int32),
                metadata=json.loads(str(payload["metadata"].item())),
            )
        bank.validate()
        return bank


_EXORL_EPISODE_RE = re.compile(r"^episode_(\d+)_(\d+)\.npz$")


@dataclass(frozen=True)
class ExorlEpisodeSlice:
    """A zero-based transition slice from one ExORL episode.

    ExORL arrays contain a dummy reset entry at array index zero.  Therefore a
    transition step ``s`` in ``[start, stop)`` is read from array index
    ``s + 1`` so that its observation/physics is the transition's next state.
    """

    path: Path
    episode_id: int
    episode_length: int
    start: int
    stop: int

    def __len__(self) -> int:
        return self.stop - self.start


def parse_exorl_episode_path(path: Path) -> Tuple[int, int]:
    match = _EXORL_EPISODE_RE.fullmatch(path.name)
    if match is None:
        raise ValueError(f"Unexpected ExORL episode filename: {path.name!r}")
    episode_id, episode_length = (int(value) for value in match.groups())
    if episode_length <= 0:
        raise ValueError(f"ExORL episode length must be positive: {path}")
    return episode_id, episode_length


def select_exorl_episode_slices(
    episode_files: Sequence[Path],
    size: int,
    seed: int,
) -> List[ExorlEpisodeSlice]:
    """Select seeded episode clusters without reading the episode payloads.

    Sampling balanced contiguous clusters keeps I/O bounded while spreading a
    20,480-transition bank over hundreds of RND episodes. Episodes are sampled
    without replacement and every selected block gets a seeded random start.
    """

    if size <= 0:
        raise ValueError(f"Bank size must be positive, got {size}")
    parsed = [parse_exorl_episode_path(path) for path in episode_files]
    total_transitions = sum(length for _, length in parsed)
    if size > total_transitions:
        raise ValueError(
            f"Requested {size} transitions but ExORL source only has "
            f"{total_transitions}"
        )

    rng = np.random.RandomState(seed)
    order = rng.permutation(len(episode_files))
    cluster_count = min(len(episode_files), size, EXORL_TARGET_EPISODE_CLUSTERS)
    base_count, extra = divmod(size, cluster_count)
    selected: List[ExorlEpisodeSlice] = []
    for position, file_index in enumerate(order[:cluster_count]):
        episode_id, episode_length = parsed[int(file_index)]
        take = base_count + int(position < extra)
        if take > episode_length:
            raise ValueError(
                f"Episode {episode_files[int(file_index)]} has {episode_length} "
                f"transitions but balanced sampling requires {take}"
            )
        start = int(rng.randint(0, episode_length - take + 1))
        selected.append(
            ExorlEpisodeSlice(
                path=episode_files[int(file_index)],
                episode_id=episode_id,
                episode_length=episode_length,
                start=start,
                stop=start + take,
            )
        )
    remaining = size - sum(len(item) for item in selected)
    if remaining:
        raise RuntimeError(f"Failed to select {size} ExORL transitions")
    return selected


def exorl_selection_digest(selection: Sequence[ExorlEpisodeSlice]) -> str:
    payload = [
        {
            "file": item.path.name,
            "episode_id": item.episode_id,
            "episode_length": item.episode_length,
            "start": item.start,
            "stop": item.stop,
            "size": item.path.stat().st_size,
            "mtime_ns": item.path.stat().st_mtime_ns,
        }
        for item in selection
    ]
    return hash_payload({"episode_slices": payload})


def prepare_exorl_source(
    exorl_root: Path,
    task: str,
    size: int,
    bank_seed: int,
) -> Tuple[Path, List[ExorlEpisodeSlice], Dict[str, Any]]:
    domain = task.split("_", maxsplit=1)[0]
    source_dir = exorl_root.resolve() / domain / "rnd" / "buffer_dino_cls"
    if not source_dir.is_dir():
        raise FileNotFoundError(source_dir)
    episode_files = sorted(source_dir.glob("episode_*.npz"))
    if not episode_files:
        raise FileNotFoundError(f"No ExORL episodes found in {source_dir}")
    selection_seed = stable_seed(domain, bank_seed, "exorl_rnd_episode_clusters")
    selection = select_exorl_episode_slices(episode_files, size, selection_seed)
    provenance = {
        "source_dir": str(source_dir),
        "source_policy": "rnd",
        "source_episode_count": len(episode_files),
        "selected_episode_count": len(selection),
        "sampling": "seeded_balanced_episode_clusters_without_replacement",
        "selection_seed": int(selection_seed),
        "selection_digest": exorl_selection_digest(selection),
    }
    return source_dir, selection, provenance


def sample_behavior_meta(agent: Any) -> OrderedDict:
    z = agent.sample_z(1, device="cpu").squeeze(0).numpy().astype(np.float32)
    return OrderedDict(z=z)


def encode_backward_features(
    agent: Any,
    cfg: omgcf.DictConfig,
    backward_inputs: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    outputs: List[np.ndarray] = []
    with torch.no_grad(), utils.eval_mode(agent):
        for start in range(0, backward_inputs.shape[0], batch_size):
            value = torch.as_tensor(
                backward_inputs[start : start + batch_size],
                device=cfg.device,
                dtype=torch.float32,
            )
            if cfg.goal_space is None and cfg.obs_type in {"pixels", "dino", "vit"}:
                # Visual-B variants may use an adapter that is intentionally
                # separate from the forward/actor adapter.  Fall back to the
                # historically shared path for older agents/checkpoints.
                backward_encode = getattr(agent, "backward_aug_and_encode", agent.aug_and_encode)
                value = backward_encode(value)
            backward = agent.backward_net(value)
            outputs.append(backward.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(outputs, axis=0)


def collect_bank(
    agent: Any,
    cfg: omgcf.DictConfig,
    dino_cache: DinoCache,
    size: int,
    bank_seed: int,
    z_resample_steps: int,
    encode_batch_size: int,
    checkpoint_fingerprint: str,
    dino_model_name: str,
    collector: str,
) -> CalibrationBank:
    utils.set_seed_everywhere(bank_seed)
    env = make_env(
        cfg,
        dino_cache,
        seed=bank_seed,
        visual_seed=stable_seed(cfg.task, bank_seed, "calibration_visual"),
    )
    time_step = env.reset()
    meta = sample_behavior_meta(agent)
    episode_id = 0
    step_in_episode = 0
    behavior_id = 0
    steps_in_behavior = 0

    backward_inputs: List[np.ndarray] = []
    rewards: List[float] = []
    episodes: List[int] = []
    episode_steps: List[int] = []
    behaviors: List[int] = []
    started = time.time()

    while len(rewards) < size:
        if steps_in_behavior >= z_resample_steps:
            meta = sample_behavior_meta(agent)
            behavior_id += 1
            steps_in_behavior = 0
        with torch.no_grad(), utils.eval_mode(agent):
            if collector == "random_z_mean":
                action = agent.act(time_step.observation, meta, 0, eval_mode=True)
            elif collector == "random_z_sample":
                # Passing exactly num_expl_steps preserves the trained actor's
                # exploration distribution while bypassing the earlier
                # uniform-random action branch (which uses a strict < check).
                action = agent.act(
                    time_step.observation,
                    meta,
                    int(agent.cfg.num_expl_steps),
                    eval_mode=False,
                )
            else:
                raise ValueError(f"Unsupported collector {collector!r}")
        time_step = env.step(action)
        if cfg.goal_space is not None:
            if not hasattr(time_step, "goal"):
                raise RuntimeError(f"Expected goal for goal_space={cfg.goal_space}")
            backward_input = time_step.goal
        else:
            backward_input = time_step.observation
        backward_inputs.append(np.asarray(backward_input, dtype=np.float32))
        rewards.append(float(time_step.reward))
        episodes.append(episode_id)
        episode_steps.append(step_in_episode)
        behaviors.append(behavior_id)
        step_in_episode += 1
        steps_in_behavior += 1

        count = len(rewards)
        if count % 1000 == 0 or count == size:
            elapsed = max(time.time() - started, 1e-9)
            print(
                f"[{cfg.task}] calibration {count}/{size} ({count / elapsed:.1f} steps/s)",
                flush=True,
            )
        if time_step.last() and count < size:
            episode_id += 1
            step_in_episode = 0
            behavior_id += 1
            steps_in_behavior = 0
            time_step = env.reset()
            meta = sample_behavior_meta(agent)

    inputs = np.stack(backward_inputs, axis=0)
    backward = encode_backward_features(agent, cfg, inputs, encode_batch_size)
    bank = CalibrationBank(
        backward=backward,
        reward=np.asarray(rewards, dtype=np.float32),
        episode_id=np.asarray(episodes, dtype=np.int32),
        step_in_episode=np.asarray(episode_steps, dtype=np.int32),
        behavior_id=np.asarray(behaviors, dtype=np.int32),
        metadata={
            "schema_version": SCHEMA_VERSION,
            "bank_source": "rollout",
            "task": str(cfg.task),
            "goal_space": None if cfg.goal_space is None else str(cfg.goal_space),
            "obs_type": str(cfg.obs_type),
            "use_cls": bool(cfg.use_cls),
            "dino_model": dino_model_name,
            "collector": collector,
            "bank_seed": int(bank_seed),
            "size": int(size),
            "z_resample_steps": int(z_resample_steps),
            "checkpoint_fingerprint": checkpoint_fingerprint,
            "backward_input": "next_goal" if cfg.goal_space is not None else "next_observation",
        },
    )
    bank.validate()
    return bank


def _npz_scalar(payload: Any, key: str) -> Any:
    if key not in payload.files:
        raise KeyError(f"ExORL episode is missing metadata field {key!r}")
    return np.asarray(payload[key]).item()


def collect_exorl_bank(
    agent: Any,
    cfg: omgcf.DictConfig,
    selection: Sequence[ExorlEpisodeSlice],
    provenance: Mapping[str, Any],
    size: int,
    bank_seed: int,
    encode_batch_size: int,
    checkpoint_fingerprint: str,
    dino_model_name: str,
) -> CalibrationBank:
    """Build a reward-labeled bank from precomputed ExORL RND episodes."""

    if str(cfg.obs_type) != "dino":
        raise ValueError("exorl_rnd currently requires obs_type=dino")
    if not bool(cfg.use_cls):
        raise ValueError("buffer_dino_cls is incompatible with use_cls=False")
    if sum(len(item) for item in selection) != size:
        raise ValueError("ExORL selection size does not match requested bank size")

    task = str(cfg.task)
    domain = task.split("_", maxsplit=1)[0]
    goal_space = None if cfg.goal_space is None else str(cfg.goal_space)
    goal_func = None
    if goal_space is not None:
        available = goal_utils.goal_spaces.funcs.get(domain, {})
        if goal_space not in available:
            raise ValueError(f"No goal space {goal_space!r} registered for {domain!r}")
        goal_func = available[goal_space]

    reward_function = goal_utils.get_reward_function(task, seed=bank_seed)
    reward_env = reward_function._env  # pylint: disable=protected-access
    goal_dim = None if goal_func is None else int(np.asarray(goal_func(reward_env)).size)
    expected_camera = 2 if domain == "quadruped" else 0
    expected_image_size = int(tuple(cfg.render_shape)[0])

    backward_inputs: List[np.ndarray] = []
    rewards: List[np.ndarray] = []
    episode_ids: List[np.ndarray] = []
    episode_steps: List[np.ndarray] = []
    started = time.time()
    loaded = 0

    for item in selection:
        data_indices = np.arange(item.start, item.stop, dtype=np.int64) + 1
        with np.load(item.path, allow_pickle=False) as payload:
            required_length = item.episode_length + 1
            if "physics" not in payload.files:
                raise KeyError(f"ExORL episode is missing physics: {item.path}")
            if payload["physics"].shape[0] < required_length:
                raise ValueError(
                    f"{item.path} has {payload['physics'].shape[0]} states, "
                    f"expected at least {required_length}"
                )
            source_model = str(_npz_scalar(payload, "dino_model"))
            source_token = str(_npz_scalar(payload, "dino_token"))
            source_image_size = int(_npz_scalar(payload, "image_size"))
            source_camera = int(_npz_scalar(payload, "camera_id"))
            if source_model != dino_model_name:
                raise ValueError(
                    f"DINO model mismatch in {item.path}: "
                    f"{source_model!r} != {dino_model_name!r}"
                )
            if source_token != "cls":
                raise ValueError(f"Expected CLS embeddings in {item.path}, got {source_token!r}")
            if source_image_size != expected_image_size:
                raise ValueError(
                    f"Image-size mismatch in {item.path}: "
                    f"{source_image_size} != {expected_image_size}"
                )
            if source_camera != expected_camera:
                raise ValueError(
                    f"Camera mismatch in {item.path}: {source_camera} != {expected_camera}"
                )

            physics = np.asarray(payload["physics"][data_indices], dtype=np.float64)
            if goal_func is None:
                if "dino_emb" not in payload.files:
                    raise KeyError(f"ExORL episode is missing dino_emb: {item.path}")
                values = np.asarray(payload["dino_emb"][data_indices], dtype=np.float32)
            else:
                assert goal_dim is not None
                values = np.empty((len(item), goal_dim), dtype=np.float32)

        item_rewards = np.empty(len(item), dtype=np.float32)
        for offset, state in enumerate(physics):
            # from_physics leaves reward_env at this exact next state.  The
            # goal, when used, is consequently computed from that same state.
            item_rewards[offset] = reward_function.from_physics(state)
            if goal_func is not None:
                values[offset] = goal_func(reward_env)

        backward_inputs.append(values)
        rewards.append(item_rewards)
        episode_ids.append(np.full(len(item), item.episode_id, dtype=np.int32))
        episode_steps.append(np.arange(item.start, item.stop, dtype=np.int32))
        loaded += len(item)
        if loaded == size or loaded % 5000 < len(item):
            elapsed = max(time.time() - started, 1e-9)
            print(
                f"[{task}] ExORL calibration {loaded}/{size} ({loaded / elapsed:.1f} steps/s)",
                flush=True,
            )

    close = getattr(reward_env, "close", None)
    if callable(close):
        close()

    inputs = np.concatenate(backward_inputs, axis=0)
    backward = encode_backward_features(agent, cfg, inputs, encode_batch_size)
    bank = CalibrationBank(
        backward=backward,
        reward=np.concatenate(rewards).astype(np.float32, copy=False),
        episode_id=np.concatenate(episode_ids),
        step_in_episode=np.concatenate(episode_steps),
        # ExORL does not store latent behavior IDs. Episode identity is an
        # explicit proxy so stratification still spreads labels over rollouts.
        behavior_id=np.concatenate(episode_ids),
        metadata={
            "schema_version": SCHEMA_VERSION,
            "bank_source": "exorl_rnd",
            "task": task,
            "goal_space": goal_space,
            "obs_type": str(cfg.obs_type),
            "use_cls": bool(cfg.use_cls),
            "dino_model": dino_model_name,
            "dino_token": "cls",
            "image_size": expected_image_size,
            "camera_id": expected_camera,
            "bank_seed": int(bank_seed),
            "size": int(size),
            "checkpoint_fingerprint": checkpoint_fingerprint,
            "backward_input": "next_goal" if goal_space is not None else "next_observation",
            "transition_alignment": "array_index=step_in_episode+1 (next state)",
            "behavior_id_semantics": "episode_proxy_no_behavior_labels",
            **dict(provenance),
        },
    )
    bank.validate()
    return bank


def validate_bank_metadata(bank: CalibrationBank, expected: Mapping[str, Any]) -> None:
    mismatches = {
        key: (bank.metadata.get(key), value)
        for key, value in expected.items()
        if bank.metadata.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Calibration bank metadata mismatch: {mismatches}")


def iid_order(size: int, seed: int) -> np.ndarray:
    return np.random.RandomState(seed).permutation(size).astype(np.int64)


def correlated_order(size: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    start = int(rng.randint(size))
    base = np.arange(size, dtype=np.int64)
    return np.concatenate((base[start:], base[:start]))


def stratified_order(behavior_id: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    groups: Dict[int, deque] = {}
    for group_id in np.unique(behavior_id):
        indices = np.flatnonzero(behavior_id == group_id)
        rng.shuffle(indices)
        groups[int(group_id)] = deque(int(index) for index in indices)
    group_ids = np.asarray(list(groups), dtype=np.int64)
    rng.shuffle(group_ids)
    output: List[int] = []
    active = list(int(group_id) for group_id in group_ids)
    while active:
        next_active: List[int] = []
        for group_id in active:
            queue = groups[group_id]
            output.append(queue.popleft())
            if queue:
                next_active.append(group_id)
        active = next_active
    order = np.asarray(output, dtype=np.int64)
    if order.shape != behavior_id.shape or np.unique(order).size != order.size:
        raise RuntimeError("Stratified order is not a full permutation")
    return order


def order_for_quality(bank: CalibrationBank, quality: str, seed: int) -> np.ndarray:
    if quality in {"iid_clean", "iid_corrupt20", "iid_corrupt50"}:
        return iid_order(len(bank), seed)
    if quality == "stratified_clean":
        return stratified_order(bank.behavior_id, seed)
    if quality == "correlated_clean":
        return correlated_order(len(bank), seed)
    raise ValueError(f"Unknown quality {quality!r}")


def corrupted_rewards(reward: np.ndarray, fraction: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    mask = rng.uniform(size=reward.shape[0]) < fraction
    donor = rng.permutation(reward.shape[0])
    output = reward.copy()
    output[mask] = reward[donor[mask]]
    return output, mask


def meta_from_features(
    backward: np.ndarray,
    reward: np.ndarray,
    z_dim: int,
    norm_z: bool,
    projection_method: str = "mean",
    ridge_alpha: float = 1e-2,
) -> Tuple[OrderedDict, Dict[str, Any]]:
    if backward.shape[0] != reward.shape[0] or backward.shape[0] == 0:
        raise ValueError("Backward features and rewards must have equal nonzero length")
    projection_diagnostics: Dict[str, Any] = {
        "projection_method": projection_method,
        "ridge_alpha": None,
        "ridge_lambda": None,
    }
    if projection_method == "mean":
        # Keep prior sensitivity outputs bit-for-bit resumable: multiplication
        # is performed in the bank dtype, followed by a float64 reduction.
        raw = np.mean(backward * reward[:, None], axis=0, dtype=np.float64).astype(
            np.float32
        )
    elif projection_method == "ridge":
        if not np.isfinite(ridge_alpha) or ridge_alpha <= 0:
            raise ValueError(f"ridge_alpha must be positive, got {ridge_alpha}")
        backward64 = np.asarray(backward, dtype=np.float64)
        reward64 = np.asarray(reward, dtype=np.float64)
        rhs = np.mean(backward64 * reward64[:, None], axis=0, dtype=np.float64)
        second_moment = backward64.T @ backward64 / backward64.shape[0]
        second_moment_scale = float(
            np.trace(second_moment) / second_moment.shape[0]
        )
        ridge_lambda = float(ridge_alpha * max(second_moment_scale, 1e-12))
        regularized = second_moment + ridge_lambda * np.eye(
            second_moment.shape[0], dtype=np.float64
        )
        raw64 = np.linalg.solve(regularized, rhs)
        raw = raw64.astype(np.float32)
        eigenvalues = np.linalg.eigvalsh(second_moment)
        projection_diagnostics.update(
            {
                "ridge_alpha": float(ridge_alpha),
                "ridge_lambda": ridge_lambda,
                "second_moment_scale": second_moment_scale,
                "second_moment_min_eigenvalue": float(eigenvalues[0]),
                "second_moment_max_eigenvalue": float(eigenvalues[-1]),
                "regularized_condition_number": float(np.linalg.cond(regularized)),
            }
        )
    else:
        raise ValueError(f"Unknown projection_method {projection_method!r}")
    raw_tensor = torch.as_tensor(raw).unsqueeze(0)
    raw_norm = float(torch.linalg.vector_norm(raw_tensor).item())
    if norm_z:
        z_tensor = math.sqrt(z_dim) * F.normalize(raw_tensor, dim=1)
    else:
        z_tensor = raw_tensor
    z = z_tensor.squeeze(0).numpy().astype(np.float32)
    meta = OrderedDict(z=z)
    diagnostics = {
        **projection_diagnostics,
        "raw_z_norm": raw_norm,
        "z_norm": float(np.linalg.norm(z)),
        "degenerate_z": bool(raw_norm <= 1e-12),
    }
    return meta, diagnostics


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1e-12:
        return 0.0
    return float(np.dot(left, right) / denominator)


def quality_fraction(quality: str) -> float:
    if quality == "iid_corrupt20":
        return 0.20
    if quality == "iid_corrupt50":
        return 0.50
    return 0.0


def build_meta(
    bank: CalibrationBank,
    quality: str,
    k: int,
    subset_seed: int,
    task: str,
    z_dim: int,
    norm_z: bool,
    reference_z: np.ndarray,
    projection_method: str = "mean",
    ridge_alpha: float = 1e-2,
) -> Tuple[OrderedDict, Dict[str, Any]]:
    if k <= 0 or k > len(bank):
        raise ValueError(f"K must be in [1, {len(bank)}], got {k}")
    iid_seed = stable_seed(task, bank.metadata["bank_seed"], subset_seed, "iid_order")
    order_seed = (
        iid_seed
        if quality in {"iid_clean", "iid_corrupt20", "iid_corrupt50"}
        else stable_seed(task, bank.metadata["bank_seed"], subset_seed, quality, "order")
    )
    order = order_for_quality(bank, quality, order_seed)
    indices = order[:k]
    reward = bank.reward
    corrupt_mask = np.zeros(len(bank), dtype=bool)
    fraction = quality_fraction(quality)
    if fraction:
        reward, corrupt_mask = corrupted_rewards(
            bank.reward,
            fraction,
            stable_seed(task, bank.metadata["bank_seed"], subset_seed, quality, "corruption"),
        )
    selected_reward = reward[indices]
    meta, diagnostics = meta_from_features(
        bank.backward[indices],
        selected_reward,
        z_dim=z_dim,
        norm_z=norm_z,
        projection_method=projection_method,
        ridge_alpha=ridge_alpha,
    )
    diagnostics.update(
        {
            "cosine_to_full_clean": cosine_similarity(meta["z"], reference_z),
            "reward_mean": float(np.mean(selected_reward)),
            "reward_std": float(np.std(selected_reward)),
            "reward_nonzero_fraction": float(np.mean(np.abs(selected_reward) > 1e-12)),
            "unique_episodes": int(np.unique(bank.episode_id[indices]).size),
            "unique_behaviors": int(np.unique(bank.behavior_id[indices]).size),
            "actual_corrupt_fraction": float(np.mean(corrupt_mask[indices])),
            "index_digest": hashlib.sha256(indices.tobytes()).hexdigest()[:16],
        }
    )
    return meta, diagnostics


def evaluate_one_episode(
    agent: Any,
    cfg: omgcf.DictConfig,
    dino_cache: DinoCache,
    meta: Mapping[str, np.ndarray],
    global_step: int,
    eval_seed: int,
    condition: Optional[str],
) -> Tuple[float, int]:
    utils.set_seed_everywhere(eval_seed)
    env = make_env(
        cfg,
        dino_cache,
        seed=eval_seed,
        visual_seed=stable_seed(cfg.task, eval_seed, condition or "clean", "visual"),
        condition=condition,
    )
    time_step = env.reset()
    total_reward = 0.0
    steps = 0
    while not time_step.last():
        with torch.no_grad(), utils.eval_mode(agent):
            action = agent.act(time_step.observation, meta, global_step, eval_mode=True)
        time_step = env.step(action)
        total_reward += float(time_step.reward)
        steps += 1
    close = getattr(env, "close", None)
    if callable(close):
        close()
    return total_reward, steps * int(cfg.action_repeat)


def cell_id(payload: Mapping[str, Any]) -> str:
    return hash_payload(payload)


def run(args: argparse.Namespace) -> None:
    run_dir = args.run_dir.resolve()
    config_path = args.config.resolve() if args.config else run_dir / ".hydra" / "config.yaml"
    checkpoint_path = args.checkpoint.resolve() if args.checkpoint else Path(
        "/mnt/data_7tb/fanfeng/controallable_agent_ckpt"
    ) / run_dir.name / "latest.pt"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    cfg = load_cfg(config_path, args.device)
    if args.task is not None:
        source_domain = str(cfg.task).split("_", maxsplit=1)[0]
        target_domain = str(args.task).split("_", maxsplit=1)[0]
        if source_domain != target_domain:
            raise ValueError(
                f"Task override must stay in checkpoint domain {source_domain!r}, "
                f"got {args.task!r}"
            )
        cfg.task = str(args.task)
    task = str(cfg.task)
    task_dir = args.output_dir.resolve() / task
    task_dir.mkdir(parents=True, exist_ok=True)
    episodes_path = task_dir / "episodes.jsonl"
    metas_path = task_dir / "metas.jsonl"
    checkpoint_fingerprint = fingerprint_file(checkpoint_path)
    exorl_selection: Optional[List[ExorlEpisodeSlice]] = None
    exorl_provenance: Optional[Dict[str, Any]] = None
    if args.bank_source == "rollout":
        # Keep this descriptor byte-for-byte compatible with banks produced
        # before --bank-source was added, so existing rollout runs can resume.
        bank_descriptor = {
            "schema_version": SCHEMA_VERSION,
            "task": task,
            "checkpoint_fingerprint": checkpoint_fingerprint,
            "dino_model": args.dino_model,
            "bank_seed": int(args.bank_seed),
            "size": int(args.bank_size),
            "z_resample_steps": int(args.z_resample_steps),
            "collector": args.collector,
        }
    else:
        _, exorl_selection, exorl_provenance = prepare_exorl_source(
            args.exorl_root,
            task,
            args.bank_size,
            args.bank_seed,
        )
        bank_descriptor = {
            "schema_version": SCHEMA_VERSION,
            "task": task,
            "checkpoint_fingerprint": checkpoint_fingerprint,
            "dino_model": args.dino_model,
            "bank_seed": int(args.bank_seed),
            "size": int(args.bank_size),
            "bank_source": "exorl_rnd",
            "source_dir": exorl_provenance["source_dir"],
            "selection_digest": exorl_provenance["selection_digest"],
        }
    bank_id = hash_payload(bank_descriptor)
    bank_path = task_dir / "banks" / f"bank_{bank_id}.npz"

    print(f"[{task}] config={config_path}", flush=True)
    print(f"[{task}] checkpoint={checkpoint_path}", flush=True)
    print(f"[{task}] dino_model={args.dino_model}", flush=True)
    print(f"[{task}] bank_source={args.bank_source}", flush=True)
    print(f"[{task}] bank={bank_path}", flush=True)

    dino_cache = DinoCache(
        args.dino_model,
        args.device,
        local_files_only=not args.allow_download,
    )
    utils.set_seed_everywhere(args.bank_seed)
    bootstrap_env = make_env(
        cfg,
        dino_cache,
        seed=args.bank_seed,
        visual_seed=stable_seed(task, args.bank_seed, "bootstrap_visual"),
    )
    agent, global_step = load_agent(cfg, bootstrap_env, checkpoint_path)
    close = getattr(bootstrap_env, "close", None)
    if callable(close):
        close()

    if args.bank_source == "rollout":
        expected_bank_metadata = {
            "schema_version": SCHEMA_VERSION,
            "task": task,
            "dino_model": args.dino_model,
            "collector": args.collector,
            "bank_seed": int(args.bank_seed),
            "size": int(args.bank_size),
            "z_resample_steps": int(args.z_resample_steps),
            "checkpoint_fingerprint": checkpoint_fingerprint,
        }
    else:
        assert exorl_provenance is not None
        expected_bank_metadata = {
            "schema_version": SCHEMA_VERSION,
            "bank_source": "exorl_rnd",
            "task": task,
            "dino_model": args.dino_model,
            "bank_seed": int(args.bank_seed),
            "size": int(args.bank_size),
            "checkpoint_fingerprint": checkpoint_fingerprint,
            "source_dir": exorl_provenance["source_dir"],
            "selection_digest": exorl_provenance["selection_digest"],
        }
    if bank_path.exists():
        bank = CalibrationBank.load(bank_path)
        validate_bank_metadata(bank, expected_bank_metadata)
        print(f"[{task}] loaded cached bank with {len(bank)} samples", flush=True)
    else:
        if args.bank_source == "rollout":
            bank = collect_bank(
                agent,
                cfg,
                dino_cache,
                size=args.bank_size,
                bank_seed=args.bank_seed,
                z_resample_steps=args.z_resample_steps,
                encode_batch_size=args.encode_batch_size,
                checkpoint_fingerprint=checkpoint_fingerprint,
                dino_model_name=args.dino_model,
                collector=args.collector,
            )
        else:
            assert exorl_selection is not None and exorl_provenance is not None
            bank = collect_exorl_bank(
                agent,
                cfg,
                selection=exorl_selection,
                provenance=exorl_provenance,
                size=args.bank_size,
                bank_seed=args.bank_seed,
                encode_batch_size=args.encode_batch_size,
                checkpoint_fingerprint=checkpoint_fingerprint,
                dino_model_name=args.dino_model,
            )
        bank.save(bank_path)
        print(f"[{task}] saved calibration bank", flush=True)

    ks = parse_int_csv(args.ks)
    qualities = parse_str_csv(args.qualities)
    subset_seeds = parse_int_csv(args.subset_seeds)
    eval_seeds = parse_int_csv(args.eval_seeds)
    if not ks or not qualities or not subset_seeds or not eval_seeds:
        raise ValueError("ks, qualities, subset_seeds, and eval_seeds must be nonempty")
    if max(ks) > len(bank):
        raise ValueError(f"Largest K={max(ks)} exceeds bank size {len(bank)}")

    reference_meta, reference_diag = meta_from_features(
        bank.backward,
        bank.reward,
        z_dim=int(agent.cfg.z_dim),
        norm_z=bool(agent.cfg.norm_z),
        projection_method=args.projection_method,
        ridge_alpha=args.ridge_alpha,
    )
    reference_z = reference_meta["z"]
    print(
        f"[{task}] full clean reference raw_norm={reference_diag['raw_z_norm']:.6g}",
        flush=True,
    )

    completed_episode_ids = {
        str(row["episode_cell_id"])
        for row in read_jsonl(episodes_path)
        if row.get("status") == "complete"
    }
    completed_meta_ids = {
        str(row["meta_cell_id"])
        for row in read_jsonl(metas_path)
        if row.get("status") == "complete"
    }
    evaluation_cache: Dict[Tuple[bytes, int, str], Tuple[float, int]] = {}

    for quality in qualities:
        for subset_seed in subset_seeds:
            for k in ks:
                meta, diagnostics = build_meta(
                    bank,
                    quality=quality,
                    k=k,
                    subset_seed=subset_seed,
                    task=task,
                    z_dim=int(agent.cfg.z_dim),
                    norm_z=bool(agent.cfg.norm_z),
                    reference_z=reference_z,
                    projection_method=args.projection_method,
                    ridge_alpha=args.ridge_alpha,
                )
                method_name = (
                    "reward_projection"
                    if args.projection_method == "mean"
                    else "reward_projection_ridge"
                )
                meta_key_payload = {
                    "schema_version": SCHEMA_VERSION,
                    "task": task,
                    "checkpoint_fingerprint": checkpoint_fingerprint,
                    "bank_id": bank_id,
                    "method": method_name,
                    "quality": quality,
                    "k": int(k),
                    "subset_seed": int(subset_seed),
                }
                if args.projection_method == "ridge":
                    meta_key_payload["ridge_alpha"] = float(args.ridge_alpha)
                meta_key = cell_id(meta_key_payload)
                if meta_key not in completed_meta_ids:
                    append_jsonl(
                        metas_path,
                        {
                            **meta_key_payload,
                            **diagnostics,
                            "meta_cell_id": meta_key,
                            "num_labels_used": int(k),
                            "global_step": int(global_step),
                            "checkpoint": str(checkpoint_path),
                            "config": str(config_path),
                            "bank_source": args.bank_source,
                            "source_policy": bank.metadata.get(
                                "source_policy", bank.metadata.get("collector")
                            ),
                            "goal_space": None if cfg.goal_space is None else str(cfg.goal_space),
                            "backward_input": bank.metadata["backward_input"],
                            "status": "complete",
                        },
                    )
                    completed_meta_ids.add(meta_key)

                for eval_seed in eval_seeds:
                    episode_payload = {
                        **meta_key_payload,
                        "eval_seed": int(eval_seed),
                        "eval_condition": args.eval_condition,
                    }
                    episode_key = cell_id(episode_payload)
                    if args.resume and episode_key in completed_episode_ids:
                        continue
                    cache_key = (
                        np.asarray(meta["z"], dtype=np.float32).tobytes(),
                        int(eval_seed),
                        args.eval_condition,
                    )
                    started = time.time()
                    try:
                        if cache_key not in evaluation_cache:
                            evaluation_cache[cache_key] = evaluate_one_episode(
                                agent,
                                cfg,
                                dino_cache,
                                meta,
                                global_step,
                                eval_seed,
                                None if args.eval_condition == "clean" else args.eval_condition,
                            )
                        episode_reward, episode_length = evaluation_cache[cache_key]
                        append_jsonl(
                            episodes_path,
                            {
                                **episode_payload,
                                "episode_cell_id": episode_key,
                                "meta_cell_id": meta_key,
                                "num_labels_used": int(k),
                                "bank_source": args.bank_source,
                                "episode_reward": float(episode_reward),
                                "episode_length": int(episode_length),
                                "duration_sec": time.time() - started,
                                "status": "complete",
                            },
                        )
                        completed_episode_ids.add(episode_key)
                        print(
                            f"[{task}] quality={quality} K={k} subset={subset_seed} "
                            f"eval={eval_seed} return={episode_reward:.4f}",
                            flush=True,
                        )
                    except Exception as exc:
                        append_jsonl(
                            task_dir / "errors.jsonl",
                            {
                                **episode_payload,
                                "episode_cell_id": episode_key,
                                "error": repr(exc),
                                "status": "error",
                            },
                        )
                        raise

    print(f"[{task}] complete", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--task",
        help="Evaluate another task from the same domain with this reward-free checkpoint.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dino-model", default=DEFAULT_DINO_MODEL)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument(
        "--bank-source",
        choices=("rollout", "exorl_rnd"),
        default="rollout",
    )
    parser.add_argument("--exorl-root", type=Path, default=DEFAULT_EXORL_ROOT)
    parser.add_argument("--bank-size", type=int, default=max(DEFAULT_KS))
    parser.add_argument("--bank-seed", type=int, default=20260727)
    parser.add_argument("--z-resample-steps", type=int, default=300)
    parser.add_argument(
        "--collector",
        choices=("random_z_sample", "random_z_mean"),
        default="random_z_sample",
    )
    parser.add_argument("--encode-batch-size", type=int, default=4096)
    parser.add_argument("--ks", default=",".join(str(value) for value in DEFAULT_KS))
    parser.add_argument("--qualities", default=",".join(DEFAULT_QUALITIES))
    parser.add_argument("--subset-seeds", default="0,1,2,3,4")
    parser.add_argument("--eval-seeds", default="1101,1102,1103,1104,1105")
    parser.add_argument("--eval-condition", default="clean")
    parser.add_argument(
        "--projection-method",
        choices=("mean", "ridge"),
        default="mean",
        help=(
            "Map reward labels to z with the original mean E[B r], or with a "
            "diagnostic ridge-regression solve using E[B B^T]."
        ),
    )
    parser.add_argument(
        "--ridge-alpha",
        type=float,
        default=1e-2,
        help=(
            "Relative ridge strength: lambda = alpha * trace(E[B B^T]) / z_dim. "
            "Used only with --projection-method ridge."
        ),
    )
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
