#!/usr/bin/env python3
"""Offline mechanism diagnostics for visual-B Cheetah checkpoints.

This is deliberately a read-only checkpoint analysis.  It never invokes an
agent update method and installs fail-closed guards on every deserialized
optimizer's ``step`` method.  The historical online replay/RNG state was not
serialized, so the common replay bank is a deterministic ExORL-RND proxy.  In
particular, its one-source-control-step transitions are not an exact reconstruction of the
historical action-repeat=2 replay.  Task latents use the repository's exact
reward-projection formula on that fixed proxy, and are labelled accordingly.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import faulthandler
import hashlib
import io
import json
import math
import os
import shutil
import signal
import sys
import types
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/controllable_agent_mpl_cache")

import matplotlib.pyplot as plt
import numpy as np
import omegaconf as omgcf
import torch
from torch import nn

from url_benchmark import utils
from url_benchmark.agent.fb_ddpg import _scale_gradient
from url_benchmark.goals import DmcReward


SCHEMA_VERSION = 2
Z_DIM = 50
FRAME_STACK = 3
EMBED_DIM = 768
TRAIN_BATCH = 1024
CNN_RENDER_CONTEXT_CHUNK_SIZE = 8
# Kept backend-specific so a validated OSMesa value can be changed without
# altering EGL behavior.  Full-episode (80-row) OSMesa contexts are currently
# too slow, so the fail-safe value remains 8 pending a complete benchmark.
CNN_OSMESA_RENDER_CONTEXT_CHUNK_SIZE = 8
CNN_RENDER_CONTEXT_POLICY = (
    "fresh DmcReward('cheetah_walk')._env physics context per contiguous "
    "episode-local selected-row chunk"
)
EXORL_EPISODE_FILENAME_TEMPLATE = "episode_{episode_id:06d}_1000.npz"
CNN_PIXEL_EPISODE_PARTS_DIRNAME = "fixed_rendered_replay_parts"
CNN_PIXEL_EPISODE_PART_FILENAME_TEMPLATE = "episode_{episode_id:06d}.npz"
CNN_PIXEL_EPISODE_PART_SCHEMA_VERSION = 1
CNN_PIXEL_EPISODE_STREAM_CHUNK_BYTES = 1 << 20
CNN_PIXEL_GLOBAL_ARRAYS_DIRNAME = "fixed_rendered_replay_global_arrays"
CNN_PIXEL_GLOBAL_ARRAYS_MANIFEST = "manifest.json"
CNN_PIXEL_GLOBAL_ARRAYS_SCHEMA_VERSION = 1
# Keep every transient float64 Q-geometry matrix well below one MiB.  On the
# shared training host, first-touching a full 20,480 x 50 anonymous conversion
# can spend minutes in the kernel even though the same computation in bounded
# buffers progresses normally.
CNN_Q_FLOAT64_CHUNK_ROWS = 128
CNN_PIXEL_EPISODE_PART_ARRAY_FIELDS = (
    "episode_id",
    "destinations",
    "steps",
    "index_bank_row",
    "obs",
    "next_obs",
    "action",
    "discount",
    "physics",
    "source_reward",
)
REPLAY_CACHE_ARRAY_FIELDS = (
    "obs",
    "next_obs",
    "action",
    "discount",
    "physics",
    "source_reward",
    "random_z",
    "episode_id",
    "step_in_episode",
    "index_bank_row",
)
DEFAULT_REPLAY_SIZE = 20_480
CNN_DEFAULT_REPLAY_SIZE = 20_480
DEFAULT_PROBE_SIZE = 1_024
DEFAULT_INFERENCE_SIZE = 5_120
DEFAULT_SEED = 20_260_827
DEFAULT_Z_SEED = 20_260_828
DEFAULT_OUTPUT = Path("analysis_outputs/pixel_b_cheetah_mechanisms_20260827")
CNN_DEFAULT_OUTPUT = Path(
    "analysis_outputs/cheetah_fb_cnn_visual_b_checkpoint_geometry_20260831"
)
DEFAULT_EXORL = Path("/mnt/data_7tb/fanfeng/exoRL_datasets/cheetah/rnd/buffer_dino_cls")
DEFAULT_BANK = Path(
    "analysis_outputs/reward_label_sensitivity_dinov2_best12_exorl_rnd_anchor_20260727/"
    "cheetah_walk/banks/bank_cd511a318fc46f35eee2.npz"
)
CNN_TARGET_FRAMES = (100_000, 500_000, 800_000, 1_000_000, 1_500_000, 2_000_000)
CNN_REWARD_TASKS = (
    "cheetah_walk",
    "cheetah_run",
    "cheetah_run_backward",
    "cheetah_walk_backward",
)
WEAK_DEFINITIONS = ("bottom5", "bottom10", "thr003", "thr010", "thr030")
PRIMARY_WEAK = "thr010"
TRAIN_METRICS = (
    "episode_reward", "q", "actor_loss", "B", "B_norm", "z_norm", "M1", "target_M",
    "F1_norm_mean", "F2_norm_mean", "F_norm_mean",
    "F1_norm_max", "F2_norm_max", "F_norm_max", "F1_abs_max", "F2_abs_max",
    "F_abs_max", "F_norm_p95", "F_norm_p99", "M_online_abs_p95",
    "M_online_abs_p99", "M_online_abs_max", "fb_diag", "fb_offdiag", "fb_loss",
    "orth_l2", "F1_grad_norm", "F2_grad_norm", "F_grad_norm", "encoder_grad_norm",
)


@dataclass(frozen=True)
class Experiment:
    key: str
    run_id: str
    task: str
    expected_fb_lr: float
    run_dir: Path
    checkpoint_dir: Path
    frames: Tuple[int, ...]
    checkpoint_hashes: Mapping[int, str]
    # Historical checkpoints have a recorded exact size.  Fresh experiment
    # lineages cannot know that size before their first checkpoint is written;
    # those still receive an exact SHA-256 provenance record at analysis time.
    expected_size: Optional[int]
    obs_type: str = "dino"
    goal_space: Optional[str] = None
    frame_stack: int = FRAME_STACK
    render_shape: Tuple[int, int] = (224, 224)
    seed: int = 1
    # Exact serialized lineage field. It is a pixel-path no-op, but remains a
    # strict provenance check; DINO/ViT uses it to select CLS versus patch data.
    expected_use_cls: bool = True
    expected_pixel_separate_fb_encoders: bool = False
    expected_dino_separate_fb_adapters: bool = False
    expected_dino_adapter_type: str = "linear"
    expected_idm_coef: float = 0.0
    expected_idm_route: str = "none"
    expected_idm_encoder_mode: str = "legacy"
    expected_idm_lr: Optional[float] = None
    expected_lr_f: Optional[float] = None
    expected_lr_b: Optional[float] = None
    analyze_gradient_attribution: bool = True


EXPERIMENTS: Tuple[Experiment, ...] = (
    Experiment(
        key="walk_lr1e4",
        run_id="dino_cls3_cheetah_walk_seed1-paramgrad",
        task="cheetah_walk",
        expected_fb_lr=1e-4,
        run_dir=Path("/mnt/data_7tb/fanfeng/controllable_agent_runs/20260817_cheetah_fb_stability_paramgrad/dino_cls3_cheetah_walk_seed1"),
        checkpoint_dir=Path("/mnt/data_7tb/fanfeng/controallable_agent_ckpt/cheetah_fb_stability/dino_cls3_cheetah_walk_seed1"),
        frames=(100_000, 200_000, 500_000, 800_000, 1_000_000, 1_500_000, 2_000_000),
        checkpoint_hashes={
            100_000: "9b695d7bee32c41d4ad24efcdb31450183c7b3e1828cfe8dc2f330f65749647c",
            200_000: "e2f711ab77e018c7b9594c2d52ae49c9e846497a8df3f192cb32d50f59d3f9ba",
            500_000: "247b17bcbe618157512a1e5c0e0be54a8928155db68e7b49f6e287703e9f3f8f",
            800_000: "6789ec92158849cdf2235e20cc51eb3bf9141535e941bedab67e71dec4b98174",
            1_000_000: "80b161f7fae3ec03a5de5a89c1f8333902d470407ce6f60bd4d7293754eb513d",
            1_500_000: "dbad503ed1a354ae1389fec7eea0fa61120fbcaa0da41187bf8105902e15c18d",
            2_000_000: "d33d1dd451595a9c407beb2e0aa3a2e569a41285b5af0057babe56f1e04ba1c1",
        },
        expected_size=131_827_130,
    ),
    Experiment(
        key="run_lr5e5",
        run_id="dino_cls3_cheetah_run_seed1_fblr5e5",
        task="cheetah_run",
        expected_fb_lr=5e-5,
        run_dir=Path("/mnt/data_7tb/fanfeng/controllable_agent_runs/20260818_cheetah_fb_lr5e5_ablation/dino_cls3_cheetah_run_seed1_fblr5e5"),
        checkpoint_dir=Path("/mnt/data_7tb/fanfeng/controallable_agent_ckpt/cheetah_fb_stability/dino_cls3_cheetah_run_seed1_fblr5e5"),
        frames=(100_000, 200_000, 500_000, 800_000, 1_000_000, 1_500_000, 2_000_000),
        checkpoint_hashes={
            100_000: "f8d89fc5ee36c75e248aca3702189bab77b6095c8c282dfbb6410a04c5d1b409",
            200_000: "f57d4b5113fe16dee697e25470adfc1b15caef9b98c5b6900326d06b7f5371b9",
            500_000: "89b724e9a889f21ae498104cc54ba44080915f60e71412446a78fe2da5f223df",
            800_000: "c28415c1afb2c59e774c274da400a364ebe1cf6d7a94fd84709c2adb23cb25c2",
            1_000_000: "463f3a73e7070879110ef86de614a3dc6aeb98915909dc556dfa9397d331351c",
            1_500_000: "56fa358fb170f0b3cacae7d8881464ad420f20ebba3d6dc4ef956464fd28129b",
            2_000_000: "6803fff41be5f2897e3dae61a4af9d593c5c35044711d7c37c1bc0573be2fb5a",
        },
        expected_size=131_827_898,
    ),
    Experiment(
        key="run_lr1e4",
        run_id="dino_cls3_cheetah_run_seed1-paramgrad",
        task="cheetah_run",
        expected_fb_lr=1e-4,
        run_dir=Path("/mnt/data_7tb/fanfeng/controllable_agent_runs/20260817_cheetah_fb_stability_paramgrad/dino_cls3_cheetah_run_seed1"),
        checkpoint_dir=Path("/mnt/data_7tb/fanfeng/controallable_agent_ckpt/cheetah_fb_stability/dino_cls3_cheetah_run_seed1"),
        frames=(100_000, 200_000, 500_000, 800_000, 1_000_000, 1_500_000),
        checkpoint_hashes={
            100_000: "e99ece2bd984bd109599049d80ada795ff35b7c74a62e8f03ab699033d001821",
            200_000: "7c66ad3ba6e5f80a55b93eb59371d1053a7f40ab813a3fd0284f0a50b79ccc5d",
            500_000: "476bf845bd6633eb786cf718018b821254867890cba886df63c361fc75426128",
            800_000: "487e8c190bd6f22724b201fbf4e22a44e0351045d04ec0eae9f0789449c915be",
            1_000_000: "5a9680a2e6f22f4ee465d1ff98c47f110d237ed441a83cffc9bc7d251803c51e",
            1_500_000: "15b4793170f6e8cc0db14ac45eaa198c86869ca252848e57d7105bf4fceaf060",
        },
        expected_size=131_827_130,
    ),
)


_DINO_SEPARATE_CAMPAIGN = (
    "20260901_cheetah_cls3_sepFB_mlpLN_onlineEnc_idm_routes_s1"
)
_DINO_SEPARATE_GROUPS = (
    ("noidm", "noIDM", 0.0, "none", "legacy", None),
    ("idmf", "idmF_c01", 0.1, "forward_adapter", "static", 1e-4),
    (
        "idmb",
        "idmBAdapter_c01",
        0.1,
        "backward_adapter",
        "static",
        1e-4,
    ),
)
DINO_SEPARATE_EXPERIMENTS: Tuple[Experiment, ...] = tuple(
    Experiment(
        key=f"dino_sep_{group_key}_{task[len('cheetah_'):]}",
        run_id=(
            f"dino_cls3_{task}_seed1_sepFB_mlpLN_onlineEnc_{run_suffix}"
        ),
        task=task,
        expected_fb_lr=1e-4,
        run_dir=Path("/mnt/data_7tb/fanfeng/controllable_agent_runs")
        / _DINO_SEPARATE_CAMPAIGN
        / f"dino_cls3_{task}_seed1_sepFB_mlpLN_onlineEnc_{run_suffix}",
        checkpoint_dir=Path(
            "/mnt/data_nvme1/fanfeng/controllable_agent_ckpt"
        )
        / _DINO_SEPARATE_CAMPAIGN
        / f"dino_cls3_{task}_seed1_sepFB_mlpLN_onlineEnc_{run_suffix}",
        frames=(
            100_000,
            200_000,
            500_000,
            800_000,
            1_000_000,
            1_500_000,
            2_000_000,
        ),
        checkpoint_hashes={},
        expected_size=None,
        expected_dino_separate_fb_adapters=True,
        expected_dino_adapter_type="mlp_ln",
        expected_idm_coef=idm_coef,
        expected_idm_route=idm_route,
        expected_idm_encoder_mode=idm_mode,
        expected_idm_lr=idm_lr,
        expected_lr_f=1e-4,
        expected_lr_b=1e-4,
        # The common B/F geometry is valid for every group.  The historical
        # raw-gradient attribution assumes idm=0 and a shared encoder, so it is
        # deliberately omitted instead of silently changing its definition.
        analyze_gradient_attribution=False,
    )
    for group_key, run_suffix, idm_coef, idm_route, idm_mode, idm_lr in _DINO_SEPARATE_GROUPS
    for task in CNN_REWARD_TASKS
)

ALL_DINO_EXPERIMENTS: Tuple[Experiment, ...] = (
    *EXPERIMENTS,
    *DINO_SEPARATE_EXPERIMENTS,
)


# These are the available fully visual (goal_space=null) CNN-B lineages.  The
# newer seed-1 ``cheetah_speed`` runs are intentionally not mixed in: their B
# network consumes a privileged low-dimensional goal, so they do not answer
# whether a learned CNN visual representation develops the DINO pathology.
CNN_EXPERIMENTS: Tuple[Experiment, ...] = (
    Experiment(
        key="cnn_cheetah_walk",
        run_id="20260504_145139_seed2009_cheetah_walk_cnn_visual_b",
        task="cheetah_walk",
        expected_fb_lr=1e-4,
        run_dir=Path(
            "/mnt/data_7tb/fanfeng/controllable_agent_runs/"
            "20260504_145139_dmc_cnn_failed15_safe_cuda0_3_4_seed2009_"
            "cheetah_walk_cnn_cuda0"
        ),
        checkpoint_dir=Path(
            "/mnt/data_7tb/fanfeng/controallable_agent_ckpt/"
            "20260504_145139_dmc_cnn_failed15_safe_cuda0_3_4_seed2009_"
            "cheetah_walk_cnn_cuda0"
        ),
        frames=CNN_TARGET_FRAMES,
        checkpoint_hashes={},
        expected_size=2_662_104_282,
        obs_type="pixels",
        render_shape=(84, 84),
        seed=2009,
    ),
    Experiment(
        key="cnn_cheetah_run",
        run_id="20260422_155246_seed1_cheetah_run_cnn_visual_b",
        task="cheetah_run",
        expected_fb_lr=1e-4,
        run_dir=Path(
            "/mnt/data_7tb/fanfeng/controllable_agent_runs/"
            "20260422_155246_cheetah_run_cuda7_cnn"
        ),
        checkpoint_dir=Path(
            "/mnt/data_7tb/fanfeng/controallable_agent_ckpt/"
            "20260422_155246_cheetah_run_cuda7_cnn"
        ),
        frames=CNN_TARGET_FRAMES,
        checkpoint_hashes={},
        expected_size=2_662_104_282,
        obs_type="pixels",
        render_shape=(84, 84),
        seed=1,
        expected_use_cls=False,
    ),
    Experiment(
        key="cnn_cheetah_run_backward",
        run_id="20260504_121354_seed2009_cheetah_run_backward_cnn_visual_b",
        task="cheetah_run_backward",
        expected_fb_lr=1e-4,
        run_dir=Path(
            "/mnt/data_7tb/fanfeng/controllable_agent_runs/"
            "20260504_121354_dmc_cnn_parallel24_cuda0_3_4_seed2009_"
            "cheetah_run_backward_cnn_cuda0_slot0"
        ),
        checkpoint_dir=Path(
            "/mnt/data_7tb/fanfeng/controallable_agent_ckpt/"
            "20260504_121354_dmc_cnn_parallel24_cuda0_3_4_seed2009_"
            "cheetah_run_backward_cnn_cuda0_slot0"
        ),
        frames=CNN_TARGET_FRAMES,
        checkpoint_hashes={},
        expected_size=2_662_104_282,
        obs_type="pixels",
        render_shape=(84, 84),
        seed=2009,
    ),
    Experiment(
        key="cnn_cheetah_walk_backward",
        run_id="20260504_145139_seed2009_cheetah_walk_backward_cnn_visual_b",
        task="cheetah_walk_backward",
        expected_fb_lr=1e-4,
        run_dir=Path(
            "/mnt/data_7tb/fanfeng/controllable_agent_runs/"
            "20260504_145139_dmc_cnn_failed15_safe_cuda0_3_4_seed2009_"
            "cheetah_walk_backward_cnn_cuda4"
        ),
        checkpoint_dir=Path(
            "/mnt/data_7tb/fanfeng/controallable_agent_ckpt/"
            "20260504_145139_dmc_cnn_failed15_safe_cuda0_3_4_seed2009_"
            "cheetah_walk_backward_cnn_cuda4"
        ),
        frames=CNN_TARGET_FRAMES,
        checkpoint_hashes={},
        expected_size=2_662_104_282,
        obs_type="pixels",
        render_shape=(84, 84),
        seed=2009,
    ),
)


# Kept out of ``CNN_EXPERIMENTS`` so the established no-argument CNN profile
# continues to mean the historical shared-encoder analysis.  Once exact
# checkpoints exist, these lineages can be selected explicitly with
# ``--experiments``; their sizes are intentionally discovered and hashed at
# analysis time instead of predicting a serialization-dependent byte count.
CNN_SEPARATE_EXPERIMENTS: Tuple[Experiment, ...] = tuple(
    Experiment(
        key=f"cnn_sep_{task[len('cheetah_'):]}",
        run_id=f"cnn_{task}_seed1_sepFBcnn_onlineEnc",
        task=task,
        expected_fb_lr=1e-4,
        run_dir=Path(
            "/mnt/data_7tb/fanfeng/controllable_agent_runs/"
            "20260901_cheetah_cnn_separate_fb_onlineEnc_s1"
        )
        / f"cnn_{task}_seed1_sepFBcnn_onlineEnc",
        checkpoint_dir=Path(
            "/mnt/data_nvme1/fanfeng/controllable_agent_ckpt/"
            "20260901_cheetah_cnn_separate_fb_onlineEnc_s1"
        )
        / f"cnn_{task}_seed1_sepFBcnn_onlineEnc",
        frames=CNN_TARGET_FRAMES,
        checkpoint_hashes={},
        expected_size=None,
        obs_type="pixels",
        render_shape=(84, 84),
        seed=1,
        expected_pixel_separate_fb_encoders=True,
        expected_lr_f=1e-4,
        expected_lr_b=1e-4,
    )
    for task in CNN_REWARD_TASKS
)

ALL_CNN_EXPERIMENTS: Tuple[Experiment, ...] = (
    *CNN_EXPERIMENTS,
    *CNN_SEPARATE_EXPERIMENTS,
)


@dataclass
class ReplayData:
    obs: np.ndarray
    next_obs: np.ndarray
    action: np.ndarray
    discount: np.ndarray
    physics: np.ndarray
    source_reward: np.ndarray
    random_z: np.ndarray
    episode_id: np.ndarray
    step_in_episode: np.ndarray
    index_bank_row: np.ndarray
    metadata: Dict[str, Any]

    def validate(self, probe_size: int, inference_size: int) -> None:
        n = self.obs.shape[0]
        obs_type = str(self.metadata.get("obs_type", "dino"))
        if obs_type == "dino":
            expected_obs_shape = (n, FRAME_STACK * EMBED_DIM)
            expected_dtype = np.dtype(np.float32)
        elif obs_type == "pixels":
            render_shape = tuple(self.metadata.get("render_shape", (84, 84)))
            expected_obs_shape = (n, 3 * FRAME_STACK, *render_shape)
            expected_dtype = np.dtype(np.uint8)
        else:
            raise ValueError(f"unsupported replay obs_type: {obs_type!r}")
        if (
            self.obs.shape != self.next_obs.shape
            or self.obs.shape != expected_obs_shape
            or self.obs.dtype != expected_dtype
            or self.next_obs.dtype != expected_dtype
        ):
            raise ValueError(f"bad replay observation shapes: {self.obs.shape}, {self.next_obs.shape}")
        expected = {
            "action": (n, 6), "discount": (n, 1), "physics": (n, 18),
            "source_reward": (n, 1), "random_z": (n, Z_DIM),
            "episode_id": (n,), "step_in_episode": (n,), "index_bank_row": (n,),
        }
        for name, shape in expected.items():
            value = getattr(self, name)
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"bad replay field {name}: {value.shape}, finite={np.isfinite(value).all()}")
        if n % TRAIN_BATCH or not (0 < probe_size <= n) or not (0 < inference_size <= n):
            raise ValueError("replay size must be divisible by 1024 and cover probe/inference subsets")
        norms = np.linalg.norm(self.random_z.astype(np.float64), axis=1)
        if not np.allclose(norms, math.sqrt(Z_DIM), rtol=2e-6, atol=2e-6):
            raise AssertionError("fixed random z is not normalized to sqrt(z_dim)")


@dataclass
class CheckpointResult:
    experiment: Experiment
    frame: int
    checkpoint_path: Path
    checkpoint_sha256: str
    checkpoint_size: int
    global_step: int
    global_episode: int
    task_z: np.ndarray
    task_z_raw: np.ndarray
    covariance: np.ndarray
    eigvals: np.ndarray
    eigvecs: np.ndarray
    projectors: Dict[str, np.ndarray]
    metrics: List[Dict[str, Any]]
    extremes: Dict[str, np.ndarray]
    extreme_json: List[Dict[str, Any]]
    gradient_rows: List[Dict[str, Any]]


def parse_csv_list(text: str) -> Tuple[str, ...]:
    return tuple(item.strip() for item in text.split(",") if item.strip())


def parse_int_list(text: str) -> Tuple[int, ...]:
    return tuple(int(item.strip()) for item in text.split(",") if item.strip())


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def hash_arrays(items: Iterable[Tuple[str, np.ndarray]]) -> str:
    digest = hashlib.sha256()
    for name, value in items:
        array = np.asarray(value)
        if array.ndim == 0:
            # Preserve np.ascontiguousarray()'s historical scalar promotion.
            array = array.reshape(1)
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        # ``ndarray.tobytes()`` makes a second allocation as large as the
        # complete array.  CNN replay observations are multi-gigabyte and the
        # host can spend minutes faulting that anonymous copy under pressure.
        # A C-contiguous memoryview has identical bytes and lets hashlib
        # consume them with a fixed-size working set.
        if array.flags.c_contiguous:
            payload = memoryview(array).cast("B")
            for offset in range(
                0, len(payload), CNN_PIXEL_EPISODE_STREAM_CHUNK_BYTES
            ):
                digest.update(
                    payload[
                        offset : offset + CNN_PIXEL_EPISODE_STREAM_CHUNK_BYTES
                    ]
                )
        else:
            elements_per_chunk = max(
                CNN_PIXEL_EPISODE_STREAM_CHUNK_BYTES // max(array.itemsize, 1),
                1,
            )
            for chunk in np.nditer(
                array,
                flags=["external_loop", "buffered", "zerosize_ok"],
                op_flags=[["readonly"]],
                buffersize=elements_per_chunk,
                order="C",
            ):
                digest.update(memoryview(np.ascontiguousarray(chunk)).cast("B"))
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with tmp.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with tmp.open("w", encoding="utf-8") as stream:
            stream.write(value)
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


def atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        torch.save(dict(payload), tmp)
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV {path}")
    fields: List[str] = []
    seen: set[str] = set()
    for row in rows:
        for field in row:
            if field not in seen:
                seen.add(field)
                fields.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with tmp.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def _frame_indices(step: np.ndarray) -> np.ndarray:
    offsets = np.arange(-FRAME_STACK + 1, 1, dtype=np.int64)
    return np.maximum(step[:, None] + offsets[None, :], 0)


_CNN_NATIVE_TRACE_EVENTS = 0


def _cnn_native_trace(message: str) -> None:
    """Emit bounded opt-in breadcrumbs around long-running native calls."""
    global _CNN_NATIVE_TRACE_EVENTS
    if os.environ.get("CNN_ANALYSIS_NATIVE_TRACE") != "1":
        return
    limit = int(os.environ.get("CNN_ANALYSIS_NATIVE_TRACE_LIMIT", "400"))
    if _CNN_NATIVE_TRACE_EVENTS >= limit:
        return
    _CNN_NATIVE_TRACE_EVENTS += 1
    print(
        f"CNN_NATIVE_TRACE[{_CNN_NATIVE_TRACE_EVENTS}/{limit}]: {message}",
        file=sys.stderr,
        flush=True,
    )


def _render_pixel_stacks(
    physics: Any,
    states: np.ndarray,
    steps: np.ndarray,
    render_shape: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Render deterministic CHW frame stacks from serialized DMC physics."""
    _cnn_native_trace(f"render_pixel_stacks begin rows={len(steps)}")
    current_indices = _frame_indices(steps)
    next_indices = _frame_indices(steps + 1)
    required = np.unique(np.concatenate((current_indices.reshape(-1), next_indices.reshape(-1))))
    _cnn_native_trace(f"render_pixel_stacks indices ready required={len(required)}")
    rendered: Dict[int, np.ndarray] = {}
    height, width = render_shape
    for state_index in required:
        _cnn_native_trace(f"state={int(state_index)} reset/set begin")
        with physics.reset_context():
            physics.set_state(states[int(state_index)])
        _cnn_native_trace(f"state={int(state_index)} reset/set done; render begin")
        image = physics.render(height=height, width=width, camera_id=0)
        _cnn_native_trace(f"state={int(state_index)} render done")
        if image.shape != (height, width, 3) or image.dtype != np.uint8:
            raise ValueError(
                f"unexpected rendered image at state {state_index}: {image.shape}/{image.dtype}"
            )
        rendered[int(state_index)] = image.transpose(2, 0, 1).copy()

    def assemble(indices: np.ndarray) -> np.ndarray:
        rows = [
            np.concatenate([rendered[int(state_index)] for state_index in row], axis=0)
            for row in indices
        ]
        return np.stack(rows).astype(np.uint8, copy=False)

    return assemble(current_indices), assemble(next_indices)


def _make_cnn_pixel_render_env() -> Any:
    """Create the exact camera/physics source used by the CNN replay renderer."""
    return DmcReward("cheetah_walk")._env


def _cnn_pixel_render_context_contract(
    render_shape: Tuple[int, int],
    context_chunk_size: Optional[int] = None,
) -> Dict[str, Any]:
    height, width = (int(item) for item in render_shape)
    backend = os.environ.get("MUJOCO_GL", "egl").strip().lower()
    if backend not in {"egl", "osmesa"}:
        raise ValueError(
            f"CNN fixed pixel replay requires MUJOCO_GL=egl or osmesa, got {backend!r}"
        )
    if context_chunk_size is None:
        context_chunk_size = (
            CNN_OSMESA_RENDER_CONTEXT_CHUNK_SIZE
            if backend == "osmesa"
            else CNN_RENDER_CONTEXT_CHUNK_SIZE
        )
    raw_egl_device = os.environ.get("MUJOCO_EGL_DEVICE_ID")
    egl_device_id = (
        raw_egl_device.strip()
        if backend == "egl" and raw_egl_device and raw_egl_device.strip()
        else None
    )
    return {
        "mujoco_gl_backend": backend,
        "mujoco_egl_device_id": egl_device_id,
        "context_chunk_size_rows": int(context_chunk_size),
        "context_policy": f"{CNN_RENDER_CONTEXT_POLICY}; backend={backend}",
        "row_chunking": (
            "contiguous positional slices of each episode's selected rows; results "
            "are assigned back through the unchanged destination indices"
        ),
        "environment_factory": "DmcReward('cheetah_walk')._env",
        "state_update": "with physics.reset_context(): physics.set_state(states[state_index])",
        "render_call": {
            "height": height,
            "width": width,
            "camera_id": 0,
        },
        "frame_assembly": "CHW oldest-to-newest; obs[s] and next_obs[s+1]",
    }


def _render_pixel_stacks_context_chunked(
    states: np.ndarray,
    steps: np.ndarray,
    render_shape: Tuple[int, int],
    *,
    context_chunk_size: Optional[int] = None,
    env_factory: Optional[Callable[[], Any]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Render exact pixel stacks while bounding each MuJoCo renderer context.

    Chunking changes only context lifetime.  Episode-local row order, state indices,
    reset/set-state semantics, camera, image shape, and stack assembly are delegated
    unchanged to :func:`_render_pixel_stacks`.
    """
    steps = np.asarray(steps, dtype=np.int64)
    if steps.ndim != 1:
        raise ValueError(f"pixel render steps must be one-dimensional, got {steps.shape}")
    height, width = (int(item) for item in render_shape)
    if context_chunk_size is None:
        context_chunk_size = int(
            _cnn_pixel_render_context_contract((height, width))[
                "context_chunk_size_rows"
            ]
        )
    if context_chunk_size <= 0:
        raise ValueError("pixel render context chunk size must be positive")
    _cnn_native_trace(f"chunked render allocation begin rows={len(steps)}")
    current = np.empty((len(steps), 3 * FRAME_STACK, height, width), dtype=np.uint8)
    following = np.empty_like(current)
    _cnn_native_trace("chunked render allocation done")
    make_env = _make_cnn_pixel_render_env if env_factory is None else env_factory
    for start in range(0, len(steps), context_chunk_size):
        stop = min(start + context_chunk_size, len(steps))
        _cnn_native_trace(f"chunk [{start}:{stop}] environment creation begin")
        render_env = make_env()
        _cnn_native_trace(f"chunk [{start}:{stop}] environment creation done")
        chunk_current, chunk_following = _render_pixel_stacks(
            render_env.physics, states, steps[start:stop], (height, width)
        )
        _cnn_native_trace(f"chunk [{start}:{stop}] render returned; assignment begin")
        current[start:stop] = chunk_current
        following[start:stop] = chunk_following
        _cnn_native_trace(f"chunk [{start}:{stop}] assignment done")
        # Dropping the final owner bounds the MuJoCo render-context lifetime
        # to this chunk; no state or pixels are retained through the environment.
        del render_env
    return current, following


def _exorl_episode_filename_contract() -> Dict[str, Any]:
    return {
        "template": EXORL_EPISODE_FILENAME_TEMPLATE,
        "episode_id_format": "zero-padded decimal width 6",
        "fixed_length_suffix": 1000,
        "resolution": (
            "exact selected episode IDs only; per-episode is_file immediately before "
            "np.load; no glob/fallback"
        ),
    }


def _resolve_selected_exorl_episode_files(
    exorl_dir: Path, episode_ids: np.ndarray
) -> Dict[int, Path]:
    """Resolve only selected fixed-contract ExORL files, with no directory scan."""
    unique_ids = [int(value) for value in np.unique(np.asarray(episode_ids))]
    invalid_ids = [
        episode_id
        for episode_id in unique_ids
        if episode_id < 0 or episode_id > 999_999
    ]
    if invalid_ids:
        raise ValueError(
            "ExORL episode IDs must fit zero-padded width 6: "
            f"{invalid_ids}"
        )
    return {
        episode_id: exorl_dir
        / EXORL_EPISODE_FILENAME_TEMPLATE.format(episode_id=episode_id)
        for episode_id in unique_ids
    }


def _require_exact_exorl_episode_file(episode_id: int, path: Path) -> Path:
    """Fail closed on one exact file immediately before its payload is loaded."""
    if not path.is_file():
        raise FileNotFoundError(
            f"missing exact ExORL episode {episode_id} under fixed filename contract "
            f"{EXORL_EPISODE_FILENAME_TEMPLATE!r}: {path}"
        )
    return path


def _cnn_pixel_episode_part_cache_contract(
    output_dir: Path, obs_type: str
) -> Optional[Dict[str, Any]]:
    """Describe the pixels-only durable per-episode replay assembly cache."""
    if obs_type != "pixels":
        return None
    parts_dir = output_dir / CNN_PIXEL_EPISODE_PARTS_DIRNAME
    return {
        "part_schema_version": CNN_PIXEL_EPISODE_PART_SCHEMA_VERSION,
        "directory": str(parts_dir.resolve()),
        "filename_template": CNN_PIXEL_EPISODE_PART_FILENAME_TEMPLATE,
        "completion_marker": (
            "the exact final .npz filename after atomic rename; hidden .tmp.<pid> "
            "files are incomplete and ignored"
        ),
        "resume_validation": (
            "fail closed on source SHA-256, index selection, render context, exact "
            "array shape/dtype, and array-content SHA-256"
        ),
        "phase_a": "all parts stream-validated before aggregate pixel allocation",
        "phase_b": (
            "bounded streaming row assembly without full per-part materialization"
        ),
        "retention": "parts are retained after the aggregate fixed replay is validated",
    }


def _cnn_pixel_episode_part_path(output_dir: Path, episode_id: int) -> Path:
    episode_id = int(episode_id)
    if episode_id < 0 or episode_id > 999_999:
        raise ValueError(
            f"pixel replay part episode ID does not fit width 6: {episode_id}"
        )
    return output_dir / CNN_PIXEL_EPISODE_PARTS_DIRNAME / (
        CNN_PIXEL_EPISODE_PART_FILENAME_TEMPLATE.format(episode_id=episode_id)
    )


def _cnn_pixel_episode_part_array_contract(
    row_count: int, render_shape: Tuple[int, int]
) -> Dict[str, Dict[str, Any]]:
    height, width = (int(item) for item in render_shape)
    specs = {
        "episode_id": ((), np.dtype(np.int64)),
        "destinations": ((row_count,), np.dtype(np.int64)),
        "steps": ((row_count,), np.dtype(np.int64)),
        "index_bank_row": ((row_count,), np.dtype(np.int32)),
        "obs": ((row_count, 3 * FRAME_STACK, height, width), np.dtype(np.uint8)),
        "next_obs": ((row_count, 3 * FRAME_STACK, height, width), np.dtype(np.uint8)),
        "action": ((row_count, 6), np.dtype(np.float32)),
        "discount": ((row_count, 1), np.dtype(np.float32)),
        "physics": ((row_count, 18), np.dtype(np.float64)),
        "source_reward": ((row_count, 1), np.dtype(np.float32)),
    }
    return {
        name: {"shape": list(shape), "dtype": dtype.name}
        for name, (shape, dtype) in specs.items()
    }


def _cnn_pixel_episode_part_contract(
    args: argparse.Namespace,
    *,
    episode_id: int,
    episode_path: Path,
    source_episode_size: int,
    source_episode_sha256: str,
    destinations: np.ndarray,
    steps: np.ndarray,
    index_bank_rows: np.ndarray,
    pixel_render_context: Mapping[str, Any],
    source_index_bank_sha256: str,
) -> Dict[str, Any]:
    """Build the exact expected contract for one selected source episode."""
    destinations = np.asarray(destinations, dtype=np.int64)
    steps = np.asarray(steps, dtype=np.int64)
    index_bank_rows = np.asarray(index_bank_rows, dtype=np.int32)
    if not (destinations.shape == steps.shape == index_bank_rows.shape):
        raise ValueError(
            "pixel episode part selection arrays must have identical one-dimensional shapes"
        )
    if destinations.ndim != 1 or destinations.size == 0:
        raise ValueError("pixel episode part selection must contain at least one row")
    selection_sha256 = hash_arrays(
        (
            ("destinations", destinations),
            ("steps", steps),
            ("index_bank_row", index_bank_rows),
        )
    )
    render_shape = tuple(int(item) for item in args.render_shape)
    return {
        "analysis_schema_version": SCHEMA_VERSION,
        "part_schema_version": CNN_PIXEL_EPISODE_PART_SCHEMA_VERSION,
        "obs_type": "pixels",
        "episode_id": int(episode_id),
        "row_count": int(destinations.size),
        "selection_sha256": selection_sha256,
        "source_episode_path": str(episode_path.resolve()),
        "source_episode_size_bytes": int(source_episode_size),
        "source_episode_sha256": str(source_episode_sha256),
        "source_episode_filename_contract": _exorl_episode_filename_contract(),
        "source_index_bank": str(args.index_bank.resolve()),
        "source_index_bank_sha256": str(source_index_bank_sha256),
        "replay_seed": int(args.replay_seed),
        "replay_size": int(args.replay_size),
        "render_shape": list(render_shape),
        "frame_stack": FRAME_STACK,
        "pixel_render_context": dict(pixel_render_context),
        "transition_alignment": (
            "obs[s], action[s+1], next_obs/physics/reward[s+1]"
        ),
        "array_contract": _cnn_pixel_episode_part_array_contract(
            int(destinations.size), render_shape
        ),
    }


def _cnn_pixel_episode_part_arrays_digest(
    arrays: Mapping[str, np.ndarray]
) -> str:
    missing = [name for name in CNN_PIXEL_EPISODE_PART_ARRAY_FIELDS if name not in arrays]
    if missing:
        raise KeyError(f"pixel episode part arrays are missing fields: {missing}")
    return hash_arrays(
        (name, np.asarray(arrays[name]))
        for name in CNN_PIXEL_EPISODE_PART_ARRAY_FIELDS
    )


def _validate_cnn_pixel_episode_part_arrays(
    arrays: Mapping[str, np.ndarray],
    contract: Mapping[str, Any],
    *,
    expected_destinations: np.ndarray,
    expected_steps: np.ndarray,
    expected_index_bank_rows: np.ndarray,
) -> None:
    _cnn_native_trace("pixel part array validation begin")
    expected_names = set(CNN_PIXEL_EPISODE_PART_ARRAY_FIELDS)
    if set(arrays) != expected_names:
        raise ValueError(
            "pixel episode part array fields mismatch: "
            f"{sorted(arrays)} != {sorted(expected_names)}"
        )
    row_count = int(contract["row_count"])
    render_shape = tuple(int(item) for item in contract["render_shape"])
    expected_array_contract = _cnn_pixel_episode_part_array_contract(
        row_count, render_shape
    )
    if contract.get("array_contract") != expected_array_contract:
        raise ValueError("pixel episode part embedded array contract mismatch")
    for name, spec in expected_array_contract.items():
        value = np.asarray(arrays[name])
        if list(value.shape) != spec["shape"] or value.dtype.name != spec["dtype"]:
            raise ValueError(
                f"pixel episode part {name} shape/dtype mismatch: "
                f"{value.shape}/{value.dtype} != {tuple(spec['shape'])}/{spec['dtype']}"
            )
        _cnn_native_trace(f"pixel part {name} finite check begin")
        if not np.isfinite(value).all():
            raise ValueError(f"pixel episode part {name} contains non-finite values")
        _cnn_native_trace(f"pixel part {name} finite check done")
    if int(np.asarray(arrays["episode_id"]).item()) != int(contract["episode_id"]):
        raise ValueError("pixel episode part episode_id payload mismatch")
    selection_arrays = {
        "destinations": np.asarray(arrays["destinations"]),
        "steps": np.asarray(arrays["steps"]),
        "index_bank_row": np.asarray(arrays["index_bank_row"]),
    }
    expected_selection = {
        "destinations": np.asarray(expected_destinations, dtype=np.int64),
        "steps": np.asarray(expected_steps, dtype=np.int64),
        "index_bank_row": np.asarray(expected_index_bank_rows, dtype=np.int32),
    }
    for name in expected_selection:
        if not np.array_equal(selection_arrays[name], expected_selection[name]):
            raise ValueError(f"pixel episode part {name} selection mismatch")
    selection_sha256 = hash_arrays(
        (name, selection_arrays[name])
        for name in ("destinations", "steps", "index_bank_row")
    )
    if selection_sha256 != contract.get("selection_sha256"):
        raise ValueError("pixel episode part selection digest mismatch")
    _cnn_native_trace("pixel part temporal overlap check begin")
    if not np.array_equal(arrays["obs"][:, 3:], arrays["next_obs"][:, :6]):
        raise ValueError("pixel episode part obs/next_obs temporal stack overlap mismatch")
    _cnn_native_trace("pixel part temporal overlap check done; array validation done")


def _save_cnn_pixel_episode_part(
    path: Path,
    arrays: Mapping[str, np.ndarray],
    contract: Mapping[str, Any],
) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite pixel episode part: {path}")
    _validate_cnn_pixel_episode_part_arrays(
        arrays,
        contract,
        expected_destinations=arrays["destinations"],
        expected_steps=arrays["steps"],
        expected_index_bank_rows=arrays["index_bank_row"],
    )
    metadata = dict(contract)
    metadata["content_sha256"] = _cnn_pixel_episode_part_arrays_digest(arrays)
    atomic_npz(
        path,
        **{name: np.asarray(arrays[name]) for name in CNN_PIXEL_EPISODE_PART_ARRAY_FIELDS},
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
    )


def _load_cnn_pixel_episode_part(
    path: Path,
    expected_contract: Mapping[str, Any],
    *,
    expected_destinations: np.ndarray,
    expected_steps: np.ndarray,
    expected_index_bank_rows: np.ndarray,
) -> Dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"pixel episode part is missing: {path}")
    _cnn_native_trace(f"pixel part np.load open begin: {path.name}")
    with np.load(path, allow_pickle=False) as payload:
        _cnn_native_trace(f"pixel part np.load open done: {path.name}")
        expected_payload_fields = set(CNN_PIXEL_EPISODE_PART_ARRAY_FIELDS) | {"metadata"}
        if set(payload.files) != expected_payload_fields:
            raise ValueError(
                f"pixel episode part payload fields mismatch at {path}: {payload.files}"
            )
        arrays = {}
        for name in CNN_PIXEL_EPISODE_PART_ARRAY_FIELDS:
            _cnn_native_trace(f"pixel part field {name} load begin")
            arrays[name] = np.asarray(payload[name])
            _cnn_native_trace(f"pixel part field {name} load done")
        _cnn_native_trace("pixel part metadata load begin")
        metadata_array = np.asarray(payload["metadata"])
        _cnn_native_trace("pixel part metadata load done")
        if metadata_array.shape != () or metadata_array.dtype.kind not in {"U", "S"}:
            raise ValueError(f"pixel episode part metadata scalar is invalid: {path}")
        metadata = json.loads(str(metadata_array.item()))
    _cnn_native_trace(f"pixel part archive close done: {path.name}")
    if not isinstance(metadata, dict):
        raise ValueError(f"pixel episode part metadata is not an object: {path}")
    stored_contract = {
        key: value for key, value in metadata.items() if key != "content_sha256"
    }
    if stored_contract != dict(expected_contract):
        keys = sorted(set(stored_contract) | set(expected_contract))
        mismatches = {
            key: (stored_contract.get(key), expected_contract.get(key))
            for key in keys
            if stored_contract.get(key) != expected_contract.get(key)
        }
        raise ValueError(
            f"pixel episode part contract mismatch at {path}: {mismatches}"
        )
    _cnn_native_trace("pixel part embedded contract validation done")
    _validate_cnn_pixel_episode_part_arrays(
        arrays,
        expected_contract,
        expected_destinations=expected_destinations,
        expected_steps=expected_steps,
        expected_index_bank_rows=expected_index_bank_rows,
    )
    _cnn_native_trace("pixel part content digest begin")
    observed_digest = _cnn_pixel_episode_part_arrays_digest(arrays)
    _cnn_native_trace("pixel part content digest done")
    if observed_digest != metadata.get("content_sha256"):
        raise ValueError(
            f"pixel episode part content digest mismatch at {path}: "
            f"{observed_digest} != {metadata.get('content_sha256')}"
        )
    return arrays


def _read_cnn_pixel_part_npy_header(
    stream: Any, path: Path, field: str
) -> Tuple[Tuple[int, ...], bool, np.dtype]:
    try:
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(
                stream
            )
        elif version == (2, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(
                stream
            )
        else:
            raise ValueError(f"unsupported NPY version {version}")
    except Exception as error:
        raise ValueError(
            f"invalid NPY header for pixel episode part {field} at {path}: {error}"
        ) from error
    dtype = np.dtype(dtype)
    if dtype.hasobject:
        raise ValueError(
            f"object dtype is forbidden in pixel episode part {field} at {path}"
        )
    return tuple(int(item) for item in shape), bool(fortran_order), dtype


def _read_exact_cnn_pixel_part_bytes(
    stream: Any, size: int, path: Path, field: str
) -> bytes:
    output = bytearray()
    while len(output) < size:
        chunk = stream.read(min(CNN_PIXEL_EPISODE_STREAM_CHUNK_BYTES, size - len(output)))
        if not chunk:
            raise ValueError(
                f"truncated pixel episode part {field} at {path}: "
                f"{len(output)} != {size} bytes"
            )
        output.extend(chunk)
    if stream.read(1):
        raise ValueError(f"trailing data in pixel episode part {field} at {path}")
    return bytes(output)


def _read_cnn_pixel_part_metadata(
    archive: zipfile.ZipFile, path: Path
) -> Dict[str, Any]:
    with archive.open("metadata.npy", "r") as stream:
        shape, fortran_order, dtype = _read_cnn_pixel_part_npy_header(
            stream, path, "metadata"
        )
        if shape != () or fortran_order or dtype.kind not in {"U", "S"}:
            raise ValueError(
                f"pixel episode part metadata header mismatch at {path}: "
                f"{shape}/{dtype}/fortran={fortran_order}"
            )
        if dtype.itemsize > CNN_PIXEL_EPISODE_STREAM_CHUNK_BYTES:
            raise ValueError(f"pixel episode part metadata is oversized at {path}")
        raw = _read_exact_cnn_pixel_part_bytes(
            stream, dtype.itemsize, path, "metadata"
        )
    value = np.frombuffer(raw, dtype=dtype).reshape(()).item()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    try:
        metadata = json.loads(str(value))
    except Exception as error:
        raise ValueError(
            f"invalid pixel episode part metadata JSON at {path}: {error}"
        ) from error
    if not isinstance(metadata, dict):
        raise ValueError(f"pixel episode part metadata is not an object: {path}")
    return metadata


def _update_cnn_pixel_stack_slice_digest(
    digest: Any,
    chunk: bytes,
    chunk_start: int,
    *,
    row_bytes: int,
    slice_start: int,
    slice_stop: int,
) -> None:
    """Hash a fixed byte slice from every C-order row intersecting ``chunk``."""
    if not chunk:
        return
    chunk_stop = chunk_start + len(chunk)
    first_row = chunk_start // row_bytes
    last_row = (chunk_stop - 1) // row_bytes
    view = memoryview(chunk)
    for row in range(first_row, last_row + 1):
        absolute_start = row * row_bytes + slice_start
        absolute_stop = row * row_bytes + slice_stop
        overlap_start = max(chunk_start, absolute_start)
        overlap_stop = min(chunk_stop, absolute_stop)
        if overlap_start < overlap_stop:
            digest.update(
                view[overlap_start - chunk_start : overlap_stop - chunk_start]
            )


def _stream_validate_cnn_pixel_episode_part(
    path: Path,
    expected_contract: Mapping[str, Any],
    *,
    expected_destinations: np.ndarray,
    expected_steps: np.ndarray,
    expected_index_bank_rows: np.ndarray,
    target_arrays: Optional[Mapping[str, np.ndarray]] = None,
) -> None:
    """Validate a part without materializing pixel members.

    When ``target_arrays`` is supplied, the same bounded stream also assembles
    each decoded row into the aggregate ReplayData destinations.  At most one
    archive member and a roughly 1 MiB decode buffer are live at a time.
    """
    if not path.is_file():
        raise FileNotFoundError(f"pixel episode part is missing: {path}")
    expected_member_names = {
        *(f"{name}.npy" for name in CNN_PIXEL_EPISODE_PART_ARRAY_FIELDS),
        "metadata.npy",
    }
    try:
        archive = zipfile.ZipFile(path, "r")
    except Exception as error:
        raise ValueError(f"invalid pixel episode part ZIP at {path}: {error}") from error
    with archive:
        member_names = archive.namelist()
        if (
            len(member_names) != len(expected_member_names)
            or set(member_names) != expected_member_names
        ):
            raise ValueError(
                f"pixel episode part archive fields mismatch at {path}: {member_names}"
            )
        metadata = _read_cnn_pixel_part_metadata(archive, path)
        stored_contract = {
            key: value for key, value in metadata.items() if key != "content_sha256"
        }
        if stored_contract != dict(expected_contract):
            keys = sorted(set(stored_contract) | set(expected_contract))
            mismatches = {
                key: (stored_contract.get(key), expected_contract.get(key))
                for key in keys
                if stored_contract.get(key) != expected_contract.get(key)
            }
            raise ValueError(
                f"pixel episode part contract mismatch at {path}: {mismatches}"
            )

        expected_array_contract = expected_contract.get("array_contract")
        if not isinstance(expected_array_contract, dict):
            raise ValueError(f"pixel episode part lacks array contract at {path}")
        content_digest = hashlib.sha256()
        obs_overlap_digest = hashlib.sha256()
        next_overlap_digest = hashlib.sha256()
        captured: Dict[str, np.ndarray] = {}
        selection_fields = {
            "episode_id", "destinations", "steps", "index_bank_row"
        }
        destinations = np.asarray(expected_destinations, dtype=np.int64)
        assignable_fields = {
            "obs", "next_obs", "action", "discount", "physics", "source_reward"
        }
        for field in CNN_PIXEL_EPISODE_PART_ARRAY_FIELDS:
            spec = expected_array_contract.get(field)
            if not isinstance(spec, dict):
                raise ValueError(
                    f"pixel episode part array contract lacks {field}: {path}"
                )
            with archive.open(f"{field}.npy", "r") as stream:
                shape, fortran_order, dtype = _read_cnn_pixel_part_npy_header(
                    stream, path, field
                )
                if (
                    list(shape) != spec.get("shape")
                    or dtype.name != spec.get("dtype")
                    or fortran_order
                ):
                    raise ValueError(
                        f"pixel episode part {field} shape/dtype/order mismatch at "
                        f"{path}: {shape}/{dtype}/fortran={fortran_order}"
                    )
                content_digest.update(field.encode())
                content_digest.update(str(dtype).encode())
                # hash_arrays() normalizes a 0-D array through
                # np.ascontiguousarray(), which changes its hashed shape from
                # () to (1,) while leaving the scalar payload bytes unchanged.
                digest_shape = (1,) if shape == () else shape
                content_digest.update(
                    np.asarray(digest_shape, dtype=np.int64).tobytes()
                )
                element_count = int(np.prod(shape, dtype=np.int64)) if shape else 1
                data_size = element_count * dtype.itemsize
                capture = bytearray() if field in selection_fields else None
                finite_remainder = b""
                assignment_remainder = b""
                assigned_rows = 0
                target = None
                row_shape: Tuple[int, ...] = ()
                row_bytes = 0
                if target_arrays is not None and field in assignable_fields:
                    if field not in target_arrays:
                        raise KeyError(f"aggregate target is missing {field}")
                    target = np.asarray(target_arrays[field])
                    if target.dtype != dtype or target.shape[1:] != shape[1:]:
                        raise ValueError(
                            f"aggregate target contract mismatch for {field}: "
                            f"{target.shape}/{target.dtype}"
                        )
                    if shape[0] != len(destinations):
                        raise ValueError(
                            f"aggregate destination count mismatch for {field} at {path}"
                        )
                    row_shape = shape[1:]
                    row_bytes = int(np.prod(row_shape, dtype=np.int64)) * dtype.itemsize
                raw_offset = 0
                remaining = data_size
                while remaining:
                    chunk = stream.read(
                        min(CNN_PIXEL_EPISODE_STREAM_CHUNK_BYTES, remaining)
                    )
                    if not chunk:
                        raise ValueError(
                            f"truncated pixel episode part {field} at {path}: "
                            f"{raw_offset} != {data_size} bytes"
                        )
                    remaining -= len(chunk)
                    content_digest.update(chunk)
                    if capture is not None:
                        capture.extend(chunk)
                    if dtype.kind in {"f", "c"}:
                        finite_data = finite_remainder + chunk
                        aligned = len(finite_data) // dtype.itemsize * dtype.itemsize
                        if aligned:
                            values = np.frombuffer(finite_data[:aligned], dtype=dtype)
                            if not np.isfinite(values).all():
                                raise ValueError(
                                    f"pixel episode part {field} contains non-finite "
                                    f"values at {path}"
                                )
                        finite_remainder = finite_data[aligned:]
                    if field in {"obs", "next_obs"}:
                        pixel_plane_bytes = int(np.prod(shape[2:], dtype=np.int64))
                        full_row_bytes = int(np.prod(shape[1:], dtype=np.int64))
                        if field == "obs":
                            _update_cnn_pixel_stack_slice_digest(
                                obs_overlap_digest,
                                chunk,
                                raw_offset,
                                row_bytes=full_row_bytes,
                                slice_start=3 * pixel_plane_bytes,
                                slice_stop=9 * pixel_plane_bytes,
                            )
                        else:
                            _update_cnn_pixel_stack_slice_digest(
                                next_overlap_digest,
                                chunk,
                                raw_offset,
                                row_bytes=full_row_bytes,
                                slice_start=0,
                                slice_stop=6 * pixel_plane_bytes,
                            )
                    if target is not None:
                        assignment_data = assignment_remainder + chunk
                        complete_rows = len(assignment_data) // row_bytes
                        for local_offset in range(complete_rows):
                            byte_offset = local_offset * row_bytes
                            row = np.frombuffer(
                                assignment_data,
                                dtype=dtype,
                                count=int(np.prod(row_shape, dtype=np.int64)),
                                offset=byte_offset,
                            ).reshape(row_shape)
                            target[destinations[assigned_rows]] = row
                            assigned_rows += 1
                        assignment_remainder = assignment_data[complete_rows * row_bytes :]
                    raw_offset += len(chunk)
                if stream.read(1):
                    raise ValueError(
                        f"trailing data in pixel episode part {field} at {path}"
                    )
                if finite_remainder:
                    raise ValueError(
                        f"unaligned numeric data in pixel episode part {field} at {path}"
                    )
                if target is not None and (
                    assignment_remainder or assigned_rows != shape[0]
                ):
                    raise ValueError(
                        f"incomplete aggregate assembly for {field} at {path}"
                    )
                if capture is not None:
                    captured[field] = np.frombuffer(
                        bytes(capture), dtype=dtype
                    ).reshape(shape).copy()

    expected_selection = {
        "destinations": np.asarray(expected_destinations, dtype=np.int64),
        "steps": np.asarray(expected_steps, dtype=np.int64),
        "index_bank_row": np.asarray(expected_index_bank_rows, dtype=np.int32),
    }
    if int(captured["episode_id"].item()) != int(expected_contract["episode_id"]):
        raise ValueError(f"pixel episode part episode_id mismatch at {path}")
    for field, expected in expected_selection.items():
        if not np.array_equal(captured[field], expected):
            raise ValueError(f"pixel episode part {field} selection mismatch at {path}")
    selection_digest = hash_arrays(
        (field, captured[field])
        for field in ("destinations", "steps", "index_bank_row")
    )
    if selection_digest != expected_contract.get("selection_sha256"):
        raise ValueError(f"pixel episode part selection digest mismatch at {path}")
    if obs_overlap_digest.digest() != next_overlap_digest.digest():
        raise ValueError(f"pixel episode part stack overlap mismatch at {path}")
    observed_content_digest = content_digest.hexdigest()
    if observed_content_digest != metadata.get("content_sha256"):
        raise ValueError(
            f"pixel episode part content digest mismatch at {path}: "
            f"{observed_content_digest} != {metadata.get('content_sha256')}"
        )


@dataclass(frozen=True)
class CnnPixelEpisodePartPlan:
    number: int
    total: int
    episode_id: int
    source_path: Path
    part_path: Path
    destinations: np.ndarray
    steps: np.ndarray
    index_bank_rows: np.ndarray
    contract: Dict[str, Any]


@dataclass
class _CnnPixelEpisodePartAssemblyState:
    plan: CnnPixelEpisodePartPlan
    metadata: Dict[str, Any]
    destinations: np.ndarray
    content_digest: Any
    obs_overlap_digest: Any
    next_overlap_digest: Any


def _update_cnn_pixel_array_content_digest(
    digest: Any, field: str, value: np.ndarray
) -> None:
    """Apply exactly one ``hash_arrays`` field update to ``digest``."""
    array = np.ascontiguousarray(value)
    digest.update(field.encode())
    digest.update(str(array.dtype).encode())
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())


def _update_cnn_pixel_stream_content_preamble(
    digest: Any, field: str, shape: Tuple[int, ...], dtype: np.dtype
) -> None:
    """Apply the non-payload portion of one ``hash_arrays`` field update."""
    # np.ascontiguousarray() promotes a scalar from shape () to shape (1,).
    digest_shape = (1,) if shape == () else shape
    digest.update(field.encode())
    digest.update(str(dtype).encode())
    digest.update(np.asarray(digest_shape, dtype=np.int64).tobytes())


def _require_cnn_pixel_native_c_member_contract(
    path: Path,
    field: str,
    shape: Tuple[int, ...],
    fortran_order: bool,
    dtype: np.dtype,
    spec: Mapping[str, Any],
) -> None:
    expected_shape = tuple(int(item) for item in spec.get("shape", ()))
    try:
        expected_dtype = np.dtype(spec["dtype"])
    except Exception as error:
        raise ValueError(
            f"invalid array dtype contract for pixel episode part {field} at "
            f"{path}: {spec.get('dtype')!r}"
        ) from error
    if (
        shape != expected_shape
        or fortran_order
        or not dtype.isnative
        or dtype != expected_dtype
    ):
        raise ValueError(
            f"pixel episode part {field} native C-order header mismatch at "
            f"{path}: {shape}/{dtype}/fortran={fortran_order} != "
            f"{expected_shape}/{expected_dtype}/fortran=False"
        )


def _read_cnn_pixel_part_small_array(
    archive: zipfile.ZipFile,
    path: Path,
    field: str,
    spec: Mapping[str, Any],
) -> np.ndarray:
    with archive.open(f"{field}.npy", "r") as stream:
        shape, fortran_order, dtype = _read_cnn_pixel_part_npy_header(
            stream, path, field
        )
        _require_cnn_pixel_native_c_member_contract(
            path, field, shape, fortran_order, dtype, spec
        )
        element_count = math.prod(shape) if shape else 1
        data_size = element_count * dtype.itemsize
        raw = _read_exact_cnn_pixel_part_bytes(stream, data_size, path, field)
    return np.frombuffer(raw, dtype=dtype).reshape(shape).copy()


def _validate_cnn_pixel_episode_part_partition(
    replay_size: int, plans: Sequence[CnnPixelEpisodePartPlan]
) -> np.ndarray:
    """Return global-row owners after proving an exact destinations partition."""
    replay_size = int(replay_size)
    if replay_size <= 0 or not plans:
        raise ValueError("pixel episode part partition must be non-empty")
    owners = np.full(replay_size, -1, dtype=np.int32)
    seen_episodes: set[int] = set()
    seen_paths: set[Path] = set()
    for plan_index, plan in enumerate(plans):
        if plan.episode_id in seen_episodes:
            raise ValueError(
                f"duplicate pixel episode part plan for episode {plan.episode_id}"
            )
        if plan.part_path in seen_paths:
            raise ValueError(f"duplicate pixel episode part path: {plan.part_path}")
        seen_episodes.add(plan.episode_id)
        seen_paths.add(plan.part_path)
        destinations = np.asarray(plan.destinations)
        if destinations.dtype != np.dtype(np.int64) or destinations.ndim != 1:
            raise ValueError(
                f"pixel episode part destinations must be one-dimensional int64: "
                f"episode {plan.episode_id} has {destinations.shape}/{destinations.dtype}"
            )
        if destinations.size == 0:
            raise ValueError(
                f"pixel episode part destinations are empty: episode {plan.episode_id}"
            )
        if np.any(destinations[1:] <= destinations[:-1]):
            raise ValueError(
                f"pixel episode part destinations are not strictly increasing: "
                f"episode {plan.episode_id}"
            )
        if destinations[0] < 0 or destinations[-1] >= replay_size:
            raise ValueError(
                f"pixel episode part destination is out of range [0,{replay_size}): "
                f"episode {plan.episode_id}"
            )
        if not (
            destinations.shape == np.asarray(plan.steps).shape
            == np.asarray(plan.index_bank_rows).shape
        ):
            raise ValueError(
                f"pixel episode part selection shapes differ: episode {plan.episode_id}"
            )
        occupied = owners[destinations]
        if np.any(occupied != -1):
            duplicate_rows = destinations[occupied != -1]
            raise ValueError(
                "pixel episode part destinations contain duplicate global rows: "
                f"{duplicate_rows[:8].tolist()}"
            )
        owners[destinations] = plan_index
    missing = np.flatnonzero(owners == -1)
    if missing.size:
        raise ValueError(
            "pixel episode part destinations do not partition [0,N); missing "
            f"global rows {missing[:8].tolist()} (count={missing.size})"
        )
    return owners


def _read_cnn_pixel_part_row(
    stream: Any, size: int, path: Path, field: str, local_row: int
) -> bytes:
    output = bytearray()
    while len(output) < size:
        try:
            chunk = stream.read(
                min(CNN_PIXEL_EPISODE_STREAM_CHUNK_BYTES, size - len(output))
            )
        except Exception as error:
            raise ValueError(
                f"failed reading pixel episode part {field} row {local_row} at "
                f"{path}: {error}"
            ) from error
        if not chunk:
            raise ValueError(
                f"truncated pixel episode part {field} row {local_row} at {path}: "
                f"{len(output)} != {size} bytes"
            )
        output.extend(chunk)
    return bytes(output)


def _open_cnn_pixel_episode_part_assembly_state(
    plan: CnnPixelEpisodePartPlan,
) -> _CnnPixelEpisodePartAssemblyState:
    """Read one part's small assembly state without retaining its archive FD."""
    if not plan.part_path.is_file():
        raise FileNotFoundError(f"pixel episode part is missing: {plan.part_path}")
    try:
        archive = zipfile.ZipFile(plan.part_path, "r")
    except Exception as error:
        raise ValueError(
            f"invalid pixel episode part ZIP at {plan.part_path}: {error}"
        ) from error
    with archive:
        expected_members = {
            *(f"{field}.npy" for field in CNN_PIXEL_EPISODE_PART_ARRAY_FIELDS),
            "metadata.npy",
        }
        member_names = archive.namelist()
        if (
            len(member_names) != len(expected_members)
            or set(member_names) != expected_members
        ):
            raise ValueError(
                f"pixel episode part archive fields mismatch at {plan.part_path}: "
                f"{member_names}"
            )
        metadata = _read_cnn_pixel_part_metadata(archive, plan.part_path)
        stored_contract = {
            key: value for key, value in metadata.items() if key != "content_sha256"
        }
        if stored_contract != dict(plan.contract):
            keys = sorted(set(stored_contract) | set(plan.contract))
            mismatches = {
                key: (stored_contract.get(key), plan.contract.get(key))
                for key in keys
                if stored_contract.get(key) != plan.contract.get(key)
            }
            raise ValueError(
                f"pixel episode part contract mismatch at {plan.part_path}: "
                f"{mismatches}"
            )
        expected_digest = metadata.get("content_sha256")
        if (
            not isinstance(expected_digest, str)
            or len(expected_digest) != 64
            or any(
                character not in "0123456789abcdef"
                for character in expected_digest
            )
        ):
            raise ValueError(
                f"pixel episode part content digest is invalid at {plan.part_path}"
            )
        array_contract = plan.contract.get("array_contract")
        if not isinstance(array_contract, dict):
            raise ValueError(
                f"pixel episode part lacks array contract at {plan.part_path}"
            )
        small_fields = CNN_PIXEL_EPISODE_PART_ARRAY_FIELDS[:4]
        small_arrays = {
            field: _read_cnn_pixel_part_small_array(
                archive, plan.part_path, field, array_contract[field]
            )
            for field in small_fields
        }
    if int(small_arrays["episode_id"].item()) != int(plan.episode_id):
        raise ValueError(
            f"pixel episode part episode_id mismatch at {plan.part_path}"
        )
    expected_selection = {
        "destinations": np.asarray(plan.destinations, dtype=np.int64),
        "steps": np.asarray(plan.steps, dtype=np.int64),
        "index_bank_row": np.asarray(plan.index_bank_rows, dtype=np.int32),
    }
    for field, expected in expected_selection.items():
        if not np.array_equal(small_arrays[field], expected):
            raise ValueError(
                f"pixel episode part {field} selection mismatch at {plan.part_path}"
            )
    selection_digest = hash_arrays(
        (field, small_arrays[field])
        for field in ("destinations", "steps", "index_bank_row")
    )
    if selection_digest != plan.contract.get("selection_sha256"):
        raise ValueError(
            f"pixel episode part selection digest mismatch at {plan.part_path}"
        )
    content_digest = hashlib.sha256()
    for field in small_fields:
        _update_cnn_pixel_array_content_digest(
            content_digest, field, small_arrays[field]
        )
    return _CnnPixelEpisodePartAssemblyState(
        plan=plan,
        metadata=metadata,
        destinations=small_arrays["destinations"],
        content_digest=content_digest,
        obs_overlap_digest=hashlib.sha256(),
        next_overlap_digest=hashlib.sha256(),
    )


def _cnn_pixel_global_array_specs(
    args: argparse.Namespace,
) -> Dict[str, Dict[str, Any]]:
    render_shape = tuple(int(item) for item in args.render_shape)
    replay_size = int(args.replay_size)
    shapes = {
        "obs": (replay_size, 3 * FRAME_STACK, *render_shape),
        "next_obs": (replay_size, 3 * FRAME_STACK, *render_shape),
        "action": (replay_size, 6),
        "discount": (replay_size, 1),
        "physics": (replay_size, 18),
        "source_reward": (replay_size, 1),
    }
    dtypes = {
        "obs": np.dtype(np.uint8),
        "next_obs": np.dtype(np.uint8),
        "action": np.dtype(np.float32),
        "discount": np.dtype(np.float32),
        "physics": np.dtype(np.float64),
        "source_reward": np.dtype(np.float32),
    }
    return {
        field: {
            "filename": f"{field}.npy",
            "shape": list(shapes[field]),
            "dtype": dtypes[field].name,
        }
        for field in CNN_PIXEL_EPISODE_PART_ARRAY_FIELDS[4:]
    }


def _cnn_pixel_global_arrays_contract(
    args: argparse.Namespace,
    states: Sequence[_CnnPixelEpisodePartAssemblyState],
) -> Dict[str, Any]:
    parts = []
    for state in states:
        canonical_contract = json.dumps(
            state.plan.contract,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        parts.append(
            {
                "number": int(state.plan.number),
                "total": int(state.plan.total),
                "episode_id": int(state.plan.episode_id),
                "part_path": str(state.plan.part_path.resolve()),
                "part_contract_sha256": hashlib.sha256(
                    canonical_contract
                ).hexdigest(),
                "part_content_sha256": state.metadata["content_sha256"],
            }
        )
    return {
        "schema_version": CNN_PIXEL_GLOBAL_ARRAYS_SCHEMA_VERSION,
        "replay_size": int(args.replay_size),
        "render_shape": [int(item) for item in args.render_shape],
        "array_specs": _cnn_pixel_global_array_specs(args),
        "parts": parts,
        "assembly_policy": (
            "one part/member stream at a time; C-order rows written with pwrite "
            "to their exact global destination in output-dir staged NPY files"
        ),
    }


def _cnn_pixel_global_arrays_path(args: argparse.Namespace) -> Path:
    return Path(args.output_dir) / CNN_PIXEL_GLOBAL_ARRAYS_DIRNAME


def _write_cnn_pixel_global_npy_header(
    path: Path, spec: Mapping[str, Any]
) -> int:
    shape = tuple(int(item) for item in spec["shape"])
    dtype = np.dtype(spec["dtype"])
    if not dtype.isnative or dtype.hasobject:
        raise ValueError(f"invalid global pixel array dtype at {path}: {dtype}")
    with path.open("xb") as stream:
        np.lib.format.write_array_header_2_0(
            stream,
            {
                "descr": np.lib.format.dtype_to_descr(dtype),
                "fortran_order": False,
                "shape": shape,
            },
        )
        data_offset = stream.tell()
        total_size = data_offset + math.prod(shape) * dtype.itemsize
        stream.truncate(total_size)
        stream.flush()
        os.fsync(stream.fileno())
    file_descriptor = os.open(path, os.O_RDWR)
    try:
        if hasattr(os, "posix_fallocate"):
            os.posix_fallocate(file_descriptor, 0, total_size)
        else:
            os.ftruncate(file_descriptor, total_size)
    finally:
        os.close(file_descriptor)
    return data_offset


def _pwrite_all_cnn_pixel_global_row(
    file_descriptor: int,
    raw: bytes,
    offset: int,
    path: Path,
    field: str,
    global_row: int,
) -> None:
    view = memoryview(raw)
    written = 0
    while written < len(view):
        try:
            count = os.pwrite(
                file_descriptor, view[written:], int(offset) + written
            )
        except Exception as error:
            raise ValueError(
                f"failed writing global pixel array {field} row {global_row} "
                f"at {path}: {error}"
            ) from error
        if count <= 0:
            raise ValueError(
                f"short write for global pixel array {field} row {global_row} "
                f"at {path}: {written} != {len(view)} bytes"
            )
        written += int(count)


def _stream_hash_cnn_pixel_global_array(
    path: Path, field: str, spec: Mapping[str, Any]
) -> str:
    if path.is_symlink():
        raise ValueError(f"global pixel array must not be a symlink: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"global pixel array is missing: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        shape, fortran_order, dtype = _read_cnn_pixel_part_npy_header(
            stream, path, field
        )
        _require_cnn_pixel_native_c_member_contract(
            path, field, shape, fortran_order, dtype, spec
        )
        _update_cnn_pixel_stream_content_preamble(digest, field, shape, dtype)
        remaining = math.prod(shape) * dtype.itemsize
        finite_remainder = b""
        while remaining:
            chunk = stream.read(
                min(CNN_PIXEL_EPISODE_STREAM_CHUNK_BYTES, remaining)
            )
            if not chunk:
                raise ValueError(
                    f"truncated global pixel array {field} at {path}"
                )
            remaining -= len(chunk)
            digest.update(chunk)
            if dtype.kind in {"f", "c"}:
                finite_data = finite_remainder + chunk
                aligned = len(finite_data) // dtype.itemsize * dtype.itemsize
                if aligned and not np.isfinite(
                    np.frombuffer(finite_data[:aligned], dtype=dtype)
                ).all():
                    raise ValueError(
                        f"global pixel array {field} contains non-finite values "
                        f"at {path}"
                    )
                finite_remainder = finite_data[aligned:]
        if stream.read(1):
            raise ValueError(f"trailing data in global pixel array {field} at {path}")
        if finite_remainder:
            raise ValueError(
                f"unaligned numeric data in global pixel array {field} at {path}"
            )
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _load_cnn_pixel_global_arrays(
    directory: Path, contract: Mapping[str, Any]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    manifest_path = directory / CNN_PIXEL_GLOBAL_ARRAYS_MANIFEST
    specs = contract["array_specs"]
    expected_entries = {
        CNN_PIXEL_GLOBAL_ARRAYS_MANIFEST,
        *(spec["filename"] for spec in specs.values()),
    }
    observed_entries = {entry.name for entry in directory.iterdir()}
    if observed_entries != expected_entries:
        raise ValueError(
            f"global pixel array directory fields mismatch at {directory}: "
            f"{sorted(observed_entries)}"
        )
    if manifest_path.is_symlink():
        raise ValueError(
            f"global pixel array manifest must not be a symlink: {manifest_path}"
        )
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"global pixel array manifest is missing: {manifest_path}"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as error:
        raise ValueError(
            f"invalid global pixel array manifest at {manifest_path}: {error}"
        ) from error
    if set(manifest) != {"contract", "field_hashes"}:
        raise ValueError(
            f"global pixel array manifest fields mismatch at {manifest_path}"
        )
    if manifest["contract"] != dict(contract):
        raise ValueError(
            f"global pixel array contract mismatch at {manifest_path}"
        )
    field_hashes = manifest["field_hashes"]
    expected_fields = set(CNN_PIXEL_EPISODE_PART_ARRAY_FIELDS[4:])
    if set(field_hashes) != expected_fields:
        raise ValueError(
            f"global pixel array hash fields mismatch at {manifest_path}"
        )
    arrays: List[np.ndarray] = []
    for field in CNN_PIXEL_EPISODE_PART_ARRAY_FIELDS[4:]:
        spec = specs[field]
        path = directory / spec["filename"]
        observed_hash = _stream_hash_cnn_pixel_global_array(path, field, spec)
        if observed_hash != field_hashes[field]:
            raise ValueError(
                f"global pixel array content digest mismatch for {field} at "
                f"{path}: {observed_hash} != {field_hashes[field]}"
            )
        try:
            # Keep the validated cache file immutable while exposing a writable
            # private mapping to PyTorch.  Passing a read-only NumPy memmap
            # straight to ``torch.as_tensor(..., device="cuda")`` can enter a
            # pathological CUDA-driver path on heavily contended hosts.  An
            # anonymous ``np.copy`` is not a viable staging fallback there
            # either: first-touch allocation can itself stall.  Mode ``c`` is
            # file-backed copy-on-write, so reads remain zero-copy and any
            # accidental writes are private and can never modify the durable
            # global-array artifact whose digest was just verified above.
            array = np.load(path, mmap_mode="c", allow_pickle=False)
        except Exception as error:
            raise ValueError(
                f"cannot memory-map global pixel array {field} at {path}: {error}"
            ) from error
        if (
            tuple(array.shape) != tuple(spec["shape"])
            or array.dtype != np.dtype(spec["dtype"])
            or not array.flags.c_contiguous
            or not array.flags.writeable
        ):
            raise ValueError(
                f"memory-mapped global pixel array contract mismatch for {field} "
                f"at {path}"
            )
        arrays.append(array)
    return tuple(arrays)  # type: ignore[return-value]


def _prepare_cnn_pixel_episode_parts(
    args: argparse.Namespace,
    episode_id: np.ndarray,
    step: np.ndarray,
    order: np.ndarray,
    pixel_render_context: Mapping[str, Any],
    source_index_bank_sha256: str,
) -> List[CnnPixelEpisodePartPlan]:
    """Phase A: make every selected pixel episode independently durable."""
    unique = np.unique(episode_id)
    episode_files = _resolve_selected_exorl_episode_files(args.exorl_dir, unique)
    parts_dir = args.output_dir / CNN_PIXEL_EPISODE_PARTS_DIRNAME
    parts_dir.mkdir(parents=True, exist_ok=True)
    plans: List[CnnPixelEpisodePartPlan] = []
    print(
        "CNN fixed replay Phase A: validating/generating durable episode parts; "
        "no aggregate pixel replay is allocated",
        flush=True,
    )
    for number, episode_value in enumerate(unique, start=1):
        ep = int(episode_value)
        episode_path = _require_exact_exorl_episode_file(ep, episode_files[ep])
        destinations = np.flatnonzero(episode_id == ep).astype(np.int64, copy=False)
        steps = step[destinations].astype(np.int64)
        index_bank_rows = order[destinations].astype(np.int32)
        print(
            f"replay Phase A episode {number}/{len(unique)} id={ep}: "
            f"checking source and durable part ({len(steps)} rows)",
            flush=True,
        )
        source_stat = episode_path.stat()
        source_episode_sha256 = sha256_file(episode_path)
        part_path = _cnn_pixel_episode_part_path(args.output_dir, ep)
        contract = _cnn_pixel_episode_part_contract(
            args,
            episode_id=ep,
            episode_path=episode_path,
            source_episode_size=source_stat.st_size,
            source_episode_sha256=source_episode_sha256,
            destinations=destinations,
            steps=steps,
            index_bank_rows=index_bank_rows,
            pixel_render_context=pixel_render_context,
            source_index_bank_sha256=source_index_bank_sha256,
        )
        plan = CnnPixelEpisodePartPlan(
            number=number,
            total=len(unique),
            episode_id=ep,
            source_path=episode_path,
            part_path=part_path,
            destinations=destinations,
            steps=steps,
            index_bank_rows=index_bank_rows,
            contract=contract,
        )
        if part_path.is_file():
            if not args.resume:
                raise FileExistsError(
                    "pixel episode replay part already exists; pass --resume to "
                    f"validate/reuse it: {part_path}"
                )
            _stream_validate_cnn_pixel_episode_part(
                part_path,
                contract,
                expected_destinations=destinations,
                expected_steps=steps,
                expected_index_bank_rows=index_bank_rows,
            )
            print(
                f"replay Phase A episode {number}/{len(unique)} id={ep}: "
                f"resumed stream-validated part {part_path}",
                flush=True,
            )
            plans.append(plan)
            continue

        print(
            f"replay Phase A episode {number}/{len(unique)} id={ep}: rendering",
            flush=True,
        )
        _cnn_native_trace(
            f"Phase A episode {number}/{len(unique)} id={ep} np.load begin "
            f"rows={len(steps)}"
        )
        with np.load(episode_path, allow_pickle=False) as payload:
            emb = np.asarray(payload["dino_emb"], dtype=np.float32)
            actions = np.asarray(payload["action"], dtype=np.float32)
            discounts = np.asarray(payload["discount"], dtype=np.float32)
            states = np.asarray(payload["physics"], dtype=np.float64)
            rewards = np.asarray(payload["reward"], dtype=np.float32)
            token = str(payload["dino_token"].item())
            model = str(payload["dino_model"].item())
            processor = str(payload["dino_processor"].item())
            image_size = int(payload["image_size"].item())
            camera_id = int(payload["camera_id"].item())
        _cnn_native_trace(
            f"Phase A episode {number}/{len(unique)} id={ep} np.load done"
        )
        if (
            token != "cls"
            or model != "facebook/dinov2-base"
            or processor != "facebook/dinov2-base"
            or image_size != 224
            or camera_id != 0
            or emb.shape[1] != EMBED_DIM
        ):
            raise ValueError(f"unexpected DINO payload in {episode_path}")
        next_index = steps + 1
        if np.any(next_index >= emb.shape[0]):
            raise IndexError(f"transition beyond episode {ep}")
        current_pixels, next_pixels = _render_pixel_stacks_context_chunked(
            states,
            steps,
            tuple(args.render_shape),
            context_chunk_size=int(pixel_render_context["context_chunk_size_rows"]),
        )
        part_arrays = {
            "episode_id": np.asarray(ep, dtype=np.int64),
            "destinations": destinations,
            "steps": steps,
            "index_bank_row": index_bank_rows,
            "obs": current_pixels,
            "next_obs": next_pixels,
            "action": np.asarray(actions[next_index], dtype=np.float32),
            "discount": np.asarray(
                0.99 * discounts[next_index].reshape(-1, 1), dtype=np.float32
            ),
            "physics": np.asarray(states[next_index], dtype=np.float64),
            "source_reward": np.asarray(
                rewards[next_index].reshape(-1, 1), dtype=np.float32
            ),
        }
        source_stat_after_render = episode_path.stat()
        source_sha256_after_render = sha256_file(episode_path)
        if (
            source_stat_after_render.st_size != contract["source_episode_size_bytes"]
            or source_sha256_after_render != contract["source_episode_sha256"]
        ):
            raise RuntimeError(
                f"source episode changed while rendering; refusing part commit: "
                f"{episode_path}"
            )
        _save_cnn_pixel_episode_part(part_path, part_arrays, contract)
        _stream_validate_cnn_pixel_episode_part(
            part_path,
            contract,
            expected_destinations=destinations,
            expected_steps=steps,
            expected_index_bank_rows=index_bank_rows,
        )
        print(
            f"replay Phase A episode {number}/{len(unique)} id={ep}: rendered, "
            f"atomically committed, and stream-validated {part_path}",
            flush=True,
        )
        plans.append(plan)
    print(
        f"CNN fixed replay Phase A complete: {len(plans)}/{len(unique)} durable "
        "episode parts validated",
        flush=True,
    )
    return plans


def _assemble_cnn_pixel_episode_parts(
    args: argparse.Namespace,
    plans: Sequence[CnnPixelEpisodePartPlan],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Atomically assemble globally ordered, file-backed CNN replay arrays.

    A single compressed part member is open at a time.  Its rows are decoded
    into bounded buffers and ``pwrite`` places each row at its proved global
    destination in an output-directory staged NPY file.  This avoids both the
    pathological complete-member ``np.load`` path and multi-gigabyte anonymous
    array first-touch under host memory pressure.
    """
    _validate_cnn_pixel_episode_part_partition(args.replay_size, plans)
    states = [
        _open_cnn_pixel_episode_part_assembly_state(plan) for plan in plans
    ]
    contract = _cnn_pixel_global_arrays_contract(args, states)
    final_directory = _cnn_pixel_global_arrays_path(args)
    if final_directory.is_symlink():
        raise ValueError(
            f"global pixel array directory must not be a symlink: {final_directory}"
        )
    if final_directory.exists():
        if not final_directory.is_dir():
            raise ValueError(
                f"global pixel array path is not a directory: {final_directory}"
            )
        print(
            "CNN fixed replay Phase B: validating and resuming file-backed "
            f"global arrays at {final_directory}",
            flush=True,
        )
        return _load_cnn_pixel_global_arrays(final_directory, contract)

    final_directory.parent.mkdir(parents=True, exist_ok=True)
    staging_directory = final_directory.with_name(
        f".{final_directory.name}.tmp.{os.getpid()}"
    )
    if staging_directory.exists() or staging_directory.is_symlink():
        raise FileExistsError(
            f"refusing to reuse global pixel array staging path: "
            f"{staging_directory}"
        )
    staging_directory.mkdir(mode=0o700)
    print(
        "CNN fixed replay Phase B: assembling file-backed global NPY arrays in "
        f"{staging_directory}; one part/member stream and bounded row at a time",
        flush=True,
    )
    try:
        specs = contract["array_specs"]
        field_hashes: Dict[str, str] = {}
        streamed_fields = tuple(CNN_PIXEL_EPISODE_PART_ARRAY_FIELDS[4:])
        for field_index, field in enumerate(streamed_fields, start=1):
            global_spec = specs[field]
            target_path = staging_directory / global_spec["filename"]
            data_offset = _write_cnn_pixel_global_npy_header(
                target_path, global_spec
            )
            target_descriptor = os.open(target_path, os.O_RDWR)
            print(
                f"replay Phase B field {field_index}/{len(streamed_fields)} "
                f"{field}: streaming {len(states)} parts to file-backed global rows",
                flush=True,
            )
            try:
                for state in states:
                    local_spec = state.plan.contract["array_contract"][field]
                    try:
                        archive = zipfile.ZipFile(state.plan.part_path, "r")
                    except Exception as error:
                        raise ValueError(
                            f"invalid pixel episode part ZIP at "
                            f"{state.plan.part_path}: {error}"
                        ) from error
                    with archive:
                        try:
                            stream_context = archive.open(f"{field}.npy", "r")
                        except Exception as error:
                            raise ValueError(
                                f"cannot open pixel episode part {field} at "
                                f"{state.plan.part_path}: {error}"
                            ) from error
                        with stream_context as stream:
                            shape, fortran_order, dtype = (
                                _read_cnn_pixel_part_npy_header(
                                    stream, state.plan.part_path, field
                                )
                            )
                            _require_cnn_pixel_native_c_member_contract(
                                state.plan.part_path,
                                field,
                                shape,
                                fortran_order,
                                dtype,
                                local_spec,
                            )
                            if (
                                not shape
                                or shape[0] != len(state.destinations)
                                or tuple(global_spec["shape"])
                                != (int(args.replay_size), *shape[1:])
                                or np.dtype(global_spec["dtype"]) != dtype
                            ):
                                raise ValueError(
                                    f"aggregate target contract mismatch for "
                                    f"{field} at {state.plan.part_path}"
                                )
                            row_shape = shape[1:]
                            bytes_per_row = math.prod(row_shape) * dtype.itemsize
                            if bytes_per_row <= 0:
                                raise ValueError(
                                    f"pixel episode part {field} has an empty row "
                                    f"at {state.plan.part_path}"
                                )
                            _update_cnn_pixel_stream_content_preamble(
                                state.content_digest, field, shape, dtype
                            )
                            for local_row, destination in enumerate(
                                state.destinations
                            ):
                                raw = _read_cnn_pixel_part_row(
                                    stream,
                                    bytes_per_row,
                                    state.plan.part_path,
                                    field,
                                    local_row,
                                )
                                state.content_digest.update(raw)
                                row = np.frombuffer(raw, dtype=dtype).reshape(
                                    row_shape
                                )
                                if dtype.kind in {"f", "c"} and not np.isfinite(
                                    row
                                ).all():
                                    raise ValueError(
                                        f"pixel episode part {field} contains "
                                        f"non-finite values at "
                                        f"{state.plan.part_path}, row {local_row}"
                                    )
                                if field in {"obs", "next_obs"}:
                                    if len(row_shape) != 3 or row_shape[0] != 9:
                                        raise ValueError(
                                            f"unexpected pixel stack row shape for "
                                            f"{field}: {row_shape}"
                                        )
                                    plane_bytes = (
                                        math.prod(row_shape[1:]) * dtype.itemsize
                                    )
                                    if field == "obs":
                                        state.obs_overlap_digest.update(
                                            raw[3 * plane_bytes : 9 * plane_bytes]
                                        )
                                    else:
                                        state.next_overlap_digest.update(
                                            raw[: 6 * plane_bytes]
                                        )
                                global_row = int(destination)
                                _pwrite_all_cnn_pixel_global_row(
                                    target_descriptor,
                                    raw,
                                    data_offset + global_row * bytes_per_row,
                                    target_path,
                                    field,
                                    global_row,
                                )
                            try:
                                trailing = stream.read(1)
                            except Exception as error:
                                raise ValueError(
                                    f"failed finalizing pixel episode part {field} "
                                    f"at {state.plan.part_path}: {error}"
                                ) from error
                            if trailing:
                                raise ValueError(
                                    f"trailing data in pixel episode part {field} "
                                    f"at {state.plan.part_path}"
                                )
                    if (
                        state.plan.number % 50 == 0
                        or state.plan.number == state.plan.total
                    ):
                        print(
                            f"replay Phase B field {field_index}/"
                            f"{len(streamed_fields)} {field}: wrote part "
                            f"{state.plan.number}/{state.plan.total}",
                            flush=True,
                        )
                os.fsync(target_descriptor)
            finally:
                os.close(target_descriptor)
            field_hashes[field] = _stream_hash_cnn_pixel_global_array(
                target_path, field, global_spec
            )
            print(
                f"replay Phase B field {field_index}/{len(streamed_fields)} "
                f"{field}: staged {args.replay_size} global rows "
                f"({field_hashes[field][:16]})",
                flush=True,
            )

        for state in states:
            if state.obs_overlap_digest.digest() != state.next_overlap_digest.digest():
                raise ValueError(
                    f"pixel episode part stack overlap mismatch at "
                    f"{state.plan.part_path}"
                )
            observed_digest = state.content_digest.hexdigest()
            expected_digest = state.metadata["content_sha256"]
            if observed_digest != expected_digest:
                raise ValueError(
                    f"pixel episode part content digest mismatch at "
                    f"{state.plan.part_path}: {observed_digest} != {expected_digest}"
                )
            if state.plan.number % 50 == 0 or state.plan.number == state.plan.total:
                print(
                    f"replay Phase B content digests validated "
                    f"{state.plan.number}/{state.plan.total}",
                    flush=True,
                )

        manifest_path = staging_directory / CNN_PIXEL_GLOBAL_ARRAYS_MANIFEST
        atomic_json(
            manifest_path,
            {"contract": contract, "field_hashes": field_hashes},
        )
        with manifest_path.open("rb") as manifest_stream:
            os.fsync(manifest_stream.fileno())
        _fsync_directory(staging_directory)
        if final_directory.exists() or final_directory.is_symlink():
            raise FileExistsError(
                f"global pixel array directory appeared during assembly: "
                f"{final_directory}"
            )
        staging_directory.replace(final_directory)
        _fsync_directory(final_directory.parent)
    finally:
        if staging_directory.exists():
            shutil.rmtree(staging_directory, ignore_errors=True)
    print(
        f"CNN fixed replay Phase B complete: {len(plans)} episode parts assembled "
        f"into {final_directory}",
        flush=True,
    )
    return _load_cnn_pixel_global_arrays(final_directory, contract)


def _build_replay_legacy(args: argparse.Namespace, obs_type: str = "dino") -> ReplayData:
    _cnn_native_trace("index bank np.load begin")
    with np.load(args.index_bank, allow_pickle=False) as bank:
        episode_all = np.asarray(bank["episode_id"], dtype=np.int32)
        step_all = np.asarray(bank["step_in_episode"], dtype=np.int32)
        bank_metadata = json.loads(str(bank["metadata"].item()))
    _cnn_native_trace(f"index bank loaded rows={len(episode_all)}")
    if args.replay_size > episode_all.size:
        raise ValueError(f"requested {args.replay_size} rows from {episode_all.size}-row index bank")
    _cnn_native_trace("replay permutation begin")
    order = np.random.RandomState(args.replay_seed).permutation(episode_all.size)[: args.replay_size]
    _cnn_native_trace("replay permutation done")
    episode_id = episode_all[order]
    step = step_all[order]
    pixel_render_context: Optional[Dict[str, Any]] = None
    if obs_type == "dino":
        obs = np.empty((args.replay_size, FRAME_STACK * EMBED_DIM), dtype=np.float32)
    elif obs_type == "pixels":
        render_shape = tuple(int(item) for item in args.render_shape)
        if len(render_shape) != 2 or min(render_shape) <= 0:
            raise ValueError(f"invalid render shape: {render_shape}")
        pixel_render_context = _cnn_pixel_render_context_contract(render_shape)
        _cnn_native_trace("full pixel replay allocation begin")
        obs = np.empty(
            (args.replay_size, 3 * FRAME_STACK, *render_shape), dtype=np.uint8
        )
        _cnn_native_trace("full pixel replay allocation done")
    else:
        raise ValueError(f"unsupported replay obs_type: {obs_type!r}")
    next_obs = np.empty_like(obs)
    action = np.empty((args.replay_size, 6), dtype=np.float32)
    discount = np.empty((args.replay_size, 1), dtype=np.float32)
    physics = np.empty((args.replay_size, 18), dtype=np.float64)
    source_reward = np.empty((args.replay_size, 1), dtype=np.float32)
    pixel_episode_part_cache: Optional[Dict[str, Any]] = None
    source_index_bank_sha256: Optional[str] = None
    if obs_type == "pixels":
        pixel_episode_part_cache = _cnn_pixel_episode_part_cache_contract(
            args.output_dir, obs_type
        )
        assert pixel_episode_part_cache is not None
        Path(pixel_episode_part_cache["directory"]).mkdir(parents=True, exist_ok=True)
        source_index_bank_sha256 = sha256_file(args.index_bank)

    unique = np.unique(episode_id)
    _cnn_native_trace(f"episode IDs ready unique={len(unique)}")
    episode_files = _resolve_selected_exorl_episode_files(args.exorl_dir, unique)
    for number, episode_value in enumerate(unique, start=1):
        ep = int(episode_value)
        _cnn_native_trace(f"episode {number}/{len(unique)} id={ep} file check begin")
        episode_path = _require_exact_exorl_episode_file(ep, episode_files[ep])
        destinations = np.flatnonzero(episode_id == ep)
        steps = step[destinations].astype(np.int64)
        index_bank_rows = order[destinations].astype(np.int32)
        pixel_part_path: Optional[Path] = None
        pixel_part_contract: Optional[Dict[str, Any]] = None
        if obs_type == "pixels":
            assert pixel_render_context is not None
            assert source_index_bank_sha256 is not None
            print(
                f"replay episode {number}/{len(unique)} id={ep}: "
                f"checking source and durable part ({len(steps)} rows)",
                flush=True,
            )
            source_stat = episode_path.stat()
            source_episode_sha256 = sha256_file(episode_path)
            pixel_part_path = _cnn_pixel_episode_part_path(args.output_dir, ep)
            pixel_part_contract = _cnn_pixel_episode_part_contract(
                args,
                episode_id=ep,
                episode_path=episode_path,
                source_episode_size=source_stat.st_size,
                source_episode_sha256=source_episode_sha256,
                destinations=destinations,
                steps=steps,
                index_bank_rows=index_bank_rows,
                pixel_render_context=pixel_render_context,
                source_index_bank_sha256=source_index_bank_sha256,
            )
            if pixel_part_path.is_file():
                if not args.resume:
                    raise FileExistsError(
                        "pixel episode replay part already exists; pass --resume to "
                        f"validate/reuse it: {pixel_part_path}"
                    )
                _cnn_native_trace(
                    f"episode {number}/{len(unique)} id={ep} part validation load begin"
                )
                part_arrays = _load_cnn_pixel_episode_part(
                    pixel_part_path,
                    pixel_part_contract,
                    expected_destinations=destinations,
                    expected_steps=steps,
                    expected_index_bank_rows=index_bank_rows,
                )
                _cnn_native_trace(
                    f"episode {number}/{len(unique)} id={ep} part validation load done"
                )
                _cnn_native_trace(
                    f"episode {number}/{len(unique)} id={ep} global obs assignment begin"
                )
                obs[destinations] = part_arrays["obs"]
                _cnn_native_trace(
                    f"episode {number}/{len(unique)} id={ep} global obs assignment done; "
                    "next_obs assignment begin"
                )
                next_obs[destinations] = part_arrays["next_obs"]
                _cnn_native_trace(
                    f"episode {number}/{len(unique)} id={ep} global next_obs assignment done; "
                    "scalar assignments begin"
                )
                action[destinations] = part_arrays["action"]
                discount[destinations] = part_arrays["discount"]
                physics[destinations] = part_arrays["physics"]
                source_reward[destinations] = part_arrays["source_reward"]
                _cnn_native_trace(
                    f"episode {number}/{len(unique)} id={ep} global scalar assignments done"
                )
                print(
                    f"replay episode {number}/{len(unique)} id={ep}: resumed "
                    f"validated part {pixel_part_path}",
                    flush=True,
                )
                continue
            print(
                f"replay episode {number}/{len(unique)} id={ep}: rendering",
                flush=True,
            )
        _cnn_native_trace(
            f"episode {number}/{len(unique)} id={ep} np.load begin rows={len(steps)}"
        )
        with np.load(episode_path, allow_pickle=False) as payload:
            emb = np.asarray(payload["dino_emb"], dtype=np.float32)
            actions = np.asarray(payload["action"], dtype=np.float32)
            discounts = np.asarray(payload["discount"], dtype=np.float32)
            states = np.asarray(payload["physics"], dtype=np.float64)
            rewards = np.asarray(payload["reward"], dtype=np.float32)
            token = str(payload["dino_token"].item())
            model = str(payload["dino_model"].item())
            processor = str(payload["dino_processor"].item())
            image_size = int(payload["image_size"].item())
            camera_id = int(payload["camera_id"].item())
        _cnn_native_trace(f"episode {number}/{len(unique)} id={ep} np.load done")
        if (token != "cls" or model != "facebook/dinov2-base"
                or processor != "facebook/dinov2-base" or image_size != 224
                or camera_id != 0 or emb.shape[1] != EMBED_DIM):
            raise ValueError(f"unexpected DINO payload in {episode_path}")
        next_index = steps + 1
        if np.any(next_index >= emb.shape[0]):
            raise IndexError(f"transition beyond episode {ep}")
        if obs_type == "dino":
            obs[destinations] = emb[_frame_indices(steps)].reshape(len(destinations), -1)
            next_obs[destinations] = emb[_frame_indices(next_index)].reshape(
                len(destinations), -1
            )
            action[destinations] = actions[next_index]
            discount[destinations] = 0.99 * discounts[next_index].reshape(-1, 1)
            physics[destinations] = states[next_index]
            source_reward[destinations] = rewards[next_index].reshape(-1, 1)
        else:
            assert pixel_render_context is not None
            assert pixel_part_path is not None
            assert pixel_part_contract is not None
            current_pixels, next_pixels = _render_pixel_stacks_context_chunked(
                states,
                steps,
                tuple(args.render_shape),
                context_chunk_size=int(
                    pixel_render_context["context_chunk_size_rows"]
                ),
            )
            part_arrays = {
                "episode_id": np.asarray(ep, dtype=np.int64),
                "destinations": destinations.astype(np.int64, copy=False),
                "steps": steps,
                "index_bank_row": index_bank_rows,
                "obs": current_pixels,
                "next_obs": next_pixels,
                "action": np.asarray(actions[next_index], dtype=np.float32),
                "discount": np.asarray(
                    0.99 * discounts[next_index].reshape(-1, 1), dtype=np.float32
                ),
                "physics": np.asarray(states[next_index], dtype=np.float64),
                "source_reward": np.asarray(
                    rewards[next_index].reshape(-1, 1), dtype=np.float32
                ),
            }
            source_stat_after_render = episode_path.stat()
            source_sha256_after_render = sha256_file(episode_path)
            if (
                source_stat_after_render.st_size
                != pixel_part_contract["source_episode_size_bytes"]
                or source_sha256_after_render
                != pixel_part_contract["source_episode_sha256"]
            ):
                raise RuntimeError(
                    f"source episode changed while rendering; refusing part commit: "
                    f"{episode_path}"
                )
            _save_cnn_pixel_episode_part(
                pixel_part_path, part_arrays, pixel_part_contract
            )
            print(
                f"replay episode {number}/{len(unique)} id={ep}: atomically "
                "committed part; reloading for validation",
                flush=True,
            )
            # Fill the aggregate only from the durable, reloaded artifact.  This
            # validates the exact resume path immediately after every commit.
            _cnn_native_trace(
                f"episode {number}/{len(unique)} id={ep} committed part validation load begin"
            )
            part_arrays = _load_cnn_pixel_episode_part(
                pixel_part_path,
                pixel_part_contract,
                expected_destinations=destinations,
                expected_steps=steps,
                expected_index_bank_rows=index_bank_rows,
            )
            _cnn_native_trace(
                f"episode {number}/{len(unique)} id={ep} committed part validation load done; "
                "global obs assignment begin"
            )
            obs[destinations] = part_arrays["obs"]
            _cnn_native_trace(
                f"episode {number}/{len(unique)} id={ep} global obs assignment done; "
                "next_obs assignment begin"
            )
            next_obs[destinations] = part_arrays["next_obs"]
            _cnn_native_trace(
                f"episode {number}/{len(unique)} id={ep} global next_obs assignment done; "
                "scalar assignments begin"
            )
            action[destinations] = part_arrays["action"]
            discount[destinations] = part_arrays["discount"]
            physics[destinations] = part_arrays["physics"]
            source_reward[destinations] = part_arrays["source_reward"]
            _cnn_native_trace(
                f"episode {number}/{len(unique)} id={ep} global scalar assignments done"
            )
            print(
                f"replay episode {number}/{len(unique)} id={ep}: rendered and "
                f"atomically committed {pixel_part_path}",
                flush=True,
            )
        if obs_type == "dino" and (number % 50 == 0 or number == len(unique)):
            print(f"replay: loaded {number}/{len(unique)} source episodes", flush=True)

    rng = np.random.RandomState(args.z_seed)
    random_z = rng.standard_normal((args.replay_size, Z_DIM)).astype(np.float32)
    random_z /= np.linalg.norm(random_z.astype(np.float64), axis=1, keepdims=True).astype(np.float32)
    random_z *= np.float32(math.sqrt(Z_DIM))
    metadata: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "obs_type": obs_type,
        "render_shape": list(args.render_shape) if obs_type == "pixels" else None,
        "role": "deterministic_exorl_rnd_proxy_not_historical_replay",
        "historical_replay_serialized": False,
        "temporal_semantics": "ExORL one-source-step stack; historical run used action_repeat=2",
        "transition_alignment": "obs[s], action[s+1], next_obs/physics/reward[s+1]",
        "source_dir": str(args.exorl_dir.resolve()),
        "source_episode_filename_contract": _exorl_episode_filename_contract(),
        "source_reward_task": "cheetah_run",
        "dino_cache_contract": {
            "model": "facebook/dinov2-base", "processor": "facebook/dinov2-base",
            "token": "cls", "image_size": 224, "camera_id": 0,
            "model_revision_or_weights_hash_available": False,
        },
        "observation_construction": (
            "cached_dinov2_base_cls3"
            if obs_type == "dino"
            else "DMC physics.set_state then camera0 84x84 render; CHW oldest-to-newest stack"
        ),
        "pixel_render_context": pixel_render_context,
        "source_index_bank": str(args.index_bank.resolve()),
        "source_index_bank_sha256": (
            source_index_bank_sha256
            if source_index_bank_sha256 is not None
            else sha256_file(args.index_bank)
        ),
        "source_index_bank_metadata": bank_metadata,
        "replay_seed": args.replay_seed,
        "z_seed": args.z_seed,
        "size": args.replay_size,
        "probe_size": args.probe_size,
        "inference_size": args.inference_size,
        "pairing": f"fixed reordered rows in consecutive {TRAIN_BATCH}x{TRAIN_BATCH} chunks",
    }
    if pixel_episode_part_cache is not None:
        metadata["pixel_episode_part_cache"] = {
            **pixel_episode_part_cache,
            "selected_episode_count": int(len(unique)),
            "selected_episode_ids_sha256": hash_arrays(
                (("episode_id", unique.astype(np.int32, copy=False)),)
            ),
        }
    replay = ReplayData(obs, next_obs, action, discount, physics, source_reward, random_z,
                        episode_id, step, order.astype(np.int32), metadata)
    replay.validate(args.probe_size, args.inference_size)
    metadata["content_sha256"] = hash_arrays(
        (name, getattr(replay, name)) for name in (
            "obs", "next_obs", "action", "discount", "physics", "source_reward",
            "random_z", "episode_id", "step_in_episode", "index_bank_row"
        )
    )
    return replay


def build_replay(args: argparse.Namespace, obs_type: str = "dino") -> ReplayData:
    # Preserve the established DINO path byte-for-byte.  Only CNN pixels use
    # the durable two-phase episode assembly below.
    if obs_type != "pixels":
        return _build_replay_legacy(args, obs_type=obs_type)

    _cnn_native_trace("two-phase index bank np.load begin")
    with np.load(args.index_bank, allow_pickle=False) as bank:
        episode_all = np.asarray(bank["episode_id"], dtype=np.int32)
        step_all = np.asarray(bank["step_in_episode"], dtype=np.int32)
        bank_metadata = json.loads(str(bank["metadata"].item()))
    _cnn_native_trace(f"two-phase index bank loaded rows={len(episode_all)}")
    if args.replay_size > episode_all.size:
        raise ValueError(
            f"requested {args.replay_size} rows from {episode_all.size}-row index bank"
        )
    order = np.random.RandomState(args.replay_seed).permutation(episode_all.size)[
        : args.replay_size
    ]
    episode_id = episode_all[order]
    step = step_all[order]
    render_shape = tuple(int(item) for item in args.render_shape)
    if len(render_shape) != 2 or min(render_shape) <= 0:
        raise ValueError(f"invalid render shape: {render_shape}")
    pixel_render_context = _cnn_pixel_render_context_contract(render_shape)
    source_index_bank_sha256 = sha256_file(args.index_bank)
    pixel_episode_part_cache = _cnn_pixel_episode_part_cache_contract(
        args.output_dir, obs_type
    )
    assert pixel_episode_part_cache is not None

    plans = _prepare_cnn_pixel_episode_parts(
        args,
        episode_id,
        step,
        order,
        pixel_render_context,
        source_index_bank_sha256,
    )
    obs, next_obs, action, discount, physics, source_reward = (
        _assemble_cnn_pixel_episode_parts(args, plans)
    )

    rng = np.random.RandomState(args.z_seed)
    random_z = rng.standard_normal((args.replay_size, Z_DIM)).astype(np.float32)
    random_z /= np.linalg.norm(
        random_z.astype(np.float64), axis=1, keepdims=True
    ).astype(np.float32)
    random_z *= np.float32(math.sqrt(Z_DIM))
    unique = np.unique(episode_id)
    metadata: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "obs_type": obs_type,
        "render_shape": list(args.render_shape),
        "role": "deterministic_exorl_rnd_proxy_not_historical_replay",
        "historical_replay_serialized": False,
        "temporal_semantics": (
            "ExORL one-source-step stack; historical run used action_repeat=2"
        ),
        "transition_alignment": (
            "obs[s], action[s+1], next_obs/physics/reward[s+1]"
        ),
        "source_dir": str(args.exorl_dir.resolve()),
        "source_episode_filename_contract": _exorl_episode_filename_contract(),
        "source_reward_task": "cheetah_run",
        "dino_cache_contract": {
            "model": "facebook/dinov2-base",
            "processor": "facebook/dinov2-base",
            "token": "cls",
            "image_size": 224,
            "camera_id": 0,
            "model_revision_or_weights_hash_available": False,
        },
        "observation_construction": (
            "DMC physics.set_state then camera0 84x84 render; CHW "
            "oldest-to-newest stack"
        ),
        "pixel_render_context": pixel_render_context,
        "source_index_bank": str(args.index_bank.resolve()),
        "source_index_bank_sha256": source_index_bank_sha256,
        "source_index_bank_metadata": bank_metadata,
        "replay_seed": args.replay_seed,
        "z_seed": args.z_seed,
        "size": args.replay_size,
        "probe_size": args.probe_size,
        "inference_size": args.inference_size,
        "pairing": (
            f"fixed reordered rows in consecutive {TRAIN_BATCH}x{TRAIN_BATCH} chunks"
        ),
        "pixel_episode_part_cache": {
            **pixel_episode_part_cache,
            "selected_episode_count": int(len(unique)),
            "selected_episode_ids_sha256": hash_arrays(
                (("episode_id", unique.astype(np.int32, copy=False)),)
            ),
        },
    }
    replay = ReplayData(
        obs,
        next_obs,
        action,
        discount,
        physics,
        source_reward,
        random_z,
        episode_id,
        step,
        order.astype(np.int32),
        metadata,
    )
    replay.validate(args.probe_size, args.inference_size)
    metadata["content_sha256"] = hash_arrays(
        (name, getattr(replay, name))
        for name in (
            "obs",
            "next_obs",
            "action",
            "discount",
            "physics",
            "source_reward",
            "random_z",
            "episode_id",
            "step_in_episode",
            "index_bank_row",
        )
    )
    return replay


def save_replay_cache(path: Path, replay: ReplayData) -> None:
    replay.validate(
        int(replay.metadata["probe_size"]), int(replay.metadata["inference_size"])
    )
    atomic_npz(
        path,
        obs=replay.obs,
        next_obs=replay.next_obs,
        action=replay.action,
        discount=replay.discount,
        physics=replay.physics,
        source_reward=replay.source_reward,
        random_z=replay.random_z,
        episode_id=replay.episode_id,
        step_in_episode=replay.step_in_episode,
        index_bank_row=replay.index_bank_row,
        metadata=np.asarray(json.dumps(replay.metadata, sort_keys=True)),
    )


def _read_replay_cache_metadata(
    archive: zipfile.ZipFile, path: Path
) -> Dict[str, Any]:
    try:
        stream_context = archive.open("metadata.npy", "r")
    except Exception as error:
        raise ValueError(
            f"cannot open fixed replay metadata at {path}: {error}"
        ) from error
    with stream_context as stream:
        shape, fortran_order, dtype = _read_cnn_pixel_part_npy_header(
            stream, path, "metadata"
        )
        if shape != () or fortran_order or dtype.kind not in {"U", "S"}:
            raise ValueError(
                f"fixed replay metadata header mismatch at {path}: "
                f"{shape}/{dtype}/fortran={fortran_order}"
            )
        if dtype.itemsize > CNN_PIXEL_EPISODE_STREAM_CHUNK_BYTES:
            raise ValueError(f"fixed replay metadata is oversized at {path}")
        raw = _read_exact_cnn_pixel_part_bytes(
            stream, dtype.itemsize, path, "metadata"
        )
    value = np.frombuffer(raw, dtype=dtype).reshape(()).item()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    try:
        metadata = json.loads(str(value))
    except Exception as error:
        raise ValueError(
            f"invalid fixed replay metadata JSON at {path}: {error}"
        ) from error
    if not isinstance(metadata, dict):
        raise ValueError(f"fixed replay metadata is not an object at {path}")
    return metadata


def _stream_validate_replay_cache_against_replay(
    path: Path, replay: ReplayData
) -> None:
    """Validate a compressed replay artifact without materializing its arrays."""
    if not path.is_file():
        raise FileNotFoundError(f"fixed replay cache is missing: {path}")
    expected_members = {
        *(f"{field}.npy" for field in REPLAY_CACHE_ARRAY_FIELDS),
        "metadata.npy",
    }
    try:
        archive = zipfile.ZipFile(path, "r")
    except Exception as error:
        raise ValueError(f"invalid fixed replay ZIP at {path}: {error}") from error
    digest = hashlib.sha256()
    with archive:
        member_names = archive.namelist()
        if (
            len(member_names) != len(expected_members)
            or set(member_names) != expected_members
        ):
            raise ValueError(
                f"fixed replay cache fields mismatch at {path}: {member_names}"
            )
        metadata = _read_replay_cache_metadata(archive, path)
        if metadata != replay.metadata:
            keys = sorted(set(metadata) | set(replay.metadata))
            mismatches = {
                key: (metadata.get(key), replay.metadata.get(key))
                for key in keys
                if metadata.get(key) != replay.metadata.get(key)
            }
            raise ValueError(
                f"fixed replay cache metadata contract mismatch at {path}: "
                f"{mismatches}"
            )
        for field in REPLAY_CACHE_ARRAY_FIELDS:
            expected = np.asarray(getattr(replay, field))
            if not expected.flags.c_contiguous:
                raise ValueError(
                    f"expected replay field {field} is not C-contiguous"
                )
            try:
                stream_context = archive.open(f"{field}.npy", "r")
            except Exception as error:
                raise ValueError(
                    f"cannot open fixed replay field {field} at {path}: {error}"
                ) from error
            with stream_context as stream:
                shape, fortran_order, dtype = _read_cnn_pixel_part_npy_header(
                    stream, path, field
                )
                if (
                    shape != expected.shape
                    or fortran_order
                    or not dtype.isnative
                    or dtype != expected.dtype
                ):
                    raise ValueError(
                        f"fixed replay field contract mismatch for {field} at "
                        f"{path}: {shape}/{dtype}/fortran={fortran_order} != "
                        f"{expected.shape}/{expected.dtype}/fortran=False"
                    )
                _update_cnn_pixel_stream_content_preamble(
                    digest, field, shape, dtype
                )
                expected_payload = memoryview(expected).cast("B")
                offset = 0
                finite_remainder = b""
                while offset < len(expected_payload):
                    requested = min(
                        CNN_PIXEL_EPISODE_STREAM_CHUNK_BYTES,
                        len(expected_payload) - offset,
                    )
                    try:
                        chunk = stream.read(requested)
                    except Exception as error:
                        raise ValueError(
                            f"failed reading fixed replay field {field} at "
                            f"{path}: {error}"
                        ) from error
                    if not chunk:
                        raise ValueError(
                            f"truncated fixed replay field {field} at {path}: "
                            f"{offset} != {len(expected_payload)} bytes"
                        )
                    expected_chunk = expected_payload[
                        offset : offset + len(chunk)
                    ]
                    if memoryview(chunk) != expected_chunk:
                        raise ValueError(
                            f"fixed replay field content mismatch for {field} at "
                            f"{path}, byte offset {offset}"
                        )
                    digest.update(chunk)
                    if dtype.kind in {"f", "c"}:
                        finite_data = finite_remainder + chunk
                        aligned = (
                            len(finite_data) // dtype.itemsize * dtype.itemsize
                        )
                        if aligned and not np.isfinite(
                            np.frombuffer(finite_data[:aligned], dtype=dtype)
                        ).all():
                            raise ValueError(
                                f"fixed replay field {field} contains non-finite "
                                f"values at {path}"
                            )
                        finite_remainder = finite_data[aligned:]
                    offset += len(chunk)
                try:
                    trailing = stream.read(1)
                except Exception as error:
                    raise ValueError(
                        f"failed finalizing fixed replay field {field} at "
                        f"{path}: {error}"
                    ) from error
                if trailing:
                    raise ValueError(
                        f"trailing data in fixed replay field {field} at {path}"
                    )
                if finite_remainder:
                    raise ValueError(
                        f"unaligned numeric data in fixed replay field {field} "
                        f"at {path}"
                    )
    observed_digest = digest.hexdigest()
    expected_digest = replay.metadata.get("content_sha256")
    if observed_digest != expected_digest:
        raise ValueError(
            f"fixed replay cache content digest mismatch at {path}: "
            f"{observed_digest} != {expected_digest}"
        )


def load_replay_cache(path: Path, args: argparse.Namespace, obs_type: str) -> ReplayData:
    with np.load(path, allow_pickle=False) as payload:
        replay = ReplayData(
            obs=np.asarray(payload["obs"]),
            next_obs=np.asarray(payload["next_obs"]),
            action=np.asarray(payload["action"], dtype=np.float32),
            discount=np.asarray(payload["discount"], dtype=np.float32),
            physics=np.asarray(payload["physics"], dtype=np.float64),
            source_reward=np.asarray(payload["source_reward"], dtype=np.float32),
            random_z=np.asarray(payload["random_z"], dtype=np.float32),
            episode_id=np.asarray(payload["episode_id"], dtype=np.int32),
            step_in_episode=np.asarray(payload["step_in_episode"], dtype=np.int32),
            index_bank_row=np.asarray(payload["index_bank_row"], dtype=np.int32),
            metadata=json.loads(str(payload["metadata"].item())),
        )
    expected = {
        "schema_version": SCHEMA_VERSION,
        "obs_type": obs_type,
        "render_shape": list(args.render_shape) if obs_type == "pixels" else None,
        "pixel_render_context": (
            _cnn_pixel_render_context_contract(tuple(args.render_shape))
            if obs_type == "pixels"
            else None
        ),
        "source_dir": str(args.exorl_dir.resolve()),
        "source_episode_filename_contract": _exorl_episode_filename_contract(),
        "source_index_bank": str(args.index_bank.resolve()),
        "source_index_bank_sha256": sha256_file(args.index_bank),
        "replay_seed": args.replay_seed,
        "z_seed": args.z_seed,
        "size": args.replay_size,
        "probe_size": args.probe_size,
        "inference_size": args.inference_size,
    }
    if obs_type == "pixels":
        unique_episode_ids = np.unique(replay.episode_id).astype(np.int32, copy=False)
        pixel_episode_part_cache = _cnn_pixel_episode_part_cache_contract(
            args.output_dir, obs_type
        )
        assert pixel_episode_part_cache is not None
        expected["pixel_episode_part_cache"] = {
            **pixel_episode_part_cache,
            "selected_episode_count": int(len(unique_episode_ids)),
            "selected_episode_ids_sha256": hash_arrays(
                (("episode_id", unique_episode_ids),)
            ),
        }
    mismatches = {
        key: (replay.metadata.get(key), value)
        for key, value in expected.items()
        if replay.metadata.get(key) != value
    }
    if mismatches:
        raise ValueError(f"fixed replay cache contract mismatch: {mismatches}")
    replay.validate(args.probe_size, args.inference_size)
    observed_digest = hash_arrays(
        (name, getattr(replay, name))
        for name in (
            "obs", "next_obs", "action", "discount", "physics", "source_reward",
            "random_z", "episode_id", "step_in_episode", "index_bank_row",
        )
    )
    if observed_digest != replay.metadata.get("content_sha256"):
        raise ValueError(
            "fixed replay cache digest mismatch: "
            f"{observed_digest} != {replay.metadata.get('content_sha256')}"
        )
    return replay


def load_or_build_replay_cache(
    args: argparse.Namespace, obs_type: str
) -> Tuple[ReplayData, Path]:
    path = args.output_dir / "fixed_rendered_replay.npz"
    if path.is_file():
        if not args.resume:
            raise FileExistsError(
                f"fixed replay cache already exists; pass --resume to validate/reuse it: {path}"
            )
        if obs_type == "pixels":
            # Reconstruct only the deterministic small fields and resume the
            # validated global NPY memmaps.  The compressed publication
            # artifact is checked member-by-member against those arrays; it is
            # never materialized into another 2x multi-gigabyte anonymous copy.
            replay = build_replay(args, obs_type=obs_type)
            _stream_validate_replay_cache_against_replay(path, replay)
            print(
                f"fixed replay: stream-validated {path} and resumed file-backed "
                f"global arrays ({replay.metadata['content_sha256'][:16]})",
                flush=True,
            )
            return replay, path
        replay = load_replay_cache(path, args, obs_type)
        print(
            f"fixed replay: resumed {path} ({replay.metadata['content_sha256'][:16]})",
            flush=True,
        )
        return replay, path
    replay = build_replay(args, obs_type=obs_type)
    save_replay_cache(path, replay)
    if obs_type == "pixels":
        # Prove the durable compressed artifact byte-for-byte while retaining
        # the validated global mmap arrays for analysis.
        _stream_validate_replay_cache_against_replay(path, replay)
        print(
            f"fixed replay: saved and stream-validated {path}; using file-backed "
            f"global arrays ({replay.metadata['content_sha256'][:16]})",
            flush=True,
        )
        return replay, path
    # Preserve the established DINO path: reload the just-written artifact.
    replay = load_replay_cache(path, args, obs_type)
    print(
        f"fixed replay: saved {path} ({replay.metadata['content_sha256'][:16]})",
        flush=True,
    )
    return replay, path


def compute_task_rewards(replay: ReplayData, task: str) -> np.ndarray:
    rewarder = DmcReward(task)
    rewards = np.empty((replay.physics.shape[0], 1), dtype=np.float32)
    for index, state in enumerate(replay.physics):
        rewards[index, 0] = rewarder.from_physics(state)
        if (index + 1) % 4096 == 0:
            print(f"task reward {task}: {index + 1}/{len(rewards)}", flush=True)
    if not np.isfinite(rewards).all():
        raise FloatingPointError(f"non-finite {task} proxy rewards")
    return rewards


def save_replay_manifest(path: Path, replay: ReplayData, task_rewards: Mapping[str, np.ndarray]) -> None:
    arrays: Dict[str, np.ndarray] = {
        "episode_id": replay.episode_id,
        "step_in_episode": replay.step_in_episode,
        "index_bank_row": replay.index_bank_row,
        "random_z": replay.random_z,
        "discount": replay.discount,
        "source_reward": replay.source_reward,
        "metadata": np.asarray(json.dumps(replay.metadata, sort_keys=True)),
    }
    arrays.update({f"proxy_reward_{task}": reward for task, reward in task_rewards.items()})
    atomic_npz(path, **arrays)


def prepare_agent(agent: Any, device: str) -> None:
    if hasattr(agent, "cfg"):
        agent.cfg.device = device
    for name in (
        "forward_encoder",
        "backward_encoder",
        "backward_encoder_target",
        "idm_head",
    ):
        if not hasattr(agent, name):
            setattr(agent, name, None)
    for value in vars(agent).values():
        if isinstance(value, nn.Module):
            value.to(device)
            value.eval()
    # CNN comparisons use an exact fixed rendered batch.  Training-time
    # random-shift augmentation would otherwise draw a different crop for
    # every checkpoint and even for the manual/repository task-z cross-check.
    # The identity crop is the deterministic, unshifted observation path.
    if str(getattr(getattr(agent, "cfg", None), "obs_type", "")) == "pixels":
        agent.aug = nn.Identity().to(device)
    agent.train(False)


def install_optimizer_guards(agent: Any) -> None:
    def forbidden_step(_self: Any, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError("optimizer.step() is forbidden in offline checkpoint analysis")
    for value in vars(agent).values():
        if isinstance(value, torch.optim.Optimizer):
            value.step = types.MethodType(forbidden_step, value)


def expected_loaded_agent_contract(
    experiment: Experiment,
    expected_obs_shape: Tuple[int, ...],
) -> Dict[str, Any]:
    return {
        "obs_type": experiment.obs_type, "obs_shape": expected_obs_shape,
        "action_shape": (6,), "z_dim": Z_DIM, "batch_size": TRAIN_BATCH,
        "num_inference_steps": DEFAULT_INFERENCE_SIZE, "goal_space": experiment.goal_space,
        "norm_z": True, "q_loss": False,
        "idm_coef": experiment.expected_idm_coef,
        "future_ratio": 0.0, "rand_weight": False, "boltzmann": False,
        "use_cls": experiment.expected_use_cls,
        "dino_separate_backward_adapter": False,
        "dino_separate_fb_adapters": (
            experiment.expected_dino_separate_fb_adapters
        ),
        "dino_adapter_type": experiment.expected_dino_adapter_type,
        "pixel_separate_fb_encoders": (
            experiment.expected_pixel_separate_fb_encoders
        ),
        "idm_route": experiment.expected_idm_route,
        "idm_encoder_mode": experiment.expected_idm_encoder_mode,
        "idm_lr": experiment.expected_idm_lr,
    }


def loaded_fb_optimizer_lrs(agent: Any) -> Dict[str, Tuple[float, ...]]:
    """Return the serialized FB optimizer topology without stepping it."""
    separate = bool(
        getattr(agent.cfg, "pixel_separate_fb_encoders", False)
        or getattr(agent.cfg, "dino_separate_fb_adapters", False)
    )
    if separate:
        optimizers = {
            "forward_fb_opt": getattr(agent, "forward_fb_opt", None),
            "backward_fb_opt": getattr(agent, "backward_fb_opt", None),
        }
        missing = [name for name, optimizer in optimizers.items() if optimizer is None]
        if missing:
            raise ValueError(
                f"separate-F/B checkpoint lacks branch optimizers: {missing}"
            )
        if getattr(agent, "fb_opt", None) is not None:
            raise ValueError("separate-F/B checkpoint unexpectedly retains fb_opt")
    else:
        optimizer = getattr(agent, "fb_opt", None)
        if optimizer is None:
            raise ValueError("shared-F/B checkpoint lacks fb_opt")
        if any(
            getattr(agent, name, None) is not None
            for name in ("forward_fb_opt", "backward_fb_opt")
        ):
            raise ValueError("shared-F/B checkpoint unexpectedly has branch optimizers")
        optimizers = {"fb_opt": optimizer}
    return {
        name: tuple(float(group["lr"]) for group in optimizer.param_groups)
        for name, optimizer in optimizers.items()
    }


def validated_checkpoint_size(path: Path, expected_size: Optional[int]) -> int:
    size = path.stat().st_size
    if expected_size is not None and size != expected_size:
        raise ValueError(
            f"checkpoint size mismatch for {path}: {size} != {expected_size}"
        )
    return size


def load_agent(experiment: Experiment, frame: int, device: str) -> Tuple[Any, Dict[str, Any]]:
    path = experiment.checkpoint_dir / f"snapshot_{frame}.pt"
    if not path.is_file():
        raise FileNotFoundError(f"requested checkpoint is missing: {path}")
    size = validated_checkpoint_size(path, experiment.expected_size)
    digest = sha256_file(path)
    expected_digest = experiment.checkpoint_hashes.get(frame)
    if expected_digest is not None and digest != expected_digest:
        raise ValueError(f"checkpoint SHA-256 mismatch for {path}: {digest}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "agent" not in payload:
        raise KeyError(f"checkpoint lacks agent: {path}")
    agent = payload["agent"]
    cfg = agent.cfg
    global_step = int(payload.get("global_step", -1))
    if 2 * global_step != frame:
        raise ValueError(f"loaded step/frame mismatch for {path}: {global_step} -> {2 * global_step}")
    contract = {
        "obs_type": str(cfg.obs_type), "obs_shape": tuple(cfg.obs_shape),
        "action_shape": tuple(cfg.action_shape), "z_dim": int(cfg.z_dim),
        "batch_size": int(cfg.batch_size), "num_inference_steps": int(cfg.num_inference_steps),
        "goal_space": cfg.goal_space, "norm_z": bool(cfg.norm_z),
        "q_loss": bool(cfg.q_loss), "idm_coef": float(cfg.idm_coef),
        "future_ratio": float(cfg.future_ratio), "rand_weight": bool(cfg.rand_weight),
        "boltzmann": bool(cfg.boltzmann), "use_cls": bool(cfg.use_cls),
        "dino_separate_backward_adapter": bool(
            getattr(cfg, "dino_separate_backward_adapter", False)
        ),
        "dino_separate_fb_adapters": bool(
            getattr(cfg, "dino_separate_fb_adapters", False)
        ),
        "dino_adapter_type": str(getattr(cfg, "dino_adapter_type", "linear")),
        "pixel_separate_fb_encoders": bool(
            getattr(cfg, "pixel_separate_fb_encoders", False)
        ),
        "idm_route": str(getattr(cfg, "idm_route", "none")),
        "idm_encoder_mode": str(getattr(cfg, "idm_encoder_mode", "legacy")),
        "idm_lr": (
            None
            if getattr(cfg, "idm_lr", None) is None
            else float(cfg.idm_lr)
        ),
    }
    expected_obs_shape: Tuple[int, ...]
    if experiment.obs_type == "dino":
        expected_obs_shape = (experiment.frame_stack * EMBED_DIM,)
    elif experiment.obs_type == "pixels":
        expected_obs_shape = (3 * experiment.frame_stack, *experiment.render_shape)
    else:
        raise ValueError(f"unsupported experiment obs_type: {experiment.obs_type!r}")
    expected_contract = expected_loaded_agent_contract(
        experiment, expected_obs_shape
    )
    if contract != expected_contract:
        raise ValueError(f"unsupported checkpoint contract for {path}: {contract}")
    serialized_fb_lr = vars(cfg).get("fb_lr", None)
    effective_fb_lr = float(cfg.lr if serialized_fb_lr is None else serialized_fb_lr)
    serialized_lr_f = vars(cfg).get("lr_f", None)
    serialized_lr_b = vars(cfg).get("lr_b", None)
    effective_lr_f = float(
        effective_fb_lr if serialized_lr_f is None else serialized_lr_f
    )
    effective_lr_b = float(
        float(cfg.lr_coef) * effective_fb_lr
        if serialized_lr_b is None
        else serialized_lr_b
    )
    optimizer_lr_contract = loaded_fb_optimizer_lrs(agent)
    if (
        experiment.expected_pixel_separate_fb_encoders
        or experiment.expected_dino_separate_fb_adapters
    ):
        expected_lr_f = (
            experiment.expected_fb_lr
            if experiment.expected_lr_f is None
            else experiment.expected_lr_f
        )
        expected_lr_b = (
            experiment.expected_fb_lr
            if experiment.expected_lr_b is None
            else experiment.expected_lr_b
        )
        lr_mismatch = (
            not math.isclose(effective_fb_lr, experiment.expected_fb_lr)
            or not math.isclose(effective_lr_f, expected_lr_f)
            or not math.isclose(effective_lr_b, expected_lr_b)
            or any(
                not math.isclose(value, expected_lr_f)
                for value in optimizer_lr_contract["forward_fb_opt"]
            )
            or any(
                not math.isclose(value, expected_lr_b)
                for value in optimizer_lr_contract["backward_fb_opt"]
            )
        )
    else:
        # Preserve the historical shared-profile check exactly: every FB
        # param-group used the single expected rate in these lineages.
        lr_mismatch = (
            not math.isclose(effective_fb_lr, experiment.expected_fb_lr)
            or any(
                not math.isclose(value, experiment.expected_fb_lr)
                for value in optimizer_lr_contract["fb_opt"]
            )
        )
    if lr_mismatch:
        raise ValueError(
            f"FB LR mismatch at {path}: base={effective_fb_lr}, "
            f"forward={effective_lr_f}, backward={effective_lr_b}, "
            f"optimizers={optimizer_lr_contract}"
        )
    prepare_agent(agent, device)
    install_optimizer_guards(agent)
    info = {
        "path": path, "sha256": digest, "size": size, "global_step": global_step,
        "global_episode": int(payload.get("global_episode", -1)),
        "serialized_fb_lr_key_present": "fb_lr" in vars(cfg),
        "serialized_fb_lr": serialized_fb_lr,
        "effective_fb_lr": effective_fb_lr,
        "effective_lr_f": effective_lr_f,
        "effective_lr_b": effective_lr_b,
        "optimizer_lrs": tuple(
            value
            for values in optimizer_lr_contract.values()
            for value in values
        ),
        "optimizer_lr_contract": optimizer_lr_contract,
        "agent_contract": contract,
    }
    del payload
    return agent, info


class MetricRecorder:
    def __init__(self, experiment: Experiment, frame: int, info: Mapping[str, Any], replay: ReplayData):
        self.rows: List[Dict[str, Any]] = []
        self.base = {
            "schema_version": SCHEMA_VERSION,
            "experiment": experiment.key,
            "run_id": experiment.run_id,
            "task": experiment.task,
            "effective_fb_lr": experiment.expected_fb_lr,
            "checkpoint_frame": frame,
            "checkpoint_step": int(info["global_step"]),
            "checkpoint_path": str(Path(info["path"]).resolve()),
            "checkpoint_sha256": str(info["sha256"]),
            "checkpoint_size_bytes": int(info["size"]),
            "fixed_replay_sha256": replay.metadata["content_sha256"],
        }

    def add(
        self,
        section: str,
        metric: str,
        value: Any,
        *,
        branch: str = "",
        z_mode: str = "",
        action_mode: str = "",
        weak_definition: str = "",
        tail_group: str = "",
        unit: str = "",
        source: str = "offline_checkpoint_analysis",
        source_frame: Any = "",
        source_step: Any = "",
        frame_delta: Any = "",
        note: str = "",
    ) -> None:
        if isinstance(value, (np.floating, np.integer)):
            value = value.item()
        self.rows.append({
            **self.base, "section": section, "z_mode": z_mode,
            "action_mode": action_mode, "branch": branch,
            "weak_definition": weak_definition, "tail_group": tail_group,
            "metric": metric, "value": value, "unit": unit, "source": source,
            "source_frame": source_frame, "source_step": source_step,
            "frame_delta": frame_delta, "note": note,
        })


def array_stats(values: np.ndarray, *, abs_quantiles: Sequence[float] = (0.95, 0.99)) -> Dict[str, float]:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    if x.size == 0:
        return {}
    output = {
        "mean": float(x.mean()), "std": float(x.std()),
        "rms": float(np.sqrt(np.mean(np.square(x)))),
        "min": float(x.min()), "max": float(x.max()),
        "abs_max": float(np.abs(x).max()),
    }
    absolute = np.abs(x)
    for quantile in abs_quantiles:
        label = f"abs_p{quantile * 100:g}".replace(".", "_")
        output[label] = float(np.quantile(absolute, quantile))
    return output


def add_stats(
    recorder: MetricRecorder,
    section: str,
    prefix: str,
    values: np.ndarray,
    **dimensions: str,
) -> None:
    for name, value in array_stats(values).items():
        recorder.add(section, f"{prefix}_{name}", value, **dimensions)


def row_norm_stats(value: np.ndarray) -> Dict[str, float]:
    norms = np.linalg.norm(np.asarray(value, dtype=np.float64), axis=1)
    return {
        "norm_mean": float(norms.mean()),
        "norm_rms": float(np.sqrt(np.mean(np.square(norms)))),
        "norm_p95": float(np.quantile(norms, 0.95)),
        "norm_p99": float(np.quantile(norms, 0.99)),
        "norm_max": float(norms.max()),
        "element_abs_max": float(np.abs(value).max()),
    }


def add_row_norm_stats(
    recorder: MetricRecorder, section: str, prefix: str, value: np.ndarray, **dimensions: str
) -> None:
    for name, scalar in row_norm_stats(value).items():
        recorder.add(section, f"{prefix}_{name}", scalar, **dimensions)


def encode_random_replay(
    agent: Any, replay: ReplayData, device: str, chunk_size: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    f1 = np.empty((len(replay.obs), Z_DIM), dtype=np.float32)
    f2 = np.empty_like(f1)
    backward = np.empty_like(f1)
    with torch.inference_mode():
        for start in range(0, len(replay.obs), chunk_size):
            stop = min(start + chunk_size, len(replay.obs))
            raw_obs = torch.as_tensor(replay.obs[start:stop], device=device)
            raw_next = torch.as_tensor(replay.next_obs[start:stop], device=device)
            action = torch.as_tensor(replay.action[start:stop], device=device)
            z = torch.as_tensor(replay.random_z[start:stop], device=device)
            obs = agent.aug_and_encode(raw_obs)
            next_goal = agent.backward_aug_and_encode(raw_next)
            current_f1, current_f2 = agent.forward_net(obs, z, action)
            current_b = agent.backward_net(next_goal)
            f1[start:stop] = current_f1.cpu().numpy()
            f2[start:stop] = current_f2.cpu().numpy()
            backward[start:stop] = current_b.cpu().numpy()
    return f1, f2, backward


def encode_forward(
    agent: Any,
    raw_obs_array: np.ndarray,
    z_array: np.ndarray,
    replay_action: np.ndarray,
    device: str,
    chunk_size: int,
    action_mode: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(raw_obs_array)
    z_values = np.asarray(z_array)
    constant_z = z_values.ndim == 1
    if constant_z:
        if z_values.shape != (Z_DIM,):
            raise ValueError(
                f"constant forward z must have shape {(Z_DIM,)}, got "
                f"{z_values.shape}"
            )
    elif z_values.shape != (n, Z_DIM):
        raise ValueError(
            f"batched forward z must have shape {(n, Z_DIM)}, got "
            f"{z_values.shape}"
        )
    f1 = np.empty((n, Z_DIM), dtype=np.float32)
    f2 = np.empty_like(f1)
    actions = np.empty((n, replay_action.shape[1]), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, n, chunk_size):
            stop = min(start + chunk_size, n)
            raw_obs = torch.as_tensor(raw_obs_array[start:stop], device=device)
            if constant_z:
                z = torch.as_tensor(z_values, device=device).unsqueeze(0).expand(
                    stop - start, -1
                )
            else:
                z = torch.as_tensor(z_values[start:stop], device=device)
            obs = agent.aug_and_encode(raw_obs)
            if action_mode == "replay":
                action = torch.as_tensor(replay_action[start:stop], device=device)
            elif action_mode == "actor_mean":
                stddev = utils.schedule(agent.cfg.stddev_schedule, 0)
                action = agent.actor(obs, z, stddev).mean
            else:
                raise ValueError(f"unknown action mode {action_mode}")
            current_f1, current_f2 = agent.forward_net(obs, z, action)
            f1[start:stop] = current_f1.cpu().numpy()
            f2[start:stop] = current_f2.cpu().numpy()
            actions[start:stop] = action.cpu().numpy()
    return f1, f2, actions


def covariance_and_projectors(
    backward_probe: np.ndarray, recorder: MetricRecorder
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    b = np.asarray(backward_probe, dtype=np.float64)
    covariance = b.T @ b / b.shape[0]
    covariance = 0.5 * (covariance + covariance.T)
    eigvals, eigvecs = np.linalg.eigh(covariance)
    if not np.all(np.diff(eigvals) >= -1e-12):
        raise AssertionError("EVD eigenvalues are not ascending")
    if not np.allclose(covariance, eigvecs @ np.diag(eigvals) @ eigvecs.T, rtol=1e-10, atol=1e-10):
        raise AssertionError("C_B eigendecomposition reconstruction failed")
    trace = float(eigvals.sum())
    if not math.isclose(trace, Z_DIM, rel_tol=5e-5, abs_tol=5e-5):
        raise AssertionError(f"normalized-B trace should equal {Z_DIM}, got {trace}")
    largest = float(eigvals[-1])
    tolerance = max(b.shape) * np.finfo(np.float64).eps * max(largest, 1.0)
    raw_condition = float("inf") if eigvals[0] <= 0 else largest / float(eigvals[0])
    effective_condition = largest / max(float(eigvals[0]), tolerance)
    participation = trace * trace / float(np.square(eigvals).sum())
    base_metrics = {
        "trace": trace, "lambda_min": float(eigvals[0]), "lambda_max": largest,
        "condition_number": raw_condition, "effective_condition_number": effective_condition,
        "rank_tolerance": tolerance, "numerical_rank": int(np.sum(eigvals > tolerance)),
        "participation_ratio": participation,
        "participation_ratio_fraction": participation / Z_DIM,
    }
    for metric, value in base_metrics.items():
        recorder.add("b_geometry", metric, value)
    add_row_norm_stats(recorder, "b_geometry", "b", b)
    for rank, value in enumerate(eigvals):
        recorder.add("b_spectrum", "eigenvalue", float(value), tail_group=f"ascending_rank_{rank:02d}")

    masks = {
        "bottom5": np.arange(Z_DIM) < 5,
        "bottom10": np.arange(Z_DIM) < 10,
        "thr003": eigvals < 0.03 * largest,
        "thr010": eigvals < 0.10 * largest,
        "thr030": eigvals < 0.30 * largest,
    }
    projectors: Dict[str, np.ndarray] = {}
    for definition, mask in masks.items():
        basis = eigvecs[:, mask]
        projector = basis @ basis.T
        if not np.allclose(projector, projector.T, rtol=0, atol=2e-12):
            raise AssertionError(f"{definition} projector is not symmetric")
        if not np.allclose(projector @ projector, projector, rtol=0, atol=2e-12):
            raise AssertionError(f"{definition} projector is not idempotent")
        if not math.isclose(float(np.trace(projector)), int(mask.sum()), abs_tol=2e-10):
            raise AssertionError(f"{definition} projector trace mismatch")
        projectors[definition] = projector
        weak_trace = float(eigvals[mask].sum())
        recorder.add("b_geometry", "weak_dimension", int(mask.sum()), weak_definition=definition)
        recorder.add("b_geometry", "weak_trace_fraction", weak_trace / trace, weak_definition=definition)
        if definition.startswith("bottom"):
            cutoff_value: Any = ""
            cutoff_note = "fixed ascending-eigenvalue rank"
        else:
            alpha = {"thr003": 0.03, "thr010": 0.10, "thr030": 0.30}[definition]
            cutoff_value = alpha * largest
            cutoff_note = f"lambda < {alpha:.2f} * lambda_max (strict inequality)"
        recorder.add("b_geometry", "weak_eigenvalue_cutoff", cutoff_value,
                     weak_definition=definition, note=cutoff_note)
    return covariance, eigvals, eigvecs, projectors


def infer_task_z_exact(
    agent: Any,
    replay: ReplayData,
    backward: np.ndarray,
    rewards: np.ndarray,
    inference_size: int,
    device: str,
) -> Tuple[np.ndarray, np.ndarray, str]:
    raw = rewards[:inference_size].astype(np.float64).T @ backward[:inference_size].astype(np.float64)
    raw = (raw / inference_size).reshape(-1)
    expected = math.sqrt(Z_DIM) * raw / np.linalg.norm(raw)
    raw_obs = torch.as_tensor(replay.next_obs[:inference_size], device=device)
    reward_tensor = torch.as_tensor(rewards[:inference_size], device=device)
    capture = io.StringIO()
    with contextlib.redirect_stdout(capture):
        meta = agent.infer_meta_from_obs_and_rewards(raw_obs, reward_tensor)
    observed = np.asarray(meta["z"], dtype=np.float64)
    if not np.allclose(observed, expected, rtol=3e-6, atol=3e-6):
        raise AssertionError(f"repository task-z path differs from manual reward projection: max={np.max(np.abs(observed-expected))}")
    if not math.isclose(float(np.linalg.norm(observed)), math.sqrt(Z_DIM), rel_tol=3e-6):
        raise AssertionError("task z is not normalized")
    return observed.astype(np.float32), raw.astype(np.float64), capture.getvalue().strip()


def infer_task_z_from_encoded_backward(
    backward: np.ndarray,
    rewards: np.ndarray,
    inference_size: int,
) -> Tuple[np.ndarray, np.ndarray, str]:
    """Chunk-safe form of the repository reward-weighted-B task-z formula."""
    if inference_size <= 0 or inference_size > len(backward) or inference_size > len(rewards):
        raise ValueError("invalid task-z inference size")
    raw = rewards[:inference_size].astype(np.float64).T @ backward[:inference_size].astype(
        np.float64
    )
    raw = (raw / inference_size).reshape(-1)
    raw_norm = float(np.linalg.norm(raw))
    if not math.isfinite(raw_norm) or raw_norm <= np.finfo(np.float64).tiny:
        raise FloatingPointError("reward-weighted B task-z has zero/non-finite raw norm")
    normalized = math.sqrt(Z_DIM) * raw / raw_norm
    note = (
        "Exact infer_meta_from_obs_and_rewards reward-weighted-B formula, with B encoded "
        "in deterministic chunks to bound CNN activation memory."
    )
    return normalized.astype(np.float32), raw, note


def analyze_task_z_geometry(
    recorder: MetricRecorder,
    task_z: np.ndarray,
    task_z_raw: np.ndarray,
    covariance: np.ndarray,
    eigvals: np.ndarray,
    eigvecs: np.ndarray,
    projectors: Mapping[str, np.ndarray],
    *,
    z_mode: str,
) -> None:
    """Record task-z energy in exactly the B covariance eigenbasis."""
    z = np.asarray(task_z, dtype=np.float64).reshape(-1)
    raw = np.asarray(task_z_raw, dtype=np.float64).reshape(-1)
    if z.shape != (Z_DIM,) or raw.shape != (Z_DIM,):
        raise ValueError(f"task-z vectors must have shape {(Z_DIM,)}, got {z.shape}/{raw.shape}")
    z_norm_sq = float(np.dot(z, z))
    if z_norm_sq <= np.finfo(np.float64).tiny:
        raise FloatingPointError("task-z has zero norm")
    coordinates = z @ eigvecs
    energy = np.square(coordinates) / z_norm_sq
    if not math.isclose(float(energy.sum()), 1.0, rel_tol=2e-10, abs_tol=2e-10):
        raise AssertionError("task-z eigendirection energy does not sum to one")
    rayleigh_direct = float(z @ covariance @ z / z_norm_sq)
    rayleigh_spectral = float(np.dot(energy, eigvals))
    if not math.isclose(rayleigh_direct, rayleigh_spectral, rel_tol=2e-10, abs_tol=2e-10):
        raise AssertionError("task-z covariance Rayleigh quotient identity failed")

    recorder.add("task_z", "raw_norm", float(np.linalg.norm(raw)), z_mode=z_mode)
    recorder.add("task_z", "normalized_norm", math.sqrt(z_norm_sq), z_mode=z_mode)
    recorder.add(
        "task_z_geometry",
        "covariance_rayleigh_quotient",
        rayleigh_direct,
        z_mode=z_mode,
        note="z^T C_B z / ||z||^2",
    )
    for rank, value in enumerate(energy):
        recorder.add(
            "task_z_spectrum",
            "eigendirection_energy_ratio",
            float(value),
            z_mode=z_mode,
            tail_group=f"ascending_rank_{rank:02d}",
        )
    for definition, projector in projectors.items():
        weak = z @ projector
        recorder.add(
            "task_z_geometry",
            "weak_energy_ratio",
            float(np.dot(weak, weak) / z_norm_sq),
            z_mode=z_mode,
            weak_definition=definition,
        )


def analyze_forward_geometry(
    recorder: MetricRecorder,
    forward: np.ndarray,
    z: np.ndarray,
    projectors: Mapping[str, np.ndarray],
    *,
    branch: str,
    z_mode: str,
    action_mode: str,
) -> None:
    f = np.asarray(forward, dtype=np.float64)
    z64 = np.asarray(z, dtype=np.float64)
    denominator = np.sum(np.square(z64), axis=1, keepdims=True)
    parallel = np.sum(f * z64, axis=1, keepdims=True) / denominator * z64
    perpendicular = f - parallel
    total_energy = float(np.square(f).sum())
    parallel_energy = float(np.square(parallel).sum())
    perpendicular_energy = float(np.square(perpendicular).sum())
    if not math.isclose(total_energy, parallel_energy + perpendicular_energy, rel_tol=2e-11, abs_tol=1e-7):
        raise AssertionError("F parallel/perpendicular Pythagorean identity failed")
    dimensions = {"branch": branch, "z_mode": z_mode, "action_mode": action_mode}
    add_row_norm_stats(recorder, "f_geometry", "f", f, **dimensions)
    add_row_norm_stats(recorder, "f_geometry", "f_parallel", parallel, **dimensions)
    add_row_norm_stats(recorder, "f_geometry", "f_perp", perpendicular, **dimensions)
    recorder.add("f_geometry", "parallel_energy_ratio", parallel_energy / total_energy, **dimensions)
    recorder.add("f_geometry", "perp_energy_ratio", perpendicular_energy / total_energy, **dimensions)
    for definition, projector in projectors.items():
        weak_f = f @ projector
        weak_perp = perpendicular @ projector
        recorder.add("f_geometry", "weak_energy_ratio", float(np.square(weak_f).sum() / total_energy),
                     weak_definition=definition, **dimensions)
        recorder.add("f_geometry", "rho_dangerous", float(np.square(weak_perp).sum() / total_energy),
                     weak_definition=definition, **dimensions)


def analyze_actions(
    recorder: MetricRecorder, actions: np.ndarray, *, z_mode: str, action_mode: str
) -> None:
    recorder.add("actor_actions", "action_element_rms", float(np.sqrt(np.mean(np.square(actions)))),
                 z_mode=z_mode, action_mode=action_mode)
    recorder.add("actor_actions", "action_vector_norm_rms",
                 float(np.sqrt(np.mean(np.sum(np.square(actions), axis=1)))),
                 z_mode=z_mode, action_mode=action_mode)
    recorder.add("actor_actions", "action_abs_p95", float(np.quantile(np.abs(actions), 0.95)),
                 z_mode=z_mode, action_mode=action_mode)
    recorder.add("actor_actions", "action_abs_p99", float(np.quantile(np.abs(actions), 0.99)),
                 z_mode=z_mode, action_mode=action_mode)
    recorder.add("actor_actions", "action_abs_max", float(np.abs(actions).max()),
                 z_mode=z_mode, action_mode=action_mode)
    recorder.add("actor_actions", "action_saturation_fraction_abs_ge_0_99",
                 float(np.mean(np.abs(actions) >= 0.99)), z_mode=z_mode, action_mode=action_mode)


def analyze_q(
    recorder: MetricRecorder,
    f1: np.ndarray,
    f2: np.ndarray,
    z: np.ndarray,
    actions: np.ndarray,
    projectors: Mapping[str, np.ndarray],
    replay: ReplayData,
    *,
    z_mode: str,
    action_mode: str,
    extremes: MutableMapping[str, np.ndarray],
    extreme_json: List[Dict[str, Any]],
) -> None:
    forwards = (np.asarray(f1), np.asarray(f2))
    z_input = np.asarray(z)
    qs = list(_q_row_dots_float64_chunked(forwards, z_input))
    qmin = np.minimum(qs[0], qs[1])
    for branch, q in zip(("f1", "f2"), qs):
        add_stats(recorder, "q", "q", q, branch=branch, z_mode=z_mode, action_mode=action_mode)
    add_stats(recorder, "q", "q", qmin, branch="twin_min", z_mode=z_mode, action_mode=action_mode)
    add_stats(recorder, "q", "twin_abs_gap", np.abs(qs[0] - qs[1]), branch="twin",
              z_mode=z_mode, action_mode=action_mode)
    analyze_actions(recorder, actions, z_mode=z_mode, action_mode=action_mode)

    for definition, projector in projectors.items():
        z_ratio, qweak_values, qstrong_values = (
            _q_projected_parts_float64_chunked(
                forwards, z_input, projector
            )
        )
        recorder.add("q_decomposition", "z_weak_energy_ratio", z_ratio, z_mode=z_mode,
                     action_mode=action_mode, weak_definition=definition)
        qweak = list(qweak_values)
        qstrong = list(qstrong_values)
        for branch, q, qw, qs_part in zip(
            ("f1", "f2"), qs, qweak, qstrong
        ):
            if not np.allclose(q, qw + qs_part, rtol=2e-9, atol=2e-7):
                raise AssertionError(f"Q decomposition identity failed for {branch}/{definition}")
            add_stats(recorder, "q_decomposition", "q_weak", qw, branch=branch, z_mode=z_mode,
                      action_mode=action_mode, weak_definition=definition)
            recorder.add("q_decomposition", "abs_qweak_over_abs_q_mean",
                         float(np.mean(np.abs(qw) / (np.abs(q) + 1e-12))), branch=branch,
                         z_mode=z_mode, action_mode=action_mode, weak_definition=definition)
        less = qs[0] < qs[1]
        equal = qs[0] == qs[1]
        w1 = less.astype(np.float64) + 0.5 * equal.astype(np.float64)
        w2 = 1.0 - w1
        min_weak = w1 * qweak[0] + w2 * qweak[1]
        min_strong = w1 * qstrong[0] + w2 * qstrong[1]
        if not np.allclose(qmin, min_weak + min_strong, rtol=2e-9, atol=2e-7):
            raise AssertionError(f"twin-min Q identity failed for {definition}")
        add_stats(recorder, "q_decomposition", "q_weak", min_weak, branch="twin_min",
                  z_mode=z_mode, action_mode=action_mode, weak_definition=definition)
        ratio = np.abs(min_weak) / (np.abs(qmin) + 1e-12)
        recorder.add("q_decomposition", "abs_qweak_over_abs_q_mean", float(ratio.mean()),
                     branch="twin_min", z_mode=z_mode, action_mode=action_mode,
                     weak_definition=definition)
        for tail_name, fraction in (("top0.1pct", 0.001), ("top0.01pct", 0.0001)):
            count = max(1, int(math.ceil(len(qmin) * fraction)))
            selected = np.argpartition(np.abs(qmin), -count)[-count:]
            recorder.add("q_extremes", "abs_qweak_over_abs_q_mean", float(ratio[selected].mean()),
                         branch="twin_min", z_mode=z_mode, action_mode=action_mode,
                         weak_definition=definition, tail_group=tail_name)
            recorder.add("q_extremes", "q_weak_abs_mean", float(np.abs(min_weak[selected]).mean()),
                         branch="twin_min", z_mode=z_mode, action_mode=action_mode,
                         weak_definition=definition, tail_group=tail_name)

        if definition == PRIMARY_WEAK and z_mode == "task_proxy" and action_mode == "actor_mean":
            top = np.argsort(np.abs(qmin), kind="mergesort")[-100:][::-1]
            prefix = "q_task_actor_top100"
            extremes[f"{prefix}_row"] = top.astype(np.int32)
            extremes[f"{prefix}_q"] = qmin[top].astype(np.float64)
            extremes[f"{prefix}_qweak"] = min_weak[top].astype(np.float64)
            extremes[f"{prefix}_action"] = actions[top].astype(np.float32)
            for rank, row in enumerate(top):
                extreme_json.append({
                    "kind": "Q", "z_mode": z_mode, "action_mode": action_mode,
                    "weak_definition": definition, "rank": rank + 1,
                    "row": int(row), "episode_id": int(replay.episode_id[row]),
                    "step_in_episode": int(replay.step_in_episode[row]),
                    "index_bank_row": int(replay.index_bank_row[row]),
                    "q": float(qmin[row]), "q_weak": float(min_weak[row]),
                    "abs_qweak_over_abs_q": float(ratio[row]),
                    "action": [float(item) for item in actions[row]],
                })


def _q_row_dots_float64_chunked(
    forwards: Sequence[np.ndarray],
    z: np.ndarray,
    chunk_rows: int = CNN_Q_FLOAT64_CHUNK_ROWS,
) -> Tuple[np.ndarray, ...]:
    """Compute row dot products in float64 without full NxZ conversions."""
    if chunk_rows <= 0:
        raise ValueError("Q float64 chunk_rows must be positive")
    z_input = np.asarray(z)
    forward_inputs = tuple(np.asarray(value) for value in forwards)
    if z_input.ndim != 2 or any(
        value.ndim != 2 or value.shape != z_input.shape
        for value in forward_inputs
    ):
        raise ValueError("Q row-dot inputs must share one two-dimensional shape")
    outputs = tuple(
        np.empty(z_input.shape[0], dtype=np.float64)
        for _ in forward_inputs
    )
    for start in range(0, z_input.shape[0], chunk_rows):
        stop = min(start + chunk_rows, z_input.shape[0])
        z_chunk = np.asarray(z_input[start:stop], dtype=np.float64)
        for value, output in zip(forward_inputs, outputs):
            forward_chunk = np.asarray(value[start:stop], dtype=np.float64)
            output[start:stop] = np.sum(
                forward_chunk * z_chunk, axis=1, dtype=np.float64
            )
    return outputs


def _q_projected_parts_float64_chunked(
    forwards: Sequence[np.ndarray],
    z: np.ndarray,
    projector: np.ndarray,
    chunk_rows: int = CNN_Q_FLOAT64_CHUNK_ROWS,
) -> Tuple[float, Tuple[np.ndarray, ...], Tuple[np.ndarray, ...]]:
    """Compute weak/strong Q terms with only bounded float64 matrices."""
    if chunk_rows <= 0:
        raise ValueError("Q float64 chunk_rows must be positive")
    z_input = np.asarray(z)
    forward_inputs = tuple(np.asarray(value) for value in forwards)
    projector64 = np.asarray(projector, dtype=np.float64)
    if z_input.ndim != 2 or any(
        value.ndim != 2 or value.shape != z_input.shape
        for value in forward_inputs
    ):
        raise ValueError("Q decomposition inputs must share one two-dimensional shape")
    if projector64.shape != (z_input.shape[1], z_input.shape[1]):
        raise ValueError(
            "Q decomposition projector must be square in the latent dimension"
        )
    weak_outputs = tuple(
        np.empty(z_input.shape[0], dtype=np.float64)
        for _ in forward_inputs
    )
    strong_outputs = tuple(
        np.empty(z_input.shape[0], dtype=np.float64)
        for _ in forward_inputs
    )
    weak_energy = 0.0
    total_energy = 0.0
    for start in range(0, z_input.shape[0], chunk_rows):
        stop = min(start + chunk_rows, z_input.shape[0])
        z_chunk = np.asarray(z_input[start:stop], dtype=np.float64)
        z_weak = z_chunk @ projector64
        z_strong = z_chunk - z_weak
        weak_energy += float(np.square(z_weak).sum(dtype=np.float64))
        total_energy += float(np.square(z_chunk).sum(dtype=np.float64))
        for value, weak_output, strong_output in zip(
            forward_inputs, weak_outputs, strong_outputs
        ):
            forward_chunk = np.asarray(
                value[start:stop], dtype=np.float64
            )
            forward_weak = forward_chunk @ projector64
            forward_strong = forward_chunk - forward_weak
            weak_output[start:stop] = np.sum(
                forward_weak * z_weak, axis=1, dtype=np.float64
            )
            strong_output[start:stop] = np.sum(
                forward_strong * z_strong, axis=1, dtype=np.float64
            )
    return weak_energy / total_energy, weak_outputs, strong_outputs


def analyze_large_m(
    recorder: MetricRecorder,
    forward: np.ndarray,
    backward: np.ndarray,
    eigvals: np.ndarray,
    eigvecs: np.ndarray,
    projectors: Mapping[str, np.ndarray],
    replay: ReplayData,
    *,
    branch: str,
    device: str,
    extremes: MutableMapping[str, np.ndarray],
    extreme_json: List[Dict[str, Any]],
) -> None:
    del projectors  # masks are recovered in the common eigenbasis below.
    n = len(forward)
    if n % TRAIN_BATCH:
        raise ValueError("large M scan requires complete training-sized chunks")
    f = np.asarray(forward, dtype=np.float32)
    b = np.asarray(backward, dtype=np.float32)
    with torch.inference_mode():
        f_tensor = torch.as_tensor(f, device=device)
        b_tensor = torch.as_tensor(b, device=device)
        eigenvectors = torch.as_tensor(eigvecs, device=device, dtype=torch.float32)
        f_coordinates_tensor = f_tensor @ eigenvectors
        b_coordinates_tensor = b_tensor @ eigenvectors
    f_coordinates = f_coordinates_tensor.cpu().numpy()
    b_coordinates = b_coordinates_tensor.cpu().numpy()
    entries_per_chunk = TRAIN_BATCH * TRAIN_BATCH
    total_entries = (n // TRAIN_BATCH) * entries_per_chunk
    m_store = np.empty(total_entries, dtype=np.float32)
    signed_sum = 0.0
    squared_sum = 0.0
    cursor = 0
    with torch.inference_mode():
        for start in range(0, n, TRAIN_BATCH):
            stop = start + TRAIN_BATCH
            matrix = f_tensor[start:stop] @ b_tensor[start:stop].T
            flat = matrix.cpu().numpy().reshape(-1)
            m_store[cursor:cursor + entries_per_chunk] = flat
            signed_sum += float(flat.astype(np.float64).sum())
            squared_sum += float(np.square(flat.astype(np.float64)).sum())
            cursor += entries_per_chunk
    absolute = np.abs(m_store)
    quantile_levels = (0.95, 0.99, 0.999, 0.9999)
    quantile_values = np.quantile(absolute, quantile_levels)
    quantiles = {q: float(value) for q, value in zip(quantile_levels, quantile_values)}
    recorder.add("large_m", "m_mean", signed_sum / total_entries, branch=branch,
                 z_mode="fixed_random", action_mode="replay")
    recorder.add("large_m", "m_rms", math.sqrt(squared_sum / total_entries), branch=branch,
                 z_mode="fixed_random", action_mode="replay")
    for quantile, value in quantiles.items():
        recorder.add("large_m", f"m_abs_p{quantile * 100:g}".replace(".", "_"), value,
                     branch=branch, z_mode="fixed_random", action_mode="replay")
    recorder.add("large_m", "m_abs_max", float(absolute.max()), branch=branch,
                 z_mode="fixed_random", action_mode="replay")
    top100_unsorted = np.argpartition(absolute, -100)[-100:]
    top100 = top100_unsorted[np.argsort(absolute[top100_unsorted], kind="mergesort")][::-1]
    group_thresholds = {
        "all": None,
        "top0.1pct": quantiles[0.999],
        "top0.01pct": quantiles[0.9999],
    }
    del absolute
    # Threshold bases are identified from the projector rank and overlap with
    # the ascending eigenbasis.  Reconstruct exact boolean membership via the
    # diagonal of V^T P V; this also validates projector/eigenbasis agreement.
    weak_masks: Dict[str, np.ndarray] = {}
    for definition in WEAK_DEFINITIONS:
        if definition == "bottom5":
            mask = np.arange(Z_DIM) < 5
        elif definition == "bottom10":
            mask = np.arange(Z_DIM) < 10
        else:
            alpha = {"thr003": 0.03, "thr010": 0.10, "thr030": 0.30}[definition]
            mask = eigvals < alpha * float(eigvals[-1])
        weak_masks[definition] = mask

    for definition, weak_mask in weak_masks.items():
        strong_mask = ~weak_mask
        fweak_ratio = np.linalg.norm(f_coordinates[:, weak_mask], axis=1) / (np.linalg.norm(f, axis=1) + 1e-12)
        bweak_ratio = np.linalg.norm(b_coordinates[:, weak_mask], axis=1) / (np.linalg.norm(b, axis=1) + 1e-12)
        accumulators = {
            name: {"count": 0, "abs_mw": 0.0, "sq_mw": 0.0, "ratio": 0.0,
                   "fw": 0.0, "bw": 0.0, "max_abs_mw": 0.0}
            for name in group_thresholds
        }
        primary_top_values: Dict[str, List[np.ndarray]] = {
            "m": [], "mw": [], "fw": [], "bw": [], "left": [], "right": []
        }
        cursor = 0
        identity_max_relative_error = 0.0
        with torch.inference_mode():
            weak_indices = torch.as_tensor(np.flatnonzero(weak_mask), device=device, dtype=torch.long)
            strong_indices = torch.as_tensor(np.flatnonzero(strong_mask), device=device, dtype=torch.long)
            for start in range(0, n, TRAIN_BATCH):
                stop = start + TRAIN_BATCH
                full_tensor = f_tensor[start:stop] @ b_tensor[start:stop].T
                weak_tensor = (
                    f_coordinates_tensor[start:stop].index_select(1, weak_indices)
                    @ b_coordinates_tensor[start:stop].index_select(1, weak_indices).T
                )
                strong_tensor = (
                    f_coordinates_tensor[start:stop].index_select(1, strong_indices)
                    @ b_coordinates_tensor[start:stop].index_select(1, strong_indices).T
                )
                error = float(torch.max(torch.abs(full_tensor - strong_tensor - weak_tensor)).item())
                scale = max(float(torch.max(torch.abs(full_tensor)).item()), 1.0)
                identity_max_relative_error = max(identity_max_relative_error, error / scale)
                if error / scale > 2e-5:
                    raise AssertionError(f"M identity failed for {branch}/{definition}: {error}/{scale}")
                full_flat = m_store[cursor:cursor + entries_per_chunk]
                weak_flat = weak_tensor.cpu().numpy().reshape(-1)
                absolute_full = np.abs(full_flat)
                ratio_flat = np.abs(weak_flat) / (absolute_full + 1e-12)
                for group_name, threshold in group_thresholds.items():
                    if threshold is None:
                        local_indices = None
                        selected_m = full_flat
                        selected_w = weak_flat
                        selected_ratio = ratio_flat
                        fw_sum = TRAIN_BATCH * float(fweak_ratio[start:stop].sum())
                        bw_sum = TRAIN_BATCH * float(bweak_ratio[start:stop].sum())
                        selected_count = entries_per_chunk
                    else:
                        local_indices = np.flatnonzero(absolute_full >= threshold)
                        selected_m = full_flat[local_indices]
                        selected_w = weak_flat[local_indices]
                        selected_ratio = ratio_flat[local_indices]
                        left_indices = local_indices // TRAIN_BATCH + start
                        right_indices = local_indices % TRAIN_BATCH + start
                        fw_sum = float(fweak_ratio[left_indices].sum())
                        bw_sum = float(bweak_ratio[right_indices].sum())
                        selected_count = int(local_indices.size)
                    state = accumulators[group_name]
                    state["count"] += selected_count
                    state["abs_mw"] += float(np.abs(selected_w).astype(np.float64).sum())
                    state["sq_mw"] += float(np.square(selected_w.astype(np.float64)).sum())
                    state["ratio"] += float(selected_ratio.astype(np.float64).sum())
                    state["fw"] += fw_sum
                    state["bw"] += bw_sum
                    if selected_w.size:
                        state["max_abs_mw"] = max(state["max_abs_mw"], float(np.abs(selected_w).max()))
                if definition == PRIMARY_WEAK:
                    within = top100[(top100 >= cursor) & (top100 < cursor + entries_per_chunk)] - cursor
                    if within.size:
                        left = within // TRAIN_BATCH + start
                        right = within % TRAIN_BATCH + start
                        primary_top_values["m"].append(full_flat[within])
                        primary_top_values["mw"].append(weak_flat[within])
                        primary_top_values["fw"].append(fweak_ratio[left])
                        primary_top_values["bw"].append(bweak_ratio[right])
                        primary_top_values["left"].append(left)
                        primary_top_values["right"].append(right)
                cursor += entries_per_chunk
        recorder.add("numerical_assertions", "m_identity_max_relative_element_error",
                     identity_max_relative_error, branch=branch, z_mode="fixed_random",
                     action_mode="replay", weak_definition=definition)

        all_ratio = accumulators["all"]["ratio"] / accumulators["all"]["count"]
        for group_name, state in accumulators.items():
            count = state["count"]
            recorder.add("large_m_tail", "mweak_abs_mean", state["abs_mw"] / count,
                         branch=branch, z_mode="fixed_random", action_mode="replay",
                         weak_definition=definition, tail_group=group_name)
            recorder.add("large_m_tail", "mweak_rms", math.sqrt(state["sq_mw"] / count),
                         branch=branch, z_mode="fixed_random", action_mode="replay",
                         weak_definition=definition, tail_group=group_name)
            recorder.add("large_m_tail", "mweak_abs_max", state["max_abs_mw"],
                         branch=branch, z_mode="fixed_random", action_mode="replay",
                         weak_definition=definition, tail_group=group_name)
            mean_ratio = state["ratio"] / count
            recorder.add("large_m_tail", "abs_mweak_over_abs_m_mean", mean_ratio,
                         branch=branch, z_mode="fixed_random", action_mode="replay",
                         weak_definition=definition, tail_group=group_name)
            recorder.add("large_m_tail", "fweak_norm_ratio_mean", state["fw"] / count,
                         branch=branch, z_mode="fixed_random", action_mode="replay",
                         weak_definition=definition, tail_group=group_name)
            recorder.add("large_m_tail", "bweak_norm_ratio_mean", state["bw"] / count,
                         branch=branch, z_mode="fixed_random", action_mode="replay",
                         weak_definition=definition, tail_group=group_name)
            recorder.add("large_m_tail", "weak_contribution_disproportion_vs_all",
                         mean_ratio / (all_ratio + 1e-12), branch=branch,
                         z_mode="fixed_random", action_mode="replay",
                         weak_definition=definition, tail_group=group_name)

        if definition == PRIMARY_WEAK:
            assembled = {name: np.concatenate(parts) for name, parts in primary_top_values.items()}
            # Re-sort because chunks were visited in chronological, not rank, order.
            rank_order = np.argsort(np.abs(assembled["m"]), kind="mergesort")[::-1]
            assembled = {name: value[rank_order] for name, value in assembled.items()}
            prefix = f"m_{branch}_top100"
            for name, value in assembled.items():
                extremes[f"{prefix}_{name}"] = value
            for rank in range(len(assembled["m"])):
                left = int(assembled["left"][rank])
                right = int(assembled["right"][rank])
                m_value = float(assembled["m"][rank])
                mw_value = float(assembled["mw"][rank])
                extreme_json.append({
                    "kind": "M", "branch": branch, "z_mode": "fixed_random",
                    "action_mode": "replay", "weak_definition": definition,
                    "rank": rank + 1, "m": m_value, "m_weak": mw_value,
                    "abs_mweak_over_abs_m": abs(mw_value) / (abs(m_value) + 1e-12),
                    "fweak_norm_ratio": float(assembled["fw"][rank]),
                    "bweak_norm_ratio": float(assembled["bw"][rank]),
                    "left_row": left, "left_episode_id": int(replay.episode_id[left]),
                    "left_step": int(replay.step_in_episode[left]),
                    "left_index_bank_row": int(replay.index_bank_row[left]),
                    "right_row": right, "right_episode_id": int(replay.episode_id[right]),
                    "right_step": int(replay.step_in_episode[right]),
                    "right_index_bank_row": int(replay.index_bank_row[right]),
                })
    del m_store, f_tensor, b_tensor, f_coordinates_tensor, b_coordinates_tensor


def _fork_rng(device: str):
    parsed = torch.device(device)
    devices: List[int] = []
    if parsed.type == "cuda":
        devices = [torch.cuda.current_device() if parsed.index is None else parsed.index]
    return torch.random.fork_rng(devices=devices)


def make_training_probe_z(agent: Any, raw_obs: torch.Tensor, seed: int) -> torch.Tensor:
    n = raw_obs.shape[0]
    if n != TRAIN_BATCH:
        raise ValueError(f"exact gradient probe requires N={TRAIN_BATCH}, got {n}")
    numpy_rng = np.random.RandomState(seed + 1)
    with torch.no_grad(), _fork_rng(str(raw_obs.device)):
        torch.manual_seed(seed)
        z = agent.sample_z(n, device=str(raw_obs.device))
        backward_input = agent.backward_aug_and_encode(raw_obs)
        # Repository update() draws this permutation on CPU, even when the
        # encoded tensor resides on CUDA.
        permutation = torch.randperm(n)
        backward_input = backward_input[permutation]
        mix_indices_np = np.flatnonzero(numpy_rng.uniform(size=n) < float(agent.cfg.mix_ratio))
        mix_indices = torch.as_tensor(mix_indices_np, device=raw_obs.device, dtype=torch.long)
        mix_z = agent.backward_net(backward_input[mix_indices]).detach()
        mix_z = math.sqrt(Z_DIM) * torch.nn.functional.normalize(mix_z, dim=1)
        z[mix_indices] = mix_z
    if z.requires_grad:
        raise AssertionError("training probe z must be detached")
    return z


def unique_parameters(*groups: Iterable[nn.Parameter]) -> Tuple[nn.Parameter, ...]:
    output: List[nn.Parameter] = []
    seen: set[int] = set()
    for group in groups:
        for parameter in group:
            if parameter.requires_grad and id(parameter) not in seen:
                seen.add(id(parameter))
                output.append(parameter)
    return tuple(output)


def optimizer_parameters(*optimizers: Optional[torch.optim.Optimizer]) -> Tuple[nn.Parameter, ...]:
    groups: List[Iterable[nn.Parameter]] = []
    for optimizer in optimizers:
        if optimizer is not None:
            groups.extend(group["params"] for group in optimizer.param_groups)
    return unique_parameters(*groups)


def normalize_grads(
    grads: Sequence[Optional[torch.Tensor]], parameters: Sequence[nn.Parameter]
) -> Tuple[torch.Tensor, ...]:
    output = tuple(torch.zeros_like(parameter) if grad is None else grad for grad, parameter in zip(grads, parameters))
    if not all(bool(torch.isfinite(grad).all()) for grad in output):
        raise FloatingPointError("non-finite functional gradient")
    return output


def subset_gradient_summary(
    left: Sequence[torch.Tensor],
    right: Sequence[torch.Tensor],
    full: Sequence[torch.Tensor],
    indices: Sequence[int],
) -> Dict[str, float]:
    left_sq = right_sq = full_sq = dot_lr = dot_lf = dot_rf = residual_sq = 0.0
    for index in indices:
        gl, gr, gf = left[index], right[index], full[index]
        left_sq += float(torch.sum(gl.double() * gl.double()).item())
        right_sq += float(torch.sum(gr.double() * gr.double()).item())
        full_sq += float(torch.sum(gf.double() * gf.double()).item())
        dot_lr += float(torch.sum(gl.double() * gr.double()).item())
        dot_lf += float(torch.sum(gl.double() * gf.double()).item())
        dot_rf += float(torch.sum(gr.double() * gf.double()).item())
        residual = gf.double() - gl.double() - gr.double()
        residual_sq += float(torch.sum(residual * residual).item())
    nl, nr, nf = math.sqrt(left_sq), math.sqrt(right_sq), math.sqrt(full_sq)
    return {
        "gstrong_norm": nl,
        "gweak_norm": nr,
        "gfull_norm": nf,
        "cosine_strong_weak": dot_lr / (nl * nr) if nl > 0 and nr > 0 else float("nan"),
        "cancellation_ratio": nf / (nl + nr) if nl + nr > 0 else float("nan"),
        "strong_projection_on_full": dot_lf / full_sq if full_sq > 0 else float("nan"),
        "weak_projection_on_full": dot_rf / full_sq if full_sq > 0 else float("nan"),
        "gradient_additivity_relerr": math.sqrt(residual_sq) / max(nf, nl + nr, 1e-12),
    }


def gradient_dot_summary(
    first: Mapping[int, torch.Tensor], second: Mapping[int, torch.Tensor]
) -> Tuple[float, float, float, float]:
    common = sorted(set(first) & set(second))
    first_sq = second_sq = dot = 0.0
    for identity in common:
        left = first[identity].double()
        right = second[identity].double()
        first_sq += float(torch.sum(left * left).item())
        second_sq += float(torch.sum(right * right).item())
        dot += float(torch.sum(left * right).item())
    first_norm, second_norm = math.sqrt(first_sq), math.sqrt(second_sq)
    cosine = dot / (first_norm * second_norm) if first_norm > 0 and second_norm > 0 else float("nan")
    return first_norm, second_norm, dot, cosine


def _append_gradient_row(
    rows: List[Dict[str, Any]],
    experiment: Experiment,
    frame: int,
    objective: str,
    scope: str,
    values: Mapping[str, Any],
    note: str,
) -> None:
    rows.append({
        "schema_version": SCHEMA_VERSION, "experiment": experiment.key,
        "run_id": experiment.run_id, "task": experiment.task,
        "effective_fb_lr": experiment.expected_fb_lr, "checkpoint_frame": frame,
        "objective": objective, "scope": scope, **values, "note": note,
    })


def analyze_gradients(
    agent: Any,
    replay: ReplayData,
    projector_np: np.ndarray,
    experiment: Experiment,
    frame: int,
    device: str,
    seed: int,
    recorder: MetricRecorder,
) -> List[Dict[str, Any]]:
    if bool(agent.cfg.q_loss) or agent.idm_head is not None or bool(agent.cfg.boltzmann):
        raise ValueError("gradient attribution only supports this run's q_loss=False/idm=0/non-Boltzmann contract")
    if not math.isclose(float(agent.cfg.backward_encoder_grad_scale), 1.0):
        raise ValueError("unexpected backward encoder gradient scale")
    n = TRAIN_BATCH
    raw_obs = torch.as_tensor(replay.obs[:n], device=device)
    raw_next = torch.as_tensor(replay.next_obs[:n], device=device)
    action = torch.as_tensor(replay.action[:n], device=device)
    discount = torch.as_tensor(replay.discount[:n], device=device)
    if discount.shape != (n, 1) or not torch.allclose(discount, torch.full_like(discount, 0.99)):
        raise AssertionError("gradient probe discount is not exact nstep=1 gamma=0.99")
    z = make_training_probe_z(agent, raw_obs, seed)
    projector = torch.as_tensor(projector_np, device=device, dtype=torch.float32).detach()
    identity = torch.eye(Z_DIM, device=device)
    strong_projector = (identity - projector).detach()
    if projector.requires_grad or not torch.allclose(projector, projector.T, atol=2e-6):
        raise AssertionError("weak projector must be detached and symmetric")
    if not torch.allclose(projector @ projector, projector, atol=2e-5):
        raise AssertionError("weak projector must be idempotent")

    encoder_params = unique_parameters(agent.encoder.parameters())
    forward_params = unique_parameters(agent.forward_net.parameters())
    f1_head_params = unique_parameters(agent.forward_net.F1.parameters())
    f2_head_params = unique_parameters(agent.forward_net.F2.parameters())
    backward_params = unique_parameters(agent.backward_net.parameters())
    fb_union = optimizer_parameters(agent.fb_opt, agent.encoder_opt, agent.backward_encoder_opt)
    if len({id(parameter) for parameter in fb_union}) != len(fb_union):
        raise AssertionError("duplicate parameters in FB gradient scope")
    union_index = {id(parameter): index for index, parameter in enumerate(fb_union)}
    scopes = {
        "shared_encoder": [union_index[id(parameter)] for parameter in encoder_params],
        "forward_map": [union_index[id(parameter)] for parameter in forward_params],
        "forward_f1_exclusive_head": [union_index[id(parameter)] for parameter in f1_head_params],
        "forward_f2_exclusive_head": [union_index[id(parameter)] for parameter in f2_head_params],
        "backward_map": [union_index[id(parameter)] for parameter in backward_params],
        "actual_fb_update_union": list(range(len(fb_union))),
    }
    before_grads = [None if parameter.grad is None else parameter.grad.detach().clone() for parameter in fb_union]

    obs = agent.aug_and_encode(raw_obs)
    next_obs = agent.aug_and_encode(raw_next)
    next_goal = _scale_gradient(next_obs, float(agent.cfg.backward_encoder_grad_scale))
    f1, f2 = agent.forward_net(obs, z, action)
    backward = agent.backward_net(next_goal)
    with torch.no_grad(), _fork_rng(device):
        torch.manual_seed(seed + 2)
        stddev = utils.schedule(agent.cfg.stddev_schedule, frame // 2)
        next_action = agent.actor(next_obs, z, stddev).sample(clip=agent.cfg.stddev_clip)
        target_f1, target_f2 = agent.forward_target_net(next_obs, z, next_action)
        target_b = agent.backward_target_net(next_obs)
    if any(value.requires_grad for value in (target_f1, target_f2, target_b, next_action)):
        raise AssertionError("target FB path must be detached")

    def split(value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return value @ strong_projector, value @ projector

    f1s, f1w = split(f1)
    f2s, f2w = split(f2)
    bs, bw = split(backward)
    tf1s, tf1w = split(target_f1)
    tf2s, tf2w = split(target_f2)
    tbs, tbw = split(target_b)
    m1s, m1w = f1s @ bs.T, f1w @ bw.T
    m2s, m2w = f2s @ bs.T, f2w @ bw.T
    m1, m2 = f1 @ backward.T, f2 @ backward.T
    tm1s, tm1w = tf1s @ tbs.T, tf1w @ tbw.T
    tm2s, tm2w = tf2s @ tbs.T, tf2w @ tbw.T
    tm1, tm2 = target_f1 @ target_b.T, target_f2 @ target_b.T
    winner1 = (tm1 < tm2).to(tm1.dtype) + 0.5 * (tm1 == tm2).to(tm1.dtype)
    winner2 = 1.0 - winner1
    target_s = winner1 * tm1s + winner2 * tm2s
    target_w = winner1 * tm1w + winner2 * tm2w
    target = torch.minimum(tm1, tm2)
    offdiag = ~torch.eye(n, device=device, dtype=torch.bool)
    if int(offdiag.sum()) != n * (n - 1):
        raise AssertionError("off-diagonal mask semantics changed")
    r1s, r1w = m1s - discount * target_s, m1w - discount * target_w
    r2s, r2w = m2s - discount * target_s, m2w - discount * target_w
    r1, r2 = m1 - discount * target, m2 - discount * target
    gram_s, gram_w = bs @ bs.T, bw @ bw.T
    gram = backward @ backward.T
    identities = (
        ("f1", f1, f1s + f1w), ("f2", f2, f2s + f2w),
        ("backward", backward, bs + bw), ("m1", m1, m1s + m1w),
        ("m2", m2, m2s + m2w), ("target_m1", tm1, tm1s + tm1w),
        ("target_m2", tm2, tm2s + tm2w), ("target_min", target, target_s + target_w),
        ("residual1", r1, r1s + r1w), ("residual2", r2, r2s + r2w),
        ("gram", gram, gram_s + gram_w),
    )
    failed = []
    identity_relative_max_errors: List[float] = []
    identity_relative_fro_errors: List[float] = []
    for name, left, right in identities:
        max_error = float(torch.max(torch.abs(left - right)).item())
        scale = max(float(torch.max(torch.abs(left)).item()), 1.0)
        relative_fro = float(
            torch.linalg.vector_norm((left - right).double()).item()
            / max(torch.linalg.vector_norm(left.double()).item(), 1e-12)
        )
        # Independent float32 GEMMs are not elementwise distributive.  Test a
        # scale-aware numerical identity rather than bit equality near zero.
        identity_relative_max_errors.append(max_error / scale)
        identity_relative_fro_errors.append(relative_fro)
        if max_error / scale > 1e-5 or relative_fro > 1e-5:
            failed.append((name, max_error, scale, relative_fro))
    if failed:
        raise AssertionError(f"FB tensor decomposition identity failed: {failed}")
    recorder.add("numerical_assertions", "fb_tensor_identity_max_relative_element_error",
                 max(identity_relative_max_errors), weak_definition=PRIMARY_WEAK)
    recorder.add("numerical_assertions", "fb_tensor_identity_max_relative_fro_error",
                 max(identity_relative_fro_errors), weak_definition=PRIMARY_WEAK)

    repo_off = 0.5 * (torch.square(r1[offdiag]).mean() + torch.square(r2[offdiag]).mean())
    repo_diag = -m1.diag().mean() - m2.diag().mean()
    repo_orth = torch.square(gram[offdiag]).mean() - 2.0 * gram.diag().mean()
    repo_loss = repo_off + repo_diag + float(agent.cfg.ortho_coef) * repo_orth
    if not all(bool(torch.isfinite(value)) for value in (repo_off, repo_diag, repo_orth, repo_loss)):
        raise FloatingPointError("non-finite repository FB loss component")

    value_off_s = 0.5 * sum(
        (torch.square(rs[offdiag]) + rs[offdiag] * rw[offdiag]).mean()
        for rs, rw in ((r1s, r1w), (r2s, r2w))
    )
    value_off_w = 0.5 * sum(
        (torch.square(rw[offdiag]) + rs[offdiag] * rw[offdiag]).mean()
        for rs, rw in ((r1s, r1w), (r2s, r2w))
    )
    value_diag_s = -m1s.diag().mean() - m2s.diag().mean()
    value_diag_w = -m1w.diag().mean() - m2w.diag().mean()
    value_orth_s = (torch.square(gram_s[offdiag]) + gram_s[offdiag] * gram_w[offdiag]).mean() - 2 * gram_s.diag().mean()
    value_orth_w = (torch.square(gram_w[offdiag]) + gram_s[offdiag] * gram_w[offdiag]).mean() - 2 * gram_w.diag().mean()
    value_s = value_off_s + value_diag_s + float(agent.cfg.ortho_coef) * value_orth_s
    value_w = value_off_w + value_diag_w + float(agent.cfg.ortho_coef) * value_orth_w

    surrogate_off_s = sum((r.detach()[offdiag] * rs[offdiag]).mean() for r, rs in ((r1, r1s), (r2, r2s)))
    surrogate_off_w = sum((r.detach()[offdiag] * rw[offdiag]).mean() for r, rw in ((r1, r1w), (r2, r2w)))
    surrogate_orth_s = 2 * (gram.detach()[offdiag] * gram_s[offdiag]).mean() - 2 * gram_s.diag().mean()
    surrogate_orth_w = 2 * (gram.detach()[offdiag] * gram_w[offdiag]).mean() - 2 * gram_w.diag().mean()
    surrogate_s = surrogate_off_s + value_diag_s + float(agent.cfg.ortho_coef) * surrogate_orth_s
    surrogate_w = surrogate_off_w + value_diag_w + float(agent.cfg.ortho_coef) * surrogate_orth_w
    attributed_s = surrogate_s + (value_s - surrogate_s).detach()
    attributed_w = surrogate_w + (value_w - surrogate_w).detach()
    if not torch.allclose(value_s + value_w, repo_loss, rtol=3e-5, atol=3e-5):
        raise AssertionError("strong+weak FB values do not reproduce repository FB loss")
    if not torch.allclose(attributed_s + attributed_w, repo_loss, rtol=3e-5, atol=3e-5):
        raise AssertionError("corrected attribution values do not reproduce FB loss")
    fb_value_metrics = {
        "repo_fb_offdiag": float(repo_off.detach().item()),
        "repo_fb_diag": float(repo_diag.detach().item()),
        "repo_orth": float(repo_orth.detach().item()),
        "repo_fb_core": float(repo_loss.detach().item()),
        "value_strong_offdiag": float(value_off_s.detach().item()),
        "value_weak_offdiag": float(value_off_w.detach().item()),
        "value_strong_diag": float(value_diag_s.detach().item()),
        "value_weak_diag": float(value_diag_w.detach().item()),
        "value_strong_orth": float(value_orth_s.detach().item()),
        "value_weak_orth": float(value_orth_w.detach().item()),
        "value_strong_total": float(value_s.detach().item()),
        "value_weak_total": float(value_w.detach().item()),
        "value_additivity_abs_error": float(torch.abs(value_s + value_w - repo_loss).detach().item()),
        "attributed_additivity_abs_error": float(torch.abs(attributed_s + attributed_w - repo_loss).detach().item()),
    }
    for metric, scalar in fb_value_metrics.items():
        recorder.add("gradient_loss_decomposition", metric, scalar,
                     weak_definition=PRIMARY_WEAK, z_mode="training_mix_probe",
                     action_mode="replay")

    gfull = normalize_grads(torch.autograd.grad(repo_loss, fb_union, retain_graph=True, allow_unused=True), fb_union)
    gstrong = normalize_grads(torch.autograd.grad(attributed_s, fb_union, retain_graph=True, allow_unused=True), fb_union)
    gweak = normalize_grads(torch.autograd.grad(attributed_w, fb_union, allow_unused=True), fb_union)
    gradient_rows: List[Dict[str, Any]] = []
    for scope, indices in scopes.items():
        summary = subset_gradient_summary(gstrong, gweak, gfull, indices)
        if not all(math.isfinite(value) for value in summary.values()):
            raise FloatingPointError(f"non-finite FB gradient summary for {scope}: {summary}")
        if summary["gradient_additivity_relerr"] > 5e-5:
            raise AssertionError(f"FB gradient additivity failed for {scope}: {summary}")
        _append_gradient_row(
            gradient_rows, experiment, frame, "exact_fb_strong_weak_path", scope,
            {**summary, **fb_value_metrics},
            "Detached Pweak; cross terms value-split symmetrically; stop-gradient path surrogate; no step.",
        )
        for metric in ("gstrong_norm", "gweak_norm", "gfull_norm", "cosine_strong_weak", "cancellation_ratio"):
            recorder.add("gradient_interference", metric, summary[metric], branch=scope,
                         weak_definition=PRIMARY_WEAK, z_mode="training_mix_probe",
                         action_mode="replay")
    fb_strong_by_id = {id(parameter): gstrong[index].detach() for index, parameter in enumerate(fb_union)}
    fb_weak_by_id = {id(parameter): gweak[index].detach() for index, parameter in enumerate(fb_union)}
    del gfull, gstrong, gweak

    # Exact actor objective attribution at the checkpoint parameter point.
    # We cannot reproduce the real post-FB-step actor parameter point because
    # optimizer steps are forbidden. Encoder input is detached exactly as in
    # update(); ForwardMap gradients are not stepped by update_actor.
    obs_actor = agent.aug_and_encode(raw_obs).detach()
    with _fork_rng(device):
        torch.manual_seed(seed + 3)
        stddev = utils.schedule(agent.cfg.stddev_schedule, frame // 2)
        actor_dist = agent.actor(obs_actor, z, stddev)
        actor_action = actor_dist.sample(clip=agent.cfg.stddev_clip)
    actor_f1, actor_f2 = agent.forward_net(obs_actor, z, actor_action)
    z_s, z_w = z @ strong_projector, z @ projector
    af1s, af1w = split(actor_f1)
    af2s, af2w = split(actor_f2)
    # Preserve update_actor's exact einsum reduction semantics; torch.sum can
    # differ enough in float32 to flip a near-tie twin winner.
    aq1 = torch.einsum("sd,sd->s", actor_f1, z)
    aq2 = torch.einsum("sd,sd->s", actor_f2, z)
    aq1s, aq1w = torch.einsum("sd,sd->s", af1s, z_s), torch.einsum("sd,sd->s", af1w, z_w)
    aq2s, aq2w = torch.einsum("sd,sd->s", af2s, z_s), torch.einsum("sd,sd->s", af2w, z_w)
    awinner1 = (aq1 < aq2).to(aq1.dtype) + 0.5 * (aq1 == aq2).to(aq1.dtype)
    awinner2 = 1.0 - awinner1
    actor_loss_s = -(awinner1 * aq1s + awinner2 * aq2s).mean()
    actor_loss_w = -(awinner1 * aq1w + awinner2 * aq2w).mean()
    actor_loss = -torch.minimum(aq1, aq2).mean()
    if not all(bool(torch.isfinite(value)) for value in (actor_loss_s, actor_loss_w, actor_loss)):
        raise FloatingPointError("non-finite actor objective component")
    if not torch.allclose(actor_loss_s + actor_loss_w, actor_loss, rtol=3e-5, atol=3e-5):
        raise AssertionError("actor strong+weak objective identity failed")
    actor_scope = optimizer_parameters(agent.actor_opt)
    actor_forward_union = unique_parameters(actor_scope, forward_params)
    actor_union_index = {id(parameter): index for index, parameter in enumerate(actor_forward_union)}
    actor_scopes = {
        "actor_optimizer_parameter_scope_at_checkpoint": [actor_union_index[id(parameter)] for parameter in actor_scope],
        "unstepped_forward_map": [actor_union_index[id(parameter)] for parameter in forward_params],
        "unstepped_forward_f1_exclusive_head": [actor_union_index[id(parameter)] for parameter in f1_head_params],
        "unstepped_forward_f2_exclusive_head": [actor_union_index[id(parameter)] for parameter in f2_head_params],
    }
    actor_full_grad = normalize_grads(torch.autograd.grad(actor_loss, actor_forward_union, retain_graph=True, allow_unused=True), actor_forward_union)
    actor_s_grad = normalize_grads(torch.autograd.grad(actor_loss_s, actor_forward_union, retain_graph=True, allow_unused=True), actor_forward_union)
    actor_w_grad = normalize_grads(torch.autograd.grad(actor_loss_w, actor_forward_union, allow_unused=True), actor_forward_union)
    for scope, indices in actor_scopes.items():
        summary = subset_gradient_summary(actor_s_grad, actor_w_grad, actor_full_grad, indices)
        if not all(math.isfinite(value) for value in summary.values()):
            raise FloatingPointError(f"non-finite actor gradient summary for {scope}: {summary}")
        if summary["gradient_additivity_relerr"] > 5e-5:
            raise AssertionError(f"actor gradient additivity failed for {scope}: {summary}")
        _append_gradient_row(
            gradient_rows, experiment, frame, "exact_actor_q_strong_weak", scope, summary,
            "Checkpoint-parameter objective only: no preceding FB step is simulated. In training actor params would step; ForwardMap gradient is discarded.",
        )
    actor_full_by_id = {
        id(parameter): actor_full_grad[index].detach()
        for index, parameter in enumerate(actor_forward_union)
        if id(parameter) in {id(item) for item in forward_params}
    }
    fs_norm, actor_norm, _, cosine_s_actor = gradient_dot_summary(fb_strong_by_id, actor_full_by_id)
    fw_norm, actor_norm_again, _, cosine_w_actor = gradient_dot_summary(fb_weak_by_id, actor_full_by_id)
    if not math.isclose(actor_norm, actor_norm_again, rel_tol=1e-8, abs_tol=1e-12):
        raise AssertionError("actor ForwardMap gradient norm changed across cosine calculations")
    if not all(math.isfinite(value) for value in (fs_norm, fw_norm, actor_norm, cosine_s_actor, cosine_w_actor)):
        raise FloatingPointError("non-finite FB/actor ForwardMap gradient comparison")
    cross = {
        "gstrong_norm": fs_norm, "gweak_norm": fw_norm,
        "gactor_norm": actor_norm, "cosine_strong_actor": cosine_s_actor,
        "cosine_weak_actor": cosine_w_actor,
        "actor_gradient_applied_to_scope": False,
    }
    _append_gradient_row(
        gradient_rows, experiment, frame, "fb_vs_actor_raw_backward", "unstepped_forward_map",
        cross, "Comparable raw gradients in ForwardMap parameter space, but actor optimizer never steps ForwardMap.",
    )
    for metric in ("gactor_norm", "cosine_strong_actor", "cosine_weak_actor"):
        recorder.add("gradient_interference", metric, cross[metric], branch="unstepped_forward_map",
                     weak_definition=PRIMARY_WEAK, z_mode="training_mix_probe", action_mode="actor_sample")
    _append_gradient_row(
        gradient_rows, experiment, frame, "repository_actor_encoder_path", "shared_encoder",
        {"gactor_norm": 0.0, "cosine_strong_actor": float("nan"), "cosine_weak_actor": float("nan"),
         "actor_gradient_applied_to_scope": False},
        "At any checkpoint parameter point update() calls update_actor(obs.detach(), z); actor-objective encoder gradient is exactly zero.",
    )
    recorder.add("gradient_interference", "gactor_norm", 0.0, branch="shared_encoder",
                 weak_definition=PRIMARY_WEAK, z_mode="training_mix_probe", action_mode="actor_sample",
                 note="Exact zero because update_actor receives obs.detach().")

    for parameter, previous in zip(fb_union, before_grads):
        if previous is None and parameter.grad is not None:
            raise AssertionError("functional autograd unexpectedly populated parameter.grad")
        if previous is not None and not torch.equal(previous, parameter.grad):
            raise AssertionError("functional autograd mutated a pre-existing parameter.grad")
    del actor_full_grad, actor_s_grad, actor_w_grad, fb_strong_by_id, fb_weak_by_id, actor_full_by_id
    return gradient_rows


def read_complete_numeric_csv(path: Path) -> Tuple[List[str], List[Dict[str, str]], int]:
    rows: List[Dict[str, str]] = []
    dropped = 0
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.reader(stream)
        try:
            header = next(reader)
        except StopIteration as error:
            raise ValueError(f"empty CSV {path}") from error
        for raw in reader:
            if len(raw) != len(header):
                dropped += 1
                continue
            rows.append(dict(zip(header, raw)))
    return header, rows, dropped


def nearest_numeric_row(
    rows: Sequence[Mapping[str, str]], metric: str, target_frame: int, max_gap: int
) -> Optional[Tuple[float, int, Optional[int]]]:
    candidates: List[Tuple[int, bool, int, float, Optional[int]]] = []
    for row in rows:
        try:
            frame = int(float(row["frame"]))
            value = float(row[metric])
            step = int(float(row["step"])) if row.get("step", "") != "" else None
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(value):
            candidates.append((abs(frame - target_frame), frame > target_frame, frame, value, step))
    if not candidates:
        return None
    gap, _, frame, value, step = min(candidates)
    if gap > max_gap:
        return None
    return value, frame, step


def add_wandb_alignment(
    recorder: MetricRecorder, experiment: Experiment, frame: int
) -> Dict[str, Any]:
    train_path = experiment.run_dir / "train.csv"
    eval_path = experiment.run_dir / "eval.csv"
    train_header, train_rows, train_dropped = read_complete_numeric_csv(train_path)
    eval_header, eval_rows, eval_dropped = read_complete_numeric_csv(eval_path)
    wandb_files = sorted(experiment.run_dir.glob("wandb/run-*/run-*.wandb"))
    if len(wandb_files) != 1:
        raise FileNotFoundError(f"expected exactly one local W&B binary under {experiment.run_dir}, got {wandb_files}")
    for metric in ("episode_reward", "episode_reward#std", "z_norm"):
        nearest = nearest_numeric_row(eval_rows, metric, frame, max_gap=10_000)
        if nearest is None:
            recorder.add("wandb_alignment", f"eval/{metric}", "", source="local_eval_csv_sibling_of_wandb",
                         note="metric missing; no value inferred")
            continue
        value, source_frame, source_step = nearest
        if metric == "episode_reward" and source_frame != frame:
            raise AssertionError(f"requested checkpoint {frame} lacks exact periodic eval reward")
        recorder.add("wandb_alignment", f"eval/{metric}", value,
                     source="local_eval_csv_sibling_of_wandb", source_frame=source_frame,
                     source_step="" if source_step is None else source_step, frame_delta=source_frame - frame,
                     note="Nearest finite local CSV metric; tie prefers earlier frame; no interpolation. W&B binary is provenance-hashed but not decoded.")
    for metric in TRAIN_METRICS:
        if metric not in train_header:
            recorder.add("wandb_alignment", f"train/{metric}", "", source="local_train_csv_sibling_of_wandb",
                         note="column absent from this run; no value inferred")
            continue
        nearest = nearest_numeric_row(train_rows, metric, frame, max_gap=1_000)
        if nearest is None:
            recorder.add("wandb_alignment", f"train/{metric}", "", source="local_train_csv_sibling_of_wandb",
                         note="no finite value within 1000 frames; no value inferred")
            continue
        value, source_frame, source_step = nearest
        recorder.add("wandb_alignment", f"train/{metric}", value,
                     source="local_train_csv_sibling_of_wandb", source_frame=source_frame,
                     source_step="" if source_step is None else source_step, frame_delta=source_frame - frame,
                     note="Nearest finite local CSV metric; tie prefers earlier frame; no interpolation. W&B binary is provenance-hashed but not decoded.")
    return {
        "train_csv": str(train_path.resolve()), "eval_csv": str(eval_path.resolve()),
        "train_header": train_header, "eval_header": eval_header,
        "train_complete_rows": len(train_rows), "eval_complete_rows": len(eval_rows),
        "train_dropped_incomplete_rows": train_dropped,
        "eval_dropped_incomplete_rows": eval_dropped,
        "wandb_binary": str(wandb_files[0].resolve()),
        "wandb_binary_size": wandb_files[0].stat().st_size,
        "wandb_binary_sha256": sha256_file(wandb_files[0]),
    }


def analyze_checkpoint(
    experiment: Experiment,
    frame: int,
    replay: ReplayData,
    task_reward: np.ndarray,
    args: argparse.Namespace,
) -> Tuple[CheckpointResult, Dict[str, Any]]:
    agent, info = load_agent(experiment, frame, args.device)
    recorder = MetricRecorder(experiment, frame, info, replay)
    recorder.add("checkpoint", "global_step", info["global_step"])
    recorder.add("checkpoint", "global_episode", info["global_episode"])
    recorder.add("checkpoint", "effective_fb_lr", info["effective_fb_lr"])
    recorder.add(
        "checkpoint",
        "agent_use_cls",
        int(info["agent_contract"]["use_cls"]),
        note=(
            "serialized dead option for the ordinary CNN Encoder; retained "
            "for exact checkpoint provenance"
        ),
    )
    recorder.add("checkpoint", "serialized_fb_lr_key_present", int(info["serialized_fb_lr_key_present"]))
    recorder.add("replay", "proxy_size", len(replay.obs))
    recorder.add("replay", "probe_size", args.probe_size)
    recorder.add("replay", "task_inference_size", args.inference_size)
    recorder.add("replay", "source_action_repeat", 1)
    recorder.add("replay", "target_run_action_repeat", 2)
    recorder.add("replay", "temporal_semantics_match", 0,
                 note="Continuous ExORL transition retained for action consistency; not historical replay timing.")
    recorder.add("gradient_probe", "base_seed", args.gradient_seed,
                 note="Derived seeds: NumPy mix mask=base+1, target action=base+2, actor action=base+3.")
    recorder.add("gradient_probe", "numpy_mix_seed", args.gradient_seed + 1)
    recorder.add("gradient_probe", "target_action_seed", args.gradient_seed + 2)
    recorder.add("gradient_probe", "actor_action_seed", args.gradient_seed + 3)

    random_f1, random_f2, backward = encode_random_replay(agent, replay, args.device, args.inference_chunk)
    covariance, eigvals, eigvecs, projectors = covariance_and_projectors(
        backward[: args.probe_size], recorder
    )
    task_z, task_z_raw, inference_log = infer_task_z_exact(
        agent, replay, backward, task_reward, args.inference_size, args.device
    )
    analyze_task_z_geometry(
        recorder,
        task_z,
        task_z_raw,
        covariance,
        eigvals,
        eigvecs,
        projectors,
        z_mode="task_proxy",
    )
    recorder.add("task_z", "reward_mean", float(task_reward[:args.inference_size].mean()), z_mode="task_proxy")
    recorder.add("task_z", "reward_max", float(task_reward[:args.inference_size].max()), z_mode="task_proxy")
    recorder.add(
        "task_z", "repository_inference_path_verified", 1, z_mode="task_proxy",
        note=("Exact infer_meta_from_obs_and_rewards method on fixed single-state reward-labelled ExORL proxy; "
              "not the unrecoverable historical eval z. " + inference_log.replace("\n", "; ")),
    )
    repeated_task_z = np.broadcast_to(task_z, (len(replay.obs), Z_DIM)).copy()
    task_replay_f1, task_replay_f2, task_replay_actions = encode_forward(
        agent, replay.obs[: args.probe_size], repeated_task_z[: args.probe_size],
        replay.action[: args.probe_size], args.device, args.inference_chunk, "replay"
    )
    task_actor_f1, task_actor_f2, task_actor_actions = encode_forward(
        agent, replay.obs, repeated_task_z, replay.action, args.device,
        args.inference_chunk, "actor_mean"
    )

    for branch, value in (("f1", random_f1), ("f2", random_f2)):
        analyze_forward_geometry(
            recorder, value[: args.probe_size], replay.random_z[: args.probe_size],
            projectors, branch=branch, z_mode="fixed_random", action_mode="replay"
        )
    for branch, value in (("f1", task_replay_f1), ("f2", task_replay_f2)):
        analyze_forward_geometry(
            recorder, value, repeated_task_z[: args.probe_size], projectors,
            branch=branch, z_mode="task_proxy", action_mode="replay"
        )
    for branch, value in (("f1", task_actor_f1), ("f2", task_actor_f2)):
        analyze_forward_geometry(
            recorder, value[: args.probe_size], repeated_task_z[: args.probe_size],
            projectors, branch=branch, z_mode="task_proxy", action_mode="actor_mean"
        )

    extremes: Dict[str, np.ndarray] = {}
    extreme_json: List[Dict[str, Any]] = []
    analyze_large_m(recorder, random_f1, backward, eigvals, eigvecs, projectors, replay,
                    branch="f1", device=args.device, extremes=extremes, extreme_json=extreme_json)
    analyze_large_m(recorder, random_f2, backward, eigvals, eigvecs, projectors, replay,
                    branch="f2", device=args.device, extremes=extremes, extreme_json=extreme_json)
    analyze_q(
        recorder, random_f1, random_f2, replay.random_z, replay.action, projectors,
        replay, z_mode="fixed_random", action_mode="replay", extremes=extremes,
        extreme_json=extreme_json,
    )
    analyze_q(
        recorder, task_replay_f1, task_replay_f2, repeated_task_z[: args.probe_size],
        task_replay_actions, projectors, replay, z_mode="task_proxy", action_mode="replay",
        extremes=extremes, extreme_json=extreme_json,
    )
    analyze_q(
        recorder, task_actor_f1, task_actor_f2, repeated_task_z, task_actor_actions,
        projectors, replay, z_mode="task_proxy", action_mode="actor_mean",
        extremes=extremes, extreme_json=extreme_json,
    )
    gradient_rows = (
        analyze_gradients(
            agent,
            replay,
            projectors[PRIMARY_WEAK],
            experiment,
            frame,
            args.device,
            args.gradient_seed,
            recorder,
        )
        if experiment.analyze_gradient_attribution
        else []
    )
    wandb_provenance = add_wandb_alignment(recorder, experiment, frame)
    for item in extreme_json:
        item.update({"experiment": experiment.key, "run_id": experiment.run_id,
                     "task": experiment.task, "checkpoint_frame": frame})
    result = CheckpointResult(
        experiment=experiment, frame=frame, checkpoint_path=Path(info["path"]),
        checkpoint_sha256=str(info["sha256"]), checkpoint_size=int(info["size"]),
        global_step=int(info["global_step"]), global_episode=int(info["global_episode"]),
        task_z=task_z, task_z_raw=task_z_raw, covariance=covariance, eigvals=eigvals,
        eigvecs=eigvecs, projectors=projectors, metrics=recorder.rows,
        extremes=extremes, extreme_json=extreme_json, gradient_rows=gradient_rows,
    )
    del agent, random_f1, random_f2, backward, task_replay_f1, task_replay_f2
    del task_actor_f1, task_actor_f2, task_actor_actions
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
    return result, wandb_provenance


def _definition_basis(result: CheckpointResult, definition: str) -> np.ndarray:
    if definition == "bottom5":
        return result.eigvecs[:, :5]
    if definition == "bottom10":
        return result.eigvecs[:, :10]
    if definition == "top10":
        return result.eigvecs[:, -10:]
    if definition == "top20":
        return result.eigvecs[:, -20:]
    alpha = {"thr003": 0.03, "thr010": 0.10, "thr030": 0.30}[definition]
    return result.eigvecs[:, result.eigvals < alpha * result.eigvals[-1]]


def _append_result_metric(
    result: CheckpointResult, section: str, metric: str, value: Any,
    *, weak_definition: str = "", note: str = ""
) -> None:
    template = dict(result.metrics[0])
    template.update({
        "section": section, "z_mode": "", "action_mode": "", "branch": "",
        "weak_definition": weak_definition, "tail_group": "", "metric": metric,
        "value": value, "unit": "", "source": "offline_checkpoint_analysis",
        "source_frame": "", "source_step": "", "frame_delta": "", "note": note,
    })
    result.metrics.append(template)


def add_consecutive_drift(results: Sequence[CheckpointResult]) -> None:
    grouped: Dict[str, List[CheckpointResult]] = {}
    for result in results:
        grouped.setdefault(result.experiment.key, []).append(result)
    definitions = (*WEAK_DEFINITIONS, "top10", "top20")
    for experiment_results in grouped.values():
        experiment_results.sort(key=lambda item: item.frame)
        previous: Optional[CheckpointResult] = None
        for current in experiment_results:
            for definition in definitions:
                current_basis = _definition_basis(current, definition)
                if previous is None:
                    metrics = {
                        "previous_frame": "", "previous_dimension": "",
                        "current_dimension": current_basis.shape[1],
                        "principal_angle_mean_deg": "", "principal_angle_rms_deg": "",
                        "principal_angle_max_deg": "", "projector_distance_fro": "",
                        "projector_distance_normalized": "",
                    }
                else:
                    previous_basis = _definition_basis(previous, definition)
                    if previous_basis.shape[1] and current_basis.shape[1]:
                        singular = np.linalg.svd(previous_basis.T @ current_basis, compute_uv=False)
                        angles = np.arccos(np.clip(singular, 0.0, 1.0))
                        degrees = np.rad2deg(angles)
                    else:
                        degrees = np.asarray([], dtype=np.float64)
                    previous_projector = previous_basis @ previous_basis.T
                    current_projector = current_basis @ current_basis.T
                    distance = float(np.linalg.norm(current_projector - previous_projector, ord="fro"))
                    normalizer = math.sqrt(max(previous_basis.shape[1] + current_basis.shape[1], 1))
                    metrics = {
                        "previous_frame": previous.frame,
                        "previous_dimension": previous_basis.shape[1],
                        "current_dimension": current_basis.shape[1],
                        "principal_angle_mean_deg": float(degrees.mean()) if degrees.size else "",
                        "principal_angle_rms_deg": float(np.sqrt(np.mean(np.square(degrees)))) if degrees.size else "",
                        "principal_angle_max_deg": float(degrees.max()) if degrees.size else "",
                        "projector_distance_fro": distance,
                        "projector_distance_normalized": distance / normalizer,
                    }
                for metric, value in metrics.items():
                    _append_result_metric(current, "b_subspace_drift", metric, value,
                                          weak_definition=definition,
                                          note="Principal angles use min(rank_prev,rank_cur); projector distance captures rank changes.")
            if previous is not None:
                cosine = float(np.dot(previous.task_z, current.task_z) /
                               (np.linalg.norm(previous.task_z) * np.linalg.norm(current.task_z)))
                _append_result_metric(current, "task_z_drift", "consecutive_cosine", cosine,
                                      note=f"previous checkpoint frame={previous.frame}")
            previous = current


SIGNATURE_FIELDS = (
    "section", "z_mode", "action_mode", "branch", "weak_definition",
    "tail_group", "metric",
)


def metric_signature(row: Mapping[str, Any]) -> Tuple[str, ...]:
    return tuple(str(row.get(field, "")) for field in SIGNATURE_FIELDS)


def numeric_value(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def get_metric(
    result: CheckpointResult,
    section: str,
    metric: str,
    *,
    branch: str = "",
    z_mode: str = "",
    action_mode: str = "",
    weak_definition: str = "",
    tail_group: str = "",
) -> float:
    matches = [
        numeric_value(row["value"])
        for row in result.metrics
        if row["section"] == section and row["metric"] == metric
        and row["branch"] == branch and row["z_mode"] == z_mode
        and row["action_mode"] == action_mode
        and row["weak_definition"] == weak_definition
        and row["tail_group"] == tail_group
    ]
    matches = [value for value in matches if value is not None]
    if len(matches) != 1:
        raise KeyError(
            f"expected one metric {section}/{metric}/{branch}/{z_mode}/{action_mode}/"
            f"{weak_definition}/{tail_group}, got {matches}"
        )
    return matches[0]


def get_metric_allow_nonfinite(
    result: CheckpointResult,
    section: str,
    metric: str,
    *,
    branch: str = "",
    z_mode: str = "",
    action_mode: str = "",
    weak_definition: str = "",
    tail_group: str = "",
) -> float:
    """Return one numeric metric while preserving a scientifically valid +/-inf."""
    matches = [
        float(row["value"])
        for row in result.metrics
        if row["section"] == section and row["metric"] == metric
        and row["branch"] == branch and row["z_mode"] == z_mode
        and row["action_mode"] == action_mode
        and row["weak_definition"] == weak_definition
        and row["tail_group"] == tail_group
    ]
    if len(matches) != 1 or math.isnan(matches[0]):
        raise KeyError(
            f"expected one numeric metric {section}/{metric}/{branch}/{z_mode}/"
            f"{action_mode}/{weak_definition}/{tail_group}, got {matches}"
        )
    return matches[0]


def save_checkpoint_artifacts(output_dir: Path, result: CheckpointResult) -> None:
    stem = f"{result.experiment.key}_{result.frame:07d}"
    directory = output_dir / "per_checkpoint"
    atomic_csv(directory / f"{stem}_metrics.csv", result.metrics)
    if result.gradient_rows:
        atomic_csv(directory / f"{stem}_gradients.csv", result.gradient_rows)
    atomic_json(directory / f"{stem}_extremes.json", result.extreme_json)
    arrays: Dict[str, np.ndarray] = {
        "covariance": result.covariance, "eigvals": result.eigvals,
        "eigvecs": result.eigvecs, "task_z": result.task_z,
        "task_z_raw": result.task_z_raw,
    }
    arrays.update({f"projector_{name}": value for name, value in result.projectors.items()})
    arrays.update(result.extremes)
    atomic_npz(directory / f"{stem}_arrays.npz", **arrays)


def build_matched_comparison(results: Sequence[CheckpointResult]) -> List[Dict[str, Any]]:
    lower = {result.frame: result for result in results if result.experiment.key == "run_lr5e5"}
    default = {result.frame: result for result in results if result.experiment.key == "run_lr1e4"}
    frames = sorted(set(lower) & set(default))
    expected = [100_000, 200_000, 500_000, 800_000, 1_000_000, 1_500_000]
    if frames != expected:
        raise AssertionError(f"matched LR frames differ from requested primary set: {frames}")
    if 2_000_000 in frames:
        raise AssertionError("lower-LR-only 2M checkpoint must not enter matched comparison")
    rows: List[Dict[str, Any]] = []
    for frame in frames:
        low_map = {metric_signature(row): row for row in lower[frame].metrics}
        default_map = {metric_signature(row): row for row in default[frame].metrics}
        for signature in sorted(set(low_map) | set(default_map)):
            low_value = numeric_value(low_map.get(signature, {}).get("value", ""))
            default_value = numeric_value(default_map.get(signature, {}).get("value", ""))
            section, z_mode, action_mode, branch, weak_definition, tail_group, metric = signature
            delta = "" if low_value is None or default_value is None else low_value - default_value
            ratio = "" if low_value is None or default_value in (None, 0.0) else low_value / default_value
            rows.append({
                "schema_version": SCHEMA_VERSION, "checkpoint_frame": frame,
                "matched_frame": True, "task": "cheetah_run",
                "lower_lr_run_id": lower[frame].experiment.run_id,
                "default_lr_run_id": default[frame].experiment.run_id,
                "lower_fb_lr": 5e-5, "default_fb_lr": 1e-4,
                "section": section, "z_mode": z_mode, "action_mode": action_mode,
                "branch": branch, "weak_definition": weak_definition,
                "tail_group": tail_group, "metric": metric,
                "lower_lr_value": "" if low_value is None else low_value,
                "default_lr_value": "" if default_value is None else default_value,
                "lower_minus_default": delta, "lower_over_default": ratio,
                "missing_policy": "blank means source metric absent/nonfinite; never inferred",
            })
    return rows


def serialize_aggregate(output_dir: Path, results: Sequence[CheckpointResult]) -> Dict[str, str]:
    long_rows = [row for result in results for row in result.metrics]
    gradient_rows = [row for result in results for row in result.gradient_rows]
    matched_rows = build_matched_comparison(results)
    long_path = output_dir / "checkpoint_diagnostics_long.csv"
    matched_path = output_dir / "matched_lr_comparison.csv"
    gradient_path = output_dir / "gradient_interference.csv"
    atomic_csv(long_path, long_rows)
    atomic_csv(matched_path, matched_rows)
    atomic_csv(gradient_path, gradient_rows)

    eigen_arrays: Dict[str, np.ndarray] = {}
    extreme_arrays: Dict[str, np.ndarray] = {}
    extreme_records: List[Dict[str, Any]] = []
    for result in results:
        prefix = f"{result.experiment.key}__frame_{result.frame}"
        eigen_arrays[f"{prefix}__covariance"] = result.covariance
        eigen_arrays[f"{prefix}__eigvals_ascending"] = result.eigvals
        eigen_arrays[f"{prefix}__eigvecs_columns_ascending"] = result.eigvecs
        eigen_arrays[f"{prefix}__task_z_proxy"] = result.task_z
        eigen_arrays[f"{prefix}__task_z_raw"] = result.task_z_raw
        for definition, projector in result.projectors.items():
            eigen_arrays[f"{prefix}__projector_{definition}"] = projector
        for name, value in result.extremes.items():
            extreme_arrays[f"{prefix}__{name}"] = value
        extreme_records.extend(result.extreme_json)
    eigen_path = output_dir / "eigen_projectors_and_task_z.npz"
    extreme_npz_path = output_dir / "extreme_examples.npz"
    extreme_json_path = output_dir / "extreme_examples.json"
    atomic_npz(eigen_path, **eigen_arrays)
    atomic_npz(extreme_npz_path, **extreme_arrays)
    atomic_json(extreme_json_path, extreme_records)
    return {
        "long_csv": str(long_path), "matched_csv": str(matched_path),
        "gradient_csv": str(gradient_path), "eigen_npz": str(eigen_path),
        "extreme_npz": str(extreme_npz_path), "extreme_json": str(extreme_json_path),
    }


def _series(
    results: Sequence[CheckpointResult], section: str, metric: str, **dimensions: str
) -> Tuple[np.ndarray, np.ndarray]:
    ordered = sorted(results, key=lambda item: item.frame)
    return (
        np.asarray([item.frame / 1e6 for item in ordered]),
        np.asarray([get_metric(item, section, metric, **dimensions) for item in ordered]),
    )


def plot_spectra(results: Sequence[CheckpointResult], output: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.4), sharey=True)
    for ax, experiment in zip(axes, EXPERIMENTS):
        selected = sorted((item for item in results if item.experiment.key == experiment.key), key=lambda item: item.frame)
        for item in selected:
            ax.semilogy(np.arange(1, Z_DIM + 1), np.maximum(item.eigvals[::-1], 1e-16),
                        label=f"{item.frame / 1e6:g}M")
        ax.set(title=experiment.key, xlabel="eigenvalue rank (largest first)")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, ncol=2)
    axes[0].set_ylabel("eigenvalue of C_B (log scale)")
    figure.suptitle("B second-moment spectra on the common 1024-state proxy")
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_walk_mechanisms(results: Sequence[CheckpointResult], output: Path) -> None:
    selected = [item for item in results if item.experiment.key == "walk_lr1e4"]
    frames = np.asarray([item.frame / 1e6 for item in sorted(selected, key=lambda x: x.frame)])
    ordered = sorted(selected, key=lambda x: x.frame)
    figure, axes = plt.subplots(3, 2, figsize=(12, 11))
    reward = [get_metric(item, "wandb_alignment", "eval/episode_reward") for item in ordered]
    axes[0, 0].plot(frames, reward, marker="o")
    axes[0, 0].set(title="Nearest (exact-frame) eval reward", ylabel="episode reward")
    condition = [get_metric(item, "b_geometry", "effective_condition_number") for item in ordered]
    pr = [get_metric(item, "b_geometry", "participation_ratio") for item in ordered]
    axes[0, 1].semilogy(frames, condition, marker="o", color="tab:red", label="effective condition")
    twin = axes[0, 1].twinx(); twin.plot(frames, pr, marker="s", color="tab:blue", label="participation ratio")
    axes[0, 1].set(title="B conditioning"); twin.set_ylabel("participation ratio")
    for definition, marker in (("bottom5", "o"), ("bottom10", "s"), ("thr010", "^")):
        values = [numeric_value(next(row["value"] for row in item.metrics
                    if row["section"] == "b_subspace_drift" and row["metric"] == "projector_distance_normalized"
                    and row["weak_definition"] == definition)) for item in ordered]
        axes[1, 0].plot(frames, [np.nan if value is None else value for value in values], marker=marker, label=definition)
    axes[1, 0].set(title="Consecutive weak-B projector drift", ylabel="normalized Frobenius distance")
    axes[1, 0].legend()
    for branch in ("f1", "f2"):
        rho = [get_metric(item, "f_geometry", "rho_dangerous", branch=branch,
                          z_mode="task_proxy", action_mode="actor_mean",
                          weak_definition=PRIMARY_WEAK) for item in ordered]
        axes[1, 1].plot(frames, rho, marker="o", label=branch.upper())
    axes[1, 1].set(title="Policy-path F dangerous energy", ylabel="rho_dangerous")
    axes[1, 1].legend()
    for branch in ("f1", "f2"):
        tail = [get_metric(item, "large_m_tail", "abs_mweak_over_abs_m_mean", branch=branch,
                           z_mode="fixed_random", action_mode="replay", weak_definition=PRIMARY_WEAK,
                           tail_group="top0.01pct") for item in ordered]
        axes[2, 0].plot(frames, tail, marker="o", label=branch.upper())
    axes[2, 0].set(title="Extreme-M weak contribution", ylabel="mean |Mweak|/(|M|+eps)")
    axes[2, 0].legend()
    qweak = [get_metric(item, "q_decomposition", "q_weak_rms", branch="twin_min",
                        z_mode="task_proxy", action_mode="actor_mean",
                        weak_definition=PRIMARY_WEAK) for item in ordered]
    axes[2, 1].plot(frames, qweak, marker="o", color="tab:purple", label="task Qweak RMS")
    grad = [get_metric(item, "gradient_interference", "gweak_norm", branch="actual_fb_update_union",
                       z_mode="training_mix_probe", action_mode="replay",
                       weak_definition=PRIMARY_WEAK) for item in ordered]
    twin2 = axes[2, 1].twinx(); twin2.semilogy(frames, grad, marker="s", color="tab:orange", label="FB gweak")
    axes[2, 1].set(title="Qweak and weak-path gradient", ylabel="Qweak RMS"); twin2.set_ylabel("gweak norm")
    for ax in axes.flat:
        ax.set_xlabel("training frames (millions)"); ax.grid(alpha=0.25)
    figure.suptitle("Cheetah Walk mechanism probes")
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_lr_comparison(results: Sequence[CheckpointResult], output: Path) -> None:
    figure, axes = plt.subplots(3, 2, figsize=(12, 11))
    styles = (("run_lr5e5", "FB LR 5e-5", "tab:blue"), ("run_lr1e4", "FB LR 1e-4", "tab:red"))
    for key, label, color in styles:
        selected = sorted((item for item in results if item.experiment.key == key), key=lambda item: item.frame)
        frames = np.asarray([item.frame / 1e6 for item in selected])
        axes[0, 0].plot(frames, [get_metric(item, "wandb_alignment", "eval/episode_reward") for item in selected],
                        marker="o", label=label, color=color)
        axes[0, 1].semilogy(frames, [get_metric(item, "b_geometry", "effective_condition_number") for item in selected],
                            marker="o", label=label, color=color)
        for branch, linestyle in (("f1", "-"), ("f2", "--")):
            branch_label = f"{label} {branch.upper()}"
            axes[1, 0].plot(frames, [get_metric(item, "f_geometry", "rho_dangerous", branch=branch,
                                  z_mode="task_proxy", action_mode="actor_mean",
                                  weak_definition=PRIMARY_WEAK) for item in selected],
                                  marker="o", linestyle=linestyle, label=branch_label, color=color)
            axes[1, 1].semilogy(frames, [get_metric(item, "large_m", "m_abs_p99_99", branch=branch,
                                     z_mode="fixed_random", action_mode="replay") for item in selected],
                                     marker="o", linestyle=linestyle, label=branch_label, color=color)
            axes[2, 0].plot(frames, [get_metric(item, "large_m_tail", "abs_mweak_over_abs_m_mean",
                                    branch=branch, z_mode="fixed_random", action_mode="replay",
                                    weak_definition=PRIMARY_WEAK, tail_group="top0.01pct")
                                    for item in selected], marker="o", linestyle=linestyle,
                                    label=branch_label, color=color)
        axes[2, 1].semilogy(frames, [get_metric(item, "gradient_interference", "gweak_norm",
                                branch="actual_fb_update_union", z_mode="training_mix_probe",
                                action_mode="replay", weak_definition=PRIMARY_WEAK) for item in selected],
                                marker="o", label=label, color=color)
    titles = (
        "Evaluation reward", "B effective condition number", "Task/actor rho_dangerous",
        "Large-M |M| p99.99", "Top-0.01% weak M contribution", "Weak-path FB gradient norm",
    )
    for ax, title in zip(axes.flat, titles):
        ax.set(title=title, xlabel="training frames (millions)"); ax.grid(alpha=0.25); ax.legend(fontsize=8)
    figure.suptitle("Cheetah Run matched FB learning-rate comparison (2M blue point is unmatched)")
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_geometry_drift_overview(results: Sequence[CheckpointResult], output: Path) -> None:
    """Cover the requested B-tail, F-energy, and weak-drift timelines for every run."""
    figure, axes = plt.subplots(3, 3, figsize=(16, 12), squeeze=False)
    for column, experiment in enumerate(EXPERIMENTS):
        selected = sorted(
            (item for item in results if item.experiment.key == experiment.key),
            key=lambda item: item.frame,
        )
        frames = np.asarray([item.frame / 1e6 for item in selected])

        axis = axes[0, column]
        axis.semilogy(
            frames,
            [get_metric(item, "b_geometry", "lambda_min") for item in selected],
            marker="o", color="tab:red", label="lambda_min",
        )
        trace_axis = axis.twinx()
        trace_axis.plot(
            frames,
            [get_metric(item, "b_geometry", "weak_trace_fraction",
                        weak_definition=PRIMARY_WEAK) for item in selected],
            marker="s", color="tab:blue", label="weak trace fraction",
        )
        axis.set_yscale("log")
        axis.set_title(experiment.key)
        if column == 0:
            axis.set_ylabel("lambda_min (log)")
        if column == 2:
            trace_axis.set_ylabel("thr010 trace fraction")
        axis.legend(loc="lower left", fontsize=7)
        trace_axis.legend(loc="upper right", fontsize=7)

        axis = axes[1, column]
        for branch, color in (("f1", "tab:blue"), ("f2", "tab:orange")):
            axis.plot(
                frames,
                [get_metric(item, "f_geometry", "weak_energy_ratio", branch=branch,
                            z_mode="task_proxy", action_mode="actor_mean",
                            weak_definition=PRIMARY_WEAK) for item in selected],
                marker="o", color=color, label=f"{branch.upper()} weak energy",
            )
            axis.plot(
                frames,
                [get_metric(item, "f_geometry", "rho_dangerous", branch=branch,
                            z_mode="task_proxy", action_mode="actor_mean",
                            weak_definition=PRIMARY_WEAK) for item in selected],
                marker="s", linestyle="--", color=color, label=f"{branch.upper()} rho dangerous",
            )
        if column == 0:
            axis.set_ylabel("energy fraction")
        axis.legend(fontsize=6.5)

        axis = axes[2, column]
        for definition, marker in (("bottom5", "o"), ("bottom10", "s"), (PRIMARY_WEAK, "^")):
            values = [
                numeric_value(next(
                    row["value"] for row in item.metrics
                    if row["section"] == "b_subspace_drift"
                    and row["metric"] == "projector_distance_normalized"
                    and row["weak_definition"] == definition
                ))
                for item in selected
            ]
            axis.plot(
                frames, [np.nan if value is None else value for value in values],
                marker=marker, label=definition,
            )
        if column == 0:
            axis.set_ylabel("raw projector distance")
        axis.legend(fontsize=7)

    for axis in axes.flat:
        axis.set_xlabel("training frames (millions)")
        axis.grid(alpha=0.25)
    figure.suptitle(
        "B-tail, task/actor F geometry, and raw weak-subspace drift "
        "(drift is not latent-gauge aligned)"
    )
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_f_m_q_tail_overview(results: Sequence[CheckpointResult], output: Path) -> None:
    """Cover F/M/Q tail statistics with F1 and F2 kept visually separate."""
    figure, axes = plt.subplots(3, 3, figsize=(16, 12), squeeze=False)
    for column, experiment in enumerate(EXPERIMENTS):
        selected = sorted(
            (item for item in results if item.experiment.key == experiment.key),
            key=lambda item: item.frame,
        )
        frames = np.asarray([item.frame / 1e6 for item in selected])

        axis = axes[0, column]
        for branch, color in (("f1", "tab:blue"), ("f2", "tab:orange")):
            axis.semilogy(
                frames,
                [get_metric(item, "f_geometry", "f_norm_p99", branch=branch,
                            z_mode="task_proxy", action_mode="actor_mean") for item in selected],
                marker="o", color=color, label=f"{branch.upper()} norm p99",
            )
        axis.set_title(experiment.key)
        if column == 0:
            axis.set_ylabel("actor F norm p99 (log)")
        axis.legend(fontsize=7)

        axis = axes[1, column]
        for branch, color in (("f1", "tab:blue"), ("f2", "tab:orange")):
            axis.semilogy(
                frames,
                [get_metric(item, "large_m", "m_abs_p99_9", branch=branch,
                            z_mode="fixed_random", action_mode="replay") for item in selected],
                marker="o", color=color, label=f"{branch.upper()} p99.9",
            )
            axis.semilogy(
                frames,
                [get_metric(item, "large_m", "m_abs_p99_99", branch=branch,
                            z_mode="fixed_random", action_mode="replay") for item in selected],
                marker="s", linestyle="--", color=color, label=f"{branch.upper()} p99.99",
            )
        if column == 0:
            axis.set_ylabel("large-replay |M| tail (log)")
        axis.legend(fontsize=6.5)

        axis = axes[2, column]
        for branch, color in (("f1", "tab:blue"), ("f2", "tab:orange")):
            axis.semilogy(
                frames,
                [get_metric(item, "q", "q_abs_p99", branch=branch,
                            z_mode="task_proxy", action_mode="actor_mean") for item in selected],
                marker="o", color=color, label=f"{branch.upper()} |Q| p99",
            )
            axis.semilogy(
                frames,
                [get_metric(item, "q_decomposition", "q_weak_abs_p99", branch=branch,
                            z_mode="task_proxy", action_mode="actor_mean",
                            weak_definition=PRIMARY_WEAK) for item in selected],
                marker="s", linestyle="--", color=color, label=f"{branch.upper()} |Qweak| p99",
            )
        if column == 0:
            axis.set_ylabel("task/actor Q tail (log)")
        axis.legend(fontsize=6.5)

    for axis in axes.flat:
        axis.set_xlabel("training frames (millions)")
        axis.grid(alpha=0.25)
    figure.suptitle("F/M/Q tail statistics on the common proxy")
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_gradient_cosines(results: Sequence[CheckpointResult], output: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    for ax, experiment in zip(axes, EXPERIMENTS):
        selected = sorted((item for item in results if item.experiment.key == experiment.key), key=lambda item: item.frame)
        frames = [item.frame / 1e6 for item in selected]
        for scope, label in (("shared_encoder", "encoder gS/gW"), ("forward_map", "ForwardMap gS/gW")):
            values = [get_metric(item, "gradient_interference", "cosine_strong_weak", branch=scope,
                                 z_mode="training_mix_probe", action_mode="replay",
                                 weak_definition=PRIMARY_WEAK) for item in selected]
            ax.plot(frames, values, marker="o", label=label)
        actor = [get_metric(item, "gradient_interference", "cosine_weak_actor",
                            branch="unstepped_forward_map", z_mode="training_mix_probe",
                            action_mode="actor_sample", weak_definition=PRIMARY_WEAK) for item in selected]
        ax.plot(frames, actor, marker="s", linestyle="--", label="gW/actor raw F grad")
        ax.axhline(0, color="black", linewidth=0.8); ax.set(title=experiment.key, xlabel="frames (M)")
        ax.grid(alpha=0.25); ax.legend(fontsize=7)
    axes[0].set_ylabel("gradient cosine")
    figure.suptitle("Gradient interference (actor/ForwardMap curve is unstepped diagnostic)")
    figure.tight_layout(); figure.savefig(output, dpi=180, bbox_inches="tight"); plt.close(figure)


def generate_report(results: Sequence[CheckpointResult], replay: ReplayData) -> str:
    by_key = {key: sorted((item for item in results if item.experiment.key == key), key=lambda x: x.frame)
              for key in ("walk_lr1e4", "run_lr5e5", "run_lr1e4")}
    walk = by_key["walk_lr1e4"]
    rewards = [get_metric(item, "wandb_alignment", "eval/episode_reward") for item in walk]
    peak_index = int(np.argmax(rewards))
    peak, final = walk[peak_index], walk[-1]
    cond_peak = get_metric(peak, "b_geometry", "effective_condition_number")
    cond_final = get_metric(final, "b_geometry", "effective_condition_number")
    branches = ("f1", "f2")
    rho_peak = {b: get_metric(peak, "f_geometry", "rho_dangerous", branch=b,
                z_mode="task_proxy", action_mode="actor_mean", weak_definition=PRIMARY_WEAK)
                for b in branches}
    rho_final = {b: get_metric(final, "f_geometry", "rho_dangerous", branch=b,
                 z_mode="task_proxy", action_mode="actor_mean", weak_definition=PRIMARY_WEAK)
                 for b in branches}
    lambda_min_peak = get_metric(peak, "b_geometry", "lambda_min")
    lambda_min_final = get_metric(final, "b_geometry", "lambda_min")
    pr_peak = get_metric(peak, "b_geometry", "participation_ratio")
    pr_final = get_metric(final, "b_geometry", "participation_ratio")
    trace_fraction_peak = get_metric(
        peak, "b_geometry", "weak_trace_fraction", weak_definition=PRIMARY_WEAK
    )
    trace_fraction_final = get_metric(
        final, "b_geometry", "weak_trace_fraction", weak_definition=PRIMARY_WEAK
    )
    m_tail_final = {b: get_metric(final, "large_m_tail", "weak_contribution_disproportion_vs_all",
                    branch=b, z_mode="fixed_random", action_mode="replay",
                    weak_definition=PRIMARY_WEAK, tail_group="top0.01pct") for b in branches}
    m_p9999_peak = {b: get_metric(peak, "large_m", "m_abs_p99_99", branch=b,
                    z_mode="fixed_random", action_mode="replay") for b in branches}
    m_p9999_final = {b: get_metric(final, "large_m", "m_abs_p99_99", branch=b,
                     z_mode="fixed_random", action_mode="replay") for b in branches}
    q_total_peak = get_metric(peak, "q", "q_rms", branch="twin_min",
                              z_mode="task_proxy", action_mode="actor_mean")
    q_total_final = get_metric(final, "q", "q_rms", branch="twin_min",
                               z_mode="task_proxy", action_mode="actor_mean")
    q_mean_peak = get_metric(peak, "q", "q_mean", branch="twin_min",
                             z_mode="task_proxy", action_mode="actor_mean")
    q_mean_final = get_metric(final, "q", "q_mean", branch="twin_min",
                              z_mode="task_proxy", action_mode="actor_mean")
    q_rms_peak = get_metric(peak, "q_decomposition", "q_weak_rms", branch="twin_min",
                            z_mode="task_proxy", action_mode="actor_mean",
                            weak_definition=PRIMARY_WEAK)
    q_rms_final = get_metric(final, "q_decomposition", "q_weak_rms", branch="twin_min",
                             z_mode="task_proxy", action_mode="actor_mean",
                             weak_definition=PRIMARY_WEAK)
    strict_qweak_final = {
        definition: get_metric(
            final, "q_decomposition", "q_weak_rms", branch="twin_min",
            z_mode="task_proxy", action_mode="actor_mean", weak_definition=definition
        )
        for definition in ("bottom5", "bottom10")
    }
    online_m_peak = get_metric(peak, "wandb_alignment", "train/M_online_abs_max")
    online_m_final = get_metric(final, "wandb_alignment", "train/M_online_abs_max")
    union_gweak_peak = get_metric(
        peak, "gradient_interference", "gweak_norm", branch="actual_fb_update_union",
        z_mode="training_mix_probe", action_mode="replay", weak_definition=PRIMARY_WEAK
    )
    union_gweak_final = get_metric(
        final, "gradient_interference", "gweak_norm", branch="actual_fb_update_union",
        z_mode="training_mix_probe", action_mode="replay", weak_definition=PRIMARY_WEAK
    )
    union_cos_peak = get_metric(
        peak, "gradient_interference", "cosine_strong_weak", branch="actual_fb_update_union",
        z_mode="training_mix_probe", action_mode="replay", weak_definition=PRIMARY_WEAK
    )
    union_cos_final = get_metric(
        final, "gradient_interference", "cosine_strong_weak", branch="actual_fb_update_union",
        z_mode="training_mix_probe", action_mode="replay", weak_definition=PRIMARY_WEAK
    )
    weak_dimensions = [int(get_metric(item, "b_geometry", "weak_dimension",
                                      weak_definition=PRIMARY_WEAK)) for item in walk]
    post_peak_drift_values = [
        get_metric(item, "b_subspace_drift", "projector_distance_normalized",
                   weak_definition=PRIMARY_WEAK)
        for item in walk[peak_index + 1:]
    ]
    common_frames = (100_000, 200_000, 500_000, 800_000, 1_000_000, 1_500_000)
    lower = {item.frame: item for item in by_key["run_lr5e5"]}
    default = {item.frame: item for item in by_key["run_lr1e4"]}
    def matched_median(section: str, metric: str, **dims: str) -> Tuple[float, float]:
        low = [get_metric(lower[frame], section, metric, **dims) for frame in common_frames]
        high = [get_metric(default[frame], section, metric, **dims) for frame in common_frames]
        return float(np.median(low)), float(np.median(high))
    def matched_values(section: str, metric: str, **dims: str) -> Tuple[np.ndarray, np.ndarray]:
        low = np.asarray([get_metric(lower[frame], section, metric, **dims)
                          for frame in common_frames], dtype=np.float64)
        high = np.asarray([get_metric(default[frame], section, metric, **dims)
                           for frame in common_frames], dtype=np.float64)
        return low, high
    low_cond, high_cond = matched_median("b_geometry", "effective_condition_number")
    low_lmin, high_lmin = matched_median("b_geometry", "lambda_min")
    low_wdim, high_wdim = matched_median(
        "b_geometry", "weak_dimension", weak_definition=PRIMARY_WEAK
    )
    low_wtrace, high_wtrace = matched_median(
        "b_geometry", "weak_trace_fraction", weak_definition=PRIMARY_WEAK
    )
    matched_rho = {b: matched_median("f_geometry", "rho_dangerous", branch=b,
                                    z_mode="task_proxy", action_mode="actor_mean",
                                    weak_definition=PRIMARY_WEAK) for b in branches}
    matched_tail = {b: matched_median("large_m", "m_abs_p99_99", branch=b,
                                     z_mode="fixed_random", action_mode="replay") for b in branches}
    low_grad, high_grad = matched_median("gradient_interference", "gweak_norm",
                                         branch="actual_fb_update_union", z_mode="training_mix_probe",
                                         action_mode="replay", weak_definition=PRIMARY_WEAK)
    low_gfull, high_gfull = matched_values(
        "gradient_interference", "gfull_norm", branch="actual_fb_update_union",
        z_mode="training_mix_probe", action_mode="replay", weak_definition=PRIMARY_WEAK
    )
    low_gweak_values, high_gweak_values = matched_values(
        "gradient_interference", "gweak_norm", branch="actual_fb_update_union",
        z_mode="training_mix_probe", action_mode="replay", weak_definition=PRIMARY_WEAK
    )
    low_relative_grad = float(np.median(low_gweak_values / low_gfull))
    high_relative_grad = float(np.median(high_gweak_values / high_gfull))
    lower_2m = by_key["run_lr5e5"][-1]
    lower_2m_cond = get_metric(lower_2m, "b_geometry", "effective_condition_number")
    lower_2m_reward = get_metric(lower_2m, "wandb_alignment", "eval/episode_reward")
    lower_15m_cond = get_metric(lower[1_500_000], "b_geometry", "effective_condition_number")
    lower_15m_reward = get_metric(lower[1_500_000], "wandb_alignment", "eval/episode_reward")
    lines = [
        "# Pixel-B Cheetah offline checkpoint diagnosis", "",
        "## Bottom line", "",
        ("**Overall verdict: partial/mixed support, not support for the complete causal chain.** The fixed-proxy "
         "results strongly support deterioration of B's extreme spectral tail and accumulation of F energy in "
         "weak-B directions alongside Q/reward collapse. They do not establish gauge-invariant B-subspace drift, "
         "weak-direction activation of the extreme-M tail, or escalating FB-gradient interference."), "",
        (f"For Walk, the selected-checkpoint reward peak is at {peak.frame/1e6:g}M ({rewards[peak_index]:.3f}) and "
         f"the final 2M reward is {rewards[-1]:.3f}. Across that interval effective B condition changes "
         f"{cond_final/cond_peak:.3g}x and lambda_min falls {lambda_min_peak/lambda_min_final:.3g}x; "
         f"task/actor rho_dangerous changes "
         f"F1 {rho_final['f1']/(rho_peak['f1']+1e-12):.3g}x and "
         f"F2 {rho_final['f2']/(rho_peak['f2']+1e-12):.3g}x."), "",
        "## Walk mechanism checks", "",
        (f"- **B extreme-tail ill-conditioning: supported, but non-monotonic and not a whole-spectrum collapse.** "
         f"Effective condition rises {cond_peak:.4g} -> {cond_final:.4g}, and lambda_min falls "
         f"{lambda_min_peak:.4g} -> {lambda_min_final:.4g}. In contrast, participation ratio rises "
         f"{pr_peak:.3f} -> {pr_final:.3f} and threshold-0.10 weak trace fraction falls "
         f"{trace_fraction_peak:.3f} -> {trace_fraction_final:.3f}."),
        (f"- **F near-nullspace accumulation: supported and rank-robust.** Primary task/actor rho_dangerous rises "
         f"F1 {rho_peak['f1']:.3f} -> {rho_final['f1']:.3f} and F2 {rho_peak['f2']:.3f} -> "
         f"{rho_final['f2']:.3f}; bottom-5, bottom-10, 0.03, 0.10, and 0.30 definitions all move upward."),
        (f"- **Structural B weak-subspace drift: inconclusive.** Raw consecutive projector movement is present "
         f"(largest post-peak threshold-0.10 normalized distance {max(post_peak_drift_values):.4g}), but checkpoints "
         "were not put into a common latent gauge. An arbitrary joint orthogonal rotation of F/B can therefore "
         "produce these principal angles without structural drift."),
        (f"- **Weak-direction extreme-M activation: not supported on the fixed large replay proxy.** |M| p99.99 "
         f"contracts from F1/F2 {m_p9999_peak['f1']:.1f}/{m_p9999_peak['f2']:.1f} to "
         f"{m_p9999_final['f1']:.1f}/{m_p9999_final['f2']:.1f}. At 2M, top-0.01% entries have only "
         f"{m_tail_final['f1']:.3f}x/{m_tail_final['f2']:.3f} the ordinary-state weak contribution. Historical "
         f"W&B M_online_abs_max nevertheless rises {online_m_peak:.1f} -> {online_m_final:.1f}; because historical "
         "replay is unavailable, that distribution mismatch remains unresolved rather than explained by weak B."),
        (f"- **Task-Q collapse/redistribution: supported; a direct strict-nullspace mechanism is not.** Twin-min Q RMS "
         f"falls {q_total_peak:.1f} -> {q_total_final:.1f} (mean {q_mean_peak:.1f} -> {q_mean_final:.1f}), while "
         f"broad threshold-0.10 Qweak RMS is {q_rms_peak:.1f} -> {q_rms_final:.1f}. At 2M the bottom-5/bottom-10 "
         f"Qweak RMS values are only {strict_qweak_final['bottom5']:.3g}/{strict_qweak_final['bottom10']:.3g}; "
         "the very large per-row |Qweak|/(|Q|+eps) values mainly reflect a near-zero/cancelled total Q denominator."),
        (f"- **Escalating FB-gradient interference: not supported.** Exact union gweak falls "
         f"{union_gweak_peak:.4g} -> {union_gweak_final:.4g}, while strong/weak cosine becomes more aligned "
         f"({union_cos_peak:.3f} -> {union_cos_final:.3f}). The checkpoint-point actor attribution has a weak "
         "opposite hint, but it is counterfactual because the forbidden preceding FB optimizer step is not simulated."),
        f"- **Reward collapse: observed.** Exact-frame reward falls {rewards[peak_index]:.3f} -> {rewards[-1]:.3f}.", "",
        "## Matched Cheetah Run LR hypotheses", "",
        (f"- **H1 — partially supported after 0.5M.** Median matched effective condition is {low_cond:.4g} at "
         f"5e-5 versus {high_cond:.4g} at 1e-4, and median lambda_min is {low_lmin:.4g} versus {high_lmin:.4g}. "
         f"The lower LR is worse at 0.1M/0.2M but much better from 0.5M onward. It does not reduce the broad weak "
         f"dimension/trace fraction (medians {low_wdim:.0f}/{low_wtrace:.3f} versus "
         f"{high_wdim:.0f}/{high_wtrace:.3f}), so the benefit is specifically preservation of the extreme bottom tail."),
        (f"- **H2 — not supported as stated.** B conditioning is not similar, and median task/actor rho_dangerous is "
         f"F1 {matched_rho['f1'][0]:.4g} versus {matched_rho['f1'][1]:.4g}, and "
         f"F2 {matched_rho['f2'][0]:.4g} versus {matched_rho['f2'][1]:.4g} (5e-5 versus 1e-4). The direction "
         "changes with frame and weak-rank definition, so lower LR does not cleanly suppress F drift."),
        (f"- **H3 — weak and stage-dependent support only.** Median matched |M| p99.99 is "
         f"F1 {matched_tail['f1'][0]:.4g} versus {matched_tail['f1'][1]:.4g}, and "
         f"F2 {matched_tail['f2'][0]:.4g} versus {matched_tail['f2'][1]:.4g}. Lower LR is smaller at four of six "
         "frames but reverses late; extreme-tail weak contribution is not consistently suppressed."),
        (f"- **H4 — weak/inconsistent checkpoint association.** Median exact union gweak is {low_grad:.4g} versus "
         f"{high_grad:.4g}, but only four of six frames favor lower LR and the median relative gweak/gfull is "
         f"{low_relative_grad:.3f} versus {high_relative_grad:.3f}. This is not evidence that relative shared-network "
         "interference is systematically reduced."), "",
        (f"The unmatched lower-LR 2M point is suggestive of delay only in the B-tail/reward component: from 1.5M to "
         f"2M its condition changes {lower_15m_cond:.4g} -> {lower_2m_cond:.4g} and reward "
         f"{lower_15m_reward:.3f} -> {lower_2m_reward:.3f}. It is never used as a matched observation. Overall, the "
         "runs share late B-tail/reward degradation, but F weak-energy and gradient/tail signatures are not a clean "
         "time-shifted copy of one mechanism."), "",
        "## Direct answers", "",
        "1. Lower FB LR mainly preserves B's extreme smallest eigenvalues after 0.5M; it does not consistently slow F weak-space accumulation.",
        "2. No: fixed-proxy extreme M is strong-subspace dominated, so hidden weak F does not explain the historical online M tail in these data.",
        "3. Some broad weak-space energy enters proxy task Q, but strict bottom-5/bottom-10 effects are tiny and the late relative dominance is driven mainly by collapse/cancellation of total Q.",
        "4. No escalating FB-gradient interference is observed; only a weak, unstepped actor-side hint remains.",
        "5. Lower LR appears to delay the extreme B-tail/reward failure, but the full mechanism is neither identical nor uniformly shifted later.", "",
        "## Critical provenance and limitations", "",
        f"- All checkpoints use identical fixed raw arrays, indices, proxy rewards, and random-z with digest `{replay.metadata['content_sha256']}`; task-z is checkpoint-specific but always inferred from that same bank.",
        "- Historical replay, historical RNG state, and historical eval task-z were not serialized and cannot be exactly recovered.",
        "- The bank is a deterministic ExORL-RND proxy. Its source cadence is action_repeat=1; target runs used action_repeat=2. Continuous source transitions preserve action alignment but not historical temporal spacing.",
        f"- Task-z uses the repository's exact `infer_meta_from_obs_and_rewards` code on {replay.metadata['inference_size']:,} fixed proxy next states labelled by current single-state `DmcReward.from_physics`. This is not native two-substep accumulated reward and not historical eval z.",
        "- Actor actions are the deterministic distribution mean used by eval, evaluated on proxy observations with checkpoint-specific proxy task-z. Actual actor updates receive `obs.detach()` and step only actor parameters; encoder actor gradient is exactly zero. ForwardMap actor gradients/cosines are reported only as unstepped raw-backward diagnostics.",
        "- C_B uses float64 and is symmetrized. FB gradient attribution uses float32 repository semantics, detached EVD projectors, exact diagonal/off-diagonal masks, exact twin-target winner selection, and no optimizer step.",
        "- Cross-checkpoint projectors/principal angles and task-z cosine are reported in each checkpoint's raw latent coordinates. No gauge/Procrustes alignment is available, so they are descriptive and cannot establish structural subspace rotation.",
        f"- The requested primary rule `lambda < 0.10 * lambda_max` spans {min(weak_dimensions)}–{max(weak_dimensions)} of 50 dimensions on Walk, so it is a broad low-relative-eigenvalue subspace, not merely a handful of null directions. Interpret it alongside bottom-5, bottom-10, 0.03, and 0.30 sensitivity rows.",
        "- Gradient attribution uses the primary threshold-0.10 projector only; it does not provide weak-rank sensitivity for H4.",
        "- Local train/eval CSV files colocated with each W&B run provide the recorded values and exact CSV source frame/step. The local W&B binary is provenance-hashed but not decoded or claimed cross-validated. Missing columns remain blank; no interpolation or inference is performed.",
    ]
    return "\n".join(lines) + "\n"


def _privileged_cnn_candidate(task: str, frame: int) -> Optional[Path]:
    root = Path(
        "/mnt/data_7tb/fanfeng/controallable_agent_ckpt/"
        f"20260822_172133_cheetah_speed_goal_cnn_s1_immediate_seed1_{task}_"
        "cnn_cheetah_speed_goal"
    )
    path = root / f"snapshot_{frame}.pt"
    return path if path.is_file() else None


def validate_cnn_args(
    args: argparse.Namespace,
) -> Tuple[List[Experiment], List[Dict[str, Any]], List[Tuple[Experiment, int]]]:
    args.output_dir = args.output_dir.resolve()
    args.exorl_dir = args.exorl_dir.resolve()
    args.index_bank = args.index_bank.resolve()
    known = {item.key: item for item in ALL_CNN_EXPERIMENTS}
    # Backward compatibility: no explicit selection retains the original four
    # shared-CNN lineages even after separate-CNN lineages become available.
    requested_keys = args.experiments or tuple(
        item.key for item in CNN_EXPERIMENTS
    )
    unknown = set(requested_keys) - set(known)
    if unknown:
        raise ValueError(f"unknown CNN experiments: {sorted(unknown)}")
    selected = [known[key] for key in requested_keys]
    requested_frames = tuple(args.frames or CNN_TARGET_FRAMES)
    outside = set(requested_frames) - set(CNN_TARGET_FRAMES)
    if outside:
        raise ValueError(f"CNN frames outside the declared target set: {sorted(outside)}")
    if args.probe_size != TRAIN_BATCH:
        raise ValueError("CNN B covariance/F geometry probe-size must remain 1024")
    if not args.allow_partial:
        fixed_sizes = {
            "replay_size": (args.replay_size, CNN_DEFAULT_REPLAY_SIZE),
            "inference_size": (args.inference_size, DEFAULT_INFERENCE_SIZE),
            "probe_size": (args.probe_size, DEFAULT_PROBE_SIZE),
        }
        mismatches = {
            name: {"actual": actual, "required": required}
            for name, (actual, required) in fixed_sizes.items()
            if actual != required
        }
        if mismatches:
            raise ValueError(
                "formal CNN diagnostics require the fixed DINO-comparable sample "
                f"sizes; pass --allow-partial only for development smoke: {mismatches}"
            )
    if args.replay_size % TRAIN_BATCH or args.replay_size < max(
        args.probe_size, args.inference_size
    ):
        raise ValueError("replay-size must be divisible by 1024 and cover probes")
    if args.inference_chunk <= 0:
        raise ValueError("inference-chunk must be positive")
    if tuple(args.render_shape) != (84, 84):
        raise ValueError("historical CNN checkpoints require render-shape=84,84")
    for path in (args.exorl_dir, args.index_bank):
        if not path.exists():
            raise FileNotFoundError(path)

    availability: List[Dict[str, Any]] = []
    work_items: List[Tuple[Experiment, int]] = []
    for experiment in selected:
        config_path = experiment.run_dir / ".hydra/config.yaml"
        if not config_path.is_file():
            raise FileNotFoundError(config_path)
        cfg = omgcf.OmegaConf.load(config_path)
        serialized_lr_f = getattr(cfg.agent, "lr_f", None)
        serialized_lr_b = getattr(cfg.agent, "lr_b", None)
        contract = {
            "task": str(cfg.task),
            "obs_type": str(cfg.obs_type),
            "use_cls": bool(cfg.use_cls),
            "goal_space": None if cfg.goal_space is None else str(cfg.goal_space),
            "frame_stack": int(cfg.frame_stack),
            "render_shape": tuple(int(item) for item in cfg.render_shape),
            "action_repeat": int(cfg.action_repeat),
            "seed": int(cfg.seed),
            "pixel_separate_fb_encoders": bool(
                getattr(cfg.agent, "pixel_separate_fb_encoders", False)
            ),
            "lr_f": None if serialized_lr_f is None else float(serialized_lr_f),
            "lr_b": None if serialized_lr_b is None else float(serialized_lr_b),
        }
        expected = {
            "task": experiment.task,
            "obs_type": "pixels",
            "use_cls": experiment.expected_use_cls,
            "goal_space": None,
            "frame_stack": FRAME_STACK,
            "render_shape": (84, 84),
            "action_repeat": 2,
            "seed": experiment.seed,
            "pixel_separate_fb_encoders": (
                experiment.expected_pixel_separate_fb_encoders
            ),
            "lr_f": experiment.expected_lr_f,
            "lr_b": experiment.expected_lr_b,
        }
        if contract != expected:
            raise ValueError(
                f"CNN visual-B run config mismatch for {experiment.run_id}: {contract}"
            )
        for frame in requested_frames:
            path = experiment.checkpoint_dir / f"snapshot_{frame}.pt"
            privileged = _privileged_cnn_candidate(experiment.task, frame)
            present = path.is_file()
            if present:
                work_items.append((experiment, frame))
            availability.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "experiment": experiment.key,
                    "run_id": experiment.run_id,
                    "task": experiment.task,
                    "seed": experiment.seed,
                    "obs_type": experiment.obs_type,
                    "pixel_separate_fb_encoders": (
                        experiment.expected_pixel_separate_fb_encoders
                    ),
                    "validated_serialized_use_cls": experiment.expected_use_cls,
                    "goal_space": "null",
                    "target_frame": frame,
                    "status": "available" if present else "missing",
                    "checkpoint_path": str(path.resolve()),
                    "privileged_goal_candidate_excluded": (
                        "" if privileged is None else str(privileged.resolve())
                    ),
                    "note": (
                        "exact requested visual-B checkpoint"
                        if present
                        else (
                            "no goal_space=null checkpoint; cheetah_speed checkpoint exists but "
                            "is not a visual-B comparator"
                            if privileged is not None
                            else "no goal_space=null checkpoint at this exact frame"
                        )
                    ),
                }
            )
    if not work_items:
        raise FileNotFoundError("none of the requested CNN visual-B checkpoints exist")
    return selected, availability, work_items


def analyze_cnn_core_checkpoint(
    experiment: Experiment,
    frame: int,
    replay: ReplayData,
    task_rewards: Mapping[str, np.ndarray],
    args: argparse.Namespace,
) -> Tuple[CheckpointResult, Dict[str, Any], Dict[str, np.ndarray]]:
    agent, info = load_agent(experiment, frame, args.device)
    recorder = MetricRecorder(experiment, frame, info, replay)
    recorder.add("checkpoint", "global_step", info["global_step"])
    recorder.add("checkpoint", "global_episode", info["global_episode"])
    recorder.add("checkpoint", "effective_fb_lr", info["effective_fb_lr"])
    recorder.add("replay", "proxy_size", len(replay.obs))
    recorder.add("replay", "probe_size", args.probe_size)
    recorder.add("replay", "task_inference_size", args.inference_size)
    recorder.add(
        "replay",
        "deterministic_identity_pixel_crop",
        1,
        note="training RandomShiftsAug disabled for cross-checkpoint fixed-input analysis",
    )
    render_context = replay.metadata.get("pixel_render_context")
    if not isinstance(render_context, dict):
        raise ValueError("CNN replay is missing the pixel render context contract")
    recorder.add(
        "replay",
        "pixel_render_context_chunk_size",
        int(render_context["context_chunk_size_rows"]),
        note=str(render_context["context_policy"]),
    )

    random_f1, random_f2, backward = encode_random_replay(
        agent, replay, args.device, args.inference_chunk
    )
    covariance, eigvals, eigvecs, projectors = covariance_and_projectors(
        backward[: args.probe_size], recorder
    )
    for branch, value in (("f1", random_f1), ("f2", random_f2)):
        analyze_forward_geometry(
            recorder,
            value[: args.probe_size],
            replay.random_z[: args.probe_size],
            projectors,
            branch=branch,
            z_mode="fixed_random",
            action_mode="replay",
        )
    analyze_q(
        recorder,
        random_f1,
        random_f2,
        replay.random_z,
        replay.action,
        projectors,
        replay,
        z_mode="fixed_random",
        action_mode="replay",
        extremes={},
        extreme_json=[],
    )

    task_z_values: Dict[str, np.ndarray] = {}
    task_z_raw_values: Dict[str, np.ndarray] = {}
    for reward_task in CNN_REWARD_TASKS:
        rewards = task_rewards[reward_task]
        z_mode = f"task_proxy:{reward_task}"
        task_z, task_z_raw, inference_note = infer_task_z_from_encoded_backward(
            backward, rewards, args.inference_size
        )
        task_z_values[reward_task] = task_z
        task_z_raw_values[reward_task] = task_z_raw
        analyze_task_z_geometry(
            recorder,
            task_z,
            task_z_raw,
            covariance,
            eigvals,
            eigvecs,
            projectors,
            z_mode=z_mode,
        )
        recorder.add(
            "task_z", "reward_mean", float(rewards[: args.inference_size].mean()),
            z_mode=z_mode,
        )
        recorder.add(
            "task_z", "reward_max", float(rewards[: args.inference_size].max()),
            z_mode=z_mode,
        )
        recorder.add(
            "task_z",
            "repository_inference_formula_verified",
            1,
            z_mode=z_mode,
            note=inference_note,
        )
        # A broadcast view is sufficient for bounded CPU geometry chunks.  The
        # encoder receives the 50-value constant directly and expands it on the
        # device, avoiding a pathological 3.9 MiB anonymous host first-touch.
        repeated_z = np.broadcast_to(task_z, (len(replay.obs), Z_DIM))
        replay_f1, replay_f2, replay_actions = encode_forward(
            agent,
            replay.obs[: args.probe_size],
            task_z,
            replay.action[: args.probe_size],
            args.device,
            args.inference_chunk,
            "replay",
        )
        actor_f1, actor_f2, actor_actions = encode_forward(
            agent,
            replay.obs,
            task_z,
            replay.action,
            args.device,
            args.inference_chunk,
            "actor_mean",
        )
        for action_mode, f1, f2, actions in (
            ("replay", replay_f1, replay_f2, replay_actions),
            ("actor_mean", actor_f1, actor_f2, actor_actions),
        ):
            for branch, value in (("f1", f1), ("f2", f2)):
                analyze_forward_geometry(
                    recorder,
                    value[: args.probe_size],
                    repeated_z[: args.probe_size],
                    projectors,
                    branch=branch,
                    z_mode=z_mode,
                    action_mode=action_mode,
                )
            analyze_q(
                recorder,
                f1,
                f2,
                repeated_z[: len(f1)],
                actions,
                projectors,
                replay,
                z_mode=z_mode,
                action_mode=action_mode,
                extremes={},
                extreme_json=[],
            )

    wandb_provenance = add_wandb_alignment(recorder, experiment, frame)
    canonical_task_z = task_z_values[experiment.task]
    canonical_task_z_raw = task_z_raw_values[experiment.task]
    result = CheckpointResult(
        experiment=experiment,
        frame=frame,
        checkpoint_path=Path(info["path"]),
        checkpoint_sha256=str(info["sha256"]),
        checkpoint_size=int(info["size"]),
        global_step=int(info["global_step"]),
        global_episode=int(info["global_episode"]),
        task_z=canonical_task_z,
        task_z_raw=canonical_task_z_raw,
        covariance=covariance,
        eigvals=eigvals,
        eigvecs=eigvecs,
        projectors=projectors,
        metrics=recorder.rows,
        extremes={},
        extreme_json=[],
        gradient_rows=[],
    )
    task_arrays = {
        f"task_z__{task}": value for task, value in task_z_values.items()
    }
    task_arrays.update(
        {f"task_z_raw__{task}": value for task, value in task_z_raw_values.items()}
    )
    del agent, random_f1, random_f2, backward
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
    return result, wandb_provenance, task_arrays


def build_cnn_core_summary(results: Sequence[CheckpointResult]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for result in sorted(results, key=lambda item: (item.experiment.task, item.frame)):
        base = {
            "schema_version": SCHEMA_VERSION,
            "experiment": result.experiment.key,
            "run_id": result.experiment.run_id,
            "checkpoint_task": result.experiment.task,
            "seed": result.experiment.seed,
            "pixel_separate_fb_encoders": (
                result.experiment.expected_pixel_separate_fb_encoders
            ),
            "checkpoint_frame": result.frame,
            "checkpoint_path": str(result.checkpoint_path.resolve()),
            "checkpoint_sha256": result.checkpoint_sha256,
            "eval_reward": get_metric(
                result, "wandb_alignment", "eval/episode_reward"
            ),
            "b_lambda_min": get_metric(result, "b_geometry", "lambda_min"),
            "b_lambda_max": get_metric(result, "b_geometry", "lambda_max"),
            "b_condition_number": get_metric_allow_nonfinite(
                result, "b_geometry", "condition_number"
            ),
            "b_effective_condition_number": get_metric(
                result, "b_geometry", "effective_condition_number"
            ),
            "b_participation_ratio": get_metric(
                result, "b_geometry", "participation_ratio"
            ),
            "b_bottom10_trace_fraction": get_metric(
                result,
                "b_geometry",
                "weak_trace_fraction",
                weak_definition="bottom10",
            ),
        }
        for branch in ("f1", "f2"):
            base[f"{branch}_norm_rms"] = get_metric(
                result,
                "f_geometry",
                "f_norm_rms",
                branch=branch,
                z_mode="fixed_random",
                action_mode="replay",
            )
            base[f"{branch}_norm_max"] = get_metric(
                result,
                "f_geometry",
                "f_norm_max",
                branch=branch,
                z_mode="fixed_random",
                action_mode="replay",
            )
            base[f"{branch}_bottom10_b_eigenspace_energy"] = get_metric(
                result,
                "f_geometry",
                "weak_energy_ratio",
                branch=branch,
                z_mode="fixed_random",
                action_mode="replay",
                weak_definition="bottom10",
            )
        for reward_task in CNN_REWARD_TASKS:
            z_mode = f"task_proxy:{reward_task}"
            row = {**base, "reward_task": reward_task}
            row["task_z_raw_norm"] = get_metric(
                result, "task_z", "raw_norm", z_mode=z_mode
            )
            row["task_z_norm"] = get_metric(
                result, "task_z", "normalized_norm", z_mode=z_mode
            )
            row["task_z_bottom10_energy"] = get_metric(
                result,
                "task_z_geometry",
                "weak_energy_ratio",
                z_mode=z_mode,
                weak_definition="bottom10",
            )
            row["task_z_covariance_rayleigh_quotient"] = get_metric(
                result,
                "task_z_geometry",
                "covariance_rayleigh_quotient",
                z_mode=z_mode,
            )
            for branch in ("f1", "f2"):
                row[f"{branch}_task_parallel_energy"] = get_metric(
                    result,
                    "f_geometry",
                    "parallel_energy_ratio",
                    branch=branch,
                    z_mode=z_mode,
                    action_mode="actor_mean",
                )
            row["task_q_rms"] = get_metric(
                result,
                "q",
                "q_rms",
                branch="twin_min",
                z_mode=z_mode,
                action_mode="actor_mean",
            )
            rows.append(row)
    return rows


def cnn_diagnosis_text(
    results: Sequence[CheckpointResult],
    availability: Sequence[Mapping[str, Any]],
    pixel_render_context: Mapping[str, Any],
) -> str:
    backend = str(pixel_render_context.get("mujoco_gl_backend", "unknown"))
    lines = [
        "# CNN visual-B checkpoint diagnosis",
        "",
        "This is a fixed-input, read-only diagnostic of `obs_type=pixels`, "
        "`frame_stack=3`, `render_shape=84x84`, `goal_space=null` checkpoints.",
        "The newer `goal_space=cheetah_speed` checkpoints are excluded because their "
        "B network consumes a privileged low-dimensional goal rather than CNN pixels.",
    ]
    if backend == "osmesa":
        lines.extend(
            [
                "The fixed DINO-physics proxy was re-rendered with OSMesa; it is not "
                "the historical EGL online pixel replay, so renderer sensitivity "
                "remains a limitation.",
            ]
        )
    lines.extend(
        [
            "The older `cheetah_run` lineage serializes `use_cls=false`; this is "
            "validated exactly but is a no-op for its ordinary raw-pixel CNN "
            "encoder (the switch only selects ViT/DINO CLS versus patch inputs).",
        ]
    )
    lines.extend(
        [
        "",
        "## Exact checkpoint availability",
        "",
        "| task | frame | status | note |",
        "|---|---:|---|---|",
        ]
    )
    for row in availability:
        lines.append(
            f"| {row['task']} | {int(row['target_frame']):,} | {row['status']} | "
            f"{row['note']} |"
        )
    lines.extend(
        [
            "",
            "## B spectral timeline",
            "",
            "| task | frame | eval reward | lambda_min | condition | PR | bottom10 trace |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for result in sorted(results, key=lambda item: (item.experiment.task, item.frame)):
        lines.append(
            f"| {result.experiment.task} | {result.frame:,} | "
            f"{get_metric(result, 'wandb_alignment', 'eval/episode_reward'):.5g} | "
            f"{get_metric(result, 'b_geometry', 'lambda_min'):.5g} | "
            f"{get_metric(result, 'b_geometry', 'effective_condition_number'):.5g} | "
            f"{get_metric(result, 'b_geometry', 'participation_ratio'):.5g} | "
            f"{get_metric(result, 'b_geometry', 'weak_trace_fraction', weak_definition='bottom10'):.5g} |"
        )
    lines.extend(
        [
            "",
            "All task-z rows are checkpoint-specific reward projections from the same "
            "5,120-state proxy bank. Full per-eigendirection energy, F geometry, and Q "
            "statistics are in `checkpoint_diagnostics_long.csv`; the required core fields "
            "are also flattened in `cnn_checkpoint_summary.csv`.",
            "",
        ]
    )
    return "\n".join(lines)


def cnn_checkpoint_bundle_path(
    output_dir: Path, experiment: Experiment, frame: int
) -> Path:
    return output_dir / "completed" / f"{experiment.key}_{frame:07d}.pt"


def cnn_checkpoint_complete_path(bundle_path: Path) -> Path:
    return bundle_path.with_suffix(".complete.json")


def cnn_bundle_contract(
    experiment: Experiment,
    frame: int,
    replay: ReplayData,
    args: argparse.Namespace,
    checkpoint_sha256: str,
) -> Dict[str, Any]:
    checkpoint = experiment.checkpoint_dir / f"snapshot_{frame}.pt"
    config = experiment.run_dir / ".hydra/config.yaml"
    runtime_contract: Dict[str, Any] = {
        "device": args.device,
        "torch_version": str(torch.__version__),
        "numpy_version": str(np.__version__),
    }
    if args.device.startswith("cuda"):
        device_index = torch.device(args.device).index
        if device_index is None:
            device_index = torch.cuda.current_device()
        runtime_contract.update(
            {
                "cuda_device_name": torch.cuda.get_device_name(device_index),
                "cuda_device_capability": list(
                    torch.cuda.get_device_capability(device_index)
                ),
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "profile": (
            "cnn_visual_b_core_geometry_development_partial"
            if args.allow_partial
            else "cnn_visual_b_core_geometry"
        ),
        "analysis_mode": "development_partial" if args.allow_partial else "formal",
        "experiment": experiment.key,
        "run_id": experiment.run_id,
        "frame": frame,
        "validated_serialized_use_cls": experiment.expected_use_cls,
        "pixel_separate_fb_encoders": (
            experiment.expected_pixel_separate_fb_encoders
        ),
        "fixed_replay_sha256": replay.metadata["content_sha256"],
        "pixel_render_context": replay.metadata.get("pixel_render_context"),
        "pixel_episode_part_cache": replay.metadata.get(
            "pixel_episode_part_cache"
        ),
        "source_episode_filename_contract": replay.metadata.get(
            "source_episode_filename_contract"
        ),
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_size": checkpoint.stat().st_size,
        "checkpoint_sha256": checkpoint_sha256,
        "run_config_path": str(config.resolve()),
        "run_config_sha256": sha256_file(config),
        "analysis_script_sha256": sha256_file(Path(__file__).resolve()),
        "dependency_sha256": {
            "fb_loss_and_actor": sha256_file(
                Path(__file__).resolve().parent / "agent/fb_ddpg.py"
            ),
            "reward_projection": sha256_file(
                Path(__file__).resolve().parent / "goals.py"
            ),
            "environment_wrappers": sha256_file(
                Path(__file__).resolve().parent / "dmc.py"
            ),
        },
        "replay_size": args.replay_size,
        "probe_size": args.probe_size,
        "inference_size": args.inference_size,
        "inference_chunk": args.inference_chunk,
        "q_float64_chunk_rows": CNN_Q_FLOAT64_CHUNK_ROWS,
        "render_shape": list(args.render_shape),
        "replay_seed": args.replay_seed,
        "z_seed": args.z_seed,
        "deterministic_pixel_augmentation": "identity/no random shift",
        "reward_tasks": list(CNN_REWARD_TASKS),
        **runtime_contract,
    }


def save_cnn_checkpoint_bundle(
    path: Path,
    result: CheckpointResult,
    provenance: Mapping[str, Any],
    task_z_arrays: Mapping[str, np.ndarray],
    replay: ReplayData,
    args: argparse.Namespace,
) -> None:
    contract = cnn_bundle_contract(
        result.experiment,
        result.frame,
        replay,
        args,
        result.checkpoint_sha256,
    )
    atomic_torch_save(
        path,
        {
            "contract": contract,
            "result": {
                "checkpoint_path": str(result.checkpoint_path.resolve()),
                "checkpoint_sha256": result.checkpoint_sha256,
                "checkpoint_size": result.checkpoint_size,
                "global_step": result.global_step,
                "global_episode": result.global_episode,
                "task_z": result.task_z,
                "task_z_raw": result.task_z_raw,
                "covariance": result.covariance,
                "eigvals": result.eigvals,
                "eigvecs": result.eigvecs,
                "projectors": result.projectors,
                "metrics": result.metrics,
            },
            "wandb_provenance": dict(provenance),
            "task_z_arrays": dict(task_z_arrays),
        },
    )
    marker = {
        **contract,
        "bundle_path": str(path.resolve()),
        "bundle_size": path.stat().st_size,
        "bundle_sha256": sha256_file(path),
    }
    # This marker is deliberately last: its existence means every required
    # result field has already reached one atomically renamed bundle.
    atomic_json(cnn_checkpoint_complete_path(path), marker)


def load_cnn_checkpoint_bundle(
    path: Path,
    experiment: Experiment,
    frame: int,
    replay: ReplayData,
    args: argparse.Namespace,
) -> Tuple[CheckpointResult, Dict[str, Any], Dict[str, np.ndarray]]:
    marker_path = cnn_checkpoint_complete_path(path)
    if not marker_path.is_file():
        raise FileNotFoundError(f"completed checkpoint marker is missing: {marker_path}")
    with marker_path.open(encoding="utf-8") as stream:
        marker = json.load(stream)
    if marker.get("bundle_path") != str(path.resolve()):
        raise ValueError(f"completed marker points at a different bundle: {marker_path}")
    if marker.get("bundle_size") != path.stat().st_size:
        raise ValueError(f"completed bundle size differs from marker: {path}")
    bundle_sha256 = sha256_file(path)
    if marker.get("bundle_sha256") != bundle_sha256:
        raise ValueError(f"completed bundle digest differs from marker: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("contract"), dict):
        raise ValueError(f"malformed completed checkpoint bundle: {path}")
    stored_contract = payload["contract"]
    checkpoint = experiment.checkpoint_dir / f"snapshot_{frame}.pt"
    validated_checkpoint_size(checkpoint, experiment.expected_size)
    checkpoint_sha256 = sha256_file(checkpoint)
    expected_contract = cnn_bundle_contract(
        experiment, frame, replay, args, checkpoint_sha256
    )
    if stored_contract != expected_contract:
        differences = {
            key: (stored_contract.get(key), value)
            for key, value in expected_contract.items()
            if stored_contract.get(key) != value
        }
        raise ValueError(f"completed checkpoint bundle contract mismatch: {differences}")
    marker_contract = {key: marker.get(key) for key in expected_contract}
    if marker_contract != expected_contract:
        raise ValueError(f"completed marker contract mismatch: {marker_path}")
    data = payload.get("result")
    if not isinstance(data, dict) or not isinstance(data.get("metrics"), list):
        raise ValueError(f"completed checkpoint bundle lacks result rows: {path}")
    covariance = np.asarray(data["covariance"], dtype=np.float64)
    eigvals = np.asarray(data["eigvals"], dtype=np.float64)
    eigvecs = np.asarray(data["eigvecs"], dtype=np.float64)
    projectors = {
        str(name): np.asarray(value, dtype=np.float64)
        for name, value in dict(data["projectors"]).items()
    }
    if covariance.shape != (Z_DIM, Z_DIM) or eigvals.shape != (Z_DIM,) or eigvecs.shape != (
        Z_DIM,
        Z_DIM,
    ):
        raise ValueError(f"invalid EVD arrays in completed bundle: {path}")
    if set(projectors) != set(WEAK_DEFINITIONS) or any(
        value.shape != (Z_DIM, Z_DIM) for value in projectors.values()
    ):
        raise ValueError(f"invalid projector arrays in completed bundle: {path}")
    if not np.allclose(
        covariance, eigvecs @ np.diag(eigvals) @ eigvecs.T, rtol=1e-10, atol=1e-10
    ):
        raise ValueError(f"EVD reconstruction failed in completed bundle: {path}")
    task_arrays = {
        str(name): np.asarray(value)
        for name, value in dict(payload.get("task_z_arrays", {})).items()
    }
    expected_task_keys = {
        *(f"task_z__{task}" for task in CNN_REWARD_TASKS),
        *(f"task_z_raw__{task}" for task in CNN_REWARD_TASKS),
    }
    if set(task_arrays) != expected_task_keys or any(
        value.shape != (Z_DIM,) for value in task_arrays.values()
    ):
        raise ValueError(f"invalid all-reward task-z arrays in completed bundle: {path}")
    result = CheckpointResult(
        experiment=experiment,
        frame=frame,
        checkpoint_path=Path(data["checkpoint_path"]),
        checkpoint_sha256=str(data["checkpoint_sha256"]),
        checkpoint_size=int(data["checkpoint_size"]),
        global_step=int(data["global_step"]),
        global_episode=int(data["global_episode"]),
        task_z=np.asarray(data["task_z"]),
        task_z_raw=np.asarray(data["task_z_raw"]),
        covariance=covariance,
        eigvals=eigvals,
        eigvecs=eigvecs,
        projectors=projectors,
        metrics=[dict(row) for row in data["metrics"]],
        extremes={},
        extreme_json=[],
        gradient_rows=[],
    )
    if 2 * result.global_step != frame or result.checkpoint_sha256 != checkpoint_sha256:
        raise ValueError(f"step/hash mismatch in completed bundle: {path}")
    provenance = payload.get("wandb_provenance")
    if not isinstance(provenance, dict):
        raise ValueError(f"completed bundle lacks W&B provenance: {path}")
    return result, provenance, task_arrays


def refresh_cnn_aggregates(
    output_dir: Path,
    results: Sequence[CheckpointResult],
    task_z_arrays: Mapping[str, np.ndarray],
    availability: Sequence[Mapping[str, Any]],
    pixel_render_context: Mapping[str, Any],
) -> None:
    if not results:
        return
    atomic_csv(
        output_dir / "checkpoint_diagnostics_long.csv",
        [row for result in results for row in result.metrics],
    )
    atomic_csv(output_dir / "cnn_checkpoint_summary.csv", build_cnn_core_summary(results))
    atomic_npz(output_dir / "task_z_all_rewards.npz", **dict(task_z_arrays))
    atomic_text(
        output_dir / "DIAGNOSIS.md",
        cnn_diagnosis_text(results, availability, pixel_render_context),
    )


def main_cnn(args: argparse.Namespace) -> None:
    selected, availability, work_items = validate_cnn_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_csv(args.output_dir / "checkpoint_availability.csv", availability)
    requested_render_context = _cnn_pixel_render_context_contract(
        tuple(args.render_shape)
    )
    print(
        "CNN fixed replay renderer: "
        f"MUJOCO_GL={requested_render_context['mujoco_gl_backend']}, "
        "MUJOCO_EGL_DEVICE_ID="
        f"{requested_render_context['mujoco_egl_device_id'] or 'none'}, "
        "context_chunk_size_rows="
        f"{requested_render_context['context_chunk_size_rows']}; "
        f"{requested_render_context['context_policy']}",
        flush=True,
    )
    if torch.cuda.is_initialized():
        raise RuntimeError(
            "CUDA was initialized before CNN fixed replay creation; refusing to risk "
            "MuJoCo-renderer/CUDA context interference"
        )
    print(
        "CNN fixed replay: CUDA initialization is deferred until cache creation completes",
        flush=True,
    )
    part_cache_contract = _cnn_pixel_episode_part_cache_contract(
        args.output_dir, "pixels"
    )
    assert part_cache_contract is not None
    print(
        "CNN fixed replay episode parts: "
        f"{part_cache_contract['directory']}; atomic exact-name commit, strict resume, "
        "parts retained after aggregate validation",
        flush=True,
    )
    replay, replay_cache_path = load_or_build_replay_cache(args, obs_type="pixels")
    args.device = _resolve_and_validate_device(args.device)
    task_rewards = {
        task: compute_task_rewards(replay, task) for task in CNN_REWARD_TASKS
    }
    if np.corrcoef(replay.source_reward[:, 0], task_rewards["cheetah_run"][:, 0])[0, 1] < 0.995:
        raise AssertionError("ExORL stored run reward provenance sanity check failed")
    save_replay_manifest(args.output_dir / "fixed_replay_manifest.npz", replay, task_rewards)

    results: List[CheckpointResult] = []
    task_z_arrays: Dict[str, np.ndarray] = {}
    wandb_provenance: Dict[str, Any] = {}
    for index, (experiment, frame) in enumerate(work_items, start=1):
        print(
            f"CNN checkpoint {index}/{len(work_items)}: {experiment.task} @ {frame:,}",
            flush=True,
        )
        bundle_path = cnn_checkpoint_bundle_path(args.output_dir, experiment, frame)
        marker_path = cnn_checkpoint_complete_path(bundle_path)
        if marker_path.is_file():
            if not args.resume:
                raise FileExistsError(
                    f"completed checkpoint exists; pass --resume to validate/reuse it: {marker_path}"
                )
            if not bundle_path.is_file():
                raise FileNotFoundError(
                    f"complete marker exists but result bundle is missing: {bundle_path}"
                )
            result, provenance, current_task_z = load_cnn_checkpoint_bundle(
                bundle_path, experiment, frame, replay, args
            )
            print(f"  resumed validated bundle: {bundle_path}", flush=True)
        else:
            if bundle_path.exists():
                if not args.resume:
                    raise FileExistsError(
                        f"unmarked checkpoint bundle exists; pass --resume to recompute it: {bundle_path}"
                    )
                print(
                    f"  ignoring unmarked/incomplete bundle and recomputing: {bundle_path}",
                    flush=True,
                )
            result, provenance, current_task_z = analyze_cnn_core_checkpoint(
                experiment, frame, replay, task_rewards, args
            )
            save_cnn_checkpoint_bundle(
                bundle_path, result, provenance, current_task_z, replay, args
            )
            print(f"  committed bundle+marker: {bundle_path}", flush=True)
        results.append(result)
        wandb_provenance[experiment.key] = provenance
        prefix = f"{experiment.key}__frame_{frame}"
        task_z_arrays.update(
            {f"{prefix}__{name}": value for name, value in current_task_z.items()}
        )
        save_checkpoint_artifacts(args.output_dir, result)
        refresh_cnn_aggregates(
            args.output_dir,
            results,
            task_z_arrays,
            availability,
            replay.metadata["pixel_render_context"],
        )
        print(
            f"  reward={get_metric(result, 'wandb_alignment', 'eval/episode_reward'):.3f}, "
            f"condition={get_metric(result, 'b_geometry', 'effective_condition_number'):.4g}",
            flush=True,
        )
    add_consecutive_drift(results)
    for result in results:
        save_checkpoint_artifacts(args.output_dir, result)

    long_path = args.output_dir / "checkpoint_diagnostics_long.csv"
    summary_path = args.output_dir / "cnn_checkpoint_summary.csv"
    task_z_path = args.output_dir / "task_z_all_rewards.npz"
    refresh_cnn_aggregates(
        args.output_dir,
        results,
        task_z_arrays,
        availability,
        replay.metadata["pixel_render_context"],
    )
    report_path = args.output_dir / "DIAGNOSIS.md"
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "profile": (
            "cnn_visual_b_core_geometry_development_partial"
            if args.allow_partial
            else "cnn_visual_b_core_geometry"
        ),
        "analysis_mode": "development_partial" if args.allow_partial else "formal",
        "scientific_artifact": not args.allow_partial,
        "script": str(Path(__file__).resolve()),
        "device": args.device,
        "q_float64_chunk_rows": CNN_Q_FLOAT64_CHUNK_ROWS,
        "deterministic_pixel_augmentation": "identity/no random shift",
        "replay": replay.metadata,
        "pixel_render_context": replay.metadata.get("pixel_render_context"),
        "pixel_episode_part_cache": replay.metadata.get(
            "pixel_episode_part_cache"
        ),
        "reward_tasks": list(CNN_REWARD_TASKS),
        "target_frames": list(CNN_TARGET_FRAMES),
        "checkpoint_count": len(results),
        "missing_checkpoint_count": sum(row["status"] == "missing" for row in availability),
        "checkpoints": [
            {
                "experiment": item.experiment.key,
                "task": item.experiment.task,
                "seed": item.experiment.seed,
                "frame": item.frame,
                "global_step": item.global_step,
                "path": str(item.checkpoint_path.resolve()),
                "size": item.checkpoint_size,
                "sha256": item.checkpoint_sha256,
            }
            for item in results
        ],
        "metric_definitions": {
            "C_B": "B.T @ B / N (uncentered second moment, float64)",
            "bottom10_trace_fraction": "sum ten smallest eigenvalues / trace(C_B)",
            "F_norm_rms": "sqrt(mean row squared L2 norm)",
            "F_bottom10_energy": "||F @ V_bottom10||_F^2 / ||F||_F^2",
            "task_z": "sqrt(z_dim) normalized E[reward * B(next_obs)]",
            "task_z_eigendirection_energy": "(z dot v_i)^2 / ||z||^2",
            "task_z_rayleigh": "z^T C_B z / ||z||^2",
            "F_parallel_energy": "||row projection of F onto task-z||_F^2 / ||F||_F^2",
            "task_Q": "row dot product F_i dot task-z, twin minimum",
        },
        "wandb_local_provenance": wandb_provenance,
        "run_configs": {
            experiment.key: str((experiment.run_dir / ".hydra/config.yaml").resolve())
            for experiment in selected
        },
        "outputs": {
            "availability": str(args.output_dir / "checkpoint_availability.csv"),
            "long_csv": str(long_path),
            "summary_csv": str(summary_path),
            "task_z_npz": str(task_z_path),
            "report": str(report_path),
            "replay_manifest": str(args.output_dir / "fixed_replay_manifest.npz"),
            "rendered_replay_cache": str(replay_cache_path),
            "rendered_replay_episode_parts": str(
                args.output_dir / CNN_PIXEL_EPISODE_PARTS_DIRNAME
            ),
        },
        "exclusions": [
            "goal_space=cheetah_speed checkpoints excluded because B is privileged-goal, not visual",
            "missing target frames are reported and never substituted",
        ],
    }
    atomic_json(args.output_dir / "analysis_manifest.json", manifest)
    print(f"CNN diagnostic complete: {args.output_dir}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("dino", "cnn"), default="dino")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--exorl-dir", type=Path, default=DEFAULT_EXORL)
    parser.add_argument("--index-bank", type=Path, default=DEFAULT_BANK)
    parser.add_argument("--experiments", type=parse_csv_list, default=())
    parser.add_argument("--frames", type=parse_int_list, default=())
    parser.add_argument("--render-shape", type=parse_int_list, default=(84, 84))
    parser.add_argument("--replay-size", type=int, default=None)
    parser.add_argument("--probe-size", type=int, default=DEFAULT_PROBE_SIZE)
    parser.add_argument("--inference-size", type=int, default=DEFAULT_INFERENCE_SIZE)
    parser.add_argument("--replay-seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--z-seed", type=int, default=DEFAULT_Z_SEED)
    parser.add_argument("--gradient-seed", type=int, default=20_260_829)
    parser.add_argument("--inference-chunk", type=int, default=256)
    parser.add_argument(
        "--device",
        default=None,
        help=(
            "compute device; defaults to cuda:0 when available, resolved only after "
            "CNN fixed replay creation"
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="validate and reuse the fixed replay cache and completed checkpoint bundles",
    )
    parser.add_argument("--allow-partial", action="store_true",
                        help="development smoke run only; skips final matched artifacts")
    return parser


def _resolve_and_validate_device(requested: Optional[str]) -> str:
    """Resolve a compute device at the profile's CUDA-safe initialization point."""
    if requested is None:
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    return requested


def _enable_opt_in_faulthandler() -> None:
    """Route SIGUSR1 stack dumps to this invocation's stderr when requested."""
    if os.environ.get("CNN_ANALYSIS_FAULTHANDLER") == "1":
        faulthandler.register(signal.SIGUSR1, file=sys.stderr, all_threads=True)


def validate_args(args: argparse.Namespace) -> List[Experiment]:
    args.output_dir = args.output_dir.resolve()
    args.exorl_dir = args.exorl_dir.resolve()
    args.index_bank = args.index_bank.resolve()
    known = {item.key: item for item in ALL_DINO_EXPERIMENTS}
    if not args.experiments:
        args.experiments = tuple(item.key for item in EXPERIMENTS)
    unknown = set(args.experiments) - set(known)
    if unknown:
        raise ValueError(f"unknown experiments: {sorted(unknown)}")
    selected = [known[key] for key in args.experiments]
    if {item.key for item in selected} != {item.key for item in EXPERIMENTS}:
        print("warning: subset run; aggregate matched-LR output requires all three experiments", flush=True)
    if args.probe_size != TRAIN_BATCH:
        raise ValueError("probe-size must remain 1024 because exact gradient attribution uses the training batch")
    if not args.allow_partial and args.inference_size != DEFAULT_INFERENCE_SIZE:
        raise ValueError("full analysis must use checkpoint num_inference_steps=5120")
    if args.replay_size % TRAIN_BATCH or args.replay_size < max(args.probe_size, args.inference_size):
        raise ValueError("replay-size must be divisible by 1024 and cover probe/inference subsets")
    if args.inference_chunk <= 0:
        raise ValueError("inference-chunk must be positive")
    for path in (args.exorl_dir, args.index_bank):
        if not path.exists():
            raise FileNotFoundError(path)
    for experiment in selected:
        config = experiment.run_dir / ".hydra/config.yaml"
        if not config.is_file():
            raise FileNotFoundError(config)
        cfg = omgcf.OmegaConf.load(config)
        if str(cfg.task) != experiment.task or str(cfg.obs_type) != "dino" or int(cfg.action_repeat) != 2:
            raise ValueError(f"run config mismatch for {experiment.run_id}")
        serialized_lr_f = getattr(cfg.agent, "lr_f", None)
        serialized_lr_b = getattr(cfg.agent, "lr_b", None)
        serialized_idm_lr = getattr(cfg.agent, "idm_lr", None)
        contract = {
            "pixel_separate_fb_encoders": bool(
                getattr(cfg.agent, "pixel_separate_fb_encoders", False)
            ),
            "dino_separate_fb_adapters": bool(
                getattr(cfg.agent, "dino_separate_fb_adapters", False)
            ),
            "dino_separate_backward_adapter": bool(
                getattr(cfg.agent, "dino_separate_backward_adapter", False)
            ),
            "dino_adapter_type": str(
                getattr(cfg.agent, "dino_adapter_type", "linear")
            ),
            "idm_coef": float(getattr(cfg.agent, "idm_coef", 0.0)),
            "idm_route": str(getattr(cfg.agent, "idm_route", "none")),
            "idm_encoder_mode": str(
                getattr(cfg.agent, "idm_encoder_mode", "legacy")
            ),
            "idm_lr": (
                None if serialized_idm_lr is None else float(serialized_idm_lr)
            ),
            "lr_f": None if serialized_lr_f is None else float(serialized_lr_f),
            "lr_b": None if serialized_lr_b is None else float(serialized_lr_b),
        }
        expected_contract = {
            "pixel_separate_fb_encoders": False,
            "dino_separate_fb_adapters": (
                experiment.expected_dino_separate_fb_adapters
            ),
            "dino_separate_backward_adapter": False,
            "dino_adapter_type": experiment.expected_dino_adapter_type,
            "idm_coef": experiment.expected_idm_coef,
            "idm_route": experiment.expected_idm_route,
            "idm_encoder_mode": experiment.expected_idm_encoder_mode,
            "idm_lr": experiment.expected_idm_lr,
            "lr_f": experiment.expected_lr_f,
            "lr_b": experiment.expected_lr_b,
        }
        if contract != expected_contract:
            raise ValueError(
                f"DINO run config mismatch for {experiment.run_id}: {contract}"
            )
        requested_frames = experiment.frames if not args.frames else tuple(
            frame for frame in experiment.frames if frame in args.frames
        )
        if args.frames and not requested_frames:
            raise ValueError(f"no requested frames apply to {experiment.key}")
        for frame in requested_frames:
            path = experiment.checkpoint_dir / f"snapshot_{frame}.pt"
            if not path.is_file():
                raise FileNotFoundError(f"requested checkpoint missing: {path}")
    selected_by_key = {item.key: item for item in selected}
    if "run_lr5e5" in selected_by_key and "run_lr1e4" in selected_by_key:
        configs = []
        for key in ("run_lr5e5", "run_lr1e4"):
            path = selected_by_key[key].run_dir / ".hydra/config.yaml"
            value = omgcf.OmegaConf.to_container(omgcf.OmegaConf.load(path), resolve=False)
            if not isinstance(value, dict) or not isinstance(value.get("agent"), dict):
                raise ValueError(f"unexpected Hydra config structure at {path}")
            value["agent"].pop("fb_lr", None)
            configs.append(value)
        if configs[0] != configs[1]:
            raise ValueError("matched Cheetah Run configs differ after removing agent.fb_lr")
    return selected


def main() -> None:
    _enable_opt_in_faulthandler()
    args = build_parser().parse_args()
    if args.output_dir is None:
        args.output_dir = CNN_DEFAULT_OUTPUT if args.profile == "cnn" else DEFAULT_OUTPUT
    if args.replay_size is None:
        args.replay_size = (
            CNN_DEFAULT_REPLAY_SIZE if args.profile == "cnn" else DEFAULT_REPLAY_SIZE
        )
    if args.profile == "cnn":
        main_cnn(args)
        return
    # Preserve the DINO profile's historical behavior: choose and validate the
    # CUDA default before validation/replay construction.  Only CNN rendering
    # needs the stricter post-replay CUDA initialization boundary.
    args.device = _resolve_and_validate_device(args.device)
    selected = validate_args(args)
    full_run = (
        set(args.experiments) == {item.key for item in EXPERIMENTS}
        and not args.frames and not args.allow_partial
    )
    if not full_run and not args.allow_partial:
        raise ValueError("final artifact generation requires all requested experiments/frames; pass --allow-partial only for a smoke test")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    replay = build_replay(args)
    task_rewards = {task: compute_task_rewards(replay, task) for task in sorted({item.task for item in selected})}
    if "cheetah_run" in task_rewards and np.corrcoef(
        replay.source_reward[:, 0], task_rewards["cheetah_run"][:, 0]
    )[0, 1] < 0.995:
        raise AssertionError("ExORL stored run reward provenance sanity check failed")
    save_replay_manifest(args.output_dir / "fixed_replay_manifest.npz", replay, task_rewards)

    results: List[CheckpointResult] = []
    wandb_provenance: Dict[str, Any] = {}
    work_items = [
        (experiment, frame)
        for experiment in selected
        for frame in (experiment.frames if not args.frames else tuple(
            value for value in experiment.frames if value in args.frames
        ))
    ]
    total = len(work_items)
    index = 0
    for experiment, frame in work_items:
        index += 1
        print(f"checkpoint {index}/{total}: {experiment.key} @ {frame:,}", flush=True)
        result, current_wandb = analyze_checkpoint(
            experiment, frame, replay, task_rewards[experiment.task], args
        )
        results.append(result)
        wandb_provenance[experiment.key] = current_wandb
        save_checkpoint_artifacts(args.output_dir, result)
        print(
            f"  reward={get_metric(result, 'wandb_alignment', 'eval/episode_reward'):.3f}, "
            f"condition={get_metric(result, 'b_geometry', 'effective_condition_number'):.4g}",
            flush=True,
        )
    add_consecutive_drift(results)
    # Drift needs the next checkpoint, so rewrite per-checkpoint metrics once
    # every consecutive pair is available.
    for result in results:
        save_checkpoint_artifacts(args.output_dir, result)
    if not full_run:
        atomic_csv(args.output_dir / "partial_checkpoint_diagnostics_long.csv",
                   [row for result in results for row in result.metrics])
        partial_gradient_rows = [
            row for result in results for row in result.gradient_rows
        ]
        if partial_gradient_rows:
            atomic_csv(
                args.output_dir / "partial_gradient_interference.csv",
                partial_gradient_rows,
            )
        print(f"partial smoke complete: {args.output_dir}", flush=True)
        return
    outputs = serialize_aggregate(args.output_dir, results)
    plot_spectra(results, args.output_dir / "b_eigenvalue_spectra.png")
    plot_walk_mechanisms(results, args.output_dir / "walk_mechanism_timeline.png")
    plot_lr_comparison(results, args.output_dir / "matched_lr_timeline.png")
    plot_geometry_drift_overview(results, args.output_dir / "geometry_drift_timelines.png")
    plot_f_m_q_tail_overview(results, args.output_dir / "f_m_q_tail_timelines.png")
    plot_gradient_cosines(results, args.output_dir / "gradient_cosines.png")
    report_path = args.output_dir / "DIAGNOSIS.md"
    report_path.write_text(generate_report(results, replay), encoding="utf-8")
    outputs.update({
        "report": str(report_path),
        "spectra_plot": str(args.output_dir / "b_eigenvalue_spectra.png"),
        "walk_plot": str(args.output_dir / "walk_mechanism_timeline.png"),
        "matched_lr_plot": str(args.output_dir / "matched_lr_timeline.png"),
        "geometry_drift_plot": str(args.output_dir / "geometry_drift_timelines.png"),
        "f_m_q_tail_plot": str(args.output_dir / "f_m_q_tail_timelines.png"),
        "gradient_plot": str(args.output_dir / "gradient_cosines.png"),
        "replay_manifest": str(args.output_dir / "fixed_replay_manifest.npz"),
    })
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()), "device": args.device,
        "code_sha256": {
            "analysis_script": sha256_file(Path(__file__).resolve()),
            "fb_loss_and_actor": sha256_file(Path(__file__).resolve().parent / "agent/fb_ddpg.py"),
            "reward_projection": sha256_file(Path(__file__).resolve().parent / "goals.py"),
            "environment_wrappers": sha256_file(Path(__file__).resolve().parent / "dmc.py"),
        },
        "software": {
            "torch_version": torch.__version__, "numpy_version": np.__version__,
            "torch_float32_matmul_precision": torch.get_float32_matmul_precision(),
        },
        "gradient_probe_seeds": {
            "base": args.gradient_seed, "numpy_mix": args.gradient_seed + 1,
            "target_action": args.gradient_seed + 2, "actor_action": args.gradient_seed + 3,
        },
        "optimizer_steps_allowed": False, "optimizer_steps_executed": 0,
        "replay": replay.metadata, "weak_definitions": list(WEAK_DEFINITIONS),
        "primary_weak_definition": PRIMARY_WEAK,
        "checkpoint_count": len(results),
        "checkpoints": [
            {
                "experiment": item.experiment.key, "run_id": item.experiment.run_id,
                "frame": item.frame, "global_step": item.global_step,
                "path": str(item.checkpoint_path.resolve()), "size": item.checkpoint_size,
                "sha256": item.checkpoint_sha256,
            }
            for item in results
        ],
        "wandb_local_provenance": wandb_provenance,
        "run_configs": {
            experiment.key: {
                "path": str((experiment.run_dir / '.hydra/config.yaml').resolve()),
                "sha256": sha256_file(experiment.run_dir / '.hydra/config.yaml'),
            }
            for experiment in selected
        },
        "outputs": outputs,
        "limitations": [
            "historical replay/RNG/task-z unavailable",
            "fixed ExORL source action_repeat=1 versus target run action_repeat=2",
            "task rewards are single-state proxies rather than native two-substep accumulated reward",
            "cross-checkpoint latent gauge is not aligned; raw projector drift is descriptive only",
        ],
    }
    atomic_json(args.output_dir / "analysis_manifest.json", manifest)
    print(f"complete: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
