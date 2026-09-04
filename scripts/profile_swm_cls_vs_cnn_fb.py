#!/usr/bin/env python3
"""Controlled, end-to-end profiler for SWM-CLS and CNN-FB.

This intentionally reuses :class:`url_benchmark.pretrain.Workspace` so that the
environment wrappers, replay buffer, FB agent, optimizers, and the otherwise
idle evaluation environment match training.  Periodic evaluation, checkpoint
serialization, videos, external logging, and diagnostic-only physics/z
statistics are not executed in the measured loop.

Formal runs use the defaults: 10k warm-up environment frames followed by three
non-overlapping 20k-frame windows.  ``--allow-short-smoke`` exists only to make
small integration checks possible; it is recorded in the output and must not be
used for reported measurements.
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import dataclasses
import json
import math
import os
os.environ["MUJOCO_GL"] = os.environ.get("MUJOCO_GL", "egl")
import platform
import shlex
import socket
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import hydra
import numpy as np
import omegaconf
import torch
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from url_benchmark import utils  # noqa: E402
from url_benchmark.agent.ddpg import Encoder  # noqa: E402
from url_benchmark.pretrain import Workspace  # noqa: E402


DINO_V2_BASE = "facebook/dinov2-base"
DEFAULT_TASKS = {
    "walker": "walker_walk",
    "quadruped": "quadruped_walk",
    "cheetah": "cheetah_walk",
}
DOMAIN_GOAL_SPACES = {
    "walker": "simplified_walker",
    "quadruped": "simplified_quadruped",
    "cheetah": None,
}
METHOD_CHOICES = ("swm_cls", "cnn_fb")
DOMAIN_CHOICES = tuple(DEFAULT_TASKS)


def _bytes_dict(value: int) -> Dict[str, Any]:
    return {"bytes": int(value), "gib": float(value / (1024 ** 3))}


def _summary(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {"n": 0, "mean": math.nan, "std": math.nan, "se": math.nan,
                "min": math.nan, "max": math.nan}
    vals = [float(x) for x in values]
    std = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return {
        "n": len(vals),
        "mean": statistics.fmean(vals),
        "std": std,
        "se": std / math.sqrt(len(vals)),
        "min": min(vals),
        "max": max(vals),
    }


def _git_metadata() -> Dict[str, Any]:
    def run(*args: str) -> Optional[str]:
        try:
            return subprocess.check_output(
                args, cwd=REPO_ROOT, stderr=subprocess.DEVNULL, text=True
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    commit = run("git", "rev-parse", "HEAD")
    status = run("git", "status", "--short")
    return {"commit": commit, "dirty": bool(status), "status_short": status}


class _NvmlMemoryInfo(ctypes.Structure):
    _fields_ = [
        ("total", ctypes.c_ulonglong),
        ("free", ctypes.c_ulonglong),
        ("used", ctypes.c_ulonglong),
    ]


class _NvmlProcessInfo(ctypes.Structure):
    _fields_ = [
        ("pid", ctypes.c_uint),
        ("used_gpu_memory", ctypes.c_ulonglong),
        ("gpu_instance_id", ctypes.c_uint),
        ("compute_instance_id", ctypes.c_uint),
    ]


class _NvmlPciInfo(ctypes.Structure):
    _fields_ = [
        ("bus_id_legacy", ctypes.c_char * 16),
        ("domain", ctypes.c_uint),
        ("bus", ctypes.c_uint),
        ("device", ctypes.c_uint),
        ("pci_device_id", ctypes.c_uint),
        ("pci_subsystem_id", ctypes.c_uint),
        ("bus_id", ctypes.c_char * 32),
    ]


class NvmlMemorySampler:
    """Samples whole-device NVML memory through libnvidia-ml, if available."""

    def __init__(self, logical_cuda_index: int, period_s: float) -> None:
        self.logical_cuda_index = int(logical_cuda_index)
        self.period_s = float(period_s)
        self.available = False
        self.error: Optional[str] = None
        self.process_error: Optional[str] = None
        self.handle = ctypes.c_void_p()
        self.device_locator: Optional[str] = None
        self.device_uuid: Optional[str] = None
        self.device_pci_bus_id: Optional[str] = None
        self.samples: List[int] = []
        self.pid_samples: List[int] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lib: Optional[ctypes.CDLL] = None
        try:
            self._initialize()
        except Exception as exc:  # NVML is optional, but the failure is reported.
            self.error = f"{type(exc).__name__}: {exc}"
            self.available = False

    def _check(self, code: int, call: str) -> None:
        if code != 0:
            raise RuntimeError(f"{call} returned NVML error {code}")

    def _initialize(self) -> None:
        lib = ctypes.CDLL("libnvidia-ml.so.1")
        lib.nvmlInit_v2.restype = ctypes.c_int
        self._check(lib.nvmlInit_v2(), "nvmlInit_v2")

        lib.nvmlDeviceGetHandleByIndex_v2.argtypes = [ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p)]
        lib.nvmlDeviceGetHandleByIndex_v2.restype = ctypes.c_int
        lib.nvmlDeviceGetHandleByUUID.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)]
        lib.nvmlDeviceGetHandleByUUID.restype = ctypes.c_int
        lib.nvmlDeviceGetMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(_NvmlMemoryInfo)]
        lib.nvmlDeviceGetMemoryInfo.restype = ctypes.c_int
        lib.nvmlDeviceGetUUID.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint]
        lib.nvmlDeviceGetUUID.restype = ctypes.c_int
        lib.nvmlDeviceGetPciInfo_v3.argtypes = [ctypes.c_void_p, ctypes.POINTER(_NvmlPciInfo)]
        lib.nvmlDeviceGetPciInfo_v3.restype = ctypes.c_int

        process_fn = getattr(lib, "nvmlDeviceGetComputeRunningProcesses_v3", None)
        if process_fn is None:
            process_fn = getattr(lib, "nvmlDeviceGetComputeRunningProcesses_v2", None)
        if process_fn is not None:
            process_fn.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint),
                ctypes.POINTER(_NvmlProcessInfo),
            ]
            process_fn.restype = ctypes.c_int
        self._process_fn = process_fn

        visible = [x.strip() for x in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if x.strip()]
        token = visible[self.logical_cuda_index] if self.logical_cuda_index < len(visible) else None
        if token and token.startswith("GPU-"):
            self._check(
                lib.nvmlDeviceGetHandleByUUID(token.encode(), ctypes.byref(self.handle)),
                "nvmlDeviceGetHandleByUUID",
            )
            self.device_locator = token
        else:
            physical_index = int(token) if token is not None and token.isdigit() else self.logical_cuda_index
            self._check(
                lib.nvmlDeviceGetHandleByIndex_v2(physical_index, ctypes.byref(self.handle)),
                "nvmlDeviceGetHandleByIndex_v2",
            )
            self.device_locator = f"physical-index:{physical_index}"

        uuid_buffer = ctypes.create_string_buffer(96)
        self._check(
            lib.nvmlDeviceGetUUID(self.handle, uuid_buffer, len(uuid_buffer)),
            "nvmlDeviceGetUUID",
        )
        self.device_uuid = uuid_buffer.value.decode()
        pci_info = _NvmlPciInfo()
        self._check(
            lib.nvmlDeviceGetPciInfo_v3(self.handle, ctypes.byref(pci_info)),
            "nvmlDeviceGetPciInfo_v3",
        )
        self.device_pci_bus_id = pci_info.bus_id.decode()
        self._lib = lib
        self.available = True

    def _read_used(self) -> int:
        assert self._lib is not None
        info = _NvmlMemoryInfo()
        self._check(
            self._lib.nvmlDeviceGetMemoryInfo(self.handle, ctypes.byref(info)),
            "nvmlDeviceGetMemoryInfo",
        )
        return int(info.used)

    def _read_pid_used(self) -> Optional[int]:
        if self._process_fn is None:
            return None
        count = ctypes.c_uint(0)
        # NVML_ERROR_INSUFFICIENT_SIZE (7) is the expected size-query result.
        code = self._process_fn(self.handle, ctypes.byref(count), None)
        if code not in (0, 7):
            self._check(code, "nvmlDeviceGetComputeRunningProcesses(size query)")
        if count.value == 0:
            return 0
        entries = (_NvmlProcessInfo * count.value)()
        self._check(
            self._process_fn(self.handle, ctypes.byref(count), entries),
            "nvmlDeviceGetComputeRunningProcesses",
        )
        pid = os.getpid()
        unavailable = (1 << 64) - 1
        for entry in entries[:count.value]:
            if entry.pid == pid:
                value = int(entry.used_gpu_memory)
                return None if value == unavailable else value
        return 0

    def _sample_once(self) -> None:
        self.samples.append(self._read_used())
        try:
            pid_used = self._read_pid_used()
        except Exception as exc:
            self.process_error = f"{type(exc).__name__}: {exc}"
            self._process_fn = None
            pid_used = None
        if pid_used is not None:
            self.pid_samples.append(pid_used)

    def start(self) -> None:
        if not self.available:
            return
        self.samples = []
        self.pid_samples = []
        try:
            self._sample_once()
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            self.available = False
            return
        self._stop.clear()

        def sample_loop() -> None:
            while not self._stop.wait(self.period_s):
                try:
                    self._sample_once()
                except Exception as exc:  # Preserve measurements obtained so far.
                    self.error = f"{type(exc).__name__}: {exc}"
                    break

        self._thread = threading.Thread(target=sample_loop, name="nvml-memory-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> Dict[str, Any]:
        if self.available:
            self._stop.set()
            if self._thread is not None:
                self._thread.join(timeout=max(1.0, 5 * self.period_s))
            try:
                self._sample_once()
            except Exception as exc:
                self.error = f"{type(exc).__name__}: {exc}"
        output: Dict[str, Any] = {
            "available": self.available,
            "error": self.error,
            "sample_period_s": self.period_s,
            "sample_count": len(self.samples),
            "device_locator": self.device_locator,
            "device_uuid": self.device_uuid,
            "device_pci_bus_id": self.device_pci_bus_id,
            "process_pid": os.getpid(),
            "process_error": self.process_error,
        }
        if self.samples:
            output.update({
                "first_used": _bytes_dict(self.samples[0]),
                "minimum_used": _bytes_dict(min(self.samples)),
                "peak_used": _bytes_dict(max(self.samples)),
                "peak_minus_first": _bytes_dict(max(self.samples) - self.samples[0]),
            })
        if self.pid_samples:
            output["process_memory"] = {
                "available": True,
                "sample_count": len(self.pid_samples),
                "first_used": _bytes_dict(self.pid_samples[0]),
                "minimum_used": _bytes_dict(min(self.pid_samples)),
                "peak_used": _bytes_dict(max(self.pid_samples)),
                "peak_minus_first": _bytes_dict(max(self.pid_samples) - self.pid_samples[0]),
            }
        else:
            output["process_memory"] = {"available": False, "sample_count": 0}
        return output


@dataclasses.dataclass
class LoopState:
    time_step: Any
    meta: Mapping[str, np.ndarray]


@dataclasses.dataclass
class StepTiming:
    reset_meta_action_s: float
    environment_step_replay_s: float
    interaction_s: float
    update_section_s: float
    actual_update_s: float
    did_gradient_update: bool
    reset_count: int


class ControlledProfile:
    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace
        self.device = torch.device(workspace.cfg.device)
        if self.device.type != "cuda":
            raise RuntimeError("Controlled profiling requires CUDA; refusing a CPU fallback")

    def sync(self) -> None:
        torch.cuda.synchronize(self.device)

    def _timed(self, fn: Callable[[], Any]) -> Tuple[Any, float]:
        self.sync()
        start = time.perf_counter()
        output = fn()
        self.sync()
        return output, time.perf_counter() - start

    def initialize_loop(self) -> LoopState:
        time_step = self.workspace.train_env.reset()
        meta = self.workspace._init_meta()  # pylint: disable=protected-access
        self.workspace.replay_loader.add(time_step, meta)
        return LoopState(time_step=time_step, meta=meta)

    def step(self, state: LoopState, timed: bool) -> Tuple[LoopState, StepTiming]:
        workspace = self.workspace
        reset_count = 0

        def action_and_reset() -> Tuple[Any, Mapping[str, np.ndarray], np.ndarray, int]:
            time_step, meta = state.time_step, state.meta
            resets = 0
            if time_step.last():
                workspace.global_episode += 1
                time_step = workspace.train_env.reset()
                meta = workspace._init_meta()  # pylint: disable=protected-access
                workspace.replay_loader.add(time_step, meta)
                resets = 1
            meta = workspace.agent.update_meta(
                meta, workspace.global_step, time_step,
                finetune=False, replay_loader=workspace.replay_loader,
            )
            with torch.no_grad(), utils.eval_mode(workspace.agent):
                action = workspace.agent.act(
                    time_step.observation, meta, workspace.global_step, eval_mode=False
                )
            return time_step, meta, action, resets

        if timed:
            (time_step, meta, action, reset_count), pre_s = self._timed(action_and_reset)
        else:
            time_step, meta, action, reset_count = action_and_reset()
            pre_s = 0.0

        seed_steps = workspace.cfg.num_seed_frames // workspace.cfg.action_repeat
        eligible_for_update = workspace.global_step >= seed_steps
        did_gradient_update = bool(
            eligible_for_update
            and workspace.global_step % workspace.agent.cfg.update_every_steps == 0
        )

        def update_agent() -> None:
            if eligible_for_update:
                workspace.agent.update(workspace.replay_loader, workspace.global_step)

        if timed:
            _, update_s = self._timed(update_agent)
        else:
            update_agent()
            update_s = 0.0

        def environment_and_replay() -> Any:
            next_time_step = workspace.train_env.step(action)
            workspace.replay_loader.add(next_time_step, meta)
            return next_time_step

        if timed:
            next_time_step, post_s = self._timed(environment_and_replay)
        else:
            next_time_step = environment_and_replay()
            post_s = 0.0

        workspace.global_step += 1
        next_state = LoopState(time_step=next_time_step, meta=meta)
        return next_state, StepTiming(
            reset_meta_action_s=pre_s,
            environment_step_replay_s=post_s,
            interaction_s=pre_s + post_s,
            update_section_s=update_s,
            actual_update_s=update_s if did_gradient_update else 0.0,
            did_gradient_update=did_gradient_update,
            reset_count=reset_count,
        )

    def run_steps(self, state: LoopState, num_steps: int, timed: bool) -> Tuple[LoopState, Dict[str, Any]]:
        timings: List[StepTiming] = []
        self.sync()
        start = time.perf_counter()
        for _ in range(num_steps):
            state, step_timing = self.step(state, timed=timed)
            if timed:
                timings.append(step_timing)
        self.sync()
        elapsed = time.perf_counter() - start
        action_repeat = int(self.workspace.cfg.action_repeat)
        frames = num_steps * action_repeat
        actual_updates = sum(int(x.did_gradient_update) for x in timings)
        reset_meta_action_s = sum(x.reset_meta_action_s for x in timings)
        environment_step_replay_s = sum(x.environment_step_replay_s for x in timings)
        interaction_s = sum(x.interaction_s for x in timings)
        update_section_s = sum(x.update_section_s for x in timings)
        actual_update_s = sum(x.actual_update_s for x in timings)
        return state, {
            "agent_steps": num_steps,
            "environment_frames": frames,
            "elapsed_s": elapsed,
            "environment_frames_per_second": frames / elapsed,
            "hours_per_1m_environment_frames": 1_000_000 / (frames / elapsed) / 3600,
            "gradient_updates": actual_updates,
            "gradient_updates_per_end_to_end_second": actual_updates / elapsed,
            "interaction_section_s": interaction_s,
            "average_interaction_ms_per_agent_step": 1000 * interaction_s / num_steps,
            "average_interaction_ms_per_environment_frame": 1000 * interaction_s / frames,
            "average_reset_meta_action_ms_per_agent_step": 1000 * reset_meta_action_s / num_steps,
            "average_environment_step_replay_ms_per_agent_step": (
                1000 * environment_step_replay_s / num_steps
            ),
            "all_update_calls_section_s": update_section_s,
            "actual_gradient_update_section_s": actual_update_s,
            "average_update_ms_per_gradient_update": (
                1000 * actual_update_s / actual_updates if actual_updates else math.nan
            ),
            "episode_resets": sum(x.reset_count for x in timings),
        }

    def pure_update(self, step: int) -> float:
        def update_agent() -> None:
            self.workspace.agent.update(self.workspace.replay_loader, step)

        _, elapsed = self._timed(update_agent)
        return elapsed


_MISSING = object()


def _direct_attr(obj: Any, name: str) -> Any:
    """Read a real attribute without triggering a wrapper's forwarding __getattr__."""
    try:
        return object.__getattribute__(obj, name)
    except AttributeError:
        return _MISSING


def _traverse_wrappers(env: Any) -> Iterable[Any]:
    seen: set[int] = set()
    current = env
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        # EnvWrapper.__getattr__ forwards private attributes to its child, and
        # some dm_control wrappers keep _env in __slots__.  Bypass forwarding
        # while supporting both __dict__ and slots.
        child = _direct_attr(current, "_env")
        if child is _MISSING:
            child = _direct_attr(current, "_environment")
        current = None if child is _MISSING else child


def _dino_backbones(env: Any) -> List[nn.Module]:
    models: List[nn.Module] = []
    for wrapper in _traverse_wrappers(env):
        model = _direct_attr(wrapper, "_model")
        if isinstance(model, nn.Module):
            models.append(model)
    return models


def _unique_parameters(items: Iterable[Any]) -> List[nn.Parameter]:
    unique: Dict[int, nn.Parameter] = {}
    for item in items:
        params: Iterable[nn.Parameter]
        if isinstance(item, nn.Parameter):
            params = (item,)
        elif isinstance(item, nn.Module):
            params = item.parameters()
        else:
            continue
        for param in params:
            unique.setdefault(id(param), param)
    return list(unique.values())


def _numel(params: Iterable[nn.Parameter]) -> int:
    return sum(int(param.numel()) for param in params)


def _optimizer_parameters(agent: Any) -> List[nn.Parameter]:
    params: List[nn.Parameter] = []
    for name in ("encoder_opt", "actor_opt", "fb_opt"):
        optimizer = getattr(agent, name, None)
        if optimizer is not None:
            for group in optimizer.param_groups:
                params.extend(group["params"])
    return _unique_parameters(params)


def _parameter_report(workspace: Workspace) -> Dict[str, Any]:
    agent_modules = [value for value in vars(workspace.agent).values() if isinstance(value, nn.Module)]
    agent_params = _unique_parameters(agent_modules)
    train_dino = _dino_backbones(workspace.train_env)
    eval_dino = _dino_backbones(workspace.eval_env)
    optimizer_params = _optimizer_parameters(workspace.agent)
    encoder_params = _unique_parameters((workspace.agent.encoder,))
    encoder_param_ids = {id(param) for param in encoder_params}
    trainable_visual = [param for param in optimizer_params if id(param) in encoder_param_ids]
    logical_dino = train_dino[:1]

    # Canonical total counts training-resident agent networks (including target
    # networks) plus one logical visual backbone.  The additional idle eval-env
    # backbone is separately exposed as the actual process-resident count.
    method_params = _unique_parameters([*agent_modules, *logical_dino])
    process_params = _unique_parameters([*agent_modules, *train_dino, *eval_dino])
    visual_front_end_params = _unique_parameters([workspace.agent.encoder, *logical_dino])
    actor_path_params = _unique_parameters([workspace.agent.encoder, workspace.agent.actor, *logical_dino])
    online_model_params = _unique_parameters([
        workspace.agent.encoder,
        workspace.agent.actor,
        workspace.agent.forward_net,
        workspace.agent.backward_net,
        *logical_dino,
    ])
    return {
        "total_parameter_count": _numel(method_params),
        "total_parameter_count_semantics": (
            "unique agent parameters including FB target networks, plus one logical visual backbone"
        ),
        "total_trainable_parameter_count": _numel(optimizer_params),
        "total_trainable_parameter_count_semantics": (
            "unique parameters present in encoder_opt, actor_opt, or fb_opt; target networks excluded"
        ),
        "trainable_visual_front_end_parameter_count": _numel(trainable_visual),
        "trainable_visual_front_end_parameter_count_semantics": (
            "optimizer-owned parameters in agent.encoder (CNN encoder or SWM LayerNorm+Linear projector)"
        ),
        "logical_visual_front_end_parameter_count": _numel(visual_front_end_params),
        "online_model_parameter_count_excluding_target_copies": _numel(online_model_params),
        "online_model_parameter_count_semantics": (
            "one visual backbone, encoder/projector, actor, online forward map, and online backward map"
        ),
        "agent_resident_parameter_count_including_targets": _numel(agent_params),
        "process_resident_parameter_count": _numel(process_params),
        "process_resident_parameter_count_semantics": (
            "unique parameters actually instantiated, including the idle evaluation environment backbone"
        ),
        "deployable_actor_path_parameter_count": _numel(actor_path_params),
        "environment_backbone_copy_count": len(train_dino) + len(eval_dino),
        "single_backbone_parameter_count": _numel(_unique_parameters(logical_dino)),
    }


def _assert_configuration(workspace: Workspace, method: str, domain: str, task: str) -> Dict[str, Any]:
    cfg = workspace.cfg
    agent = workspace.agent
    assert cfg.task == task and task.startswith(domain + "_")
    assert cfg.device == "cuda"
    assert int(cfg.action_repeat) == 2
    assert int(agent.cfg.batch_size) == 1024
    assert int(agent.cfg.update_every_steps) == 2
    assert int(cfg.num_seed_frames) == 4000
    assert math.isclose(float(cfg.discount), 0.99)
    assert math.isclose(float(cfg.future), 0.99)
    assert bool(cfg.update_encoder) and bool(agent.cfg.update_encoder)
    assert cfg.goal_space == DOMAIN_GOAL_SPACES[domain]
    assert not cfg.use_tb and not cfg.use_wandb and not cfg.use_hiplog
    assert not cfg.save_video and not cfg.save_train_video
    expected_agent_values = {
        "lr": 1e-4,
        "lr_coef": 1.0,
        "fb_target_tau": 0.01,
        "hidden_dim": 1024,
        "backward_hidden_dim": 526,
        "feature_dim": 512,
        "z_dim": 50,
        "mix_ratio": 0.5,
        "future_ratio": 0.0,
    }
    for name, expected in expected_agent_values.items():
        actual = getattr(agent.cfg, name)
        if isinstance(expected, float):
            assert math.isclose(float(actual), expected), (name, actual, expected)
        else:
            assert int(actual) == expected, (name, actual, expected)
    assert str(agent.cfg.stddev_schedule) == "0.2"
    optimizers = {
        name: getattr(agent, name)
        for name in ("encoder_opt", "actor_opt", "fb_opt")
    }
    assert all(isinstance(optimizer, torch.optim.Adam) for optimizer in optimizers.values())
    assert all(
        math.isclose(float(group["lr"]), 1e-4)
        for optimizer in optimizers.values()
        for group in optimizer.param_groups
    )
    agent_modules = [value for value in vars(agent).values() if isinstance(value, nn.Module)]
    assert all(not hasattr(module, "_orig_mod") for module in agent_modules)
    assert {param.dtype for param in _unique_parameters(agent_modules)} == {torch.float32}

    details: Dict[str, Any] = {
        "common_assertions": {
            "action_repeat": 2,
            "batch_size": 1024,
            "update_every_agent_steps": 2,
            "seed_frames": 4000,
            "discount": 0.99,
            "future_sampling_discount": 0.99,
            "goal_space": DOMAIN_GOAL_SPACES[domain],
            "external_logging": False,
            "optimizer": "Adam",
            "optimizer_lr": 1e-4,
            "lr_coef": 1.0,
            "fb_target_tau": 0.01,
            "hidden_dim": 1024,
            "backward_hidden_dim": 526,
            "feature_dim": 512,
            "z_dim": 50,
            "mix_ratio": 0.5,
            "future_ratio": 0.0,
            "stddev_schedule": "0.2",
            "parameter_dtype": "torch.float32",
            "torch_compile": False,
        }
    }
    if method == "swm_cls":
        assert cfg.obs_type == "dino"
        assert cfg.dino_model_name == DINO_V2_BASE
        assert bool(cfg.use_cls)
        assert tuple(cfg.render_shape) == (224, 224)
        # Historical run configs carry frame_stack=3, but dmc.make ignores it
        # for dino observations and explicitly installs EmbedStackWrapper(..., 1).
        assert int(cfg.frame_stack) == 3
        assert agent.cfg.dino_use_adapter
        assert agent.cfg.dino_adapter_type == "linear"
        assert int(agent.cfg.dino_adapter_output_dim) == 512
        assert isinstance(agent.encoder, nn.Sequential)
        assert len(agent.encoder) == 2
        assert isinstance(agent.encoder[0], nn.LayerNorm)
        assert isinstance(agent.encoder[1], nn.Linear)
        assert agent.encoder[1].in_features == 768 and agent.encoder[1].out_features == 512
        train_models = _dino_backbones(workspace.train_env)
        eval_models = _dino_backbones(workspace.eval_env)
        assert len(train_models) == 1 and len(eval_models) == 1
        for model in [*train_models, *eval_models]:
            assert not model.training
            assert all(not param.requires_grad for param in model.parameters())
            assert int(model.config.hidden_size) == 768
            assert {param.dtype for param in model.parameters()} == {torch.float32}
            assert not hasattr(model, "_orig_mod")
        stack_sizes = []
        for wrapper in _traverse_wrappers(workspace.train_env):
            num_frames = _direct_attr(wrapper, "_num_frames")
            if num_frames is not _MISSING:
                stack_sizes.append(int(num_frames))
        assert stack_sizes == [1], stack_sizes
        details["method_assertions"] = {
            "backbone": DINO_V2_BASE,
            "backbone_copy_count": 2,
            "backbones_eval_mode": True,
            "backbone_gradients_disabled": True,
            "backbone_parameter_dtype": "torch.float32",
            "cls_token": True,
            "input_shape": [3, 224, 224],
            "environment_embedding_stack": 1,
            "configured_frame_stack_ignored_for_dino": 3,
            "adapter": "LayerNorm(768) + Linear(768, 512)",
        }
    else:
        assert cfg.obs_type == "pixels"
        assert tuple(cfg.render_shape) == (84, 84)
        assert int(cfg.frame_stack) == 3
        assert isinstance(agent.encoder, Encoder)
        assert getattr(agent.aug, "pad", None) == 4
        convs = [module for module in agent.encoder.modules() if isinstance(module, nn.Conv2d)]
        assert len(convs) == 4
        assert all(conv.out_channels == 32 and conv.kernel_size == (3, 3) for conv in convs)
        assert convs[0].in_channels == 9 and convs[0].stride == (2, 2)
        assert all(conv.stride == (1, 1) for conv in convs[1:])
        assert not _dino_backbones(workspace.train_env)
        assert not _dino_backbones(workspace.eval_env)
        details["method_assertions"] = {
            "input_shape": [9, 84, 84],
            "frame_stack": 3,
            "random_shift_padding": 4,
            "convolutions": 4,
            "convolution_channels": 32,
            "visual_encoder_trained_end_to_end": True,
            "visual_encoder_parameter_dtype": "torch.float32",
        }
    return details


def _compose_config(args: argparse.Namespace) -> omegaconf.DictConfig:
    obs_type = "dino" if args.method == "swm_cls" else "pixels"
    render_size = 224 if args.method == "swm_cls" else 84
    # Preserve the completed-run config field for SWM.  The DINO environment
    # path always hardcodes one effective embedding frame, asserted below.
    frame_stack = 3
    goal_space = DOMAIN_GOAL_SPACES[args.domain]
    goal_space_override = "null" if goal_space is None else goal_space
    overrides = [
        f"task={args.task}",
        f"seed={args.seed}",
        "device=cuda",
        f"experiment=controlled_profile_{args.method}_{args.domain}",
        f"obs_type={obs_type}",
        f"frame_stack={frame_stack}",
        "action_repeat=2",
        f"render_shape=[{render_size},{render_size}]",
        f"goal_space={goal_space_override}",
        "append_goal_to_observation=false",
        "use_cls=true",
        "num_seed_frames=4000",
        "replay_buffer_episodes=5000",
        "update_encoder=true",
        "agent.batch_size=1024",
        "agent.update_every_steps=2",
        "agent.dino_use_adapter=true",
        "agent.dino_adapter_type=linear",
        "agent.dino_adapter_output_dim=512",
        "save_video=false",
        "save_train_video=false",
        "use_tb=false",
        "use_wandb=false",
        "use_hiplog=false",
        "checkpoint_root=null",
        "load_model=null",
        "load_replay_buffer=null",
        "final_tests=0",
        "eval_every_frames=10000",
        "num_train_frames=2000010",
    ]
    if args.method == "swm_cls":
        overrides.append(f"dino_model_name={DINO_V2_BASE}")
    with hydra.initialize_config_dir(
        config_dir=str(REPO_ROOT / "url_benchmark"), version_base="1.1"
    ):
        return hydra.compose(config_name="base_config", overrides=overrides)


def _torch_memory_report(device: torch.device) -> Dict[str, Any]:
    return {
        "max_memory_allocated": _bytes_dict(torch.cuda.max_memory_allocated(device)),
        "max_memory_reserved": _bytes_dict(torch.cuda.max_memory_reserved(device)),
        "memory_allocated_at_end": _bytes_dict(torch.cuda.memory_allocated(device)),
        "memory_reserved_at_end": _bytes_dict(torch.cuda.memory_reserved(device)),
    }


def _machine_metadata(device: torch.device) -> Dict[str, Any]:
    props = torch.cuda.get_device_properties(device)
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_logical_index": device.index if device.index is not None else torch.cuda.current_device(),
        "gpu_name": props.name,
        "gpu_total_memory": _bytes_dict(props.total_memory),
        "gpu_compute_capability": [props.major, props.minor],
        "cpu_affinity": affinity,
        "cpu_affinity_count": len(affinity) if affinity is not None else None,
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
    }


def _progress(stage: str, **values: Any) -> None:
    print(json.dumps({"profile_progress": stage, **values}, sort_keys=True), flush=True)


def _run(args: argparse.Namespace, cfg: omegaconf.DictConfig) -> Dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    torch.cuda.set_device(0 if args.cuda_logical_index is None else args.cuda_logical_index)
    device = torch.device("cuda", torch.cuda.current_device())
    props = torch.cuda.get_device_properties(device)
    if not args.allow_non_h100 and "H100" not in props.name.upper():
        raise RuntimeError(f"Formal profiling requires an NVIDIA H100, found {props.name!r}")

    work_dir = args.work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=False)
    os.chdir(work_dir)

    workspace = Workspace(cfg)
    assertions = _assert_configuration(workspace, args.method, args.domain, args.task)
    profiler = ControlledProfile(workspace)
    parameter_report = _parameter_report(workspace)
    _progress("initialized", method=args.method, domain=args.domain, task=args.task)
    state = profiler.initialize_loop()

    warmup_steps = args.warmup_env_frames // int(cfg.action_repeat)
    state, warmup_raw = profiler.run_steps(state, warmup_steps, timed=False)
    warmup = {
        "agent_steps": warmup_raw["agent_steps"],
        "environment_frames": warmup_raw["environment_frames"],
        "elapsed_s": warmup_raw["elapsed_s"],
        "start_environment_frame": 0,
        "end_environment_frame": workspace.global_frame,
        "completed_replay_episodes": len(workspace.replay_loader),
    }
    _progress("warmup_complete", environment_frame=workspace.global_frame,
              elapsed_s=warmup["elapsed_s"])
    # Warm-up is deliberately not a throughput result: without per-section
    # synchronization, elapsed time can include asynchronous overlap.
    profiler.sync()
    torch.cuda.reset_peak_memory_stats(device)
    nvml = NvmlMemorySampler(torch.cuda.current_device(), args.nvml_sample_period_s)
    nvml.start()

    measured_steps = args.measured_env_frames // int(cfg.action_repeat)
    steps_per_window = measured_steps // args.num_windows
    windows: List[Dict[str, Any]] = []
    for window_index in range(args.num_windows):
        start_frame = workspace.global_frame
        state, window = profiler.run_steps(state, steps_per_window, timed=True)
        window.update({
            "window_index": window_index,
            "start_environment_frame": start_frame,
            "end_environment_frame": workspace.global_frame,
        })
        windows.append(window)
        _progress(
            "end_to_end_window_complete",
            window_index=window_index,
            end_environment_frame=workspace.global_frame,
            environment_frames_per_second=window["environment_frames_per_second"],
        )
    nvml_measured = nvml.stop()
    torch_measured = _torch_memory_report(device)

    total_frames = sum(int(window["environment_frames"]) for window in windows)
    total_elapsed = sum(float(window["elapsed_s"]) for window in windows)
    total_updates = sum(int(window["gradient_updates"]) for window in windows)
    total_interaction = sum(float(window["interaction_section_s"]) for window in windows)
    total_update_time = sum(float(window["actual_gradient_update_section_s"]) for window in windows)
    total_agent_steps = sum(int(window["agent_steps"]) for window in windows)
    measured_aggregate = {
        "environment_frames": total_frames,
        "elapsed_s": total_elapsed,
        "environment_frames_per_second": total_frames / total_elapsed,
        "hours_per_1m_environment_frames": 1_000_000 / (total_frames / total_elapsed) / 3600,
        "gradient_updates": total_updates,
        "gradient_updates_per_end_to_end_second": total_updates / total_elapsed,
        "average_interaction_ms_per_agent_step": (
            1000 * total_interaction / total_agent_steps
        ),
        "average_interaction_ms_per_environment_frame": 1000 * total_interaction / total_frames,
        "average_reset_meta_action_ms_per_agent_step": sum(
            float(window["average_reset_meta_action_ms_per_agent_step"]) * int(window["agent_steps"])
            for window in windows
        ) / total_agent_steps,
        "average_environment_step_replay_ms_per_agent_step": sum(
            float(window["average_environment_step_replay_ms_per_agent_step"]) * int(window["agent_steps"])
            for window in windows
        ) / total_agent_steps,
        "average_update_ms_per_gradient_update": 1000 * total_update_time / total_updates,
        "window_variability": {
            "environment_frames_per_second": _summary(
                [float(window["environment_frames_per_second"]) for window in windows]
            ),
            "hours_per_1m_environment_frames": _summary(
                [float(window["hours_per_1m_environment_frames"]) for window in windows]
            ),
            "gradient_updates_per_end_to_end_second": _summary(
                [float(window["gradient_updates_per_end_to_end_second"]) for window in windows]
            ),
            "average_interaction_ms_per_agent_step": _summary(
                [float(window["average_interaction_ms_per_agent_step"]) for window in windows]
            ),
            "average_interaction_ms_per_environment_frame": _summary(
                [float(window["average_interaction_ms_per_environment_frame"]) for window in windows]
            ),
            "average_reset_meta_action_ms_per_agent_step": _summary(
                [float(window["average_reset_meta_action_ms_per_agent_step"]) for window in windows]
            ),
            "average_environment_step_replay_ms_per_agent_step": _summary(
                [float(window["average_environment_step_replay_ms_per_agent_step"]) for window in windows]
            ),
            "average_update_ms_per_gradient_update": _summary(
                [float(window["average_update_ms_per_gradient_update"]) for window in windows]
            ),
        },
    }

    # Pure optimization benchmark.  Every supplied step is divisible by the
    # configured update interval, so each call performs one real gradient update.
    update_interval = int(workspace.agent.cfg.update_every_steps)
    pure_step = ((workspace.global_step + update_interval - 1) // update_interval) * update_interval
    for _ in range(args.pure_update_warmup):
        profiler.pure_update(pure_step)
        pure_step += update_interval
    profiler.sync()
    torch.cuda.reset_peak_memory_stats(device)
    pure_nvml = NvmlMemorySampler(torch.cuda.current_device(), args.nvml_sample_period_s)
    pure_nvml.start()
    pure_windows: List[Dict[str, Any]] = []
    for window_index in range(args.num_windows):
        durations: List[float] = []
        profiler.sync()
        window_start = time.perf_counter()
        for _ in range(args.pure_updates_per_window):
            durations.append(profiler.pure_update(pure_step))
            pure_step += update_interval
        profiler.sync()
        window_elapsed = time.perf_counter() - window_start
        pure_windows.append({
            "window_index": window_index,
            "gradient_updates": args.pure_updates_per_window,
            "elapsed_s": window_elapsed,
            "gradient_updates_per_second": args.pure_updates_per_window / window_elapsed,
            "average_update_ms": 1000 * sum(durations) / len(durations),
        })
        _progress(
            "pure_update_window_complete",
            window_index=window_index,
            gradient_updates_per_second=pure_windows[-1]["gradient_updates_per_second"],
        )
    nvml_pure = pure_nvml.stop()
    torch_pure = _torch_memory_report(device)

    pure_total_updates = sum(int(window["gradient_updates"]) for window in pure_windows)
    pure_total_elapsed = sum(float(window["elapsed_s"]) for window in pure_windows)
    pure_aggregate = {
        "gradient_updates": pure_total_updates,
        "elapsed_s": pure_total_elapsed,
        "gradient_updates_per_second": pure_total_updates / pure_total_elapsed,
        "window_variability": {
            "gradient_updates_per_second": _summary(
                [float(window["gradient_updates_per_second"]) for window in pure_windows]
            ),
            "average_update_ms": _summary(
                [float(window["average_update_ms"]) for window in pure_windows]
            ),
        },
    }

    resolved_cfg = omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    return {
        "schema_version": 1,
        "valid_formal_profile": not args.allow_short_smoke and not args.allow_non_h100,
        "method": args.method,
        "domain": args.domain,
        "task": args.task,
        "seed": args.seed,
        "command": shlex.join([sys.executable, *sys.argv]),
        "repository": _git_metadata(),
        "machine": _machine_metadata(device),
        "configuration": resolved_cfg,
        "configuration_assertions": assertions,
        "parameter_counts": parameter_report,
        "measurement_protocol": {
            "warmup_environment_frames": args.warmup_env_frames,
            "measured_environment_frames": args.measured_env_frames,
            "measured_non_overlapping_windows": args.num_windows,
            "pure_update_warmup_updates": args.pure_update_warmup,
            "pure_updates_per_window": args.pure_updates_per_window,
            "cuda_synchronize_before_and_after_each_timed_section": True,
            "mixed_precision": False,
            "torch_compile": False,
            "extra_dino_feature_cache": False,
            "interaction_time_definition": (
                "environment reset/step and rendering, observation wrapper/encoder, meta/action selection, "
                "and replay insertion; reset/meta/action and env-step/replay are also reported separately"
            ),
            "update_time_definition": (
                "one actual agent.update call including replay sampling, host-to-device transfer, "
                "augmentation/encoding, FB update, actor update, and target update"
            ),
            "included": [
                "environment reset/step, rendering, and observation wrappers",
                "DINO preprocessing/forward or CNN action encoder forward",
                "meta update and action selection",
                "replay-buffer insertion and sampling",
                "RandomShiftsAug and CNN encoder optimization",
                "FB and actor updates",
            ],
            "excluded": [
                "periodic policy evaluation",
                "checkpoint serialization",
                "external logging (W&B, TensorBoard, hiplog)",
                "video recording",
                "diagnostic-only PhysicsAggregator and compute_z_correl",
            ],
            "hours_per_1m_is_controlled_window_extrapolation": True,
        },
        "warmup": warmup,
        "end_to_end": {
            "windows": windows,
            "aggregate": measured_aggregate,
            "torch_cuda_memory": torch_measured,
            "nvml_device_memory": nvml_measured,
        },
        "pure_replay_update": {
            "windows": pure_windows,
            "aggregate": pure_aggregate,
            "torch_cuda_memory": torch_pure,
            "nvml_device_memory": nvml_pure,
        },
    }


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHOD_CHOICES, required=True)
    parser.add_argument("--domain", choices=DOMAIN_CHOICES, required=True)
    parser.add_argument("--task", help="DMC task; defaults to <domain>_walk")
    parser.add_argument("--seed", type=int, default=2009)
    parser.add_argument("--warmup-env-frames", type=int, default=10_000)
    parser.add_argument("--measured-env-frames", type=int, default=60_000)
    parser.add_argument("--num-windows", type=int, default=3)
    parser.add_argument("--pure-update-warmup", type=int, default=10)
    parser.add_argument("--pure-updates-per-window", type=int, default=100)
    parser.add_argument("--nvml-sample-period-s", type=float, default=0.02)
    parser.add_argument("--cuda-logical-index", type=int, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument(
        "--allow-short-smoke", action="store_true",
        help="Allow non-reportable short runs or a non-H100 for integration testing",
    )
    parser.add_argument("--allow-non-h100", action="store_true")
    parser.add_argument(
        "--print-config", action="store_true",
        help="Compose and print the resolved Hydra config without creating envs or loading DINO",
    )
    args = parser.parse_args(argv)
    args.task = args.task or DEFAULT_TASKS[args.domain]
    if not args.task.startswith(args.domain + "_"):
        parser.error(f"--task {args.task!r} does not belong to domain {args.domain!r}")
    if args.num_windows != 3:
        parser.error("--num-windows must be exactly 3")
    if args.pure_update_warmup < 0 or args.pure_updates_per_window <= 0:
        parser.error("pure-update counts must be positive (warm-up may be zero)")
    if args.warmup_env_frames <= 0 or args.measured_env_frames <= 0:
        parser.error("frame counts must be positive")
    divisor = 2 * args.num_windows
    if args.measured_env_frames % divisor:
        parser.error(
            f"--measured-env-frames must be divisible by action_repeat*num_windows ({divisor})"
        )
    if args.warmup_env_frames % 2:
        parser.error("--warmup-env-frames must be divisible by action repeat 2")
    if not args.allow_short_smoke:
        if args.warmup_env_frames != 10_000:
            parser.error("formal profiling requires exactly 10,000 warm-up environment frames")
        if args.measured_env_frames < 50_000:
            parser.error("formal profiling requires at least 50,000 measured environment frames")
    args.output = args.output.resolve()
    if args.work_dir is None:
        args.work_dir = args.output.parent / f"work_{args.method}_{args.domain}_{int(time.time())}"
    return args


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args(argv)
    cfg = _compose_config(args)
    if args.print_config:
        print(omegaconf.OmegaConf.to_yaml(cfg, resolve=True))
        return
    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = _run(args, cfg)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps({
        "output": str(output_path),
        "method": result["method"],
        "domain": result["domain"],
        "fps": result["end_to_end"]["aggregate"]["environment_frames_per_second"],
        "hours_per_1m": result["end_to_end"]["aggregate"]["hours_per_1m_environment_frames"],
        "pure_updates_per_second": result["pure_replay_update"]["aggregate"]["gradient_updates_per_second"],
    }, indent=2))


if __name__ == "__main__":
    main()
