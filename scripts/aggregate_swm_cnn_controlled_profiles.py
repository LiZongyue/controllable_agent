#!/usr/bin/env python3
"""Validate and aggregate controlled SWM-CLS versus CNN-FB profiles.

Formal mode is deliberately strict: it writes no report unless exactly one valid
profile exists for every method/domain pair and all six profiles share the same
H100, PCI address, CPU affinity, software stack, protocol, and common training
configuration.  ``--allow-smoke`` is only for exercising this reporting pipeline
with shortened integration profiles; it requires an explicit output directory and
watermarks every generated artifact as development-only.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT_ROOT = REPO_ROOT / "analysis_outputs" / "swm_cls_vs_cnn_fb_20260727"
DEFAULT_PROFILE_DIR = DEFAULT_REPORT_ROOT / "controlled_profiles"

METHODS = ("swm_cls", "cnn_fb")
DOMAINS = ("walker", "quadruped", "cheetah")
EXPECTED_COVERAGE = {(method, domain) for method in METHODS for domain in DOMAINS}
METHOD_LABELS = {"swm_cls": "SWM-CLS", "cnn_fb": "CNN-FB"}
DOMAIN_LABELS = {domain: domain.capitalize() for domain in DOMAINS}
EXPECTED_TASKS = {domain: f"{domain}_walk" for domain in DOMAINS}
EXPECTED_GOAL_SPACES = {
    "walker": "simplified_walker",
    "quadruped": "simplified_quadruped",
    "cheetah": None,
}
EXPECTED_ACTION_DIMS = {"walker": 6, "quadruped": 12, "cheetah": 6}
EXPECTED_CPU_AFFINITY = list(range(8, 16))
EXPECTED_DINO = "facebook/dinov2-base"


class ValidationError(RuntimeError):
    """Raised when a profile cannot support the controlled comparison."""


@dataclass(frozen=True)
class Profile:
    path: Path
    data: Mapping[str, Any]

    @property
    def key(self) -> Tuple[str, str]:
        return str(self.data["method"]), str(self.data["domain"])


def _fail(message: str) -> None:
    raise ValidationError(message)


def _require(condition: bool, message: str) -> None:
    if not condition:
        _fail(message)


def _nested(data: Mapping[str, Any], path: str, source: Path) -> Any:
    value: Any = data
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            _fail(f"{source}: missing required field {path!r}")
        value = value[part]
    return value


def _optional(data: Mapping[str, Any], path: str, default: Any = None) -> Any:
    value: Any = data
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return default
        value = value[part]
    return value


def _equal(actual: Any, expected: Any, label: str, source: Path) -> None:
    if actual != expected:
        _fail(f"{source}: {label} is {actual!r}; expected {expected!r}")


def _close(actual: Any, expected: Any, label: str, source: Path,
           rel_tol: float = 1e-7, abs_tol: float = 1e-9) -> None:
    try:
        actual_f = float(actual)
        expected_f = float(expected)
    except (TypeError, ValueError) as exc:
        _fail(f"{source}: {label} is not numeric: {actual!r} ({exc})")
    if not math.isfinite(actual_f) or not math.isclose(
        actual_f, expected_f, rel_tol=rel_tol, abs_tol=abs_tol
    ):
        _fail(f"{source}: {label} is {actual_f!r}; expected {expected_f!r}")


def _load_candidates(profile_dir: Path) -> List[Profile]:
    _require(profile_dir.is_dir(), f"profile directory does not exist: {profile_dir}")
    profiles: List[Profile] = []
    for path in sorted(profile_dir.glob("*.json")):
        try:
            with path.open(encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            _fail(f"could not read {path}: {exc}")
        if not isinstance(data, Mapping):
            continue
        if data.get("method") in METHODS and data.get("domain") in DOMAINS:
            profiles.append(Profile(path=path.resolve(), data=data))
    _require(profiles, f"no profiler JSON files found in {profile_dir}")
    return profiles


def _unique_by_key(profiles: Iterable[Profile], description: str) -> Dict[Tuple[str, str], Profile]:
    output: Dict[Tuple[str, str], Profile] = {}
    for profile in profiles:
        if profile.key in output:
            _fail(
                f"duplicate {description} profiles for {profile.key}: "
                f"{output[profile.key].path} and {profile.path}"
            )
        output[profile.key] = profile
    return output


def _select_profiles(candidates: Sequence[Profile], allow_smoke: bool) -> List[Profile]:
    formal = _unique_by_key(
        (profile for profile in candidates if profile.data.get("valid_formal_profile") is True),
        "valid formal",
    )
    if not allow_smoke:
        missing = sorted(EXPECTED_COVERAGE - set(formal))
        extra = sorted(set(formal) - EXPECTED_COVERAGE)
        if missing or extra or len(formal) != len(EXPECTED_COVERAGE):
            details = [
                "formal aggregation requires exactly six valid_formal_profile=true JSONs",
                f"found={len(formal)}",
            ]
            if missing:
                details.append(f"missing={missing}")
            if extra:
                details.append(f"unexpected={extra}")
            details.append("short smoke profiles are accepted only with --allow-smoke")
            _fail("; ".join(details))
        return [formal[(method, domain)] for domain in DOMAINS for method in METHODS]

    # Once all six formal profiles exist, prefer them.  Before then, use the
    # smoke set by itself when available, rather than mixing incompatible short
    # and formal protocols in a nominally shared comparison.
    if set(formal) == EXPECTED_COVERAGE:
        return [formal[(method, domain)] for domain in DOMAINS for method in METHODS]
    smoke = _unique_by_key(
        (
            profile for profile in candidates if profile.data.get("valid_formal_profile") is not True
        ),
        "smoke",
    )
    selected = smoke or formal
    _require(selected, "--allow-smoke found no usable profiles")
    return [
        selected[(method, domain)]
        for domain in DOMAINS
        for method in METHODS
        if (method, domain) in selected
    ]


def _check_byte_value(value: Any, label: str, source: Path) -> None:
    _require(isinstance(value, Mapping), f"{source}: {label} must be an object")
    byte_count = value.get("bytes")
    gib = value.get("gib")
    _require(isinstance(byte_count, int) and byte_count >= 0,
             f"{source}: {label}.bytes must be a non-negative integer")
    _close(gib, byte_count / (1024 ** 3), f"{label}.gib", source)


def _check_memory(section: Mapping[str, Any], label: str, source: Path) -> None:
    torch_memory = _nested(section, "torch_cuda_memory", source)
    for name in (
        "max_memory_allocated",
        "max_memory_reserved",
        "memory_allocated_at_end",
        "memory_reserved_at_end",
    ):
        _check_byte_value(_nested(torch_memory, name, source), f"{label}.torch.{name}", source)
    allocated = _nested(torch_memory, "max_memory_allocated.bytes", source)
    reserved = _nested(torch_memory, "max_memory_reserved.bytes", source)
    _require(allocated <= reserved,
             f"{source}: {label} max_memory_allocated exceeds max_memory_reserved")

    nvml = _nested(section, "nvml_device_memory", source)
    _equal(nvml.get("available"), True, f"{label} NVML availability", source)
    _require(bool(nvml.get("device_uuid")), f"{source}: {label} lacks NVML GPU UUID")
    _require(bool(nvml.get("device_pci_bus_id")), f"{source}: {label} lacks NVML PCI bus ID")
    for name in ("first_used", "minimum_used", "peak_used", "peak_minus_first"):
        _check_byte_value(_nested(nvml, name, source), f"{label}.nvml.{name}", source)
    _require(
        _nested(nvml, "peak_used.bytes", source) >= _nested(nvml, "minimum_used.bytes", source),
        f"{source}: {label} NVML peak is below its minimum",
    )
    process_memory = nvml.get("process_memory", {})
    if isinstance(process_memory, Mapping) and process_memory.get("available") is True:
        for name in ("first_used", "minimum_used", "peak_used", "peak_minus_first"):
            _check_byte_value(
                _nested(process_memory, name, source), f"{label}.nvml.process_memory.{name}", source
            )


def _check_configuration(profile: Profile) -> None:
    data, source = profile.data, profile.path
    method, domain = profile.key
    cfg = _nested(data, "configuration", source)
    agent = _nested(cfg, "agent", source)

    common_expected = {
        "action_repeat": 2,
        "seed": 2009,
        "device": "cuda",
        "frame_stack": 3,
        "num_seed_frames": 4000,
        "num_train_frames": 2_000_010,
        "eval_every_frames": 10_000,
        "update_encoder": True,
        "append_goal_to_observation": False,
        "reward_free": True,
        "custom_reward": None,
        "replay_buffer_episodes": 5000,
        "use_tb": False,
        "use_wandb": False,
        "use_hiplog": False,
        "save_video": False,
        "save_train_video": False,
        "final_tests": 0,
        "checkpoint_root": None,
        "load_model": None,
        "load_replay_buffer": None,
    }
    for name, expected in common_expected.items():
        _equal(cfg.get(name), expected, f"configuration.{name}", source)
    _equal(cfg.get("task"), EXPECTED_TASKS[domain], "configuration.task", source)
    _equal(cfg.get("goal_space"), EXPECTED_GOAL_SPACES[domain], "configuration.goal_space", source)
    _close(cfg.get("discount"), 0.99, "configuration.discount", source)
    _close(cfg.get("future"), 0.99, "configuration.future", source)

    agent_expected = {
        "batch_size": 1024,
        "name": "fb_ddpg",
        "update_every_steps": 2,
        "update_encoder": True,
        "hidden_dim": 1024,
        "backward_hidden_dim": 526,
        "feature_dim": 512,
        "z_dim": 50,
        "dino_use_adapter": True,
        "dino_adapter_type": "linear",
        "dino_adapter_output_dim": 512,
        "stddev_schedule": "0.2",
    }
    for name, expected in agent_expected.items():
        _equal(agent.get(name), expected, f"configuration.agent.{name}", source)
    _equal(agent.get("goal_space"), EXPECTED_GOAL_SPACES[domain],
           "configuration.agent.goal_space", source)
    _equal(agent.get("action_shape"), [EXPECTED_ACTION_DIMS[domain]],
           "configuration.agent.action_shape", source)
    for name, expected in {
        "lr": 1e-4,
        "lr_coef": 1.0,
        "fb_target_tau": 0.01,
        "mix_ratio": 0.5,
        "future_ratio": 0.0,
    }.items():
        _close(agent.get(name), expected, f"configuration.agent.{name}", source)

    assertions = _nested(data, "configuration_assertions", source)
    common = _nested(assertions, "common_assertions", source)
    _equal(common.get("optimizer"), "Adam", "optimizer assertion", source)
    _close(common.get("optimizer_lr"), 1e-4, "optimizer LR assertion", source)
    _equal(common.get("parameter_dtype"), "torch.float32", "parameter dtype assertion", source)
    _equal(common.get("torch_compile"), False, "torch.compile assertion", source)

    method_assertions = _nested(assertions, "method_assertions", source)
    if method == "swm_cls":
        _equal(cfg.get("obs_type"), "dino", "SWM observation type", source)
        _equal(agent.get("obs_type"), "dino", "SWM agent observation type", source)
        _equal(cfg.get("render_shape"), [224, 224], "SWM render shape", source)
        _equal(cfg.get("dino_model_name"), EXPECTED_DINO, "SWM DINO model", source)
        _equal(cfg.get("use_cls"), True, "SWM CLS selection", source)
        expected = {
            "backbone": EXPECTED_DINO,
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
        _equal(cfg.get("obs_type"), "pixels", "CNN observation type", source)
        _equal(agent.get("obs_type"), "pixels", "CNN agent observation type", source)
        _equal(cfg.get("render_shape"), [84, 84], "CNN render shape", source)
        expected = {
            "input_shape": [9, 84, 84],
            "frame_stack": 3,
            "random_shift_padding": 4,
            "convolutions": 4,
            "convolution_channels": 32,
            "visual_encoder_trained_end_to_end": True,
            "visual_encoder_parameter_dtype": "torch.float32",
        }
    for name, expected_value in expected.items():
        _equal(method_assertions.get(name), expected_value, f"method assertion {name}", source)


def _check_parameters(profile: Profile) -> None:
    params = _nested(profile.data, "parameter_counts", profile.path)
    names = (
        "online_model_parameter_count_excluding_target_copies",
        "total_parameter_count",
        "total_trainable_parameter_count",
        "trainable_visual_front_end_parameter_count",
        "logical_visual_front_end_parameter_count",
        "agent_resident_parameter_count_including_targets",
        "process_resident_parameter_count",
        "deployable_actor_path_parameter_count",
        "environment_backbone_copy_count",
        "single_backbone_parameter_count",
    )
    for name in names:
        value = params.get(name)
        _require(isinstance(value, int) and value >= 0,
                 f"{profile.path}: parameter_counts.{name} must be a non-negative integer")
    online = params["online_model_parameter_count_excluding_target_copies"]
    method_total = params["total_parameter_count"]
    trainable = params["total_trainable_parameter_count"]
    process = params["process_resident_parameter_count"]
    _require(0 < trainable <= online <= method_total <= process,
             f"{profile.path}: inconsistent online/trainable/target/process parameter counts")
    _require(params["trainable_visual_front_end_parameter_count"] <= trainable,
             f"{profile.path}: trainable visual parameters exceed all trainable parameters")
    if profile.data["method"] == "swm_cls":
        _equal(params["environment_backbone_copy_count"], 2, "SWM DINO copy count", profile.path)
        _require(params["single_backbone_parameter_count"] > 0,
                 f"{profile.path}: SWM must report a nonzero DINO backbone")
    else:
        _equal(params["environment_backbone_copy_count"], 0, "CNN DINO copy count", profile.path)
        _equal(params["single_backbone_parameter_count"], 0, "CNN DINO parameter count", profile.path)


def _check_profile(profile: Profile, allow_smoke: bool) -> None:
    data, source = profile.data, profile.path
    _equal(data.get("schema_version"), 1, "schema_version", source)
    _require(profile.key in EXPECTED_COVERAGE, f"{source}: unexpected method/domain {profile.key}")
    _equal(data.get("task"), EXPECTED_TASKS[profile.key[1]], "task", source)
    _equal(data.get("seed"), 2009, "profiling seed", source)
    if not allow_smoke:
        _equal(data.get("valid_formal_profile"), True, "valid_formal_profile", source)

    machine = _nested(data, "machine", source)
    _require("H100" in str(machine.get("gpu_name", "")).upper(),
             f"{source}: formal comparison requires an NVIDIA H100")
    _equal(machine.get("cuda_visible_devices"), "2", "CUDA_VISIBLE_DEVICES", source)
    _equal(machine.get("cuda_logical_index"), 0, "logical CUDA index", source)
    _equal(machine.get("cpu_affinity"), EXPECTED_CPU_AFFINITY, "CPU affinity", source)
    _equal(machine.get("cpu_affinity_count"), 8, "CPU affinity count", source)
    _equal(machine.get("omp_num_threads"), "8", "OMP_NUM_THREADS", source)
    _equal(machine.get("mkl_num_threads"), "8", "MKL_NUM_THREADS", source)

    protocol = _nested(data, "measurement_protocol", source)
    warmup_frames = protocol.get("warmup_environment_frames")
    measured_frames = protocol.get("measured_environment_frames")
    if data.get("valid_formal_profile") is True:
        _equal(warmup_frames, 10_000, "formal warm-up frames", source)
        _require(isinstance(measured_frames, int) and measured_frames >= 50_000,
                 f"{source}: formal measured frames must be at least 50,000")
        _require(protocol.get("pure_update_warmup_updates", 0) >= 10,
                 f"{source}: formal pure-update warm-up must contain at least 10 updates")
        _require(protocol.get("pure_updates_per_window", 0) >= 100,
                 f"{source}: formal pure-update windows must contain at least 100 updates")
    else:
        _require(allow_smoke, f"{source}: shortened profile requires --allow-smoke")
        _require(isinstance(warmup_frames, int) and warmup_frames > 0,
                 f"{source}: smoke warm-up frames must be positive")
        _require(isinstance(measured_frames, int) and measured_frames > 0,
                 f"{source}: smoke measured frames must be positive")
    _equal(protocol.get("measured_non_overlapping_windows"), 3, "timing-window count", source)
    _equal(protocol.get("cuda_synchronize_before_and_after_each_timed_section"), True,
           "CUDA synchronization protocol", source)
    _equal(protocol.get("mixed_precision"), False, "mixed precision protocol", source)
    _equal(protocol.get("torch_compile"), False, "compile protocol", source)
    _equal(protocol.get("extra_dino_feature_cache"), False, "feature-cache protocol", source)
    _equal(protocol.get("hours_per_1m_is_controlled_window_extrapolation"), True,
           "hours/1M extrapolation label", source)
    required_exclusions = {
        "periodic policy evaluation",
        "checkpoint serialization",
        "external logging (W&B, TensorBoard, hiplog)",
        "video recording",
        "diagnostic-only PhysicsAggregator and compute_z_correl",
    }
    _require(required_exclusions <= set(protocol.get("excluded", [])),
             f"{source}: required exclusions are not completely recorded")
    required_inclusions = {
        "environment reset/step, rendering, and observation wrappers",
        "DINO preprocessing/forward or CNN action encoder forward",
        "replay-buffer insertion and sampling",
        "FB and actor updates",
    }
    _require(required_inclusions <= set(protocol.get("included", [])),
             f"{source}: required end-to-end components are not completely recorded")

    warmup = _nested(data, "warmup", source)
    _equal(warmup.get("environment_frames"), warmup_frames, "warm-up frame count", source)
    _equal(warmup.get("start_environment_frame"), 0, "warm-up start frame", source)
    _equal(warmup.get("end_environment_frame"), warmup_frames, "warm-up end frame", source)

    end_to_end = _nested(data, "end_to_end", source)
    windows = _nested(end_to_end, "windows", source)
    _require(isinstance(windows, list) and len(windows) == 3,
             f"{source}: end-to-end section must contain three windows")
    expected_start = warmup_frames
    window_frame_counts: List[int] = []
    total_elapsed = 0.0
    total_frames = 0
    total_updates = 0
    total_agent_steps = 0
    total_interaction = 0.0
    total_actual_update = 0.0
    for index, window in enumerate(windows):
        _equal(window.get("window_index"), index, f"end-to-end window {index} index", source)
        _equal(window.get("start_environment_frame"), expected_start,
               f"end-to-end window {index} start", source)
        frame_count = window.get("environment_frames")
        agent_steps = window.get("agent_steps")
        _require(isinstance(frame_count, int) and frame_count > 0,
                 f"{source}: window {index} frame count must be positive")
        _equal(frame_count, 2 * agent_steps, f"end-to-end window {index} action repeat", source)
        expected_end = expected_start + frame_count
        _equal(window.get("end_environment_frame"), expected_end,
               f"end-to-end window {index} end", source)
        elapsed = float(window.get("elapsed_s"))
        _require(elapsed > 0, f"{source}: window {index} elapsed time must be positive")
        updates = window.get("gradient_updates")
        _equal(updates, agent_steps // 2, f"end-to-end window {index} update frequency", source)
        _close(window.get("environment_frames_per_second"), frame_count / elapsed,
               f"end-to-end window {index} FPS", source)
        _close(window.get("hours_per_1m_environment_frames"), 1_000_000 / (frame_count / elapsed) / 3600,
               f"end-to-end window {index} hours/1M", source)
        _close(window.get("gradient_updates_per_end_to_end_second"), updates / elapsed,
               f"end-to-end window {index} update throughput", source)
        actual_update_s = float(window.get("actual_gradient_update_section_s"))
        _close(window.get("average_update_ms_per_gradient_update"),
               1000 * actual_update_s / updates,
               f"end-to-end window {index} average update time", source)
        interaction_s = float(window.get("interaction_section_s"))
        _close(window.get("average_interaction_ms_per_agent_step"),
               1000 * interaction_s / agent_steps,
               f"end-to-end window {index} interaction time", source)
        total_elapsed += elapsed
        window_frame_counts.append(frame_count)
        total_frames += frame_count
        total_updates += updates
        total_agent_steps += agent_steps
        total_interaction += interaction_s
        total_actual_update += actual_update_s
        expected_start = expected_end
    _equal(total_frames, measured_frames, "sum of measured end-to-end frames", source)
    _require(len(set(window_frame_counts)) == 1,
             f"{source}: the three timing windows must contain equal frame counts")

    aggregate = _nested(end_to_end, "aggregate", source)
    _equal(aggregate.get("environment_frames"), total_frames, "aggregate frames", source)
    _equal(aggregate.get("gradient_updates"), total_updates, "aggregate updates", source)
    _close(aggregate.get("elapsed_s"), total_elapsed, "aggregate elapsed time", source)
    _close(aggregate.get("environment_frames_per_second"), total_frames / total_elapsed,
           "aggregate FPS", source)
    _close(aggregate.get("hours_per_1m_environment_frames"),
           1_000_000 / (total_frames / total_elapsed) / 3600,
           "aggregate hours/1M", source)
    _close(aggregate.get("gradient_updates_per_end_to_end_second"), total_updates / total_elapsed,
           "aggregate end-to-end updates/s", source)
    _close(aggregate.get("average_interaction_ms_per_agent_step"),
           1000 * total_interaction / total_agent_steps,
           "aggregate interaction time", source)
    _close(aggregate.get("average_update_ms_per_gradient_update"),
           1000 * total_actual_update / total_updates,
           "aggregate update time", source)
    variability = _nested(aggregate, "window_variability", source)
    for name in (
        "environment_frames_per_second",
        "hours_per_1m_environment_frames",
        "gradient_updates_per_end_to_end_second",
        "average_interaction_ms_per_agent_step",
        "average_update_ms_per_gradient_update",
    ):
        stats = _nested(variability, name, source)
        _equal(stats.get("n"), 3, f"{name} variability window count", source)
        _require(float(stats.get("se")) >= 0, f"{source}: {name} SE must be non-negative")
    _check_memory(end_to_end, "end_to_end", source)

    pure = _nested(data, "pure_replay_update", source)
    pure_windows = _nested(pure, "windows", source)
    _require(isinstance(pure_windows, list) and len(pure_windows) == 3,
             f"{source}: pure-update section must contain three windows")
    pure_total_updates = 0
    pure_total_elapsed = 0.0
    for index, window in enumerate(pure_windows):
        _equal(window.get("window_index"), index, f"pure-update window {index} index", source)
        updates = window.get("gradient_updates")
        _equal(updates, protocol.get("pure_updates_per_window"),
               f"pure-update window {index} update count", source)
        elapsed = float(window.get("elapsed_s"))
        _require(elapsed > 0, f"{source}: pure-update window {index} elapsed must be positive")
        _close(window.get("gradient_updates_per_second"), updates / elapsed,
               f"pure-update window {index} throughput", source)
        _require(float(window.get("average_update_ms")) > 0,
                 f"{source}: pure-update window {index} average update time must be positive")
        pure_total_updates += updates
        pure_total_elapsed += elapsed
    pure_aggregate = _nested(pure, "aggregate", source)
    _equal(pure_aggregate.get("gradient_updates"), pure_total_updates,
           "pure-update aggregate count", source)
    _close(pure_aggregate.get("elapsed_s"), pure_total_elapsed,
           "pure-update aggregate elapsed", source)
    _close(pure_aggregate.get("gradient_updates_per_second"),
           pure_total_updates / pure_total_elapsed,
           "pure-update aggregate throughput", source)
    _equal(_nested(pure_aggregate, "window_variability.gradient_updates_per_second.n", source), 3,
           "pure-update variability window count", source)
    _equal(_nested(pure_aggregate, "window_variability.average_update_ms.n", source), 3,
           "pure-update duration variability window count", source)
    _check_memory(pure, "pure_replay_update", source)

    for section_name in ("end_to_end", "pure_replay_update"):
        nvml = _nested(data, f"{section_name}.nvml_device_memory", source)
        _equal(nvml.get("device_locator"), "physical-index:2",
               f"{section_name} NVML device locator", source)
        _close(nvml.get("sample_period_s"), 0.02,
               f"{section_name} NVML sample period", source)
    e2e_nvml = _nested(data, "end_to_end.nvml_device_memory", source)
    pure_nvml = _nested(data, "pure_replay_update.nvml_device_memory", source)
    _equal(pure_nvml.get("device_uuid"), e2e_nvml.get("device_uuid"),
           "within-profile NVML UUID", source)
    _equal(pure_nvml.get("device_pci_bus_id"), e2e_nvml.get("device_pci_bus_id"),
           "within-profile NVML PCI bus ID", source)

    _check_configuration(profile)
    _check_parameters(profile)


def _canonical_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    value: MutableMapping[str, Any] = copy.deepcopy(dict(config))
    for name in ("task", "goal_space", "experiment", "obs_type", "render_shape", "dino_model_name"):
        value.pop(name, None)
    agent = value.get("agent")
    if isinstance(agent, MutableMapping):
        for name in ("obs_shape", "obs_type", "goal_space", "action_shape"):
            agent.pop(name, None)
    return value


def _cross_validate(profiles: Sequence[Profile], allow_smoke: bool) -> Mapping[str, Any]:
    for profile in profiles:
        _check_profile(profile, allow_smoke=allow_smoke)

    coverage = {profile.key for profile in profiles}
    if not allow_smoke:
        _equal(coverage, EXPECTED_COVERAGE, "formal method/domain coverage", Path("<all profiles>"))

    reference = profiles[0]
    ref_machine = reference.data["machine"]
    ref_e2e_nvml = reference.data["end_to_end"]["nvml_device_memory"]
    ref_protocol = reference.data["measurement_protocol"]
    ref_config = _canonical_config(reference.data["configuration"])
    ref_repo = reference.data.get("repository", {})
    hardware_fields = (
        "hostname",
        "gpu_name",
        "gpu_total_memory",
        "gpu_compute_capability",
        "cpu_affinity",
        "cpu_affinity_count",
        "omp_num_threads",
        "mkl_num_threads",
        "torch",
        "torch_cuda",
        "cudnn",
    )
    protocol_fields = (
        "warmup_environment_frames",
        "measured_environment_frames",
        "measured_non_overlapping_windows",
        "pure_update_warmup_updates",
        "pure_updates_per_window",
        "cuda_synchronize_before_and_after_each_timed_section",
        "mixed_precision",
        "torch_compile",
        "extra_dino_feature_cache",
    )
    for profile in profiles[1:]:
        machine = profile.data["machine"]
        nvml = profile.data["end_to_end"]["nvml_device_memory"]
        for name in hardware_fields:
            _equal(machine.get(name), ref_machine.get(name), f"shared machine field {name}", profile.path)
        _equal(nvml.get("device_uuid"), ref_e2e_nvml.get("device_uuid"),
               "shared H100 UUID", profile.path)
        _equal(nvml.get("device_pci_bus_id"), ref_e2e_nvml.get("device_pci_bus_id"),
               "shared H100 PCI bus ID", profile.path)
        for name in protocol_fields:
            _equal(profile.data["measurement_protocol"].get(name), ref_protocol.get(name),
                   f"shared protocol field {name}", profile.path)
        _equal(_canonical_config(profile.data["configuration"]), ref_config,
               "common configuration after declared method/domain differences", profile.path)
        _equal(profile.data.get("repository", {}).get("commit"), ref_repo.get("commit"),
               "repository commit", profile.path)
        _equal(profile.data.get("repository", {}).get("dirty"), ref_repo.get("dirty"),
               "repository dirty-worktree flag", profile.path)
        _equal(profile.data.get("repository", {}).get("status_short"), ref_repo.get("status_short"),
               "repository recorded worktree state", profile.path)

    return {
        "coverage": sorted([list(key) for key in coverage]),
        "missing_coverage": sorted([list(key) for key in EXPECTED_COVERAGE - coverage]),
        "gpu_name": ref_machine["gpu_name"],
        "gpu_uuid": ref_e2e_nvml["device_uuid"],
        "gpu_pci_bus_id": ref_e2e_nvml["device_pci_bus_id"],
        "cpu_affinity": ref_machine["cpu_affinity"],
        "hostname": ref_machine["hostname"],
        "protocol": {name: ref_protocol.get(name) for name in protocol_fields},
        "repository_commit": ref_repo.get("commit"),
        "repository_dirty": ref_repo.get("dirty"),
    }


def _base_csv_row(profile: Profile, section: str, row_type: str, window_index: Any) -> Dict[str, Any]:
    data = profile.data
    machine = data["machine"]
    nvml = data[section]["nvml_device_memory"]
    protocol = data["measurement_protocol"]
    params = data["parameter_counts"]
    return {
        "profile_file": str(profile.path),
        "profile_valid_formal": data["valid_formal_profile"],
        "method": data["method"],
        "domain": data["domain"],
        "task": data["task"],
        "seed": data["seed"],
        "section": section,
        "row_type": row_type,
        "window_index": window_index,
        "warmup_environment_frames": protocol["warmup_environment_frames"],
        "measured_environment_frames": protocol["measured_environment_frames"],
        "measured_non_overlapping_windows": protocol["measured_non_overlapping_windows"],
        "pure_update_warmup_updates": protocol["pure_update_warmup_updates"],
        "pure_updates_per_window": protocol["pure_updates_per_window"],
        "hours_per_1m_is_extrapolated": True,
        "gpu_name": machine["gpu_name"],
        "gpu_uuid": nvml["device_uuid"],
        "gpu_pci_bus_id": nvml["device_pci_bus_id"],
        "cuda_visible_devices": machine["cuda_visible_devices"],
        "cpu_affinity": "-".join(map(str, machine["cpu_affinity"])),
        "repository_commit": data.get("repository", {}).get("commit", ""),
        "online_model_parameter_count_excluding_target_copies": params[
            "online_model_parameter_count_excluding_target_copies"
        ],
        "target_copy_parameter_count": (
            params["total_parameter_count"]
            - params["online_model_parameter_count_excluding_target_copies"]
        ),
        "total_trainable_parameter_count": params["total_trainable_parameter_count"],
        "trainable_visual_front_end_parameter_count": params[
            "trainable_visual_front_end_parameter_count"
        ],
        "logical_visual_front_end_parameter_count": params[
            "logical_visual_front_end_parameter_count"
        ],
        "agent_resident_parameter_count_including_targets": params[
            "agent_resident_parameter_count_including_targets"
        ],
        "method_parameter_count_including_targets_and_one_logical_backbone": params[
            "total_parameter_count"
        ],
        "process_resident_parameter_count": params["process_resident_parameter_count"],
        "deployable_actor_path_parameter_count": params["deployable_actor_path_parameter_count"],
        "environment_backbone_copy_count": params["environment_backbone_copy_count"],
        "single_backbone_parameter_count": params["single_backbone_parameter_count"],
    }


def _memory_csv_fields(section_data: Mapping[str, Any]) -> Dict[str, Any]:
    torch_memory = section_data["torch_cuda_memory"]
    nvml = section_data["nvml_device_memory"]
    process = nvml.get("process_memory", {})
    return {
        "torch_peak_allocated_bytes": torch_memory["max_memory_allocated"]["bytes"],
        "torch_peak_allocated_gib": torch_memory["max_memory_allocated"]["gib"],
        "torch_peak_reserved_bytes": torch_memory["max_memory_reserved"]["bytes"],
        "torch_peak_reserved_gib": torch_memory["max_memory_reserved"]["gib"],
        "nvml_device_peak_used_bytes": nvml["peak_used"]["bytes"],
        "nvml_device_peak_used_gib": nvml["peak_used"]["gib"],
        "nvml_device_peak_minus_first_bytes": nvml["peak_minus_first"]["bytes"],
        "nvml_device_peak_minus_first_gib": nvml["peak_minus_first"]["gib"],
        "nvml_process_memory_available": process.get("available", False),
        "nvml_process_peak_used_bytes": _optional(process, "peak_used.bytes", ""),
        "nvml_process_peak_used_gib": _optional(process, "peak_used.gib", ""),
        "nvml_process_peak_minus_first_bytes": _optional(process, "peak_minus_first.bytes", ""),
        "nvml_process_peak_minus_first_gib": _optional(process, "peak_minus_first.gib", ""),
    }


def _csv_rows(profiles: Sequence[Profile]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for profile in profiles:
        end = profile.data["end_to_end"]
        variability = end["aggregate"]["window_variability"]
        for window in end["windows"]:
            row = _base_csv_row(profile, "end_to_end", "window", window["window_index"])
            row.update({
                "environment_frames": window["environment_frames"],
                "elapsed_s": window["elapsed_s"],
                "environment_frames_per_second": window["environment_frames_per_second"],
                "hours_per_1m_environment_frames_extrapolated": window[
                    "hours_per_1m_environment_frames"
                ],
                "gradient_updates": window["gradient_updates"],
                "gradient_updates_per_second": window[
                    "gradient_updates_per_end_to_end_second"
                ],
                "average_interaction_ms_per_agent_step": window[
                    "average_interaction_ms_per_agent_step"
                ],
                "average_interaction_ms_per_environment_frame": window[
                    "average_interaction_ms_per_environment_frame"
                ],
                "average_reset_meta_action_ms_per_agent_step": window.get(
                    "average_reset_meta_action_ms_per_agent_step", ""
                ),
                "average_environment_step_replay_ms_per_agent_step": window.get(
                    "average_environment_step_replay_ms_per_agent_step", ""
                ),
                "average_update_ms_per_gradient_update": window[
                    "average_update_ms_per_gradient_update"
                ],
            })
            rows.append(row)
        agg = end["aggregate"]
        row = _base_csv_row(profile, "end_to_end", "aggregate", "")
        row.update({
            "environment_frames": agg["environment_frames"],
            "elapsed_s": agg["elapsed_s"],
            "environment_frames_per_second": agg["environment_frames_per_second"],
            "environment_frames_per_second_window_se": variability[
                "environment_frames_per_second"
            ]["se"],
            "hours_per_1m_environment_frames_extrapolated": agg[
                "hours_per_1m_environment_frames"
            ],
            "hours_per_1m_environment_frames_window_se": variability[
                "hours_per_1m_environment_frames"
            ]["se"],
            "gradient_updates": agg["gradient_updates"],
            "gradient_updates_per_second": agg["gradient_updates_per_end_to_end_second"],
            "gradient_updates_per_second_window_se": variability[
                "gradient_updates_per_end_to_end_second"
            ]["se"],
            "average_interaction_ms_per_agent_step": agg[
                "average_interaction_ms_per_agent_step"
            ],
            "average_interaction_ms_per_agent_step_window_se": variability[
                "average_interaction_ms_per_agent_step"
            ]["se"],
            "average_interaction_ms_per_environment_frame": agg[
                "average_interaction_ms_per_environment_frame"
            ],
            "average_reset_meta_action_ms_per_agent_step": agg.get(
                "average_reset_meta_action_ms_per_agent_step", ""
            ),
            "average_environment_step_replay_ms_per_agent_step": agg.get(
                "average_environment_step_replay_ms_per_agent_step", ""
            ),
            "average_update_ms_per_gradient_update": agg[
                "average_update_ms_per_gradient_update"
            ],
            "average_update_ms_per_gradient_update_window_se": variability[
                "average_update_ms_per_gradient_update"
            ]["se"],
        })
        row.update(_memory_csv_fields(end))
        rows.append(row)

        pure = profile.data["pure_replay_update"]
        pure_variability = pure["aggregate"]["window_variability"]
        for window in pure["windows"]:
            row = _base_csv_row(profile, "pure_replay_update", "window", window["window_index"])
            row.update({
                "elapsed_s": window["elapsed_s"],
                "gradient_updates": window["gradient_updates"],
                "gradient_updates_per_second": window["gradient_updates_per_second"],
                "average_update_ms_per_gradient_update": window["average_update_ms"],
            })
            rows.append(row)
        pure_agg = pure["aggregate"]
        row = _base_csv_row(profile, "pure_replay_update", "aggregate", "")
        row.update({
            "elapsed_s": pure_agg["elapsed_s"],
            "gradient_updates": pure_agg["gradient_updates"],
            "gradient_updates_per_second": pure_agg["gradient_updates_per_second"],
            "gradient_updates_per_second_window_se": pure_variability[
                "gradient_updates_per_second"
            ]["se"],
            "average_update_ms_per_gradient_update": pure_variability["average_update_ms"]["mean"],
            "average_update_ms_per_gradient_update_window_se": pure_variability[
                "average_update_ms"
            ]["se"],
        })
        row.update(_memory_csv_fields(pure))
        rows.append(row)
    return rows


CSV_COLUMNS = (
    "profile_file",
    "profile_valid_formal",
    "method",
    "domain",
    "task",
    "seed",
    "section",
    "row_type",
    "window_index",
    "warmup_environment_frames",
    "measured_environment_frames",
    "measured_non_overlapping_windows",
    "pure_update_warmup_updates",
    "pure_updates_per_window",
    "hours_per_1m_is_extrapolated",
    "environment_frames",
    "elapsed_s",
    "environment_frames_per_second",
    "environment_frames_per_second_window_se",
    "hours_per_1m_environment_frames_extrapolated",
    "hours_per_1m_environment_frames_window_se",
    "gradient_updates",
    "gradient_updates_per_second",
    "gradient_updates_per_second_window_se",
    "average_interaction_ms_per_agent_step",
    "average_interaction_ms_per_agent_step_window_se",
    "average_interaction_ms_per_environment_frame",
    "average_reset_meta_action_ms_per_agent_step",
    "average_environment_step_replay_ms_per_agent_step",
    "average_update_ms_per_gradient_update",
    "average_update_ms_per_gradient_update_window_se",
    "torch_peak_allocated_bytes",
    "torch_peak_allocated_gib",
    "torch_peak_reserved_bytes",
    "torch_peak_reserved_gib",
    "nvml_device_peak_used_bytes",
    "nvml_device_peak_used_gib",
    "nvml_device_peak_minus_first_bytes",
    "nvml_device_peak_minus_first_gib",
    "nvml_process_memory_available",
    "nvml_process_peak_used_bytes",
    "nvml_process_peak_used_gib",
    "nvml_process_peak_minus_first_bytes",
    "nvml_process_peak_minus_first_gib",
    "online_model_parameter_count_excluding_target_copies",
    "target_copy_parameter_count",
    "total_trainable_parameter_count",
    "trainable_visual_front_end_parameter_count",
    "logical_visual_front_end_parameter_count",
    "agent_resident_parameter_count_including_targets",
    "method_parameter_count_including_targets_and_one_logical_backbone",
    "process_resident_parameter_count",
    "deployable_actor_path_parameter_count",
    "environment_backbone_copy_count",
    "single_backbone_parameter_count",
    "gpu_name",
    "gpu_uuid",
    "gpu_pci_bus_id",
    "cuda_visible_devices",
    "cpu_affinity",
    "repository_commit",
)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in CSV_COLUMNS})


def _latex_pm(value: float, se: float, digits: int = 2) -> str:
    return f"{value:.{digits}f} $\\pm$ {se:.{digits}f}"


def _latex_table(profiles: Sequence[Profile], development: bool) -> str:
    by_key = {profile.key: profile for profile in profiles}
    present_domains = [
        domain for domain in DOMAINS
        if any((method, domain) in by_key for method in METHODS)
    ]
    status = " DEVELOPMENT-ONLY SMOKE RESULTS; DO NOT CITE." if development else ""
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        (
            r"\caption{Controlled runtime, memory, and parameter comparison on one NVIDIA H100."
            + status
            + r" Runtime entries are pooled estimates with $\pm$ standard error over three "
            r"non-overlapping timing windows. Hours/1M is an extrapolation from the controlled "
            r"window, not an observed one-million-frame run.}"
        ),
        r"\label{tab:swm-cnn-controlled-cost}",
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"Method & Domain & Env. FPS $\uparrow$ & h/1M$^{\dagger}$ $\downarrow$ & "
        r"E2E upd./s $\uparrow$ & Pure upd./s $\uparrow$ & Interact. ms/step $\downarrow$ & "
        r"Update ms/update $\downarrow$ \\",
        r"\midrule",
    ]
    for domain in present_domains:
        for method in METHODS:
            profile = by_key.get((method, domain))
            if profile is None:
                continue
            e2e = profile.data["end_to_end"]["aggregate"]
            var = e2e["window_variability"]
            pure = profile.data["pure_replay_update"]["aggregate"]
            pure_var = pure["window_variability"]
            cells = [
                METHOD_LABELS[method],
                DOMAIN_LABELS[domain],
                _latex_pm(e2e["environment_frames_per_second"],
                          var["environment_frames_per_second"]["se"]),
                _latex_pm(e2e["hours_per_1m_environment_frames"],
                          var["hours_per_1m_environment_frames"]["se"]),
                _latex_pm(e2e["gradient_updates_per_end_to_end_second"],
                          var["gradient_updates_per_end_to_end_second"]["se"]),
                _latex_pm(pure["gradient_updates_per_second"],
                          pure_var["gradient_updates_per_second"]["se"]),
                _latex_pm(e2e["average_interaction_ms_per_agent_step"],
                          var["average_interaction_ms_per_agent_step"]["se"]),
                _latex_pm(e2e["average_update_ms_per_gradient_update"],
                          var["average_update_ms_per_gradient_update"]["se"]),
            ]
            lines.append(" & ".join(cells) + r" \\")
        if domain != present_domains[-1]:
            lines.append(r"\addlinespace")
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{0.5em}",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Method & Domain & Torch alloc. GiB & Torch reserv. GiB & "
        r"NVML process GiB & NVML device GiB \\",
        r"\midrule",
    ])
    for domain in present_domains:
        for method in METHODS:
            profile = by_key.get((method, domain))
            if profile is None:
                continue
            e2e = profile.data["end_to_end"]
            torch_memory = e2e["torch_cuda_memory"]
            nvml = e2e["nvml_device_memory"]
            process_peak = _optional(nvml, "process_memory.peak_used.gib", None)
            process_cell = "--" if process_peak is None else f"{float(process_peak):.2f}"
            lines.append(
                " & ".join([
                    METHOD_LABELS[method],
                    DOMAIN_LABELS[domain],
                    f"{torch_memory['max_memory_allocated']['gib']:.2f}",
                    f"{torch_memory['max_memory_reserved']['gib']:.2f}",
                    process_cell,
                    f"{nvml['peak_used']['gib']:.2f}",
                ]) + r" \\"
            )
        if domain != present_domains[-1]:
            lines.append(r"\addlinespace")
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{0.5em}",
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"Method & Domain & Online total M$^{\ddagger}$ & Trainable M & Train. visual M & "
        r"Target copies M & Process resident M & DINO copies \\",
        r"\midrule",
    ])
    for domain in present_domains:
        for method in METHODS:
            profile = by_key.get((method, domain))
            if profile is None:
                continue
            params = profile.data["parameter_counts"]
            online = params["online_model_parameter_count_excluding_target_copies"]
            target = params["total_parameter_count"] - online
            lines.append(
                " & ".join([
                    METHOD_LABELS[method],
                    DOMAIN_LABELS[domain],
                    f"{online / 1e6:.3f}",
                    f"{params['total_trainable_parameter_count'] / 1e6:.3f}",
                    f"{params['trainable_visual_front_end_parameter_count'] / 1e6:.3f}",
                    f"{target / 1e6:.3f}",
                    f"{params['process_resident_parameter_count'] / 1e6:.3f}",
                    str(params["environment_backbone_copy_count"]),
                ]) + r" \\"
            )
        if domain != present_domains[-1]:
            lines.append(r"\addlinespace")
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{minipage}{0.98\linewidth}",
        r"\footnotesize",
        r"$^{\dagger}$Controlled-window extrapolation; evaluation, checkpoint serialization, "
        r"external logging, videos, and diagnostic-only statistics are excluded. Historical 2M "
        r"timestamps are ineligible for the primary comparison because their GPU/CPU concurrency "
        r"was uncontrolled and their timers include different work. \\",
        r"$^{\ddagger}$The headline total is the online model with one logical visual backbone and "
        r"without target copies. Target-copy parameters and actual process-resident parameters are "
        r"reported separately. SWM's process count includes the otherwise idle evaluation-environment "
        r"DINO copy instantiated by the training workspace. Peak memory is measured after warm-up; "
        r"NVML device memory is whole-device usage, while NVML process memory is PID-scoped when available.",
        r"\end{minipage}",
        r"\end{table*}",
        "",
    ])
    return "\n".join(lines)


def _interpretation(profiles: Sequence[Profile], development: bool,
                    validation: Mapping[str, Any]) -> str:
    by_key = {profile.key: profile for profile in profiles}
    lines = ["# Controlled profiling interpretation", ""]
    if development:
        lines.extend([
            "> **DEVELOPMENT ONLY — shortened smoke profiles; do not cite these numbers.**",
            "",
            "This output only verifies the aggregation/validation pipeline. It is not a formal runtime "
            "comparison because at least one input uses a shortened warm-up or measurement window and "
            "the six method/domain combinations are incomplete.",
            "",
        ])
    else:
        lines.extend([
            "All six inputs passed the formal checks: one run for each method/domain pair, the same "
            f"{validation['gpu_name']} (UUID `{validation['gpu_uuid']}`, PCI "
            f"`{validation['gpu_pci_bus_id']}`), CPU affinity `{validation['cpu_affinity']}`, fixed "
            "seed 2009, and identical common FB/actor/replay/optimizer settings.",
            "",
        ])

    complete_domains = [
        domain for domain in DOMAINS
        if ("swm_cls", domain) in by_key and ("cnn_fb", domain) in by_key
    ]
    if complete_domains:
        lines.extend(["## Measured outcomes", ""])
    for domain in complete_domains:
        swm_profile = by_key[("swm_cls", domain)]
        cnn_profile = by_key[("cnn_fb", domain)]
        swm_e2e = swm_profile.data["end_to_end"]["aggregate"]
        cnn_e2e = cnn_profile.data["end_to_end"]["aggregate"]
        swm_pure = swm_profile.data["pure_replay_update"]["aggregate"]
        cnn_pure = cnn_profile.data["pure_replay_update"]["aggregate"]
        swm_mem = swm_profile.data["end_to_end"]["torch_cuda_memory"]
        cnn_mem = cnn_profile.data["end_to_end"]["torch_cuda_memory"]
        swm_nvml = swm_profile.data["end_to_end"]["nvml_device_memory"]
        cnn_nvml = cnn_profile.data["end_to_end"]["nvml_device_memory"]
        lines.extend([
            f"- **{DOMAIN_LABELS[domain]}.** SWM-CLS: "
            f"{swm_e2e['environment_frames_per_second']:.2f} env FPS, "
            f"{swm_e2e['hours_per_1m_environment_frames']:.2f} extrapolated h/1M, "
            f"{swm_pure['gradient_updates_per_second']:.2f} pure updates/s, "
            f"{swm_mem['max_memory_allocated']['gib']:.2f} GiB Torch peak allocated, and "
            f"{swm_nvml['peak_used']['gib']:.2f} GiB NVML device peak. CNN-FB: "
            f"{cnn_e2e['environment_frames_per_second']:.2f} env FPS, "
            f"{cnn_e2e['hours_per_1m_environment_frames']:.2f} extrapolated h/1M, "
            f"{cnn_pure['gradient_updates_per_second']:.2f} pure updates/s, "
            f"{cnn_mem['max_memory_allocated']['gib']:.2f} GiB Torch peak allocated, and "
            f"{cnn_nvml['peak_used']['gib']:.2f} GiB NVML device peak.",
        ])
        faster = "SWM-CLS" if swm_e2e["environment_frames_per_second"] > cnn_e2e[
            "environment_frames_per_second"
        ] else "CNN-FB"
        fast = max(swm_e2e["environment_frames_per_second"], cnn_e2e[
            "environment_frames_per_second"
        ])
        slow = min(swm_e2e["environment_frames_per_second"], cnn_e2e[
            "environment_frames_per_second"
        ])
        lines.append(
            f"  In end-to-end throughput, {faster} was {fast / slow:.2f}x faster in this domain."
        )
    if complete_domains:
        lines.append("")

    lines.extend(["## Parameter accounting", ""])
    for domain in DOMAINS:
        for method in METHODS:
            profile = by_key.get((method, domain))
            if profile is None:
                continue
            params = profile.data["parameter_counts"]
            online = params["online_model_parameter_count_excluding_target_copies"]
            target = params["total_parameter_count"] - online
            lines.append(
                f"- **{METHOD_LABELS[method]}, {DOMAIN_LABELS[domain]}:** {online / 1e6:.3f}M online "
                f"parameters (the paper's headline total), "
                f"{params['total_trainable_parameter_count'] / 1e6:.3f}M trainable, "
                f"{params['trainable_visual_front_end_parameter_count'] / 1e6:.3f}M trainable "
                f"visual-front-end, {target / 1e6:.3f}M target-copy parameters, and "
                f"{params['process_resident_parameter_count'] / 1e6:.3f}M parameters instantiated "
                "in the profiling process."
            )
    lines.extend([
        "",
        "The standard total deliberately excludes target copies and counts one logical visual backbone. "
        "The process-resident count is an implementation-footprint diagnostic: for SWM-CLS it includes "
        "both the training-environment DINO and the otherwise idle evaluation-environment DINO created "
        "by the unchanged training workspace.",
        "",
        "## Scope and interpretation constraints",
        "",
        "- The end-to-end windows include environment interaction/rendering, visual preprocessing and "
        "encoder execution, action selection, replay insertion/sampling, FB updates, actor updates, and "
        "target updates. DINO preprocessing/forward is therefore included in SWM-CLS interaction time.",
        "- The pure replay-update benchmark starts from populated replay. SWM replay already contains the "
        "online DINO embeddings produced during interaction, so this section intentionally measures the "
        "projector/FB/actor optimization path rather than another DINO forward. CNN-FB still performs its "
        "configured RandomShiftsAug and trainable CNN encoding inside each update.",
        "- Periodic evaluation, checkpoint serialization, external logging, videos, and diagnostic-only "
        "statistics are excluded from the primary throughput windows, as requested.",
        "- Every reported h/1M value is a controlled-window extrapolation, not an observed one-million-frame "
        "wall-clock duration. The historical completed 2M-run timestamps are ineligible as primary evidence "
        "because those jobs used uncontrolled and unequal GPU/CPU concurrency and their timers include "
        "evaluation/checkpoint work.",
        "- Torch allocated/reserved peaks and both PID-scoped and whole-device NVML peaks are retained in the "
        "CSV. Whole-device NVML values include the baseline device allocation; PID-scoped values are the more "
        "specific process-footprint diagnostic when NVML exposes them.",
        "- This report contains no interaction sample-efficiency scores and makes no inference about them.",
        "- Runtime/memory efficiency is distinct from robustness and task performance. If the 224x224 DINO "
        "forward raises wall-clock cost in a formal result, that cost must be stated directly rather than "
        "conflated with reduced task-specific visual optimization or visual-shift robustness.",
        "",
    ])
    return "\n".join(lines)


def _write_outputs(output_dir: Path, profiles: Sequence[Profile], development: bool,
                   validation: Mapping[str, Any]) -> Mapping[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "csv": output_dir / "controlled_profile_results.csv",
        "latex": output_dir / "controlled_runtime_params_table.tex",
        "interpretation": output_dir / "controlled_profile_interpretation.md",
        "validation": output_dir / "controlled_profile_validation.json",
    }
    rows = _csv_rows(profiles)
    _write_csv(paths["csv"], rows)
    paths["latex"].write_text(_latex_table(profiles, development), encoding="utf-8")
    paths["interpretation"].write_text(
        _interpretation(profiles, development, validation), encoding="utf-8"
    )
    manifest = {
        "schema_version": 1,
        "status": "development_smoke_only" if development else "valid_formal_comparison",
        "formal_results_ready": not development,
        "profile_count": len(profiles),
        "input_profiles": [str(profile.path) for profile in profiles],
        "validation": validation,
        "hours_per_1m_is_controlled_window_extrapolation": True,
        "historical_full_run_timing_primary_eligible": False,
        "historical_ineligibility_reason": (
            "uncontrolled/unequal GPU and CPU concurrency; historical timers include evaluation and "
            "checkpoint work"
        ),
        "csv_row_count": len(rows),
    }
    with paths["validation"].open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return paths


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-dir", type=Path, default=DEFAULT_PROFILE_DIR)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--allow-smoke",
        action="store_true",
        help="accept shortened/incomplete profiles for development-only pipeline testing",
    )
    args = parser.parse_args(argv)
    args.profile_dir = args.profile_dir.resolve()
    if args.output_dir is None:
        if args.allow_smoke:
            parser.error("--allow-smoke requires an explicit --output-dir to protect formal artifacts")
        args.output_dir = DEFAULT_REPORT_ROOT
    args.output_dir = args.output_dir.resolve()
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    try:
        candidates = _load_candidates(args.profile_dir)
        profiles = _select_profiles(candidates, allow_smoke=args.allow_smoke)
        selected_coverage = {profile.key for profile in profiles}
        development = (
            any(profile.data.get("valid_formal_profile") is not True for profile in profiles)
            or selected_coverage != EXPECTED_COVERAGE
        )
        if not args.allow_smoke and development:
            _fail("formal mode selected a non-formal profile (internal selection error)")
        validation = _cross_validate(profiles, allow_smoke=args.allow_smoke)
        paths = _write_outputs(args.output_dir, profiles, development, validation)
    except ValidationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({
        "status": "development_smoke_only" if development else "valid_formal_comparison",
        "profiles": len(profiles),
        "outputs": {name: str(path) for name, path in paths.items()},
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
