#!/usr/bin/env python3
"""Read-only optimizer/topology preflight for the 24 Walker/Quadruped runs.

``audit_agent`` also runs against each real, freshly initialized GPU agent. It
does not sample, forward, backpropagate, step, or change the random-number state.
The CLI constructs CPU agents with the real network and observation dimensions;
it never creates an environment, W&B run, or training process.
"""

from __future__ import annotations

import argparse
from collections import Counter
import gc
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import torch
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TASKS = (
    "walker_stand", "walker_walk", "walker_run", "walker_flip",
    "quadruped_stand", "quadruped_walk", "quadruped_run", "quadruped_jump",
)
GROUPS = {(False, 1e-5), (True, 5e-5), (True, 1e-5)}
LR_FIELDS = ("lr", "fb_lr", "lr_f", "lr_b", "lr_actor")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _ids(module: nn.Module) -> set[int]:
    return {id(parameter) for parameter in module.parameters()}


def _state_hash(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in module.state_dict().items():
        digest.update(name.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _identical_distinct(left: nn.Module, right: nn.Module, label: str) -> None:
    left_state, right_state = left.state_dict(), right.state_dict()
    _require(left_state.keys() == right_state.keys(), f"{label}: state keys differ")
    for name in left_state:
        _require(torch.equal(left_state[name], right_state[name]),
                 f"{label}: initial values differ at {name}")
        _require(left_state[name].data_ptr() != right_state[name].data_ptr(),
                 f"{label}: shared storage at {name}")
    _require(_ids(left).isdisjoint(_ids(right)), f"{label}: shared parameter objects")


def audit_agent(agent: Any, expected_lr: float, separated: bool) -> dict[str, Any]:
    """Assert actual optimizer ownership, LR, and fresh DINO CLS3 topology."""
    expected_lr = float(expected_lr)
    _require(math.isfinite(expected_lr) and expected_lr > 0, "invalid expected LR")
    _require(type(separated) is bool, "separated must be a bool")
    cfg = agent.cfg
    for field in LR_FIELDS:
        _require(getattr(cfg, field) is not None
                 and float(getattr(cfg, field)) == expected_lr,
                 f"agent.{field} must explicitly equal {expected_lr}")
    _require(float(cfg.lr_coef) == 1.0, "lr_coef must equal 1")
    _require(cfg.obs_type == "dino" and cfg.use_cls and cfg.dino_frame_stack == 3,
             "expected DINO CLS3 observations")
    _require(tuple(cfg.obs_shape) == (2304,), "expected 3 x 768 DINOv2 CLS input")
    _require(cfg.goal_space is None, "privileged goal space must be disabled")
    _require(cfg.dino_separate_fb_adapters is separated, "separation config mismatch")
    _require(not cfg.dino_separate_backward_adapter and not cfg.dino_flare_b
             and not cfg.pixel_separate_fb_encoders,
             "legacy target adapter / FLARE / pixel separation must be disabled")
    _require(cfg.dino_use_adapter and cfg.update_encoder, "adapter must be trainable")
    _require(cfg.dino_adapter_type == "linear", "expected ordinary LN + Linear adapter")
    _require(float(cfg.idm_coef) == 0.0 and cfg.idm_route == "none",
             "IDM must be disabled")
    for name in ("idm_head", "idm_optimizer", "flare_b_encoder", "flare_b_optimizer",
                 "backward_encoder_target", "encoder_target", "forward_adapter_target",
                 "backward_adapter_target", "encoder_scheduler"):
        _require(getattr(agent, name, None) is None, f"unexpected active {name}")

    adapter = agent.forward_adapter
    _require(adapter is agent.encoder, "F/actor must share the online F adapter")
    _require(isinstance(adapter, nn.Sequential) and len(adapter) == 2
             and isinstance(adapter[0], nn.LayerNorm)
             and isinstance(adapter[1], nn.Linear), "unexpected adapter architecture")
    _require(tuple(adapter[0].normalized_shape) == (2304,)
             and adapter[1].in_features == 2304
             and adapter[1].out_features == cfg.dino_adapter_output_dim,
             "adapter input/output dimensions differ from config")

    modules = {"F": agent.forward_net, "B": agent.backward_net,
               "actor": agent.actor, "forward_adapter": adapter}
    if separated:
        _require(agent.backward_adapter is not None
                 and agent.backward_adapter is agent.backward_encoder,
                 "missing separate online B adapter")
        _identical_distinct(adapter, agent.backward_adapter, "F/B adapters")
        modules["backward_adapter"] = agent.backward_adapter
        expected_owners = {
            "forward_fb_opt": {"F", "forward_adapter"},
            "backward_fb_opt": {"B", "backward_adapter"},
            "actor_opt": {"actor"},
        }
    else:
        _require(agent.backward_adapter is None and agent.backward_encoder is None,
                 "shared mode unexpectedly has a separate B adapter")
        expected_owners = {
            "fb_opt": {"F", "B"}, "encoder_opt": {"forward_adapter"},
            "actor_opt": {"actor"},
        }

    module_ids = {name: _ids(module) for name, module in modules.items()}
    flattened = [parameter_id for ids in module_ids.values() for parameter_id in ids]
    _require(len(flattened) == len(set(flattened)), "online modules share parameters")
    online_ids = set(flattened)
    for name, module in modules.items():
        _require(all(p.requires_grad for p in module.parameters()), f"{name} has frozen parameters")

    target_modules = {name: value for name, value in vars(agent).items()
                      if isinstance(value, nn.Module) and "target" in name.lower()}
    _require(set(target_modules) == {"forward_target_net", "backward_target_net"},
             f"unexpected target topology: {sorted(target_modules)}")
    _identical_distinct(agent.forward_net, agent.forward_target_net, "F/target F")
    _identical_distinct(agent.backward_net, agent.backward_target_net, "B/target B")
    target_ids = set().union(*(_ids(module) for module in target_modules.values()))
    _require(online_ids.isdisjoint(target_ids), "target shares online parameter objects")
    all_modules = [value for value in vars(agent).values() if isinstance(value, nn.Module)]
    all_module_ids = set().union(*(_ids(module) for module in all_modules))
    _require(all_module_ids == online_ids | target_ids, "unexpected extra parameterized module")

    optimizers = {name: value for name, value in vars(agent).items()
                  if isinstance(value, torch.optim.Optimizer)}
    _require(set(optimizers) == set(expected_owners),
             f"unexpected optimizers: {sorted(optimizers)}")
    ownership: Counter[int] = Counter()
    optimizer_report = {}
    for name, optimizer in optimizers.items():
        _require(len(optimizer.state) == 0, f"{name} is not freshly initialized")
        actual = []
        groups = []
        for index, group in enumerate(optimizer.param_groups):
            _require(float(group["lr"]) == expected_lr,
                     f"{name} group {index}: LR {group['lr']} != {expected_lr}")
            params = list(group["params"])
            _require(bool(params), f"{name} group {index} is empty")
            ids = [id(parameter) for parameter in params]
            actual.extend(ids)
            ownership.update(ids)
            groups.append({"index": index, "lr": float(group["lr"]),
                           "parameter_tensors": len(params),
                           "parameter_elements": sum(p.numel() for p in params),
                           "modules": sorted(module_name for module_name, values in module_ids.items()
                                             if values.intersection(ids))})
        expected = set().union(*(module_ids[module] for module in expected_owners[name]))
        _require(set(actual) == expected, f"{name} has incorrect parameter membership")
        _require(len(actual) == len(set(actual)), f"{name} duplicates parameters")
        optimizer_report[name] = groups
    _require(set(ownership) == online_ids and all(count == 1 for count in ownership.values()),
             "every online parameter must belong to exactly one optimizer")
    _require(set(ownership).isdisjoint(target_ids), "target parameter present in optimizer")

    adapter_hashes = {"forward": _state_hash(adapter)}
    if separated:
        adapter_hashes["backward"] = _state_hash(agent.backward_adapter)
    return {
        "status": "PASS", "expected_lr": expected_lr, "separated": separated,
        "device": str(next(adapter.parameters()).device),
        "obs_shape": list(cfg.obs_shape), "action_shape": list(cfg.action_shape),
        "configured_lrs": {field: float(getattr(cfg, field)) for field in LR_FIELDS},
        "optimizers": optimizer_report, "adapter_initial_sha256": adapter_hashes,
        "adapter_initial_values_equal": True if separated else None,
        "adapter_storage_independent": True if separated else None,
        "all_online_parameters_owned_exactly_once": True,
        "all_adapters_trainable": True,
        "targets": sorted(target_modules), "no_target_adapter_idm_or_flare": True,
        "module_parameter_elements": {
            name: sum(parameter.numel() for parameter in module.parameters())
            for name, module in modules.items()
        },
    }


def preflight_manifest(manifest_path: Path, output_path: Path) -> dict[str, Any]:
    """Compose every requested launch config and audit full-size CPU optimizers."""
    import hydra
    import numpy as np
    from dm_env import specs
    from omegaconf import OmegaConf
    from url_benchmark import pretrain  # registers the production Hydra schema

    manifest = json.loads(manifest_path.read_text())
    runs = manifest["runs"]
    _require(len(runs) == 24, f"expected 24 runs, got {len(runs)}")
    keys = [(run["task"], run["separated"], float(run["lr"])) for run in runs]
    expected = {(task, separated, lr) for task in TASKS for separated, lr in GROUPS}
    _require(len(set(keys)) == 24 and set(keys) == expected, "missing, duplicate, or unexpected run")
    _require(len({run["run_dir"] for run in runs}) == 24, "duplicate run directories")
    report = {"status": "RUNNING", "manifest": str(manifest_path), "runs": []}
    torch.set_num_threads(1)
    with hydra.initialize_config_dir(config_dir=str(REPO_ROOT / "url_benchmark"), version_base="1.1"):
        for index, run in enumerate(runs):
            cfg = hydra.compose(config_name="base_config", overrides=run["overrides"])
            _require(cfg.task == run["task"], "manifest/config task mismatch")
            _require(cfg.dino_model_name == "facebook/dinov2-base", "expected frozen DINOv2 base")
            _require(not cfg.auto_resume and cfg.load_model is None and cfg.load_replay_buffer is None,
                     "training must start fresh")
            _require(not cfg.append_goal_to_observation and cfg.goal_space is None,
                     "privileged observation/goal must be disabled")
            _require(cfg.use_wandb, "W&B must be enabled in launch config")
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
            audit = audit_agent(agent, float(run["lr"]), run["separated"])
            report["runs"].append({"task": run["task"], "group": run["group"],
                                   "run_dir": run["run_dir"], "resolved_config": resolved,
                                   "audit": audit})
            print(f"[{index + 1}/24] PASS {run['task']} {run['group']} lr={run['lr']}", flush=True)
            del agent, cfg
            gc.collect()
    report["status"] = "PASS"
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    manifest_path = args.manifest.resolve()
    output_path = args.output or manifest_path.with_name("preflight.json")
    preflight_manifest(manifest_path, output_path)
    print(f"PASS: all 24 full-size CPU agents audited; {output_path}", flush=True)


if __name__ == "__main__":
    main()
