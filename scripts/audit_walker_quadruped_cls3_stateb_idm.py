#!/usr/bin/env python3
"""Audit CLS3/state-B IDM controls without changing the training algorithm.

``audit_agent`` is read-only and safe before a live agent's first training step.
Only ``preflight_manifest`` performs synthetic forward/backward/update checks;
those checks use independent, disposable CPU agents, never live training state.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
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
GROUPS = {"idm0": 0.0, "idm0p1": 0.1}
COMMON_MODULES = (
    "encoder", "actor", "forward_net", "backward_net",
    "forward_target_net", "backward_target_net",
)


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


def audit_agent(agent: Any, expected_coef: float) -> dict[str, Any]:
    """Assert a fresh production agent's topology/optimizer state; no execution."""
    expected_coef = float(expected_coef)
    _require(expected_coef in GROUPS.values(), "expected coefficient must be 0 or 0.1")
    cfg = agent.cfg
    _require(cfg.obs_type == "dino" and cfg.use_cls and cfg.dino_frame_stack == 3,
             "expected DINO three-frame CLS input")
    _require(tuple(cfg.obs_shape) == (2304,), "observation must contain 3 x 768 CLS values")
    _require(cfg.goal_space in {"simplified_walker", "simplified_quadruped"},
             "B must use the task's state-derived goal space")
    goal_dim = 3 if cfg.goal_space == "simplified_walker" else 2
    action_dim = 6 if goal_dim == 3 else 12
    _require(tuple(cfg.action_shape) == (action_dim,), "goal/action domain mismatch")
    _require(cfg.dino_use_adapter and cfg.update_encoder, "visual adapter must train")
    _require(cfg.dino_adapter_type == "linear" and cfg.dino_adapter_output_dim == 512,
             "expected linear DINO adapter with 512 outputs")
    _require(not cfg.dino_separate_fb_adapters and not cfg.dino_separate_backward_adapter
             and not cfg.pixel_separate_fb_encoders and not cfg.dino_flare_b,
             "state-B must not enable visual-B separation or FLARE")
    _require(float(cfg.idm_coef) == expected_coef and cfg.idm_encoder_mode == "legacy"
             and cfg.idm_route == "none", "incorrect legacy IDM setting")
    _require(cfg.idm_encoder_burnin_steps == 0 and cfg.idm_encoder_ramp_steps == 0
             and cfg.idm_grad_ratio_target is None, "unexpected IDM schedule/balancing")
    for field, value in {"lr": 1e-4, "lr_coef": 1.0, "batch_size": 1024,
                         "z_dim": 50, "ortho_coef": 1.0,
                         "update_every_steps": 2, "num_inference_steps": 5120}.items():
        _require(getattr(cfg, field) == value, f"incorrect agent.{field}")
    _require(not cfg.q_loss and cfg.nstep == 1, "Q-loss must be disabled and replay one-step")
    for name in ("backward_encoder", "backward_adapter", "backward_encoder_target",
                 "flare_b_encoder", "flare_b_optimizer", "encoder_scheduler",
                 "forward_fb_opt", "backward_fb_opt", "backward_encoder_opt"):
        _require(getattr(agent, name, None) is None, f"unexpected active {name}")
    adapter = agent.encoder
    _require(agent.forward_adapter is adapter, "F/actor must share this visual adapter")
    _require(isinstance(adapter, nn.Sequential) and len(adapter) == 2
             and isinstance(adapter[0], nn.LayerNorm) and isinstance(adapter[1], nn.Linear),
             "expected LayerNorm + Linear adapter")
    _require(tuple(adapter[0].normalized_shape) == (2304,)
             and adapter[1].in_features == 2304 and adapter[1].out_features == 512,
             "incorrect adapter dimensions")
    _require(agent.obs_dim == 512 and agent.actor.obs_dim == 512
             and agent.forward_net.obs_dim == 512, "F/actor must consume adapter output")
    _require(agent.backward_net.obs_dim == goal_dim
             and agent.backward_target_net.obs_dim == goal_dim,
             "B/target B must consume low-dimensional state goals")

    modules = {name: getattr(agent, name) for name in COMMON_MODULES[:4]}
    owners = {"encoder_opt": {"encoder"}, "actor_opt": {"actor"},
              "fb_opt": {"forward_net", "backward_net"}}
    if expected_coef:
        _require(isinstance(agent.idm_head, nn.Linear)
                 and agent.idm_head.in_features == 1024
                 and agent.idm_head.out_features == action_dim,
                 "IDM must map concatenated two 512-d adapter outputs to action")
        modules["idm_head"] = agent.idm_head
        owners["idm_optimizer"] = {"idm_head"}
    else:
        _require(agent.idm_head is None and agent.idm_optimizer is None,
                 "disabled IDM must instantiate no auxiliary head/optimizer")
    module_ids = {name: _ids(module) for name, module in modules.items()}
    all_online = [item for ids in module_ids.values() for item in ids]
    _require(len(all_online) == len(set(all_online)), "online modules share parameters")
    online_ids = set(all_online)
    _require(all(p.requires_grad for module in modules.values() for p in module.parameters()),
             "online module unexpectedly frozen")
    targets = {name: value for name, value in vars(agent).items()
               if isinstance(value, nn.Module) and "target" in name.lower()}
    _require(set(targets) == {"forward_target_net", "backward_target_net"},
             "unexpected target/EMA adapter")
    for online_name, target_name in (("forward_net", "forward_target_net"),
                                     ("backward_net", "backward_target_net")):
        left, right = getattr(agent, online_name), getattr(agent, target_name)
        _require(_ids(left).isdisjoint(_ids(right)), "target shares online parameter storage")
        left_state, right_state = left.state_dict(), right.state_dict()
        _require(left_state.keys() == right_state.keys(), "target state keys differ")
        for name, value in left_state.items():
            _require(torch.equal(value, right_state[name])
                     and value.data_ptr() != right_state[name].data_ptr(),
                     f"fresh target state mismatch/shared storage at {name}")
    target_ids = set().union(*(_ids(module) for module in targets.values()))
    all_ids = set().union(*(_ids(module) for module in vars(agent).values()
                           if isinstance(module, nn.Module)))
    _require(all_ids == online_ids | target_ids, "unexpected parameterized module")
    optimizers = {name: value for name, value in vars(agent).items()
                  if isinstance(value, torch.optim.Optimizer)}
    _require(set(optimizers) == set(owners), "unexpected optimizer topology")
    ownership: Counter[int] = Counter()
    optimizer_report = {}
    for name, optimizer in optimizers.items():
        _require(not optimizer.state, f"{name} contains non-fresh optimizer state")
        actual, groups = [], []
        for index, group in enumerate(optimizer.param_groups):
            _require(float(group["lr"]) == 1e-4, f"{name}[{index}] LR must be 1e-4")
            parameters = list(group["params"])
            _require(bool(parameters), f"{name}[{index}] has no parameters")
            ids = [id(p) for p in parameters]
            actual.extend(ids)
            ownership.update(ids)
            groups.append({"index": index, "lr": float(group["lr"]),
                           "parameter_tensors": len(parameters),
                           "parameter_elements": sum(p.numel() for p in parameters),
                           "modules": sorted(n for n, values in module_ids.items()
                                             if values.intersection(ids))})
        expected = set().union(*(module_ids[n] for n in owners[name]))
        _require(set(actual) == expected and len(actual) == len(set(actual)),
                 f"{name} has incorrect/duplicate parameter ownership")
        optimizer_report[name] = groups
    _require(set(ownership) == online_ids and all(n == 1 for n in ownership.values()),
             "all trainable online parameters must belong to exactly one optimizer")
    _require(set(ownership).isdisjoint(target_ids), "targets must not belong to optimizers")
    _require(getattr(agent, "_idm_update_count", 0) == 0, "IDM update count is not fresh")
    return {"status": "PASS", "expected_coef": expected_coef, "obs_shape": [2304],
            "goal_space": cfg.goal_space, "goal_dim": goal_dim,
            "action_shape": [action_dim], "adapter_output_dim": 512,
            "idm_encoder_mode": "legacy", "idm_route": "none",
            "optimizers": optimizer_report, "targets": sorted(targets),
            "initial_sha256": {name: _state_hash(getattr(agent, name))
                               for name in COMMON_MODULES},
            "all_online_parameters_owned_exactly_once": True,
            "read_only_no_forward_backward_or_optimizer_step": True}


def _scratch_gradient_audit(agent: Any) -> dict[str, Any]:
    """Exercise production paths only on an expendable preflight CPU agent."""
    from url_benchmark.replay_buffer import EpisodeBatch
    _require(str(next(agent.encoder.parameters()).device) == "cpu", "scratch audit is CPU only")
    generator = torch.Generator().manual_seed(9182)
    action_dim, goal_dim = agent.action_dim, agent.backward_net.obs_dim
    result = {"scope": "disposable CPU agent only", "idm_only_gradient_norms": None}
    if agent.idm_head is not None:
        raw = torch.randn(8, 2304, generator=generator)
        raw_next = torch.randn(8, 2304, generator=generator)
        action = torch.randn(8, action_dim, generator=generator).tanh()
        loss, _, coefficient = agent._compute_idm_objective(
            agent.aug_and_encode(raw), agent.aug_and_encode(raw_next), action, 0)
        _require(coefficient == 0.1, "unexpected effective legacy coefficient")
        (agent.cfg.idm_coef * loss).backward()
        gradient_norms = {}
        for name in COMMON_MODULES + ("idm_head",):
            module = getattr(agent, name)
            norm = math.sqrt(sum(float(p.grad.detach().square().sum())
                                 for p in module.parameters() if p.grad is not None))
            gradient_norms[name] = norm
            _require((norm > 0) if name in {"encoder", "idm_head"} else norm == 0,
                     f"IDM-only gradient reached incorrect module: {name}: {norm}")
            module.zero_grad(set_to_none=True)
        result["idm_only_gradient_norms"] = gradient_norms
    count = agent.cfg.batch_size
    batch = EpisodeBatch(
        obs=torch.randn(count, 2304, generator=generator),
        next_obs=torch.randn(count, 2304, generator=generator),
        action=torch.randn(count, action_dim, generator=generator).tanh(),
        reward=torch.randn(count, 1, generator=generator),
        discount=torch.full((count, 1), 0.99),
        goal=torch.randn(count, goal_dim, generator=generator),
        next_goal=torch.randn(count, goal_dim, generator=generator),
        future_goal=torch.randn(count, goal_dim, generator=generator),
    )
    captured = {"B": [], "target_B": []}
    def capture(label: str):
        def hook(module: nn.Module, inputs: tuple) -> None:
            value = inputs[0]
            _require(value.ndim == 2 and value.shape[-1] == goal_dim,
                     f"{label} received visual features instead of state goal")
            captured[label].append(value.detach().clone())
        return hook
    handles = [agent.backward_net.register_forward_pre_hook(capture("B")),
               agent.backward_target_net.register_forward_pre_hook(capture("target_B"))]
    original_update_fb = agent.update_fb
    def checked_update_fb(**kwargs: Any) -> dict[str, float]:
        _require(torch.equal(kwargs["next_goal"], batch.next_goal)
                 and torch.equal(kwargs["target_next_goal"], batch.next_goal),
                 "FB received a goal other than replay.next_goal")
        _require(tuple(kwargs["obs"].shape) == (count, 512)
                 and tuple(kwargs["next_obs"].shape) == (count, 512),
                 "F/actor did not receive visual adapter output")
        _require(kwargs["idm_obs"] is None and kwargs["idm_next_obs"] is None,
                 "an explicit IDM route was activated")
        return original_update_fb(**kwargs)
    agent.update_fb = checked_update_fb
    class Replay:
        def sample(self, batch_size: int) -> Any:
            _require(batch_size == count, "unexpected synthetic replay batch size")
            return batch
    try:
        metrics = agent.update(Replay(), step=0)
    finally:
        agent.update_fb = original_update_fb
        for handle in handles:
            handle.remove()
    for label in captured:
        _require(captured[label] and torch.equal(captured[label][-1], batch.next_goal),
                 f"actual {label} network did not receive replay.next_goal")
    if agent.idm_head is not None:
        _require(metrics["idm_loss"] > 0 and math.isclose(
            metrics["total_loss"], metrics["fb_loss"] + 0.1 * metrics["idm_loss"],
            rel_tol=2e-6, abs_tol=2e-6), "legacy total objective differs from FB + 0.1 IDM")
    result.update({"production_update_executed": True, "batch_size": count,
                   "F_adapter_feature_shape": [count, 512],
                   "B_actual_input_shape": [count, goal_dim],
                   "B_and_target_B_received_exact_replay_next_goal": True,
                   "explicit_idm_route_inputs_absent": True,
                   "legacy_loss_formula_verified": bool(agent.idm_head is not None)})
    return result


def preflight_manifest(manifest_path: Path, output_path: Path) -> dict[str, Any]:
    """Compose/audit all 16 full-size CPU agents and compare each matched pair."""
    import hydra
    import numpy as np
    from dm_env import specs
    from omegaconf import OmegaConf
    from url_benchmark import pretrain
    manifest_path, output_path = Path(manifest_path), Path(output_path)
    manifest = json.loads(manifest_path.read_text())
    runs = manifest["runs"]
    keys = [(run["task"], run["group"]) for run in runs]
    _require(len(keys) == 16 and len(set(keys)) == 16
             and set(keys) == {(task, group) for task in TASKS for group in GROUPS},
             "expected precisely 16 unique task/group controls")
    for field in ("run_id", "run_dir", "checkpoint_dir"):
        _require(len({run[field] for run in runs}) == 16, f"duplicate {field}")
    torch.set_num_threads(1)
    report = {"status": "RUNNING", "manifest": str(manifest_path), "runs": [],
              "pairs": {}, "torch_num_threads": torch.get_num_threads()}
    pairs = {}
    with hydra.initialize_config_dir(config_dir=str(REPO_ROOT / "url_benchmark"), version_base="1.1"):
        for index, run in enumerate(runs):
            cfg = hydra.compose(config_name="base_config", overrides=run["overrides"])
            expected_goal = "simplified_" + run["task"].split("_")[0]
            expected_coef = GROUPS[run["group"]]
            _require(cfg.task == run["task"] and cfg.seed == run["seed"] == 1,
                     "task/seed differs from manifest")
            _require(cfg.goal_space == cfg.agent.goal_space == expected_goal
                     and run.get("goal_space", expected_goal) == expected_goal,
                     "root and agent state-goal spaces must agree")
            _require(float(run["idm_coef"]) == expected_coef, "manifest IDM coefficient differs")
            for field, value in {"obs_type": "dino", "use_cls": True, "dino_frame_stack": 3,
                                 "dino_model_name": "facebook/dinov2-base", "action_repeat": 2,
                                 "discount": 0.99, "reward_free": True,
                                 "num_train_frames": 2000010, "eval_every_frames": 10000,
                                 "num_eval_episodes": 10, "checkpoint_every": 100000,
                                 "update_encoder": True, "append_goal_to_observation": False,
                                 "use_wandb": True}.items():
                _require(getattr(cfg, field) == value, f"incorrect root config {field}")
            _require(not cfg.auto_resume and cfg.load_model is None
                     and cfg.load_replay_buffer is None, "training must start entirely fresh")
            pretrain._validate_idm_training_config(cfg)
            action_dim = 6 if run["task"].startswith("walker_") else 12
            obs_spec = specs.Array((2304,), np.float32, "observation")
            action_spec = specs.Array((action_dim,), np.float32, "action")
            cfg.agent.obs_type, cfg.agent.obs_shape = cfg.obs_type, obs_spec.shape
            cfg.agent.action_shape = action_spec.shape
            cfg.agent.num_expl_steps = cfg.num_seed_frames // cfg.action_repeat
            resolved = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
            comparable = copy.deepcopy(resolved)
            comparable.pop("experiment", None)
            comparable.pop("checkpoint_root", None)
            comparable["agent"].pop("idm_coef")
            cfg.device = cfg.agent.device = "cpu"
            np.random.seed(int(cfg.seed))
            torch.manual_seed(int(cfg.seed))
            agent = pretrain.make_agent(cfg.obs_type, obs_spec, action_spec,
                                        cfg.num_seed_frames // cfg.action_repeat, cfg.agent)
            audit = audit_agent(agent, expected_coef)
            rng_sha256 = hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()
            pairs.setdefault(run["task"], []).append({
                "group": run["group"], "common_config": comparable,
                "initial_sha256": audit["initial_sha256"], "cpu_rng_sha256": rng_sha256})
            gradient_audit = _scratch_gradient_audit(agent)
            report["runs"].append({"task": run["task"], "group": run["group"],
                                   "run_id": run["run_id"], "resolved_config": resolved,
                                   "audit": audit, "scratch_gradient_audit": gradient_audit})
            print(f"[{index + 1}/16] PASS {run['task']} {run['group']} stateB legacy", flush=True)
            del agent, cfg
            gc.collect()
    for task, pair in pairs.items():
        left, right = pair
        _require(left["common_config"] == right["common_config"],
                 f"{task}: matched training/eval configs differ beyond IDM/identities")
        _require(left["initial_sha256"] == right["initial_sha256"],
                 f"{task}: common initial network weights differ")
        report["pairs"][task] = {
            "common_training_eval_config_identical": True,
            "common_initial_weights_identical": True,
            "initial_sha256": left["initial_sha256"],
            "post_init_cpu_rng_identical": left["cpu_rng_sha256"] == right["cpu_rng_sha256"],
            "rng_note": "Existing legacy IDM head initialization consumes extra CPU RNG after all common networks are initialized; no RNG reset or algorithm change applied.",
        }
    report["status"] = "PASS"
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.manifest.with_name("preflight.json")
    preflight_manifest(args.manifest, output)
    print(f"PASS: 16 state-B agents, matched weights and legacy IDM gradients; {output}", flush=True)


if __name__ == "__main__":
    main()
