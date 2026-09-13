#!/usr/bin/env python3
"""Deterministic preflight checks for the online separate-F/B DINO adapters."""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
import sys
import typing as tp

import numpy as np
import torch
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from url_benchmark import replay_buffer as rb  # noqa: E402
from url_benchmark.agent import fb_ddpg  # noqa: E402


EXPECTED_RUNS = {
    "dino_cls3_cheetah_walk_seed1_sepFB_lrF1e4_lrB1e4_ortho1": "cheetah_walk",
    "dino_cls3_cheetah_run_seed1_sepFB_lrF1e4_lrB1e4_ortho1": "cheetah_run",
    "dino_cls3_cheetah_run_backward_seed1_sepFB_lrF1e4_lrB1e4_ortho1": "cheetah_run_backward",
    "dino_cls3_cheetah_walk_backward_seed1_sepFB_lrF1e4_lrB1e4_ortho1": "cheetah_walk_backward",
}


def _make_config() -> fb_ddpg.FBDDPGAgentConfig:
    # The deliberately unusual width proves that adapter construction consumes
    # obs_shape rather than relying on a hard-coded DINO/CLS stack dimension.
    return fb_ddpg.FBDDPGAgentConfig(
        obs_shape=(21,),
        action_shape=(2,),
        obs_type="dino",
        device="cpu",
        num_expl_steps=0,
        goal_space=None,
        use_cls=True,
        use_tb=True,
        use_wandb=False,
        use_hiplog=False,
        update_encoder=True,
        batch_size=4,
        hidden_dim=32,
        backward_hidden_dim=32,
        feature_dim=512,
        z_dim=8,
        dino_use_adapter=True,
        dino_adapter_type="linear",
        dino_adapter_output_dim=512,
        dino_separate_fb_adapters=True,
        dino_separate_backward_adapter=False,
        lr=1e-4,
        fb_lr=1e-4,
        lr_f=1e-4,
        lr_b=1e-4,
        lr_actor=1e-4,
        lr_coef=1.0,
        ortho_coef=1.0,
        mix_ratio=0.5,
    )


def _make_agent() -> fb_ddpg.FBDDPGAgent:
    torch.manual_seed(17)
    np.random.seed(17)
    return fb_ddpg.FBDDPGAgent(**dataclasses.asdict(_make_config()))


def _optimizer_parameter_ids(optimizer: torch.optim.Optimizer) -> tp.Set[int]:
    return {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }


def _module_parameter_ids(module: nn.Module) -> tp.Set[int]:
    return {id(parameter) for parameter in module.parameters()}


def _has_nonzero_gradient(module: nn.Module) -> bool:
    return any(
        parameter.grad is not None
        and torch.count_nonzero(parameter.grad.detach()).item() > 0
        for parameter in module.parameters()
    )


def _all_gradients_absent(module: nn.Module) -> bool:
    return all(parameter.grad is None for parameter in module.parameters())


def _zero_all_gradients(agent: fb_ddpg.FBDDPGAgent) -> None:
    for optimizer in (
        agent.forward_fb_opt,
        agent.backward_fb_opt,
        agent.actor_opt,
    ):
        assert optimizer is not None
        optimizer.zero_grad(set_to_none=True)


def _clone_parameters(module: nn.Module) -> tp.List[torch.Tensor]:
    return [parameter.detach().clone() for parameter in module.parameters()]


def _parameters_equal(before: tp.Sequence[torch.Tensor], module: nn.Module) -> bool:
    return all(
        torch.equal(expected, actual.detach())
        for expected, actual in zip(before, module.parameters())
    )


def _synthetic_batch() -> rb.EpisodeBatch:
    rng = np.random.RandomState(23)
    return rb.EpisodeBatch(
        obs=rng.randn(4, 21).astype(np.float32),
        action=rng.uniform(-1, 1, size=(4, 2)).astype(np.float32),
        reward=rng.randn(4, 1).astype(np.float32),
        next_obs=rng.randn(4, 21).astype(np.float32),
        discount=np.full((4, 1), 0.99, dtype=np.float32),
        future_obs=rng.randn(4, 21).astype(np.float32),
    )


class _Replay:
    def __init__(self, batch: rb.EpisodeBatch) -> None:
        self.batch = batch

    def sample(self, batch_size: int) -> rb.EpisodeBatch:
        assert batch_size == self.batch.obs.shape[0]
        return self.batch


# REDUNDANCY REVIEW: also invoked verbatim by
# url_benchmark/agent/test_separate_fb_adapters.py as its entire test body -- that pytest
# file adds no coverage beyond calling this function.
def check_agent_contract() -> tp.Dict[str, bool]:
    agent = _make_agent()
    assert agent.forward_adapter is not None
    assert agent.backward_adapter is not None
    assert agent.forward_fb_opt is not None
    assert agent.backward_fb_opt is not None

    forward_adapter = agent.forward_adapter
    backward_adapter = agent.backward_adapter

    assert isinstance(forward_adapter, nn.Sequential)
    assert len(forward_adapter) == 2
    assert isinstance(forward_adapter[0], nn.LayerNorm)
    assert isinstance(forward_adapter[1], nn.Linear)
    assert forward_adapter[0].normalized_shape == (21,)
    assert forward_adapter[1].in_features == 21
    assert forward_adapter[1].out_features == 512

    forward_state = forward_adapter.state_dict()
    backward_state = backward_adapter.state_dict()
    assert forward_state.keys() == backward_state.keys()
    assert all(
        torch.equal(forward_state[name], backward_state[name])
        for name in forward_state
    )
    assert all(
        forward is not backward and forward.data_ptr() != backward.data_ptr()
        for forward, backward in zip(
            forward_adapter.parameters(), backward_adapter.parameters()
        )
    )
    assert all(parameter.requires_grad for parameter in forward_adapter.parameters())
    assert all(parameter.requires_grad for parameter in backward_adapter.parameters())

    forward_ids = _optimizer_parameter_ids(agent.forward_fb_opt)
    backward_ids = _optimizer_parameter_ids(agent.backward_fb_opt)
    actor_ids = _optimizer_parameter_ids(agent.actor_opt)
    expected_forward_ids = (
        _module_parameter_ids(agent.forward_net)
        | _module_parameter_ids(forward_adapter)
    )
    expected_backward_ids = (
        _module_parameter_ids(agent.backward_net)
        | _module_parameter_ids(backward_adapter)
    )
    assert forward_ids == expected_forward_ids
    assert backward_ids == expected_backward_ids
    assert forward_ids.isdisjoint(backward_ids)
    assert actor_ids.isdisjoint(forward_ids | backward_ids)
    assert agent.fb_opt is None
    assert agent.encoder_opt is None
    assert agent.backward_encoder_opt is None
    assert agent.backward_encoder_target is None

    raw = torch.randn(4, 21)
    _zero_all_gradients(agent)
    backward_embedding = agent.backward_net(backward_adapter(raw))
    covariance = backward_embedding @ backward_embedding.T
    off_diagonal = ~torch.eye(4, dtype=torch.bool)
    orth_loss = (
        covariance[off_diagonal].square().mean()
        - 2.0 * covariance.diag().mean()
    )
    orth_loss.backward()
    assert _has_nonzero_gradient(backward_adapter)
    assert _has_nonzero_gradient(agent.backward_net)
    assert _all_gradients_absent(forward_adapter)
    assert _all_gradients_absent(agent.forward_net)

    _zero_all_gradients(agent)
    agent.cfg.ortho_coef = 0.0
    obs = forward_adapter(torch.randn(4, 21))
    next_obs = forward_adapter(torch.randn(4, 21))
    next_goal = backward_adapter(torch.randn(4, 21))
    metrics = agent.update_fb(
        obs=obs,
        action=torch.empty(4, 2).uniform_(-1, 1),
        discount=torch.full((4, 1), 0.99),
        next_obs=next_obs,
        next_goal=next_goal,
        target_next_goal=next_goal.detach(),
        z=agent.sample_z(4),
        step=0,
    )
    assert _has_nonzero_gradient(forward_adapter)
    assert _has_nonzero_gradient(agent.forward_net)
    assert _has_nonzero_gradient(backward_adapter)
    assert _has_nonzero_gradient(agent.backward_net)
    required_metrics = {
        "lr_f",
        "lr_b",
        "forward_adapter_grad_norm",
        "backward_adapter_grad_norm",
        "F_grad_norm",
        "B_grad_norm",
        "F_abs_max",
        "F_norm_max",
        "M_online_abs_max",
        "orth_l2",
        "orth_loss_offdiag",
        "fb_loss",
        "fb_offdiag",
        "fb_diag",
    }
    assert required_metrics <= metrics.keys()
    assert metrics["lr_f"] == 1e-4
    assert metrics["lr_b"] == 1e-4
    assert all(np.isfinite(metrics[name]) for name in required_metrics)

    routing_agent = _make_agent()
    assert routing_agent.forward_adapter is not None
    assert routing_agent.backward_adapter is not None
    forward_before = _clone_parameters(routing_agent.forward_adapter)
    backward_before = _clone_parameters(routing_agent.backward_adapter)
    batch = _synthetic_batch()
    backward_inputs: tp.List[torch.Tensor] = []

    def _capture_backward_input(_module: nn.Module, inputs: tp.Tuple[torch.Tensor, ...]) -> None:
        backward_inputs.append(inputs[0].detach().clone())

    hook = routing_agent.backward_adapter.register_forward_pre_hook(
        _capture_backward_input
    )
    original_update_fb = routing_agent.update_fb
    routing_agent.update_fb = lambda **_kwargs: {}  # type: ignore[method-assign]
    try:
        actor_metrics = routing_agent.update(_Replay(batch), step=0)  # type: ignore[arg-type]
    finally:
        routing_agent.update_fb = original_update_fb  # type: ignore[method-assign]
        hook.remove()

    assert "actor_loss" in actor_metrics
    assert _parameters_equal(forward_before, routing_agent.forward_adapter)
    assert _parameters_equal(backward_before, routing_agent.backward_adapter)
    assert _all_gradients_absent(routing_agent.forward_adapter)
    assert _all_gradients_absent(routing_agent.backward_adapter)
    assert len(backward_inputs) == 3
    expected_backward_inputs = (
        torch.from_numpy(batch.next_obs),
        torch.from_numpy(batch.obs),
        torch.from_numpy(batch.future_obs),
    )
    assert all(
        any(torch.equal(actual, expected) for actual in backward_inputs)
        for expected in expected_backward_inputs
    )

    return {
        "linear_adapter_uses_runtime_obs_shape": True,
        "adapters_start_numerically_identical": True,
        "adapter_parameter_objects_are_distinct": True,
        "branch_optimizer_membership_is_exact_and_disjoint": True,
        "orth_only_updates_backward_branch_only": True,
        "fb_loss_produces_both_branch_gradients": True,
        "actor_update_does_not_modify_adapters": True,
        "goal_space_null_routes_dino_through_backward_adapter": True,
        "no_target_or_ema_dino_adapter": True,
        "requested_diagnostics_are_finite": True,
    }


# REDUNDANCY REVIEW: also called by url_benchmark/test_separate_fb_launcher.py after that
# test dry-runs launch_cheetah_separate_fb_adapters.sh purely to reach this same function.
def check_launch_plan(launch_dir: Path) -> tp.Dict[str, bool]:
    manifest = launch_dir / "manifest.tsv"
    jobs_dir = launch_dir / "jobs"
    assert manifest.is_file(), f"missing launch manifest: {manifest}"
    job_files = sorted(jobs_dir.glob("*.sh"))
    assert len(job_files) == 4, f"expected four jobs, found {len(job_files)}"

    manifest_text = manifest.read_text()
    assert len(manifest_text.strip().splitlines()) == 5
    for run_name, task in EXPECTED_RUNS.items():
        assert run_name in manifest_text
        matching_jobs = [path for path in job_files if run_name in path.name]
        assert len(matching_jobs) == 1
        job_text = matching_jobs[0].read_text()
        # Job generators use printf %q, which escapes Hydra list brackets for
        # the shell without changing the argv received by Python.
        normalized_job_text = job_text.replace("\\", "")
        required_fragments = (
            "WANDB_ENTITY=lmu_rl",
            "WANDB_PROJECT=controllable_agent_cheetah_fb_stability",
            f"WANDB_RUN_NAME={run_name}",
            f"WANDB_RUN_ID={run_name}",
            "WANDB_RESUME=never",
            f"task={task}",
            "seed=1",
            "agent=fb_ddpg",
            "device=cuda",
            "use_wandb=True",
            "use_tb=False",
            "use_hiplog=False",
            "save_video=False",
            "save_train_video=False",
            "save_replay_buffer_in_checkpoint=False",
            "auto_resume=False",
            "load_model=null",
            "load_replay_buffer=null",
            "checkpoint_every=100000",
            "obs_type=dino",
            "dino_model_name=facebook/dinov2-base",
            "use_cls=True",
            "frame_stack=3",
            "dino_frame_stack=3",
            "render_shape=[224,224]",
            "action_repeat=2",
            "goal_space=null",
            "custom_reward=null",
            "append_goal_to_observation=False",
            "discount=0.99",
            "future=0.99",
            "reward_free=True",
            "agent.dino_use_adapter=True",
            "agent.dino_adapter_type=linear",
            "agent.dino_adapter_output_dim=512",
            "agent.dino_separate_fb_adapters=True",
            "agent.dino_separate_backward_adapter=False",
            "agent.backward_encoder_grad_scale=1.0",
            "agent.lr=0.0001",
            "agent.fb_lr=0.0001",
            "agent.lr_f=0.0001",
            "agent.lr_b=0.0001",
            "agent.lr_actor=0.0001",
            "agent.lr_coef=1.0",
            "agent.ortho_coef=1.0",
            "agent.fb_target_tau=0.01",
            "agent.z_dim=50",
            "agent.mix_ratio=0.5",
            "agent.batch_size=1024",
            "agent.update_every_steps=2",
            "agent.num_inference_steps=5120",
            "agent.hidden_dim=1024",
            "agent.backward_hidden_dim=526",
            "agent.feature_dim=512",
            "agent.stddev_schedule=0.2",
            "agent.stddev_clip=0.3",
            "agent.update_z_every_step=300",
            "agent.update_z_proba=1.0",
            "agent.nstep=1",
            "agent.future_ratio=0.0",
            "agent.rand_weight=False",
            "agent.preprocess=True",
            "agent.norm_z=True",
            "agent.q_loss=False",
            "agent.q_loss_coef=0.01",
            "agent.boltzmann=False",
            "agent.add_trunk=False",
            "agent.idm_coef=0.0",
            "agent.idm_lr=null",
            "agent.idm_encoder_mode=legacy",
            "update_encoder=True",
            "num_seed_frames=4000",
            "replay_buffer_episodes=5000",
            "num_train_frames=2000010",
            "eval_every_frames=10000",
            "num_eval_episodes=10",
            "final_tests=10",
            "experiment=cheetah_fb_stability",
            "snapshot_at=[100000,200000,500000,800000,1000000,1500000,2000000]",
        )
        for fragment in required_fragments:
            assert fragment in normalized_job_text, f"{matching_jobs[0]} missing {fragment}"
        assert "2304" not in normalized_job_text
        assert "dino_frame_stack=1" not in normalized_job_text

    return {
        "exactly_four_requested_runs": True,
        "all_four_runs_use_three_cls_frames": True,
        "all_four_jobs_match_fixed_mechanism_config": True,
        "no_job_hard_codes_adapter_input_width": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--launch-dir",
        type=Path,
        help="Optionally validate a generated four-job launch directory.",
    )
    args = parser.parse_args()

    torch.set_num_threads(1)
    checks = check_agent_contract()
    if args.launch_dir is not None:
        checks.update(check_launch_plan(args.launch_dir.resolve()))
    print(json.dumps({"status": "PASS", "checks": checks}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
