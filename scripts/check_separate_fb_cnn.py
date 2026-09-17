#!/usr/bin/env python3
"""Deterministic preflight checks for separate pixel FB CNN encoders."""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import typing as tp

import numpy as np
import torch
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from url_benchmark import replay_buffer as rb  # noqa: E402
from url_benchmark.agent import fb_ddpg  # noqa: E402
from url_benchmark.agent.ddpg import Encoder  # noqa: E402


EXPECTED_RUNS = {
    "cnn_cheetah_walk_seed1_sepFBcnn_onlineEnc": "cheetah_walk",
    "cnn_cheetah_run_seed1_sepFBcnn_onlineEnc": "cheetah_run",
    "cnn_cheetah_run_backward_seed1_sepFBcnn_onlineEnc": "cheetah_run_backward",
    "cnn_cheetah_walk_backward_seed1_sepFBcnn_onlineEnc": "cheetah_walk_backward",
}


def _make_config(*, separate: bool = True) -> fb_ddpg.FBDDPGAgentConfig:
    return fb_ddpg.FBDDPGAgentConfig(
        obs_shape=(9, 84, 84),
        action_shape=(2,),
        obs_type="pixels",
        device="cpu",
        num_expl_steps=0,
        goal_space=None,
        use_cls=True,
        use_tb=True,
        use_wandb=False,
        use_hiplog=False,
        update_encoder=True,
        batch_size=4,
        hidden_dim=8,
        backward_hidden_dim=8,
        feature_dim=8,
        z_dim=4,
        pixel_separate_fb_encoders=separate,
        lr=1e-4,
        fb_lr=1e-4,
        lr_f=1e-4 if separate else None,
        lr_b=1e-4 if separate else None,
        lr_actor=1e-4,
        lr_coef=1.0,
        ortho_coef=1.0,
        mix_ratio=0.5,
        idm_coef=0.0,
    )


def _make_agent(*, separate: bool = True, seed: int = 17) -> fb_ddpg.FBDDPGAgent:
    torch.manual_seed(seed)
    np.random.seed(seed)
    return fb_ddpg.FBDDPGAgent(
        **dataclasses.asdict(_make_config(separate=separate))
    )


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


def _gradients_finite(agent: fb_ddpg.FBDDPGAgent) -> bool:
    modules = (
        agent.encoder,
        agent.backward_encoder,
        agent.forward_net,
        agent.backward_net,
        agent.actor,
    )
    return all(
        torch.isfinite(parameter.grad).all().item()
        for module in modules
        if module is not None
        for parameter in module.parameters()
        if parameter.grad is not None
    )


def _synthetic_batch(
    seed: int,
    *,
    batch_size: int = 4,
    action_dim: int = 2,
) -> rb.EpisodeBatch[np.ndarray]:
    rng = np.random.RandomState(seed)
    shape = (batch_size, 9, 84, 84)
    return rb.EpisodeBatch(
        obs=rng.randint(0, 256, size=shape, dtype=np.uint8),
        action=rng.uniform(-1, 1, size=(batch_size, action_dim)).astype(np.float32),
        reward=rng.randn(batch_size, 1).astype(np.float32),
        next_obs=rng.randint(0, 256, size=shape, dtype=np.uint8),
        discount=np.full((batch_size, 1), 0.99, dtype=np.float32),
        future_obs=rng.randint(0, 256, size=shape, dtype=np.uint8),
    )


class _Replay:
    def __init__(self, batch: rb.EpisodeBatch[np.ndarray]) -> None:
        self.batch = batch

    def sample(self, batch_size: int) -> rb.EpisodeBatch[np.ndarray]:
        assert batch_size == self.batch.obs.shape[0]
        return self.batch


def _check_routing() -> None:
    agent = _make_agent(seed=29)
    assert agent.forward_encoder is not None
    assert agent.backward_encoder is not None
    batch = _synthetic_batch(31)
    forward_inputs: tp.List[torch.Tensor] = []
    backward_inputs: tp.List[torch.Tensor] = []
    backward_outputs: tp.List[torch.Tensor] = []
    augmentation_outputs: tp.List[torch.Tensor] = []
    update_fb_kwargs: tp.Dict[str, torch.Tensor] = {}

    def _capture_forward(
        _module: nn.Module, inputs: tp.Tuple[torch.Tensor, ...]
    ) -> None:
        forward_inputs.append(inputs[0].detach().clone())

    def _capture_backward_input(
        _module: nn.Module, inputs: tp.Tuple[torch.Tensor, ...]
    ) -> None:
        backward_inputs.append(inputs[0].detach().clone())

    def _capture_backward_output(
        _module: nn.Module,
        _inputs: tp.Tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        backward_outputs.append(output.detach().clone())

    def _capture_augmentation(
        _module: nn.Module,
        _inputs: tp.Tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        augmentation_outputs.append(output.detach().clone())

    def _capture_update_fb(**kwargs: torch.Tensor) -> tp.Dict[str, float]:
        update_fb_kwargs.update(kwargs)
        return {}

    handles = (
        agent.forward_encoder.register_forward_pre_hook(_capture_forward),
        agent.backward_encoder.register_forward_pre_hook(_capture_backward_input),
        agent.backward_encoder.register_forward_hook(_capture_backward_output),
        agent.aug.register_forward_hook(_capture_augmentation),
    )
    original_update_fb = agent.update_fb
    agent.update_fb = _capture_update_fb  # type: ignore[method-assign]
    try:
        agent.update(_Replay(batch), step=0)  # type: ignore[arg-type]
    finally:
        agent.update_fb = original_update_fb  # type: ignore[method-assign]
        for handle in handles:
            handle.remove()

    assert len(forward_inputs) == 2
    assert len(backward_inputs) == 3
    assert len(augmentation_outputs) == 3
    # Encoder call order is F(obs), F(next), B(next), B(obs), B(future).
    assert torch.equal(forward_inputs[0], backward_inputs[1])
    assert torch.equal(forward_inputs[1], backward_inputs[0])
    assert torch.equal(forward_inputs[0], augmentation_outputs[0])
    assert torch.equal(forward_inputs[1], augmentation_outputs[1])
    assert torch.equal(backward_inputs[2], augmentation_outputs[2])
    assert torch.equal(update_fb_kwargs["target_next_goal"], backward_outputs[0])
    assert not update_fb_kwargs["target_next_goal"].requires_grad

    calls = 0

    def _count_backward(_module: nn.Module, _inputs: tp.Tuple[torch.Tensor, ...]) -> None:
        nonlocal calls
        calls += 1

    handle = agent.backward_encoder.register_forward_pre_hook(_count_backward)
    try:
        goal = batch.obs[0]
        agent.get_goal_meta(goal)
        agent.infer_meta_from_obs_and_rewards(
            torch.from_numpy(batch.obs).float(),
            torch.from_numpy(batch.reward),
        )
        agent.compute_z_correl(
            SimpleNamespace(observation=goal, goal=None),
            agent.init_meta(),
        )
    finally:
        handle.remove()
    assert calls == 3


# REDUNDANCY REVIEW: also invoked verbatim by url_benchmark/agent/test_separate_fb_cnn.py
# as its entire test body -- that pytest file adds no coverage beyond calling this function.
def check_agent_contract() -> tp.Dict[str, bool]:
    agent = _make_agent()
    assert agent.forward_encoder is agent.encoder
    assert isinstance(agent.forward_encoder, Encoder)
    assert isinstance(agent.backward_encoder, Encoder)
    assert agent.forward_fb_opt is not None
    assert agent.backward_fb_opt is not None

    forward_state = agent.forward_encoder.state_dict()
    backward_state = agent.backward_encoder.state_dict()
    assert forward_state.keys() == backward_state.keys()
    assert all(
        torch.equal(forward_state[name], backward_state[name])
        for name in forward_state
    )
    assert all(
        forward is not backward and forward.data_ptr() != backward.data_ptr()
        for forward, backward in zip(
            agent.forward_encoder.parameters(), agent.backward_encoder.parameters()
        )
    )

    forward_ids = _optimizer_parameter_ids(agent.forward_fb_opt)
    backward_ids = _optimizer_parameter_ids(agent.backward_fb_opt)
    actor_ids = _optimizer_parameter_ids(agent.actor_opt)
    assert forward_ids == (
        _module_parameter_ids(agent.forward_net)
        | _module_parameter_ids(agent.forward_encoder)
    )
    assert backward_ids == (
        _module_parameter_ids(agent.backward_net)
        | _module_parameter_ids(agent.backward_encoder)
    )
    assert forward_ids.isdisjoint(backward_ids)
    assert actor_ids.isdisjoint(forward_ids | backward_ids)
    assert agent.fb_opt is None
    assert agent.encoder_opt is None
    assert agent.backward_encoder_opt is None
    assert agent.backward_encoder_target is None

    _zero_all_gradients(agent)
    raw = torch.randint(0, 256, (4, 9, 84, 84), dtype=torch.uint8)
    backward_embedding = agent.backward_net(agent.backward_aug_and_encode(raw))
    covariance = backward_embedding @ backward_embedding.T
    off_diagonal = ~torch.eye(4, dtype=torch.bool)
    orth_loss = (
        covariance[off_diagonal].square().mean()
        - 2.0 * covariance.diag().mean()
    )
    orth_loss.backward()
    assert _has_nonzero_gradient(agent.backward_encoder)
    assert _has_nonzero_gradient(agent.backward_net)
    assert _all_gradients_absent(agent.forward_encoder)
    assert _all_gradients_absent(agent.forward_net)

    _zero_all_gradients(agent)
    obs = agent.aug_and_encode(raw)
    next_obs = agent.aug_and_encode(torch.flip(raw, dims=(0,)))
    next_goal = agent.backward_aug_and_encode(torch.roll(raw, shifts=1, dims=0))
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
    assert _has_nonzero_gradient(agent.forward_encoder)
    assert _has_nonzero_gradient(agent.backward_encoder)
    required_metrics = {
        "forward_encoder_grad_norm",
        "backward_encoder_grad_norm",
        "lr_f",
        "lr_b",
        "F_abs_max",
        "F_norm_max",
        "F_norm_p95",
        "F_norm_p99",
        "M_online_abs_max",
        "M_online_abs_p95",
        "M_online_abs_p99",
        "orth_l2",
        "orth_loss",
        "orth_loss_offdiag",
        "fb_loss",
        "fb_diag",
        "fb_offdiag",
    }
    assert required_metrics <= metrics.keys()
    assert metrics["lr_f"] == 1e-4
    assert metrics["lr_b"] == 1e-4
    assert all(np.isfinite(metrics[name]) for name in required_metrics)

    _zero_all_gradients(agent)
    forward_before = _clone_parameters(agent.forward_encoder)
    backward_before = _clone_parameters(agent.backward_encoder)
    encoded_obs = agent.aug_and_encode(raw).detach()
    agent.update_actor(encoded_obs, agent.sample_z(4), step=0)
    assert _parameters_equal(forward_before, agent.forward_encoder)
    assert _parameters_equal(backward_before, agent.backward_encoder)
    assert _all_gradients_absent(agent.forward_encoder)
    assert _all_gradients_absent(agent.backward_encoder)

    _check_routing()

    smoke_agent = _make_agent(seed=37)
    smoke_metric_names = ("fb_loss", "orth_loss", "actor_loss")
    for update_index in range(3):
        smoke_metrics = smoke_agent.update(
            _Replay(_synthetic_batch(41 + update_index)),
            step=2 * update_index,
        )
        assert all(
            name in smoke_metrics and np.isfinite(smoke_metrics[name])
            for name in smoke_metric_names
        )
        assert _gradients_finite(smoke_agent)

    with tempfile.TemporaryDirectory(prefix="sep_fb_cnn_") as tmp_dir:
        checkpoint = Path(tmp_dir) / "checkpoint.pt"
        torch.save({"agent": smoke_agent}, checkpoint, pickle_protocol=4)
        assert checkpoint.stat().st_size > 0
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        restored = _make_agent(seed=43)
        restored.init_from(payload["agent"])
        assert restored.forward_encoder is not None
        assert restored.backward_encoder is not None
        for source, target in zip(
            smoke_agent.forward_encoder.parameters(),
            restored.forward_encoder.parameters(),
        ):
            assert torch.equal(source, target)
        for source, target in zip(
            smoke_agent.backward_encoder.parameters(),
            restored.backward_encoder.parameters(),
        ):
            assert torch.equal(source, target)
        assert restored.forward_fb_opt is not None
        assert restored.backward_fb_opt is not None
        assert restored.forward_fb_opt.state_dict()["state"]
        assert restored.backward_fb_opt.state_dict()["state"]
        restored_metrics = restored.update(_Replay(_synthetic_batch(47)), step=6)
        assert all(
            name in restored_metrics and np.isfinite(restored_metrics[name])
            for name in smoke_metric_names
        )

    shared_agent = _make_agent(separate=False, seed=53)
    assert shared_agent.cfg.pixel_separate_fb_encoders is False
    assert shared_agent.forward_encoder is None
    assert shared_agent.backward_encoder is None
    assert shared_agent.encoder_opt is not None
    assert shared_agent.fb_opt is not None
    assert shared_agent.forward_fb_opt is None
    assert shared_agent.backward_fb_opt is None

    return {
        "cnns_start_numerically_identical": True,
        "cnn_parameter_objects_are_distinct": True,
        "branch_optimizer_membership_is_exact_and_disjoint": True,
        "orth_only_reaches_backward_map_and_cnn_b": True,
        "fb_update_reaches_both_cnns": True,
        "actor_update_does_not_update_cnn_b": True,
        "all_visual_b_routes_use_cnn_b": True,
        "f_and_b_share_each_pixel_augmentation_realization": True,
        "target_b_features_use_detached_online_cnn_b": True,
        "multi_update_losses_and_gradients_are_finite": True,
        "checkpoint_save_load_and_resume_update_work": True,
        "shared_cnn_default_topology_is_unchanged": True,
        "requested_training_metrics_are_finite": True,
    }


def check_formal_shape_smoke(device: str) -> tp.Dict[str, bool]:
    """Run three updates with the exact Cheetah model and batch dimensions."""
    config = dataclasses.replace(
        _make_config(),
        action_shape=(6,),
        device=device,
        batch_size=1024,
        hidden_dim=1024,
        backward_hidden_dim=526,
        feature_dim=512,
        z_dim=50,
    )
    torch.manual_seed(59)
    np.random.seed(59)
    agent = fb_ddpg.FBDDPGAgent(**dataclasses.asdict(config))
    for update_index in range(3):
        metrics = agent.update(
            _Replay(
                _synthetic_batch(
                    61 + update_index,
                    batch_size=1024,
                    action_dim=6,
                )
            ),
            step=2 * update_index,
        )
        assert all(
            name in metrics and np.isfinite(metrics[name])
            for name in (
                "fb_loss",
                "orth_loss",
                "actor_loss",
                "forward_encoder_grad_norm",
                "backward_encoder_grad_norm",
            )
        )
        assert _gradients_finite(agent)
    return {
        "formal_batch_1024_three_update_losses_are_finite": True,
        "formal_batch_1024_three_update_gradients_are_finite": True,
    }


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
        normalized = matching_jobs[0].read_text().replace("\\", "")
        required = (
            "WANDB_ENTITY=lmu_rl",
            "WANDB_PROJECT=controllable_agent_cheetah_fb_stability",
            f"WANDB_RUN_NAME={run_name}",
            f"task={task}",
            "seed=1",
            "obs_type=pixels",
            "frame_stack=3",
            "render_shape=[84,84]",
            "action_repeat=2",
            "goal_space=null",
            "agent.pixel_separate_fb_encoders=True",
            "agent.dino_separate_fb_adapters=False",
            "agent.dino_separate_backward_adapter=False",
            "agent.lr_f=0.0001",
            "agent.lr_b=0.0001",
            "agent.lr_actor=0.0001",
            "agent.ortho_coef=1.0",
            "agent.fb_target_tau=0.01",
            "agent.idm_coef=0.0",
            "agent.z_dim=50",
            "agent.mix_ratio=0.5",
            "agent.batch_size=1024",
            "num_train_frames=2000010",
            "eval_every_frames=10000",
            "num_eval_episodes=10",
            "snapshot_at=[100000,200000,500000,800000,1000000,1500000,2000000]",
        )
        for fragment in required:
            assert fragment in normalized, f"{matching_jobs[0]} missing {fragment}"
    return {
        "exactly_four_requested_runs": True,
        "all_jobs_enable_only_the_pixel_separate_cnn_ablation": True,
        "all_jobs_match_the_fixed_training_contract": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--launch-dir", type=Path)
    parser.add_argument(
        "--formal-device",
        help="Also run three exact-shape batch-1024 updates, e.g. cuda or cpu.",
    )
    args = parser.parse_args()
    torch.set_num_threads(1)
    checks = check_agent_contract()
    if args.formal_device is not None:
        checks.update(check_formal_shape_smoke(args.formal_device))
    if args.launch_dir is not None:
        checks.update(check_launch_plan(args.launch_dir.resolve()))
    print(json.dumps({"status": "PASS", "checks": checks}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
