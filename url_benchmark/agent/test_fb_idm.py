# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from url_benchmark import replay_buffer as rb
from . import fb_ddpg


_OMITTED = object()


def _make_agent(
    idm_coef=_OMITTED,
    idm_lr=None,
    goal_space=None,
) -> fb_ddpg.FBDDPGAgent:
    kwargs = dict(
        obs_shape=(6,),
        action_shape=(2,),
        obs_type="dino",
        device="cpu",
        num_expl_steps=0,
        goal_space=goal_space,
        use_cls=True,
        use_tb=True,
        use_wandb=False,
        use_hiplog=False,
        update_encoder=True,
        batch_size=4,
        hidden_dim=8,
        backward_hidden_dim=8,
        feature_dim=4,
        z_dim=4,
        dino_adapter_output_dim=4,
    )
    if idm_coef is not _OMITTED:
        kwargs["idm_coef"] = idm_coef
        kwargs["idm_lr"] = idm_lr
    cfg = fb_ddpg.FBDDPGAgentConfig(**kwargs)
    return fb_ddpg.FBDDPGAgent(**dataclasses.asdict(cfg))


def _fixed_batch():
    generator = torch.Generator().manual_seed(2026)
    raw_obs = torch.randn(4, 6, generator=generator)
    raw_next_obs = torch.randn(4, 6, generator=generator)
    action = torch.randn(4, 2, generator=generator).tanh()
    discount = torch.full((4, 1), 0.98)
    z = torch.randn(4, 4, generator=generator)
    return raw_obs, raw_next_obs, action, discount, z


def _update_fb(agent: fb_ddpg.FBDDPGAgent, batch, seed: int = 99):
    raw_obs, raw_next_obs, action, discount, z = batch
    obs = agent.aug_and_encode(raw_obs)
    next_obs = agent.aug_and_encode(raw_next_obs)
    torch.manual_seed(seed)
    return agent.update_fb(
        obs=obs,
        action=action,
        discount=discount,
        next_obs=next_obs,
        next_goal=next_obs,
        target_next_goal=next_obs.detach(),
        z=z,
        step=0,
    )


def _assert_nested_equal(left, right) -> None:
    assert type(left) is type(right)
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right):
            _assert_nested_equal(left_item, right_item)
    else:
        assert left == right


def _assert_module_equal(left: torch.nn.Module, right: torch.nn.Module) -> None:
    _assert_nested_equal(left.state_dict(), right.state_dict())


def _has_nonzero_grad(module: torch.nn.Module) -> bool:
    return any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad).item() > 0
        for parameter in module.parameters()
    )


def test_idm_config_validation_and_learning_rate() -> None:
    with pytest.raises(ValueError, match="idm_coef must be non-negative"):
        _make_agent(idm_coef=-0.1)
    with pytest.raises(ValueError, match="idm_lr must be positive"):
        _make_agent(idm_coef=0.1, idm_lr=0.0)

    default_lr = _make_agent(idm_coef=0.1)
    assert default_lr.idm_optimizer is not None
    assert default_lr.idm_optimizer.param_groups[0]["lr"] == default_lr.cfg.lr

    explicit_lr = _make_agent(idm_coef=0.1, idm_lr=3e-4)
    assert explicit_lr.idm_optimizer is not None
    assert explicit_lr.idm_optimizer.param_groups[0]["lr"] == 3e-4
    idm_parameter_ids = {
        id(parameter)
        for group in explicit_lr.idm_optimizer.param_groups
        for parameter in group["params"]
    }
    assert explicit_lr.idm_head is not None
    assert idm_parameter_ids == {
        id(parameter) for parameter in explicit_lr.idm_head.parameters()
    }
    other_parameter_ids = {
        id(parameter)
        for optimizer in (
            explicit_lr.encoder_opt,
            explicit_lr.actor_opt,
            explicit_lr.fb_opt,
        )
        if optimizer is not None
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    assert idm_parameter_ids.isdisjoint(other_parameter_ids)


def test_zero_coef_is_a_strict_baseline_noop() -> None:
    torch.manual_seed(7)
    baseline = _make_agent()
    baseline_rng_state = torch.get_rng_state().clone()
    torch.manual_seed(7)
    disabled = _make_agent(idm_coef=0.0)
    disabled_rng_state = torch.get_rng_state().clone()

    assert disabled.idm_head is None
    assert disabled.idm_optimizer is None
    torch.testing.assert_close(baseline_rng_state, disabled_rng_state, rtol=0, atol=0)
    for name in (
        "encoder",
        "actor",
        "forward_net",
        "backward_net",
        "forward_target_net",
        "backward_target_net",
    ):
        _assert_module_equal(getattr(baseline, name), getattr(disabled, name))

    batch = _fixed_batch()
    baseline_metrics = _update_fb(baseline, batch)
    with mock.patch.object(
        disabled,
        "_compute_idm_loss",
        side_effect=AssertionError("IDM path was used"),
    ):
        disabled_metrics = _update_fb(disabled, batch)

    assert disabled_metrics["idm_loss"] == 0.0
    assert disabled_metrics["total_loss"] == disabled_metrics["fb_loss"]
    assert baseline_metrics.keys() == disabled_metrics.keys()
    for key in baseline_metrics:
        assert baseline_metrics[key] == disabled_metrics[key]
    for name in ("encoder", "forward_net", "backward_net"):
        _assert_module_equal(getattr(baseline, name), getattr(disabled, name))
    for name in ("encoder_opt", "fb_opt"):
        _assert_nested_equal(
            getattr(baseline, name).state_dict(),
            getattr(disabled, name).state_dict(),
        )


def test_idm_loss_shape_and_gradient_boundary() -> None:
    torch.manual_seed(11)
    agent = _make_agent(idm_coef=0.1)
    assert agent.idm_head is not None

    raw_obs, raw_next_obs, action, _, _ = _fixed_batch()
    obs = agent.aug_and_encode(raw_obs)
    next_obs = agent.aug_and_encode(raw_next_obs)
    obs.retain_grad()
    next_obs.retain_grad()
    captured = {}

    def capture_values(_module, inputs, output) -> None:
        captured["input"] = inputs[0].detach().clone()
        captured["output"] = output.shape
        captured["prediction"] = output

    handle = agent.idm_head.register_forward_hook(capture_values)
    try:
        loss = agent._compute_idm_loss(obs, next_obs, action)  # pylint: disable=protected-access
        loss.backward()
    finally:
        handle.remove()

    assert captured["input"].shape == (4, 2 * agent.obs_dim)
    torch.testing.assert_close(
        captured["input"],
        torch.cat([obs.detach(), next_obs.detach()], dim=-1),
    )
    assert captured["output"] == action.shape
    assert agent.idm_head.weight.shape == (agent.action_dim, 2 * agent.obs_dim)
    torch.testing.assert_close(loss, F.mse_loss(captured["prediction"], action))
    assert obs.grad is not None and torch.count_nonzero(obs.grad).item() > 0
    assert next_obs.grad is not None and torch.count_nonzero(next_obs.grad).item() > 0
    assert _has_nonzero_grad(agent.encoder)
    assert _has_nonzero_grad(agent.idm_head)
    for module in (agent.actor, agent.forward_net, agent.backward_net):
        assert all(parameter.grad is None for parameter in module.parameters())


def test_update_fb_uses_next_observation_and_updates_idm() -> None:
    torch.manual_seed(13)
    agent = _make_agent(idm_coef=0.1)
    assert agent.idm_head is not None
    assert agent.idm_optimizer is not None

    raw_obs, raw_next_obs, action, discount, z = _fixed_batch()
    obs = agent.aug_and_encode(raw_obs)
    next_obs = agent.aug_and_encode(raw_next_obs)
    # This deliberately differs from next_obs.  Goal-conditioned Walker and
    # Quadruped runs use a low-dimensional next_goal, which must never be fed
    # to the inverse-dynamics head in place of the next observation feature.
    next_goal = torch.randn_like(next_obs)
    head_before = {key: value.detach().clone() for key, value in agent.idm_head.state_dict().items()}

    with mock.patch.object(
        agent,
        "_compute_idm_loss",
        wraps=agent._compute_idm_loss,  # pylint: disable=protected-access
    ) as compute_idm_loss:
        metrics = agent.update_fb(
            obs=obs,
            action=action,
            discount=discount,
            next_obs=next_obs,
            next_goal=next_goal,
            target_next_goal=next_goal.detach(),
            z=z,
            step=0,
        )

    assert compute_idm_loss.call_count == 1
    assert compute_idm_loss.call_args.args[0] is obs
    assert compute_idm_loss.call_args.args[1] is next_obs
    assert compute_idm_loss.call_args.args[1] is not next_goal
    assert metrics["total_loss"] == pytest.approx(
        metrics["fb_loss"] + agent.cfg.idm_coef * metrics["idm_loss"]
    )
    assert agent.idm_optimizer.state_dict()["state"]
    assert any(
        not torch.equal(head_before[key], value)
        for key, value in agent.idm_head.state_dict().items()
    )


def test_idm_update_supports_low_dimensional_task_goals() -> None:
    agent = _make_agent(idm_coef=0.1, goal_space="simplified_walker")
    batch = rb.EpisodeBatch(
        obs=np.random.randn(4, 6).astype(np.float32),
        action=np.random.uniform(-1, 1, size=(4, 2)).astype(np.float32),
        reward=np.random.randn(4, 1).astype(np.float32),
        next_obs=np.random.randn(4, 6).astype(np.float32),
        discount=np.full((4, 1), 0.99, dtype=np.float32),
        goal=np.random.randn(4, 3).astype(np.float32),
        next_goal=np.random.randn(4, 3).astype(np.float32),
        future_goal=np.random.randn(4, 3).astype(np.float32),
    )

    class _Replay:
        def sample(self, batch_size: int) -> rb.EpisodeBatch:
            assert batch_size == 4
            return batch

    metrics = agent.update(_Replay(), step=0)  # type: ignore[arg-type]
    assert metrics["idm_loss"] > 0
    assert metrics["total_loss"] == pytest.approx(
        metrics["fb_loss"] + agent.cfg.idm_coef * metrics["idm_loss"]
    )


def test_idm_init_from_restores_head_and_optimizer_and_accepts_legacy_agent(
    tmp_path: Path,
) -> None:
    torch.manual_seed(17)
    source = _make_agent(idm_coef=0.1)
    _update_fb(source, _fixed_batch())
    assert source.idm_head is not None
    assert source.idm_optimizer is not None

    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"agent": source}, checkpoint, pickle_protocol=4)
    payload = torch.load(checkpoint, weights_only=False)

    restored = _make_agent(idm_coef=0.1)
    restored.init_from(payload["agent"])
    assert restored.idm_head is not None
    assert restored.idm_optimizer is not None
    _assert_module_equal(source.idm_head, restored.idm_head)
    _assert_nested_equal(
        source.idm_optimizer.state_dict(),
        restored.idm_optimizer.state_dict(),
    )

    # Loading optimizer state is only useful if the resumed trajectory remains
    # identical.  Exercise one more update from the restored checkpoint state.
    for agent in (source, restored):
        for optimizer in (agent.encoder_opt, agent.fb_opt, agent.idm_optimizer):
            assert optimizer is not None
            optimizer.zero_grad(set_to_none=True)
    continuation_batch = _fixed_batch()
    source_metrics = _update_fb(source, continuation_batch, seed=314)
    restored_metrics = _update_fb(restored, continuation_batch, seed=314)
    assert source_metrics == restored_metrics
    for name in ("encoder", "forward_net", "backward_net", "idm_head"):
        _assert_module_equal(getattr(source, name), getattr(restored, name))
    for name in ("encoder_opt", "fb_opt", "idm_optimizer"):
        _assert_nested_equal(
            getattr(source, name).state_dict(),
            getattr(restored, name).state_dict(),
        )

    legacy = _make_agent(idm_coef=0.0)
    del legacy.idm_head
    del legacy.idm_optimizer
    head_before = {
        key: value.detach().clone()
        for key, value in restored.idm_head.state_dict().items()
    }
    restored.init_from(legacy)
    for key, value in restored.idm_head.state_dict().items():
        torch.testing.assert_close(value, head_before[key], rtol=0, atol=0)


@pytest.mark.parametrize("missing", ["idm_head", "idm_optimizer"])
def test_idm_init_from_rejects_incomplete_idm_checkpoint(missing: str) -> None:
    source = _make_agent(idm_coef=0.1)
    destination = _make_agent(idm_coef=0.1)
    delattr(source, missing)
    with pytest.raises(ValueError, match=missing):
        destination.init_from(source)
