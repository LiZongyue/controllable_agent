# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
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
    idm_diagnostics_interval=500,
    idm_encoder_mode="legacy",
    idm_encoder_burnin_steps=0,
    idm_encoder_ramp_steps=0,
    idm_grad_ratio_target=None,
    idm_grad_ratio_ema=0.9,
    idm_coef_min=0.1,
    idm_coef_max=200.0,
    idm_coef_slew_rate=2.0,
    update_encoder=True,
    logging_enabled=True,
    lr=1e-4,
) -> fb_ddpg.FBDDPGAgent:
    kwargs = dict(
        obs_shape=(6,),
        action_shape=(2,),
        obs_type="dino",
        device="cpu",
        lr=lr,
        num_expl_steps=0,
        goal_space=goal_space,
        use_cls=True,
        use_tb=logging_enabled,
        use_wandb=False,
        use_hiplog=False,
        update_encoder=update_encoder,
        batch_size=4,
        hidden_dim=8,
        backward_hidden_dim=8,
        feature_dim=4,
        z_dim=4,
        dino_adapter_output_dim=4,
        idm_diagnostics_interval=idm_diagnostics_interval,
        idm_encoder_mode=idm_encoder_mode,
        idm_encoder_burnin_steps=idm_encoder_burnin_steps,
        idm_encoder_ramp_steps=idm_encoder_ramp_steps,
        idm_grad_ratio_target=idm_grad_ratio_target,
        idm_grad_ratio_ema=idm_grad_ratio_ema,
        idm_coef_min=idm_coef_min,
        idm_coef_max=idm_coef_max,
        idm_coef_slew_rate=idm_coef_slew_rate,
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


def _update_fb(agent: fb_ddpg.FBDDPGAgent, batch, seed: int = 99, step: int = 0):
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
        step=step,
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
    with pytest.raises(ValueError, match="idm_diagnostics_interval must be positive"):
        _make_agent(idm_coef=0.1, idm_diagnostics_interval=0)
    with pytest.raises(ValueError, match="requires update_encoder=True"):
        _make_agent(idm_coef=0.1, update_encoder=False)

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


def test_idm_encoder_mode_validation_and_schedule() -> None:
    with pytest.raises(ValueError, match="idm_encoder_mode"):
        _make_agent(idm_coef=1.0, idm_encoder_mode="unknown")
    with pytest.raises(ValueError, match="require idm_coef > 0"):
        _make_agent(idm_coef=0.0, idm_encoder_mode="static")
    with pytest.raises(ValueError, match="burn-in/ramp"):
        _make_agent(idm_coef=0.1, idm_encoder_burnin_steps=1)
    with pytest.raises(ValueError, match="finite positive idm_grad_ratio_target"):
        _make_agent(idm_coef=1.0, idm_encoder_mode="balanced")
    with pytest.raises(ValueError, match="does not use idm_encoder_ramp_steps"):
        _make_agent(
            idm_coef=1.0,
            idm_encoder_mode="balanced",
            idm_grad_ratio_target=0.01,
            idm_encoder_ramp_steps=1,
        )

    static = _make_agent(
        idm_coef=10.0,
        idm_encoder_mode="static",
        idm_encoder_burnin_steps=5,
        idm_encoder_ramp_steps=10,
    )
    assert static._idm_encoder_coefficient(4) == 0.0  # pylint: disable=protected-access
    assert static._idm_encoder_coefficient(5) == 0.0  # pylint: disable=protected-access
    assert static._idm_encoder_coefficient(10) == 5.0  # pylint: disable=protected-access
    assert static._idm_encoder_coefficient(15) == 10.0  # pylint: disable=protected-access


def test_encoder_only_mode_scales_encoder_but_not_idm_head_gradient() -> None:
    torch.manual_seed(41)
    unit = _make_agent(idm_coef=1.0, idm_encoder_mode="static")
    torch.manual_seed(41)
    scaled = _make_agent(idm_coef=10.0, idm_encoder_mode="static")
    assert unit.idm_head is not None and scaled.idm_head is not None
    _assert_module_equal(unit.encoder, scaled.encoder)
    _assert_module_equal(unit.idm_head, scaled.idm_head)

    raw_obs, raw_next_obs, action, _, _ = _fixed_batch()

    def idm_gradients(agent):
        obs = agent.aug_and_encode(raw_obs)
        next_obs = agent.aug_and_encode(raw_next_obs)
        coefficient = agent._idm_encoder_coefficient(0)  # pylint: disable=protected-access
        idm_obs = fb_ddpg._scale_idm_encoder_gradient(obs, coefficient)
        idm_next_obs = fb_ddpg._scale_idm_encoder_gradient(next_obs, coefficient)
        loss = agent._compute_idm_loss(  # pylint: disable=protected-access
            idm_obs,
            idm_next_obs,
            action,
        )
        loss.backward()
        encoder_grads = tuple(parameter.grad for parameter in agent.encoder.parameters())
        head_grads = tuple(parameter.grad for parameter in agent.idm_head.parameters())
        return loss.detach(), encoder_grads, head_grads

    unit_loss, unit_encoder_grads, unit_head_grads = idm_gradients(unit)
    scaled_loss, scaled_encoder_grads, scaled_head_grads = idm_gradients(scaled)
    torch.testing.assert_close(unit_loss, scaled_loss, rtol=0, atol=0)
    for unit_grad, scaled_grad in zip(unit_head_grads, scaled_head_grads):
        torch.testing.assert_close(unit_grad, scaled_grad, rtol=0, atol=0)
    for unit_grad, scaled_grad in zip(unit_encoder_grads, scaled_encoder_grads):
        torch.testing.assert_close(scaled_grad, 10.0 * unit_grad)


def test_balanced_encoder_mode_hits_target_and_runs_without_logging() -> None:
    kwargs = dict(
        idm_coef=1.0,
        idm_encoder_mode="balanced",
        idm_grad_ratio_target=0.05,
        idm_grad_ratio_ema=0.0,
        idm_coef_min=1e-4,
        idm_coef_max=1e4,
        idm_coef_slew_rate=1e4,
        idm_diagnostics_interval=1,
    )
    agent = _make_agent(**kwargs)
    metrics = _update_fb(agent, _fixed_batch())
    expected = (
        agent.cfg.idm_grad_ratio_target
        * metrics["encoder_grad_norm_fb"]
        / (metrics["encoder_grad_norm_idm_unweighted"] + 1e-12)
    )
    assert metrics["idm_encoder_coef_used"] == 1.0
    assert metrics["idm_weighted_loss"] == pytest.approx(metrics["idm_loss"])
    assert metrics["total_loss"] == pytest.approx(
        metrics["fb_loss"] + metrics["idm_weighted_loss"]
    )
    assert metrics["idm_encoder_coef_next"] == pytest.approx(expected)
    assert agent._idm_effective_coef == pytest.approx(expected)  # pylint: disable=protected-access

    silent = _make_agent(**kwargs, logging_enabled=False)
    with mock.patch("torch.autograd.grad", wraps=torch.autograd.grad) as functional_grad:
        silent_metrics = _update_fb(silent, _fixed_batch())
    assert functional_grad.call_count == 2
    assert silent_metrics == {}
    assert silent._idm_effective_coef != 1.0  # pylint: disable=protected-access


def test_encoder_only_burnin_reports_raw_gradient_while_encoder_weight_is_zero() -> None:
    agent = _make_agent(
        idm_coef=10.0,
        idm_encoder_mode="static",
        idm_encoder_burnin_steps=5,
    )
    metrics = _update_fb(agent, _fixed_batch())
    assert metrics["idm_encoder_coef_used"] == 0.0
    assert metrics["idm_weighted_loss"] == pytest.approx(metrics["idm_loss"])
    assert metrics["idm_encoder_loss_proxy"] == 0.0
    assert metrics["encoder_grad_norm_idm_unweighted"] > 0.0
    assert metrics["encoder_grad_norm_idm_weighted"] == 0.0
    assert metrics["encoder_grad_ratio_idm_fb"] == 0.0
    assert metrics["idm_head_grad_norm"] > 0.0


def test_balanced_burnin_keeps_coef_next_in_first_logger_schema() -> None:
    agent = _make_agent(
        idm_coef=1.0,
        idm_encoder_mode="balanced",
        idm_encoder_burnin_steps=5,
        idm_grad_ratio_target=0.01,
        idm_diagnostics_interval=1,
    )
    burnin_metrics = _update_fb(agent, _fixed_batch(), step=0)
    assert burnin_metrics["idm_encoder_coef_used"] == 0.0
    assert burnin_metrics["idm_encoder_coef_next"] == 1.0

    balanced_metrics = _update_fb(agent, _fixed_batch(), step=5)
    assert burnin_metrics.keys() == balanced_metrics.keys()
    assert balanced_metrics["idm_encoder_coef_used"] == 1.0


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


def test_idm_prediction_representation_and_gradient_diagnostics() -> None:
    torch.manual_seed(23)
    agent = _make_agent(idm_coef=0.3, idm_diagnostics_interval=2)
    assert agent.idm_head is not None
    assert agent.encoder_opt is not None
    assert agent.idm_optimizer is not None

    raw_obs, raw_next_obs, action, discount, z = _fixed_batch()
    obs = agent.aug_and_encode(raw_obs)
    next_obs = agent.aug_and_encode(raw_next_obs)
    with torch.no_grad():
        prediction = agent._predict_idm_action(obs, next_obs)  # pylint: disable=protected-access
        error = prediction - action
        action_var = action.var(dim=0, unbiased=False).mean()
        pred_var = prediction.var(dim=0, unbiased=False).mean()
        error_var = error.var(dim=0, unbiased=False).mean()
        expected_prediction_metrics = {
            "idm_action_mae": error.abs().mean().item(),
            "idm_nmse": (F.mse_loss(prediction, action) / (action_var + 1e-12)).item(),
            "idm_explained_variance": (1.0 - error_var / (action_var + 1e-12)).item(),
            "action_std": action_var.sqrt().item(),
            "idm_pred_std": pred_var.sqrt().item(),
            "h_norm": (0.5 * (obs.norm(dim=-1).mean() + next_obs.norm(dim=-1).mean())).item(),
            "delta_h_norm": (next_obs - obs).norm(dim=-1).mean().item(),
        }

    encoder_grad_at_step = []
    idm_head_grad_at_step = []
    functional_encoder_grads = []
    original_encoder_step = agent.encoder_opt.step
    original_idm_step = agent.idm_optimizer.step
    original_functional_grad = torch.autograd.grad

    def encoder_step(*args, **kwargs):
        encoder_grad_at_step.append(
            tuple(
                None if parameter.grad is None else parameter.grad.detach().clone()
                for parameter in agent.encoder.parameters()
            )
        )
        return original_encoder_step(*args, **kwargs)

    def idm_step(*args, **kwargs):
        idm_head_grad_at_step.append(
            tuple(
                None if parameter.grad is None else parameter.grad.detach().clone()
                for parameter in agent.idm_head.parameters()
            )
        )
        return original_idm_step(*args, **kwargs)

    def functional_grad_call(*args, **kwargs):
        gradients = original_functional_grad(*args, **kwargs)
        functional_encoder_grads.append(
            tuple(None if grad is None else grad.detach().clone() for grad in gradients)
        )
        return gradients

    with mock.patch.object(agent.encoder_opt, "step", side_effect=encoder_step) as encoder_step_mock, \
            mock.patch.object(agent.idm_optimizer, "step", side_effect=idm_step) as idm_step_mock, \
            mock.patch("torch.autograd.grad", side_effect=functional_grad_call) as functional_grad:
        torch.manual_seed(99)
        metrics = agent.update_fb(
            obs=obs,
            action=action,
            discount=discount,
            next_obs=next_obs,
            next_goal=next_obs,
            target_next_goal=next_obs.detach(),
            z=z,
            step=0,
        )

    assert functional_grad.call_count == 2
    assert encoder_step_mock.call_count == 1
    assert idm_step_mock.call_count == 1
    assert metrics["idm_weighted_loss"] == pytest.approx(
        agent.cfg.idm_coef * metrics["idm_loss"]
    )
    for key, expected in expected_prediction_metrics.items():
        assert metrics[key] == pytest.approx(expected)

    assert metrics["encoder_grad_norm_idm_weighted"] == pytest.approx(
        agent.cfg.idm_coef * metrics["encoder_grad_norm_idm_unweighted"]
    )
    assert metrics["encoder_grad_ratio_idm_fb"] == pytest.approx(
        metrics["encoder_grad_norm_idm_weighted"]
        / (metrics["encoder_grad_norm_fb"] + 1e-12)
    )
    fb_gradient, idm_gradient = functional_encoder_grads
    expected_cosine = agent.cfg.idm_coef * fb_ddpg._tensor_grad_dot(
        fb_gradient, idm_gradient
    ) / (
        fb_ddpg._tensor_grad_norm(fb_gradient)
        * agent.cfg.idm_coef
        * fb_ddpg._tensor_grad_norm(idm_gradient)
        + 1e-12
    )
    assert metrics["encoder_grad_norm_fb"] == pytest.approx(
        fb_ddpg._tensor_grad_norm(fb_gradient)
    )
    assert metrics["encoder_grad_norm_idm_unweighted"] == pytest.approx(
        fb_ddpg._tensor_grad_norm(idm_gradient)
    )
    assert metrics["encoder_grad_cosine_fb_idm"] == pytest.approx(expected_cosine)

    actual_total_gradient = encoder_grad_at_step[0]
    for actual, fb_grad, idm_grad in zip(
        actual_total_gradient, fb_gradient, idm_gradient
    ):
        assert actual is not None
        expected = torch.zeros_like(actual)
        if fb_grad is not None:
            expected.add_(fb_grad)
        if idm_grad is not None:
            expected.add_(idm_grad, alpha=agent.cfg.idm_coef)
        torch.testing.assert_close(actual, expected)
    total_gradient_norm = fb_ddpg._tensor_grad_norm(actual_total_gradient)
    head_gradient_norm = fb_ddpg._tensor_grad_norm(idm_head_grad_at_step[0])
    assert metrics["encoder_grad_norm_total"] == pytest.approx(total_gradient_norm)
    assert metrics["encoder_grad_norm"] == pytest.approx(total_gradient_norm)
    assert metrics["idm_head_grad_norm"] == pytest.approx(head_gradient_norm)
    assert metrics["encoder_grad_norm_fb"] > 0
    assert metrics["encoder_grad_norm_idm_unweighted"] > 0
    assert metrics["idm_head_grad_norm"] > 0


def test_idm_diagnostics_are_sparse_and_do_not_change_training_gradient() -> None:
    torch.manual_seed(29)
    diagnosed = _make_agent(idm_coef=0.1, idm_diagnostics_interval=500)
    torch.manual_seed(29)
    undiagnosed = _make_agent(idm_coef=0.1, idm_diagnostics_interval=500)
    # Both agents start identically, but only the zero-indexed first update is
    # due for diagnostics.
    undiagnosed._idm_update_count = 1  # pylint: disable=protected-access
    undiagnosed._idm_diagnostics_pending = False  # pylint: disable=protected-access

    for name in ("encoder", "forward_net", "backward_net", "idm_head"):
        _assert_module_equal(getattr(diagnosed, name), getattr(undiagnosed, name))
    for name in ("encoder_opt", "fb_opt", "idm_optimizer"):
        _assert_nested_equal(
            getattr(diagnosed, name).state_dict(),
            getattr(undiagnosed, name).state_dict(),
        )

    def update_and_capture_encoder_gradient(agent):
        assert agent.encoder_opt is not None
        captured = []
        original_step = agent.encoder_opt.step

        def step(*args, **kwargs):
            captured.append(
                tuple(
                    None if parameter.grad is None else parameter.grad.detach().clone()
                    for parameter in agent.encoder.parameters()
                )
            )
            return original_step(*args, **kwargs)

        with mock.patch.object(agent.encoder_opt, "step", side_effect=step):
            update_metrics = _update_fb(agent, _fixed_batch(), seed=101)
        assert len(captured) == 1
        return update_metrics, captured[0]

    diagnosed_metrics, diagnosed_gradient = update_and_capture_encoder_gradient(diagnosed)
    with mock.patch(
        "torch.autograd.grad",
        side_effect=AssertionError("functional diagnostics unexpectedly ran"),
    ):
        undiagnosed_metrics, undiagnosed_gradient = update_and_capture_encoder_gradient(undiagnosed)

    diagnostic_keys = {
        "encoder_grad_norm_fb",
        "encoder_grad_norm_idm_unweighted",
        "encoder_grad_norm_idm_weighted",
        "encoder_grad_norm_total",
        "encoder_grad_ratio_idm_fb",
        "encoder_grad_cosine_fb_idm",
        "idm_head_grad_norm",
    }
    assert diagnostic_keys <= diagnosed_metrics.keys()
    assert diagnostic_keys.isdisjoint(undiagnosed_metrics.keys())
    _assert_nested_equal(diagnosed_gradient, undiagnosed_gradient)
    for name in ("encoder", "forward_net", "backward_net", "idm_head"):
        _assert_module_equal(getattr(diagnosed, name), getattr(undiagnosed, name))
    for name in ("encoder_opt", "fb_opt", "idm_optimizer"):
        _assert_nested_equal(
            getattr(diagnosed, name).state_dict(),
            getattr(undiagnosed, name).state_dict(),
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
    assert restored._idm_diagnostics_pending  # pylint: disable=protected-access
    _assert_module_equal(source.idm_head, restored.idm_head)
    _assert_nested_equal(
        source.idm_optimizer.state_dict(),
        restored.idm_optimizer.state_dict(),
    )

    # Loading optimizer state is only useful if the resumed trajectory remains
    # identical.  Exercise one more update from the restored checkpoint state.
    # Suppress the restored process's one-time schema diagnostic here so the
    # complete metric dictionaries are directly comparable; a dedicated test
    # below verifies that the diagnostic normally runs.
    restored._idm_diagnostics_pending = False  # pylint: disable=protected-access
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
    del legacy._idm_update_count  # pylint: disable=protected-access
    del legacy.cfg.idm_coef
    del legacy.cfg.idm_lr
    head_before = {
        key: value.detach().clone()
        for key, value in restored.idm_head.state_dict().items()
    }
    restored.init_from(legacy)
    assert restored._idm_update_count == 0  # pylint: disable=protected-access
    assert restored._idm_diagnostics_pending  # pylint: disable=protected-access
    for key, value in restored.idm_head.state_dict().items():
        torch.testing.assert_close(value, head_before[key], rtol=0, atol=0)


def test_idm_resume_forces_diagnostics_on_first_post_load_update() -> None:
    source = _make_agent(idm_coef=0.1, idm_diagnostics_interval=500)
    _update_fb(source, _fixed_batch())
    assert source._idm_update_count == 1  # pylint: disable=protected-access
    assert not source._idm_diagnostics_pending  # pylint: disable=protected-access

    restored = _make_agent(idm_coef=0.1, idm_diagnostics_interval=500)
    restored.init_from(source)
    assert restored._idm_update_count == 1  # pylint: disable=protected-access
    assert restored._idm_diagnostics_pending  # pylint: disable=protected-access

    with mock.patch("torch.autograd.grad", wraps=torch.autograd.grad) as functional_grad:
        metrics = _update_fb(restored, _fixed_batch(), seed=2718)

    assert functional_grad.call_count == 2
    assert "encoder_grad_norm_fb" in metrics
    assert "encoder_grad_norm_idm_unweighted" in metrics
    assert not restored._idm_diagnostics_pending  # pylint: disable=protected-access


def test_balanced_idm_resume_restores_controller_state_and_rejects_mismatch() -> None:
    kwargs = dict(
        idm_coef=1.0,
        idm_encoder_mode="balanced",
        idm_grad_ratio_target=0.01,
        idm_diagnostics_interval=1,
    )
    source = _make_agent(**kwargs)
    _update_fb(source, _fixed_batch())
    restored = _make_agent(**kwargs)
    restored.init_from(source)
    assert restored._idm_update_count == source._idm_update_count  # pylint: disable=protected-access
    assert restored._idm_effective_coef == source._idm_effective_coef  # pylint: disable=protected-access
    assert restored._idm_fb_grad_norm_ema == source._idm_fb_grad_norm_ema  # pylint: disable=protected-access
    assert restored._idm_raw_grad_norm_ema == source._idm_raw_grad_norm_ema  # pylint: disable=protected-access

    continuation = _fixed_batch()
    source_metrics = _update_fb(source, continuation, seed=123)
    restored_metrics = _update_fb(restored, continuation, seed=123)
    assert source_metrics == restored_metrics
    for name in ("encoder", "forward_net", "backward_net", "idm_head"):
        _assert_module_equal(getattr(source, name), getattr(restored, name))
    for name in ("encoder_opt", "fb_opt", "idm_optimizer"):
        _assert_nested_equal(
            getattr(source, name).state_dict(),
            getattr(restored, name).state_dict(),
        )
    assert restored._idm_effective_coef == source._idm_effective_coef  # pylint: disable=protected-access
    assert restored._idm_fb_grad_norm_ema == source._idm_fb_grad_norm_ema  # pylint: disable=protected-access
    assert restored._idm_raw_grad_norm_ema == source._idm_raw_grad_norm_ema  # pylint: disable=protected-access

    mismatched = _make_agent(**{**kwargs, "idm_grad_ratio_target": 0.05})
    encoder_before = copy.deepcopy(mismatched.encoder.state_dict())
    with pytest.raises(ValueError, match="idm_grad_ratio_target"):
        mismatched.init_from(source)
    _assert_nested_equal(encoder_before, mismatched.encoder.state_dict())

    malformed = _make_agent(**kwargs)
    del malformed._idm_update_count  # pylint: disable=protected-access
    destination = _make_agent(**kwargs)
    with pytest.raises(ValueError, match="_idm_update_count"):
        destination.init_from(malformed)


@pytest.mark.parametrize("missing", ["idm_head", "idm_optimizer"])
def test_idm_init_from_rejects_incomplete_idm_checkpoint(missing: str) -> None:
    source = _make_agent(idm_coef=0.1)
    destination = _make_agent(idm_coef=0.1)
    delattr(source, missing)
    with pytest.raises(ValueError, match=missing):
        destination.init_from(source)


@pytest.mark.parametrize(
    ("source_coef", "source_lr", "destination_coef", "destination_lr", "match"),
    [
        (0.1, 1e-4, 0.3, 1e-4, "idm_coef"),
        (0.1, 1e-4, 0.0, 1e-4, "idm_coef"),
        (0.0, 1e-4, 0.1, 1e-4, "idm_coef"),
        (0.1, 1e-4, 0.1, 3e-4, "effective idm_lr"),
    ],
)
def test_idm_init_from_rejects_sweep_mismatch_before_copying(
    source_coef: float,
    source_lr: float,
    destination_coef: float,
    destination_lr: float,
    match: str,
) -> None:
    torch.manual_seed(31)
    source = _make_agent(idm_coef=source_coef, idm_lr=source_lr)
    _update_fb(source, _fixed_batch())
    torch.manual_seed(37)
    destination = _make_agent(idm_coef=destination_coef, idm_lr=destination_lr)
    destination_before = {
        name: copy.deepcopy(module.state_dict())
        for name, module in (
            ("encoder", destination.encoder),
            ("forward_net", destination.forward_net),
            ("backward_net", destination.backward_net),
            ("idm_head", destination.idm_head),
        )
        if module is not None
    }

    with pytest.raises(ValueError, match=match):
        destination.init_from(source)

    for name, state in destination_before.items():
        _assert_nested_equal(state, getattr(destination, name).state_dict())


def test_idm_init_from_accepts_equivalent_effective_lr_and_rejects_optimizer_lr_drift() -> None:
    source = _make_agent(idm_coef=0.1, idm_lr=None)
    destination = _make_agent(idm_coef=0.1, idm_lr=source.cfg.lr)
    destination.init_from(source)

    assert source.idm_optimizer is not None
    source.idm_optimizer.param_groups[0]["lr"] = 3e-4
    destination = _make_agent(idm_coef=0.1, idm_lr=source.cfg.lr)
    with pytest.raises(ValueError, match="optimizer/config mismatch"):
        destination.init_from(source)


def test_idm_init_from_does_not_apply_idm_guards_to_disabled_agents() -> None:
    source = _make_agent(idm_coef=0.0, lr=1e-4)
    destination = _make_agent(idm_coef=0.0, lr=3e-4)
    # There is no IDM head or IDM optimizer whose sweep configuration could be
    # corrupted, so ordinary baseline warm-start behavior remains unchanged.
    destination.init_from(source)
