# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
import typing as tp
from unittest import mock

import numpy as np
import pytest
import torch
from torch import nn

from url_benchmark import replay_buffer as rb
from . import fb_ddpg

# REDUNDANCY REVIEW: see scripts/check_idm_adapter_routing.py's check_contract() --
# its _check_architecture / _check_optimizer_ownership / _check_isolated_idm /
# _check_complete_update helpers duplicate most of what this file's tests already assert.


def _make_agent(
    route: str = "none",
    idm_coef: float = 0.0,
    *,
    separate: bool = True,
    adapter_hidden_dim: int = 16,
    adapter_output_dim: int = 8,
    adapter_type: str = "mlp_ln",
    logging_enabled: bool = True,
) -> fb_ddpg.FBDDPGAgent:
    cfg = fb_ddpg.FBDDPGAgentConfig(
        obs_shape=(11,),
        action_shape=(2,),
        obs_type="dino",
        device="cpu",
        lr=1e-4,
        num_expl_steps=0,
        goal_space=None,
        use_cls=True,
        use_tb=logging_enabled,
        use_wandb=False,
        use_hiplog=False,
        update_encoder=True,
        batch_size=4,
        hidden_dim=16,
        backward_hidden_dim=16,
        feature_dim=8,
        z_dim=4,
        dino_use_adapter=True,
        dino_adapter_type=adapter_type,
        dino_adapter_hidden_dim=adapter_hidden_dim,
        dino_adapter_output_dim=adapter_output_dim,
        dino_separate_fb_adapters=separate,
        dino_separate_backward_adapter=False,
        idm_coef=idm_coef,
        idm_lr=1e-4 if idm_coef > 0 else None,
        idm_encoder_mode="static" if route != "none" else "legacy",
        idm_route=route,
        idm_diagnostics_interval=1,
        ortho_coef=1.0,
        mix_ratio=0.5,
    )
    return fb_ddpg.FBDDPGAgent(**dataclasses.asdict(cfg))


def _module_parameters(module: nn.Module) -> tp.List[torch.Tensor]:
    return [parameter.detach().clone() for parameter in module.parameters()]


def _module_unchanged(before: tp.Sequence[torch.Tensor], module: nn.Module) -> bool:
    return all(
        torch.equal(expected, actual.detach())
        for expected, actual in zip(before, module.parameters())
    )


def _module_changed(before: tp.Sequence[torch.Tensor], module: nn.Module) -> bool:
    return any(
        not torch.equal(expected, actual.detach())
        for expected, actual in zip(before, module.parameters())
    )


def _has_nonzero_gradient(module: nn.Module) -> bool:
    return any(
        parameter.grad is not None
        and torch.count_nonzero(parameter.grad.detach()).item() > 0
        for parameter in module.parameters()
    )


def _has_no_gradient(module: nn.Module) -> bool:
    return all(parameter.grad is None for parameter in module.parameters())


def _zero_all_gradients(agent: fb_ddpg.FBDDPGAgent) -> None:
    for optimizer in (
        agent.forward_fb_opt,
        agent.backward_fb_opt,
        agent.actor_opt,
        agent.idm_optimizer,
        agent.encoder_opt,
        agent.backward_encoder_opt,
        agent.fb_opt,
    ):
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)


def _fixed_tensors() -> tp.Tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(20260831)
    raw_obs = torch.randn(4, 11, generator=generator)
    raw_next_obs = torch.randn(4, 11, generator=generator)
    action = torch.randn(4, 2, generator=generator).tanh()
    discount = torch.full((4, 1), 0.99)
    z = torch.randn(4, 4, generator=generator)
    return raw_obs, raw_next_obs, action, discount, z


def _optimizer_parameter_ids(
    optimizer: torch.optim.Optimizer,
) -> tp.Set[int]:
    return {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }


def test_mlp_ln_has_requested_shape_and_distinct_identical_copies() -> None:
    torch.manual_seed(1)
    agent = _make_agent(
        adapter_hidden_dim=1024,
        adapter_output_dim=512,
    )
    assert agent.forward_adapter is not None
    assert agent.backward_adapter is not None
    forward = agent.forward_adapter
    backward = agent.backward_adapter

    assert isinstance(forward, nn.Sequential)
    assert tuple(type(layer) for layer in forward) == (
        nn.LayerNorm,
        nn.Linear,
        nn.GELU,
        nn.Linear,
        nn.LayerNorm,
    )
    assert forward[0].normalized_shape == (11,)
    assert (forward[1].in_features, forward[1].out_features) == (11, 1024)
    assert (forward[3].in_features, forward[3].out_features) == (1024, 512)
    assert forward[4].normalized_shape == (512,)
    assert forward.state_dict().keys() == backward.state_dict().keys()
    for name, value in forward.state_dict().items():
        torch.testing.assert_close(value, backward.state_dict()[name], rtol=0, atol=0)
    for forward_parameter, backward_parameter in zip(
        forward.parameters(), backward.parameters()
    ):
        assert forward_parameter is not backward_parameter
        assert forward_parameter.data_ptr() != backward_parameter.data_ptr()


def test_explicit_route_validation_and_legacy_shared_compatibility() -> None:
    assert fb_ddpg.FBDDPGAgentConfig().idm_route == "none"
    with pytest.raises(ValueError, match="explicit idm_route"):
        _make_agent(route="none", idm_coef=0.1)
    with pytest.raises(ValueError, match="dino_separate_fb_adapters=True"):
        _make_agent(route="forward_adapter", idm_coef=0.1, separate=False)
    with pytest.raises(ValueError, match="idm_route must be one of"):
        _make_agent(route="typo", idm_coef=0.1)

    # Old IDM configs had no route field and used the one shared DINO adapter.
    legacy = _make_agent(route="none", idm_coef=0.1, separate=False)
    assert legacy.idm_head is not None
    assert legacy.encoder_opt is not None
    assert legacy.cfg.idm_route == "none"


def test_optimizer_ownership_is_exact_and_globally_disjoint() -> None:
    agent = _make_agent(route="forward_adapter", idm_coef=0.1)
    assert agent.forward_adapter is not None
    assert agent.backward_adapter is not None
    assert agent.forward_fb_opt is not None
    assert agent.backward_fb_opt is not None
    assert agent.idm_optimizer is not None

    forward_ids = _optimizer_parameter_ids(agent.forward_fb_opt)
    backward_ids = _optimizer_parameter_ids(agent.backward_fb_opt)
    actor_ids = _optimizer_parameter_ids(agent.actor_opt)
    idm_ids = _optimizer_parameter_ids(agent.idm_optimizer)
    assert forward_ids == {
        id(parameter)
        for module in (agent.forward_net, agent.forward_adapter)
        for parameter in module.parameters()
    }
    assert backward_ids == {
        id(parameter)
        for module in (agent.backward_net, agent.backward_adapter)
        for parameter in module.parameters()
    }
    assert actor_ids == {id(parameter) for parameter in agent.actor.parameters()}
    assert agent.idm_head is not None
    assert idm_ids == {id(parameter) for parameter in agent.idm_head.parameters()}
    all_optimizer_ids = [forward_ids, backward_ids, actor_ids, idm_ids]
    for index, left in enumerate(all_optimizer_ids):
        for right in all_optimizer_ids[index + 1:]:
            assert left.isdisjoint(right)
    assert agent.fb_opt is None
    assert agent.encoder_opt is None
    assert agent.backward_encoder_opt is None


@pytest.mark.parametrize(
    ("route", "selected_name", "route_id"),
    [
        ("forward_adapter", "forward_adapter", 1.0),
        ("backward_adapter", "backward_adapter", 2.0),
    ],
)
def test_isolated_idm_updates_only_selected_adapter_and_head(
    route: str,
    selected_name: str,
    route_id: float,
) -> None:
    torch.manual_seed(3)
    agent = _make_agent(route=route, idm_coef=0.1)
    assert agent.forward_adapter is not None
    assert agent.backward_adapter is not None
    assert agent.idm_head is not None
    assert agent.idm_optimizer is not None
    selected = getattr(agent, selected_name)
    unselected = (
        agent.backward_adapter
        if route == "forward_adapter"
        else agent.forward_adapter
    )
    branch_optimizer = (
        agent.forward_fb_opt
        if route == "forward_adapter"
        else agent.backward_fb_opt
    )
    assert selected is not None and unselected is not None
    assert branch_optimizer is not None

    modules = {
        "selected": selected,
        "unselected": unselected,
        "head": agent.idm_head,
        "forward_map": agent.forward_net,
        "backward_map": agent.backward_net,
        "actor": agent.actor,
    }
    before = {name: _module_parameters(module) for name, module in modules.items()}
    raw_obs, raw_next_obs, action, _, _ = _fixed_tensors()
    _zero_all_gradients(agent)
    h_t = selected(raw_obs)
    h_tp1 = selected(raw_next_obs)
    idm_loss, _, coefficient = agent._compute_idm_objective(  # pylint: disable=protected-access
        h_t,
        h_tp1,
        action,
        step=0,
    )
    assert coefficient == 0.1
    assert torch.isfinite(idm_loss)
    idm_loss.backward()

    assert _has_nonzero_gradient(selected)
    assert _has_nonzero_gradient(agent.idm_head)
    for module in (
        unselected,
        agent.forward_net,
        agent.backward_net,
        agent.actor,
    ):
        assert _has_no_gradient(module)

    branch_optimizer.step()
    agent.idm_optimizer.step()
    assert _module_changed(before["selected"], selected)
    assert _module_changed(before["head"], agent.idm_head)
    for name in ("unselected", "forward_map", "backward_map", "actor"):
        assert _module_unchanged(before[name], modules[name])
    assert fb_ddpg.IDM_ROUTE_IDS[route] == route_id


def test_static_route_scales_adapter_gradient_but_not_head_gradient() -> None:
    raw_obs, raw_next_obs, action, _, _ = _fixed_tensors()

    def gradients(coefficient: float):
        torch.manual_seed(5)
        agent = _make_agent(route="forward_adapter", idm_coef=coefficient)
        assert agent.forward_adapter is not None
        assert agent.idm_head is not None
        h_t = agent.forward_adapter(raw_obs)
        h_tp1 = agent.forward_adapter(raw_next_obs)
        loss, _, used = agent._compute_idm_objective(  # pylint: disable=protected-access
            h_t,
            h_tp1,
            action,
            step=0,
        )
        loss.backward()
        adapter_grads = tuple(
            parameter.grad.detach().clone()
            for parameter in agent.forward_adapter.parameters()
        )
        head_grads = tuple(
            parameter.grad.detach().clone()
            for parameter in agent.idm_head.parameters()
        )
        return loss.detach(), used, adapter_grads, head_grads

    scaled_loss, scaled_used, scaled_adapter, scaled_head = gradients(0.1)
    unit_loss, unit_used, unit_adapter, unit_head = gradients(1.0)
    torch.testing.assert_close(scaled_loss, unit_loss, rtol=0, atol=0)
    assert scaled_used == 0.1
    assert unit_used == 1.0
    for scaled, unit in zip(scaled_head, unit_head):
        torch.testing.assert_close(scaled, unit, rtol=0, atol=0)
    for scaled, unit in zip(scaled_adapter, unit_adapter):
        torch.testing.assert_close(scaled, 0.1 * unit)


@pytest.mark.parametrize(
    ("route", "route_id"),
    [("forward_adapter", 1.0), ("backward_adapter", 2.0)],
)
def test_full_fb_update_has_finite_route_and_gradient_diagnostics(
    route: str,
    route_id: float,
) -> None:
    torch.manual_seed(7)
    agent = _make_agent(route=route, idm_coef=0.1)
    assert agent.forward_adapter is not None
    assert agent.backward_adapter is not None
    raw_obs, raw_next_obs, action, discount, z = _fixed_tensors()
    obs = agent.forward_adapter(raw_obs)
    next_obs = agent.forward_adapter(raw_next_obs)
    backward_obs = agent.backward_adapter(raw_obs)
    backward_next_obs = agent.backward_adapter(raw_next_obs)
    idm_obs, idm_next_obs = (
        (obs, next_obs)
        if route == "forward_adapter"
        else (backward_obs, backward_next_obs)
    )
    metrics = agent.update_fb(
        obs=obs,
        action=action,
        discount=discount,
        next_obs=next_obs,
        next_goal=backward_next_obs,
        target_next_goal=backward_next_obs.detach(),
        z=z,
        step=0,
        idm_obs=idm_obs,
        idm_next_obs=idm_next_obs,
    )
    required = {
        "idm_loss",
        "idm_weighted_loss",
        "idm_route",
        "idm_encoder_coef_used",
        "forward_adapter_grad_norm",
        "backward_adapter_grad_norm",
        "idm_adapter_grad_norm",
        "fb_adapter_grad_norm",
        "idm_fb_adapter_grad_cosine",
        "idm_head_grad_norm",
        "fb_loss",
        "total_loss",
    }
    assert required <= metrics.keys()
    assert metrics["idm_route"] == route_id
    assert metrics["idm_encoder_coef_used"] == 0.1
    assert metrics["idm_weighted_loss"] == pytest.approx(
        0.1 * metrics["idm_loss"]
    )
    assert all(np.isfinite(metrics[name]) for name in metrics)
    assert metrics["forward_adapter_grad_norm"] > 0
    assert metrics["backward_adapter_grad_norm"] > 0
    assert metrics["idm_adapter_grad_norm"] > 0
    assert metrics["fb_adapter_grad_norm"] > 0


def test_orth_only_and_actor_updates_respect_branch_boundaries() -> None:
    torch.manual_seed(11)
    agent = _make_agent()
    assert agent.forward_adapter is not None
    assert agent.backward_adapter is not None
    raw_obs, _, _, _, z = _fixed_tensors()

    _zero_all_gradients(agent)
    backward_embedding = agent.backward_net(agent.backward_adapter(raw_obs))
    covariance = backward_embedding @ backward_embedding.T
    off_diagonal = ~torch.eye(4, dtype=torch.bool)
    orth_loss = (
        covariance[off_diagonal].square().mean()
        - 2.0 * covariance.diag().mean()
    )
    orth_loss.backward()
    assert _has_nonzero_gradient(agent.backward_adapter)
    assert _has_nonzero_gradient(agent.backward_net)
    assert _has_no_gradient(agent.forward_adapter)
    assert _has_no_gradient(agent.forward_net)

    backward_before = _module_parameters(agent.backward_adapter)
    forward_before = _module_parameters(agent.forward_adapter)
    obs = agent.forward_adapter(raw_obs).detach()
    agent.update_actor(obs, z, step=0)
    assert _module_unchanged(backward_before, agent.backward_adapter)
    assert _module_unchanged(forward_before, agent.forward_adapter)


class _Replay:
    def __init__(self, batch: rb.EpisodeBatch) -> None:
        self.batch = batch

    def sample(self, batch_size: int) -> rb.EpisodeBatch:
        assert batch_size == self.batch.obs.shape[0]
        return self.batch


@pytest.mark.parametrize("route", ["forward_adapter", "backward_adapter"])
def test_update_routes_raw_transition_through_the_selected_adapter(route: str) -> None:
    torch.manual_seed(13)
    agent = _make_agent(route=route, idm_coef=0.1)
    assert agent.forward_adapter is not None
    assert agent.backward_adapter is not None
    rng = np.random.RandomState(13)
    batch = rb.EpisodeBatch(
        obs=rng.randn(4, 11).astype(np.float32),
        action=rng.uniform(-1, 1, size=(4, 2)).astype(np.float32),
        reward=rng.randn(4, 1).astype(np.float32),
        next_obs=rng.randn(4, 11).astype(np.float32),
        discount=np.full((4, 1), 0.99, dtype=np.float32),
        future_obs=rng.randn(4, 11).astype(np.float32),
    )
    forward_outputs: tp.List[torch.Tensor] = []
    backward_outputs: tp.List[torch.Tensor] = []
    forward_hook = agent.forward_adapter.register_forward_hook(
        lambda _module, _inputs, output: forward_outputs.append(output)
    )
    backward_hook = agent.backward_adapter.register_forward_hook(
        lambda _module, _inputs, output: backward_outputs.append(output)
    )
    try:
        with mock.patch.object(agent, "update_fb", return_value={}) as update_fb, \
                mock.patch.object(agent, "update_actor", return_value={}):
            agent.update(_Replay(batch), step=0)  # type: ignore[arg-type]
    finally:
        forward_hook.remove()
        backward_hook.remove()

    assert update_fb.call_count == 1
    routed_obs = update_fb.call_args.kwargs["idm_obs"]
    routed_next_obs = update_fb.call_args.kwargs["idm_next_obs"]
    assert len(forward_outputs) == 2
    assert len(backward_outputs) == 3
    if route == "forward_adapter":
        assert routed_obs is forward_outputs[0]
        assert routed_next_obs is forward_outputs[1]
    else:
        # B calls are next_obs, obs, future_obs. The IDM pair deliberately uses
        # the adapter outputs and never the final BackwardMap B(s) embedding.
        assert routed_obs is backward_outputs[1]
        assert routed_next_obs is backward_outputs[0]
        assert routed_obs.shape[-1] == agent.obs_dim
        assert routed_obs.shape[-1] != agent.cfg.z_dim


def test_checkpoint_restore_rejects_idm_route_change() -> None:
    torch.manual_seed(17)
    source = _make_agent(route="forward_adapter", idm_coef=0.1)
    torch.manual_seed(17)
    destination = _make_agent(route="backward_adapter", idm_coef=0.1)
    with pytest.raises(ValueError, match="idm_route"):
        destination.init_from(source)
