#!/usr/bin/env python3
"""Dependency-free preflight for separate DINO adapters and IDM routing."""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path
import sys
import typing as tp

import torch
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from url_benchmark.agent import fb_ddpg  # noqa: E402


def _make_agent(route: str) -> fb_ddpg.FBDDPGAgent:
    enabled = route != "none"
    cfg = fb_ddpg.FBDDPGAgentConfig(
        # An unusual runtime width makes accidental CLS3/2304 hard-coding fail.
        obs_shape=(21,),
        action_shape=(2,),
        obs_type="dino",
        device="cpu",
        lr=1e-4,
        num_expl_steps=0,
        goal_space=None,
        use_cls=True,
        use_tb=True,
        use_wandb=False,
        use_hiplog=False,
        update_encoder=True,
        batch_size=4,
        hidden_dim=16,
        backward_hidden_dim=16,
        feature_dim=8,
        z_dim=4,
        dino_use_adapter=True,
        dino_adapter_type="mlp_ln",
        dino_adapter_hidden_dim=1024,
        dino_adapter_output_dim=512,
        dino_separate_fb_adapters=True,
        dino_separate_backward_adapter=False,
        idm_coef=0.1 if enabled else 0.0,
        idm_lr=1e-4 if enabled else None,
        idm_encoder_mode="static" if enabled else "legacy",
        idm_route=route,
        idm_diagnostics_interval=1,
        ortho_coef=1.0,
        mix_ratio=0.5,
    )
    return fb_ddpg.FBDDPGAgent(**dataclasses.asdict(cfg))


def _parameter_ids(module: nn.Module) -> tp.Set[int]:
    return {id(parameter) for parameter in module.parameters()}


def _optimizer_parameter_ids(
    optimizer: torch.optim.Optimizer,
) -> tp.Set[int]:
    return {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }


def _clone(module: nn.Module) -> tp.List[torch.Tensor]:
    return [parameter.detach().clone() for parameter in module.parameters()]


def _unchanged(before: tp.Sequence[torch.Tensor], module: nn.Module) -> bool:
    return all(
        torch.equal(expected, actual.detach())
        for expected, actual in zip(before, module.parameters())
    )


def _changed(before: tp.Sequence[torch.Tensor], module: nn.Module) -> bool:
    return any(
        not torch.equal(expected, actual.detach())
        for expected, actual in zip(before, module.parameters())
    )


def _has_gradient(module: nn.Module) -> bool:
    return any(
        parameter.grad is not None
        and torch.count_nonzero(parameter.grad.detach()).item() > 0
        for parameter in module.parameters()
    )


def _no_gradient(module: nn.Module) -> bool:
    return all(parameter.grad is None for parameter in module.parameters())


def _zero_gradients(agent: fb_ddpg.FBDDPGAgent) -> None:
    for optimizer in (
        agent.forward_fb_opt,
        agent.backward_fb_opt,
        agent.actor_opt,
        agent.idm_optimizer,
    ):
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)


def _batch() -> tp.Tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(20260831)
    return (
        torch.randn(4, 21, generator=generator),
        torch.randn(4, 21, generator=generator),
        torch.randn(4, 2, generator=generator).tanh(),
        torch.full((4, 1), 0.99),
        torch.randn(4, 4, generator=generator),
    )


def _check_architecture(agent: fb_ddpg.FBDDPGAgent) -> None:
    forward = agent.forward_adapter
    backward = agent.backward_adapter
    assert isinstance(forward, nn.Sequential)
    assert isinstance(backward, nn.Sequential)
    assert tuple(type(layer) for layer in forward) == (
        nn.LayerNorm,
        nn.Linear,
        nn.GELU,
        nn.Linear,
        nn.LayerNorm,
    )
    assert forward[0].normalized_shape == (21,)
    assert (forward[1].in_features, forward[1].out_features) == (21, 1024)
    assert (forward[3].in_features, forward[3].out_features) == (1024, 512)
    assert forward[4].normalized_shape == (512,)
    for name, value in forward.state_dict().items():
        torch.testing.assert_close(
            value,
            backward.state_dict()[name],
            rtol=0,
            atol=0,
        )
    for forward_parameter, backward_parameter in zip(
        forward.parameters(), backward.parameters()
    ):
        assert forward_parameter is not backward_parameter
        assert forward_parameter.data_ptr() != backward_parameter.data_ptr()


def _check_optimizer_ownership(agent: fb_ddpg.FBDDPGAgent) -> None:
    assert agent.forward_adapter is not None
    assert agent.backward_adapter is not None
    assert agent.forward_fb_opt is not None
    assert agent.backward_fb_opt is not None
    forward_ids = _optimizer_parameter_ids(agent.forward_fb_opt)
    backward_ids = _optimizer_parameter_ids(agent.backward_fb_opt)
    actor_ids = _optimizer_parameter_ids(agent.actor_opt)
    assert forward_ids == (
        _parameter_ids(agent.forward_net)
        | _parameter_ids(agent.forward_adapter)
    )
    assert backward_ids == (
        _parameter_ids(agent.backward_net)
        | _parameter_ids(agent.backward_adapter)
    )
    assert actor_ids == _parameter_ids(agent.actor)
    optimizer_ids = [forward_ids, backward_ids, actor_ids]
    if agent.idm_optimizer is not None:
        assert agent.idm_head is not None
        idm_ids = _optimizer_parameter_ids(agent.idm_optimizer)
        assert idm_ids == _parameter_ids(agent.idm_head)
        optimizer_ids.append(idm_ids)
    for index, left in enumerate(optimizer_ids):
        for right in optimizer_ids[index + 1:]:
            assert left.isdisjoint(right)
    assert agent.fb_opt is None
    assert agent.encoder_opt is None
    assert agent.backward_encoder_opt is None
    assert agent.backward_encoder_target is None


def _check_isolated_idm(agent: fb_ddpg.FBDDPGAgent, route: str) -> None:
    assert agent.forward_adapter is not None
    assert agent.backward_adapter is not None
    assert agent.idm_head is not None
    assert agent.idm_optimizer is not None
    selected = (
        agent.forward_adapter
        if route == "forward_adapter"
        else agent.backward_adapter
    )
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
    assert branch_optimizer is not None
    modules = {
        "selected": selected,
        "unselected": unselected,
        "head": agent.idm_head,
        "forward_map": agent.forward_net,
        "backward_map": agent.backward_net,
        "actor": agent.actor,
    }
    before = {name: _clone(module) for name, module in modules.items()}
    raw_obs, raw_next_obs, action, _, _ = _batch()
    _zero_gradients(agent)
    loss, _, coefficient = agent._compute_idm_objective(  # pylint: disable=protected-access
        selected(raw_obs),
        selected(raw_next_obs),
        action,
        step=0,
    )
    assert coefficient == 0.1 and torch.isfinite(loss)
    loss.backward()
    assert _has_gradient(selected)
    assert _has_gradient(agent.idm_head)
    for module in (
        unselected,
        agent.forward_net,
        agent.backward_net,
        agent.actor,
    ):
        assert _no_gradient(module)
    branch_optimizer.step()
    agent.idm_optimizer.step()
    assert _changed(before["selected"], selected)
    assert _changed(before["head"], agent.idm_head)
    for name in ("unselected", "forward_map", "backward_map", "actor"):
        assert _unchanged(before[name], modules[name])


def _check_complete_update(agent: fb_ddpg.FBDDPGAgent, route: str) -> None:
    assert agent.forward_adapter is not None
    assert agent.backward_adapter is not None
    raw_obs, raw_next_obs, action, discount, z = _batch()
    obs = agent.forward_adapter(raw_obs)
    next_obs = agent.forward_adapter(raw_next_obs)
    backward_obs = agent.backward_adapter(raw_obs)
    backward_next_obs = agent.backward_adapter(raw_next_obs)
    idm_obs: tp.Optional[torch.Tensor] = None
    idm_next_obs: tp.Optional[torch.Tensor] = None
    if route == "forward_adapter":
        idm_obs, idm_next_obs = obs, next_obs
    elif route == "backward_adapter":
        idm_obs, idm_next_obs = backward_obs, backward_next_obs
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
    assert metrics["idm_route"] == fb_ddpg.IDM_ROUTE_IDS[route]
    assert all(math.isfinite(float(value)) for value in metrics.values())
    assert metrics["forward_adapter_grad_norm"] > 0
    assert metrics["backward_adapter_grad_norm"] > 0
    if route != "none":
        required = {
            "idm_loss",
            "idm_weighted_loss",
            "idm_encoder_coef_used",
            "idm_adapter_grad_norm",
            "fb_adapter_grad_norm",
            "idm_fb_adapter_grad_cosine",
            "idm_head_grad_norm",
        }
        assert required <= metrics.keys()
        assert math.isclose(
            metrics["idm_weighted_loss"],
            0.1 * metrics["idm_loss"],
            rel_tol=1e-6,
        )
        assert metrics["idm_adapter_grad_norm"] > 0
        assert metrics["fb_adapter_grad_norm"] > 0


# REDUNDANCY REVIEW: _check_architecture / _check_optimizer_ownership / _check_isolated_idm /
# _check_complete_update below re-implement, as a standalone preflight, essentially the same
# assertions already covered by url_benchmark/agent/test_idm_adapter_routing.py's
# test_mlp_ln_has_requested_shape_and_distinct_identical_copies,
# test_optimizer_ownership_is_exact_and_globally_disjoint,
# test_isolated_idm_updates_only_selected_adapter_and_head and
# test_full_fb_update_has_finite_route_and_gradient_diagnostics. This script is still used
# operationally as ROUTING_CHECKER in launch_cheetah_sep_mlp_ln_idm_routes_12.sh and
# launch_cheetah_sep_mlp_ln_onlineenc_idm_routes_12.sh, so it is not dead code, but the same
# contract is being verified twice in two different formats.
def check_contract() -> tp.Dict[str, bool]:
    torch.set_num_threads(1)
    checks: tp.Dict[str, bool] = {}
    for index, route in enumerate(
        ("none", "forward_adapter", "backward_adapter")
    ):
        torch.manual_seed(101 + index)
        agent = _make_agent(route)
        assert agent.cfg.goal_space is None
        assert agent.cfg.idm_route == route
        _check_architecture(agent)
        _check_optimizer_ownership(agent)
        if route != "none":
            _check_isolated_idm(agent, route)
        _check_complete_update(agent, route)
        checks[f"{route}_config_instantiates"] = True
    checks.update(
        {
            "mlp_ln_matches_ln_linear_gelu_linear_ln": True,
            "adapter_input_uses_runtime_obs_shape": True,
            "adapters_start_equal_but_are_distinct": True,
            "optimizer_parameter_sets_are_exact_and_disjoint": True,
            "idm_f_updates_only_forward_adapter_and_head": True,
            "idm_b_updates_only_backward_adapter_and_head": True,
            "full_updates_and_diagnostics_are_finite": True,
            "goal_space_null_uses_visual_backward_adapter": True,
            "route_metric_is_stably_numeric": True,
        }
    )
    return checks


def main() -> None:
    checks = check_contract()
    print(json.dumps({"status": "PASS", "checks": checks}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
