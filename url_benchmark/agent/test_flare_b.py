# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import typing as tp
from unittest import mock

import numpy as np
import pytest
import torch
from torch import nn

from url_benchmark import replay_buffer as rb
from . import fb_ddpg


_BATCH_SIZE = 4
_FRAME_DIM = 4
_RAW_DIM = 3 * _FRAME_DIM
_ADAPTER_DIM = 8


def _agent_kwargs(**overrides: tp.Any) -> tp.Dict[str, tp.Any]:
    kwargs: tp.Dict[str, tp.Any] = dict(
        obs_shape=(_RAW_DIM,),
        action_shape=(2,),
        obs_type="dino",
        device="cpu",
        lr=1e-4,
        lr_f=3e-4,
        num_expl_steps=0,
        goal_space=None,
        use_cls=True,
        use_tb=False,
        use_wandb=False,
        use_hiplog=False,
        update_encoder=True,
        update_every_steps=1,
        batch_size=_BATCH_SIZE,
        hidden_dim=16,
        backward_hidden_dim=16,
        feature_dim=8,
        z_dim=4,
        dino_use_adapter=True,
        dino_adapter_output_dim=_ADAPTER_DIM,
        dino_frame_stack=3,
        dino_flare_b=True,
        dino_separate_fb_adapters=False,
        dino_separate_backward_adapter=False,
        idm_coef=0.0,
        mix_ratio=0.0,
        future_ratio=0.0,
    )
    kwargs.update(overrides)
    return kwargs


def _make_agent(**overrides: tp.Any) -> fb_ddpg.FBDDPGAgent:
    return fb_ddpg.FBDDPGAgent(**_agent_kwargs(**overrides))


def _optimizer_parameter_ids(
    optimizer: torch.optim.Optimizer,
) -> tp.Set[int]:
    return {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }


def _module_parameters(module: nn.Module) -> tp.Tuple[torch.Tensor, ...]:
    return tuple(parameter.detach().clone() for parameter in module.parameters())


def _module_changed(before: tp.Sequence[torch.Tensor], module: nn.Module) -> bool:
    return any(
        not torch.equal(expected, actual.detach())
        for expected, actual in zip(before, module.parameters())
    )


def _module_unchanged(before: tp.Sequence[torch.Tensor], module: nn.Module) -> bool:
    return all(
        torch.equal(expected, actual.detach())
        for expected, actual in zip(before, module.parameters())
    )


def _assert_modules_equal(left: nn.Module, right: nn.Module) -> None:
    left_state = left.state_dict()
    right_state = right.state_dict()
    assert left_state.keys() == right_state.keys()
    for name, expected in left_state.items():
        torch.testing.assert_close(expected, right_state[name], rtol=0, atol=0)


def _fixed_raw_batch(offset: float = 0.0) -> torch.Tensor:
    values = torch.arange(_BATCH_SIZE * _RAW_DIM, dtype=torch.float32)
    return values.reshape(_BATCH_SIZE, _RAW_DIM) / 17.0 + offset


def test_flare_b_architecture_shape_shared_projector_and_frame_order() -> None:
    torch.manual_seed(1)
    encoder = fb_ddpg.FlareBEncoder(frame_dim=_FRAME_DIM)

    assert isinstance(encoder.projector, nn.Sequential)
    assert tuple(type(layer) for layer in encoder.projector) == (
        nn.LayerNorm,
        nn.Linear,
    )
    assert encoder.projector[0].normalized_shape == (_FRAME_DIM,)
    assert (
        encoder.projector[1].in_features,
        encoder.projector[1].out_features,
    ) == (_FRAME_DIM, 512)

    assert isinstance(encoder.fusion, nn.Sequential)
    assert tuple(type(layer) for layer in encoder.fusion) == (
        nn.Linear,
        nn.LayerNorm,
    )
    assert (encoder.fusion[0].in_features, encoder.fusion[0].out_features) == (
        2048,
        512,
    )
    assert encoder.fusion[1].normalized_shape == (512,)

    # There is one D->512 projector, called three times, rather than three
    # independently initialized frame projectors.
    frame_projection_linears = [
        module
        for module in encoder.modules()
        if isinstance(module, nn.Linear)
        and module.in_features == _FRAME_DIM
        and module.out_features == 512
    ]
    assert frame_projection_linears == [encoder.projector[1]]

    phi0 = torch.tensor([[0.0, 1.0, 4.0, -2.0], [2.0, -3.0, 5.0, 7.0]])
    phi1 = torch.tensor([[8.0, -1.0, 3.0, 2.0], [-4.0, 9.0, 1.0, 6.0]])
    phi2 = torch.tensor([[5.0, 2.0, -7.0, 4.0], [3.0, 8.0, -2.0, 1.0]])
    raw_stack = torch.cat((phi0, phi1, phi2), dim=-1)
    projector_inputs: tp.List[torch.Tensor] = []
    projector_outputs: tp.List[torch.Tensor] = []
    fusion_inputs: tp.List[torch.Tensor] = []

    def capture_projector(
        _module: nn.Module,
        inputs: tp.Tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        projector_inputs.append(inputs[0])
        projector_outputs.append(output)

    projector_hook = encoder.projector.register_forward_hook(
        capture_projector
    )
    fusion_hook = encoder.fusion.register_forward_pre_hook(
        lambda _module, inputs: fusion_inputs.append(inputs[0])
    )
    try:
        output = encoder(raw_stack)
    finally:
        projector_hook.remove()
        fusion_hook.remove()

    assert output.shape == (2, 512)
    assert len(projector_inputs) == len(projector_outputs) == 3
    assert len(fusion_inputs) == 1
    for actual, expected in zip(projector_inputs, (phi0, phi1, phi2)):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    u0, u1, u2 = projector_outputs
    expected_fusion_input = torch.cat(
        (u1, u2, u1 - u0.detach(), u2 - u1.detach()), dim=-1
    )
    torch.testing.assert_close(
        fusion_inputs[0], expected_fusion_input, rtol=0, atol=0
    )


def test_flare_b_exact_assembly_and_asymmetric_stop_gradient() -> None:
    generator = torch.Generator().manual_seed(2)
    u0, u1, u2 = (
        torch.randn(2, 512, generator=generator, requires_grad=True)
        for _ in range(3)
    )
    flare = fb_ddpg.FlareBEncoder._assemble_flare(u0, u1, u2)
    expected = torch.cat(
        (u1, u2, u1 - u0.detach(), u2 - u1.detach()), dim=-1
    )
    assert flare.shape == (2, 2048)
    torch.testing.assert_close(flare, expected, rtol=0, atol=0)

    d1 = flare[:, 1024:1536]
    d2 = flare[:, 1536:2048]
    d1_grads = torch.autograd.grad(
        d1.sum(), (u0, u1, u2), allow_unused=True, retain_graph=True
    )
    assert d1_grads[0] is None
    torch.testing.assert_close(d1_grads[1], torch.ones_like(u1))
    torch.testing.assert_close(d1_grads[2], torch.zeros_like(u2))

    d2_grads = torch.autograd.grad(
        d2.sum(), (u0, u1, u2), allow_unused=True
    )
    assert d2_grads[0] is None
    # Slicing the concatenation leaves a zero-valued graph connection to the
    # other non-detached fields. The key assertion is zero rather than -1:
    # the subtracting u1 branch of d2 is detached.
    torch.testing.assert_close(d2_grads[1], torch.zeros_like(u1))
    torch.testing.assert_close(d2_grads[2], torch.ones_like(u2))


@pytest.mark.parametrize(
    "override",
    [
        {"obs_type": "state"},
        {"use_cls": False},
        {"dino_frame_stack": 1},
        {"goal_space": "simplified_walker"},
    ],
)
def test_flare_b_rejects_invalid_configs_before_model_construction(
    override: tp.Dict[str, tp.Any],
) -> None:
    torch.manual_seed(3)
    rng_before = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="dino_flare_b"):
        _make_agent(**override)
    torch.testing.assert_close(torch.get_rng_state(), rng_before, rtol=0, atol=0)


def test_flare_b_rejects_legacy_backward_adapter_before_construction() -> None:
    torch.manual_seed(4)
    rng_before = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="mutually exclusive"):
        _make_agent(dino_separate_backward_adapter=True)
    torch.testing.assert_close(torch.get_rng_state(), rng_before, rtol=0, atol=0)


def test_flare_b_has_no_legacy_backward_adapter_or_target_module() -> None:
    agent = _make_agent()
    assert agent.flare_b_encoder is not None
    assert agent.backward_encoder is None
    assert agent.backward_encoder_target is None
    assert agent.backward_adapter is None
    assert agent.backward_encoder_opt is None
    assert not any(
        "flare" in name.lower() and "target" in name.lower()
        for name in vars(agent)
    )
    assert not hasattr(agent, "P_B_target")
    assert not hasattr(agent, "Fusion_B_target")


def test_flare_b_gets_raw_stack_while_f_and_actor_keep_the_adapter_path() -> None:
    torch.manual_seed(5)
    agent = _make_agent()
    torch.manual_seed(5)
    baseline = _make_agent(dino_flare_b=False)
    assert agent.flare_b_encoder is not None

    raw = _fixed_raw_batch()
    with mock.patch.object(
        agent.encoder, "forward", wraps=agent.encoder.forward
    ) as forward_adapter, mock.patch.object(
        agent.flare_b_encoder,
        "forward",
        wraps=agent.flare_b_encoder.forward,
    ) as flare_b:
        forward_features = agent.aug_and_encode(raw)
        assert forward_adapter.call_count == 1
        assert flare_b.call_count == 0
        backward_features = agent.backward_aug_and_encode(raw)
        assert forward_adapter.call_count == 1
        assert flare_b.call_count == 1
        assert flare_b.call_args.args[0] is raw

    assert raw.shape == (_BATCH_SIZE, 3 * _FRAME_DIM)
    assert forward_features.shape == (_BATCH_SIZE, _ADAPTER_DIM)
    assert backward_features.shape == (_BATCH_SIZE, 512)
    assert agent.obs_dim == baseline.obs_dim == _ADAPTER_DIM
    assert agent.actor.obs_dim == baseline.actor.obs_dim == _ADAPTER_DIM
    assert agent.forward_net.obs_dim == baseline.forward_net.obs_dim == _ADAPTER_DIM
    assert agent.backward_net.obs_dim == 512
    for name in ("encoder", "actor", "forward_net", "forward_target_net"):
        _assert_modules_equal(getattr(agent, name), getattr(baseline, name))

    z = torch.randn(_BATCH_SIZE, agent.cfg.z_dim)
    action = torch.zeros(_BATCH_SIZE, agent.action_dim)
    with mock.patch.object(
        agent.flare_b_encoder,
        "forward",
        side_effect=AssertionError("F/actor unexpectedly used FLARE-B"),
    ):
        agent.actor(forward_features, z, std=0.2)
        agent.forward_net(forward_features, z, action)


def test_reward_and_goal_inference_route_visual_inputs_through_flare_b() -> None:
    agent = _make_agent()
    assert agent.flare_b_encoder is not None
    raw = _fixed_raw_batch(2.0)
    reward = torch.linspace(0.1, 0.4, _BATCH_SIZE).unsqueeze(1)
    flare_inputs: tp.List[torch.Tensor] = []
    forward_inputs: tp.List[torch.Tensor] = []
    flare_hook = agent.flare_b_encoder.register_forward_pre_hook(
        lambda _module, inputs: flare_inputs.append(inputs[0])
    )
    forward_hook = agent.encoder.register_forward_pre_hook(
        lambda _module, inputs: forward_inputs.append(inputs[0])
    )
    try:
        inferred = agent.infer_meta_from_obs_and_rewards(raw, reward)
        goal_meta = agent.get_goal_meta(raw[0].numpy())
    finally:
        flare_hook.remove()
        forward_hook.remove()

    assert not forward_inputs
    assert len(flare_inputs) == 2
    torch.testing.assert_close(flare_inputs[0], raw, rtol=0, atol=0)
    torch.testing.assert_close(flare_inputs[1], raw[:1], rtol=0, atol=0)
    assert inferred["z"].shape == (agent.cfg.z_dim,)
    assert goal_meta["z"].shape == (agent.cfg.z_dim,)


class _Replay:
    def __init__(self, batch: rb.EpisodeBatch[np.ndarray]) -> None:
        self.batch = batch

    def sample(self, batch_size: int) -> rb.EpisodeBatch[np.ndarray]:
        assert batch_size == self.batch.obs.shape[0]
        return self.batch


def test_update_routes_every_visual_b_input_through_raw_flare_features() -> None:
    torch.manual_seed(6)
    agent = _make_agent(mix_ratio=1.0, future_ratio=1.0)
    assert agent.flare_b_encoder is not None
    raw_obs = _fixed_raw_batch(0.0).numpy()
    raw_next_obs = _fixed_raw_batch(10.0).numpy()
    raw_future_obs = _fixed_raw_batch(20.0).numpy()
    batch = rb.EpisodeBatch(
        obs=raw_obs,
        action=np.linspace(-0.8, 0.8, _BATCH_SIZE * 2, dtype=np.float32).reshape(
            _BATCH_SIZE, 2
        ),
        reward=np.zeros((_BATCH_SIZE, 1), dtype=np.float32),
        next_obs=raw_next_obs,
        discount=np.full((_BATCH_SIZE, 1), 0.99, dtype=np.float32),
        future_obs=raw_future_obs,
    )

    forward_records: tp.List[tp.Tuple[torch.Tensor, torch.Tensor]] = []
    flare_records: tp.List[tp.Tuple[torch.Tensor, torch.Tensor]] = []
    backward_map_inputs: tp.List[torch.Tensor] = []
    forward_hook = agent.encoder.register_forward_hook(
        lambda _module, inputs, output: forward_records.append((inputs[0], output))
    )
    flare_hook = agent.flare_b_encoder.register_forward_hook(
        lambda _module, inputs, output: flare_records.append((inputs[0], output))
    )
    backward_hook = agent.backward_net.register_forward_pre_hook(
        lambda _module, inputs: backward_map_inputs.append(inputs[0])
    )
    try:
        with mock.patch.object(agent, "update_fb", return_value={}) as update_fb, \
                mock.patch.object(agent, "update_actor", return_value={}) as update_actor, \
                mock.patch.object(
                    torch,
                    "randperm",
                    return_value=torch.arange(_BATCH_SIZE),
                ):
            agent.update(_Replay(batch), step=0)  # type: ignore[arg-type]
    finally:
        forward_hook.remove()
        flare_hook.remove()
        backward_hook.remove()

    raw_tensors = tuple(
        torch.from_numpy(value)
        for value in (raw_obs, raw_next_obs, raw_future_obs)
    )
    assert len(forward_records) == 2
    for (actual, _output), expected in zip(
        forward_records, (raw_tensors[0], raw_tensors[1])
    ):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    # All three B sources (successor, current, and future observations) reach
    # FLARE at raw 3D width. No compressed forward-adapter tensor reaches it.
    for expected in (raw_tensors[1], raw_tensors[0], raw_tensors[2]):
        assert any(torch.equal(actual, expected) for actual, _ in flare_records)
    assert flare_records
    assert all(actual.shape[-1] == _RAW_DIM for actual, _ in flare_records)

    def flare_outputs_for(raw: torch.Tensor) -> tp.List[torch.Tensor]:
        return [
            output
            for actual, output in flare_records
            if torch.equal(actual, raw)
        ]

    routed = update_fb.call_args.kwargs
    torch.testing.assert_close(routed["obs"], forward_records[0][1])
    torch.testing.assert_close(routed["next_obs"], forward_records[1][1])
    next_flare_outputs = flare_outputs_for(raw_tensors[1])
    assert next_flare_outputs
    assert any(
        torch.equal(routed["next_goal"], output)
        for output in next_flare_outputs
    )
    assert not routed["target_next_goal"].requires_grad
    assert any(
        torch.equal(routed["target_next_goal"], output.detach())
        for output in next_flare_outputs
    )

    obs_flare_outputs = flare_outputs_for(raw_tensors[0])
    future_flare_outputs = flare_outputs_for(raw_tensors[2])
    assert obs_flare_outputs and future_flare_outputs
    assert any(
        torch.equal(actual, output)
        for actual in backward_map_inputs
        for output in obs_flare_outputs
    )
    assert any(
        torch.equal(actual, output)
        for actual in backward_map_inputs
        for output in future_flare_outputs
    )
    actor_obs = update_actor.call_args.args[0]
    torch.testing.assert_close(actor_obs, forward_records[0][1].detach())


def test_flare_b_optimizer_owns_only_flare_and_all_optimizers_are_disjoint() -> None:
    agent = _make_agent()
    assert agent.flare_b_encoder is not None
    assert agent.flare_b_optimizer is not None
    assert agent.fb_opt is not None
    assert agent.encoder_opt is not None
    assert agent.forward_fb_opt is None
    assert agent.backward_fb_opt is None

    optimizers = {
        name: value
        for name, value in vars(agent).items()
        if isinstance(value, torch.optim.Optimizer)
    }
    optimizer_ids = {
        name: _optimizer_parameter_ids(optimizer)
        for name, optimizer in optimizers.items()
    }
    for name, optimizer in optimizers.items():
        parameter_ids = [
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        ]
        assert len(parameter_ids) == len(set(parameter_ids)), name
    for left_name, right_name in itertools.combinations(optimizer_ids, 2):
        assert optimizer_ids[left_name].isdisjoint(optimizer_ids[right_name]), (
            left_name,
            right_name,
        )

    flare_ids = {id(parameter) for parameter in agent.flare_b_encoder.parameters()}
    forward_ids = {id(parameter) for parameter in agent.forward_net.parameters()}
    backward_ids = {id(parameter) for parameter in agent.backward_net.parameters()}
    encoder_ids = {id(parameter) for parameter in agent.encoder.parameters()}
    actor_ids = {id(parameter) for parameter in agent.actor.parameters()}
    assert optimizer_ids["flare_b_optimizer"] == flare_ids
    assert optimizer_ids["encoder_opt"] == encoder_ids
    assert optimizer_ids["actor_opt"] == actor_ids
    assert optimizer_ids["fb_opt"] == forward_ids | backward_ids
    assert agent.flare_b_optimizer.param_groups[0]["lr"] == agent.cfg.lr_f


def test_flare_b_and_forward_adapter_gradient_paths_are_disjoint() -> None:
    agent = _make_agent(norm_z=False)
    assert agent.flare_b_encoder is not None
    raw = _fixed_raw_batch(0.5)

    backward = agent.backward_net(agent.backward_aug_and_encode(raw))
    backward[:, 0].sum().backward()
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad)
        for parameter in agent.flare_b_encoder.parameters()
    )
    assert all(parameter.grad is None for parameter in agent.encoder.parameters())

    for module in (agent.encoder, agent.flare_b_encoder, agent.forward_net):
        for parameter in module.parameters():
            parameter.grad = None
    forward_features = agent.aug_and_encode(raw)
    z = torch.randn(_BATCH_SIZE, agent.cfg.z_dim)
    action = torch.randn(_BATCH_SIZE, agent.action_dim)
    f1, f2 = agent.forward_net(forward_features, z, action)
    (f1[:, 0].sum() + f2[:, 0].sum()).backward()
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad)
        for parameter in agent.encoder.parameters()
    )
    assert all(
        parameter.grad is None for parameter in agent.flare_b_encoder.parameters()
    )


def test_flare_b_replaces_plural_backward_adapter_without_changing_map_owners() -> None:
    agent = _make_agent(dino_separate_fb_adapters=True)
    assert agent.flare_b_encoder is not None
    assert agent.flare_b_optimizer is not None
    assert agent.backward_encoder is None
    assert agent.backward_adapter is None
    assert agent.backward_encoder_opt is None
    assert agent.encoder_opt is None
    assert agent.fb_opt is None
    assert agent.forward_fb_opt is not None
    assert agent.backward_fb_opt is not None

    flare_ids = {id(parameter) for parameter in agent.flare_b_encoder.parameters()}
    forward_map_ids = {id(parameter) for parameter in agent.forward_net.parameters()}
    forward_adapter_ids = {id(parameter) for parameter in agent.encoder.parameters()}
    backward_map_ids = {id(parameter) for parameter in agent.backward_net.parameters()}
    assert _optimizer_parameter_ids(agent.flare_b_optimizer) == flare_ids
    assert _optimizer_parameter_ids(agent.forward_fb_opt) == (
        forward_map_ids | forward_adapter_ids
    )
    assert _optimizer_parameter_ids(agent.backward_fb_opt) == backward_map_ids


def test_flare_b_optimizer_steps_during_one_fb_update() -> None:
    torch.manual_seed(7)
    agent = _make_agent()
    assert agent.flare_b_encoder is not None
    assert agent.flare_b_optimizer is not None
    assert agent.fb_opt is not None
    raw_obs = _fixed_raw_batch(0.25)
    raw_next_obs = _fixed_raw_batch(1.25)
    obs = agent.aug_and_encode(raw_obs)
    next_obs = agent.aug_and_encode(raw_next_obs)
    next_goal = agent.backward_aug_and_encode(raw_next_obs)
    generator = torch.Generator().manual_seed(77)
    action = torch.randn(_BATCH_SIZE, agent.action_dim, generator=generator).tanh()
    discount = torch.full((_BATCH_SIZE, 1), 0.99)
    z = torch.randn(_BATCH_SIZE, agent.cfg.z_dim, generator=generator)
    flare_before = _module_parameters(agent.flare_b_encoder)
    backward_before = _module_parameters(agent.backward_net)
    actor_before = _module_parameters(agent.actor)

    with mock.patch.object(
        agent.flare_b_optimizer,
        "step",
        wraps=agent.flare_b_optimizer.step,
    ) as flare_step:
        agent.update_fb(
            obs=obs,
            action=action,
            discount=discount,
            next_obs=next_obs,
            next_goal=next_goal,
            target_next_goal=next_goal.detach(),
            z=z,
            step=0,
        )

    assert flare_step.call_count == 1
    assert _module_changed(flare_before, agent.flare_b_encoder)
    assert _module_changed(backward_before, agent.backward_net)
    assert _module_unchanged(actor_before, agent.actor)
    assert agent.flare_b_optimizer.state_dict()["state"]


def test_disabled_flare_b_is_an_rng_and_routing_noop() -> None:
    kwargs = _agent_kwargs(dino_flare_b=False)
    omitted_kwargs = dict(kwargs)
    del omitted_kwargs["dino_flare_b"]

    torch.manual_seed(8)
    baseline = fb_ddpg.FBDDPGAgent(**omitted_kwargs)
    baseline_rng_state = torch.get_rng_state().clone()
    torch.manual_seed(8)
    disabled = fb_ddpg.FBDDPGAgent(**kwargs)
    disabled_rng_state = torch.get_rng_state().clone()

    assert disabled.flare_b_encoder is None
    assert disabled.flare_b_optimizer is None
    torch.testing.assert_close(
        disabled_rng_state, baseline_rng_state, rtol=0, atol=0
    )
    for name in (
        "encoder",
        "actor",
        "forward_net",
        "backward_net",
        "forward_target_net",
        "backward_target_net",
    ):
        _assert_modules_equal(getattr(baseline, name), getattr(disabled, name))

    raw = _fixed_raw_batch()
    torch.testing.assert_close(
        disabled.backward_aug_and_encode(raw),
        disabled.aug_and_encode(raw),
        rtol=0,
        atol=0,
    )

    # Checkpoints created before this flag existed do not have either instance
    # attribute; their disabled/shared behavior must still remain usable.
    del disabled.flare_b_encoder
    del disabled.flare_b_optimizer
    disabled.train(False)
    torch.testing.assert_close(
        disabled.backward_aug_and_encode(raw),
        disabled.aug_and_encode(raw),
        rtol=0,
        atol=0,
    )
    assert fb_ddpg.FBDDPGAgentConfig().dino_flare_b is False
