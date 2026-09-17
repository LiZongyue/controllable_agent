# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import inspect
import dataclasses
from types import ModuleType
import numpy as np
import torch
from url_benchmark import replay_buffer as rb
from url_benchmark import agent as agents
from . import fb_ddpg
from . import fb_modules


def get_cfg() -> fb_ddpg.FBDDPGAgentConfig:
    # hopefully this can get simpler soon
    return fb_ddpg.FBDDPGAgentConfig(
        obs_shape=(4,), action_shape=(3,), obs_type="state", device="cpu", num_expl_steps=1, goal_space=None
    )


def test_agent_init() -> None:
    cfg = get_cfg()
    agent = fb_ddpg.FBDDPGAgent(**dataclasses.asdict(cfg))
    b = 12
    shapes = dict(obs=(b, 4), next_obs=(b, 4), action=(b, 4), reward=(b,), discount=(b,))
    iterator = (rb.EpisodeBatch(**{x: np.random.rand(*y).astype(np.float32)
                for x, y in shapes.items()}) for _ in range(100))  # type: ignore
    meta = agent.init_meta()
    with torch.no_grad():
        action = agent.act(next(iterator).obs[0], meta, 0, eval_mode=False)
    assert action.shape == (3,)


def test_scale_gradient() -> None:
    for scale in (0.0, 0.1, 1.0):
        value = torch.randn(4, requires_grad=True)
        output = fb_ddpg._scale_gradient(value, scale)  # pylint: disable=protected-access
        assert torch.equal(output, value)
        output.sum().backward()
        assert value.grad is not None
        assert torch.allclose(value.grad, torch.full_like(value, scale))


def test_dino_separate_backward_adapter_one_update() -> None:
    cfg = fb_ddpg.FBDDPGAgentConfig(
        obs_shape=(8,),
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
        hidden_dim=16,
        backward_hidden_dim=16,
        feature_dim=8,
        z_dim=8,
        dino_adapter_output_dim=8,
        dino_separate_backward_adapter=True,
    )
    agent = fb_ddpg.FBDDPGAgent(**dataclasses.asdict(cfg))
    assert agent.backward_encoder is not None
    assert agent.backward_encoder_target is not None
    assert agent.encoder_opt is not None
    assert agent.backward_encoder_opt is not None
    assert not any(
        left.data_ptr() == right.data_ptr()
        for left, right in zip(agent.encoder.parameters(), agent.backward_encoder.parameters())
    )

    # A backward-only graph must not touch the forward/actor adapter.
    agent.encoder_opt.zero_grad(set_to_none=True)
    agent.backward_encoder_opt.zero_grad(set_to_none=True)
    agent.backward_aug_and_encode(torch.randn(4, 8)).square().mean().backward()
    assert all(param.grad is None for param in agent.encoder.parameters())
    assert any(param.grad is not None for param in agent.backward_encoder.parameters())

    batch = rb.EpisodeBatch(
        obs=np.random.randn(4, 8).astype(np.float32),
        action=np.random.uniform(-1, 1, size=(4, 2)).astype(np.float32),
        reward=np.random.randn(4, 1).astype(np.float32),
        next_obs=np.random.randn(4, 8).astype(np.float32),
        discount=np.full((4, 1), 0.99, dtype=np.float32),
        future_obs=np.random.randn(4, 8).astype(np.float32),
    )

    class _Replay:
        def sample(self, batch_size: int) -> rb.EpisodeBatch:
            assert batch_size == 4
            return batch

    forward_before = [param.detach().clone() for param in agent.encoder.parameters()]
    backward_before = [param.detach().clone() for param in agent.backward_encoder.parameters()]
    backward_target_before = [
        param.detach().clone() for param in agent.backward_encoder_target.parameters()
    ]
    target_calls = 0

    def _count_target_calls(_module, _inputs, _output) -> None:
        nonlocal target_calls
        target_calls += 1

    target_hook = agent.backward_encoder_target.register_forward_hook(_count_target_calls)
    metrics = agent.update(_Replay(), step=0)  # type: ignore[arg-type]
    target_hook.remove()
    assert "fb_loss" in metrics
    assert "actor_loss" in metrics
    assert target_calls == 1
    assert any(
        not torch.equal(before, after)
        for before, after in zip(forward_before, agent.encoder.parameters())
    )
    assert any(
        not torch.equal(before, after)
        for before, after in zip(backward_before, agent.backward_encoder.parameters())
    )
    for before, online, target in zip(
        backward_target_before,
        agent.backward_encoder.parameters(),
        agent.backward_encoder_target.parameters(),
    ):
        expected = cfg.fb_target_tau * online + (1 - cfg.fb_target_tau) * before
        assert torch.allclose(target, expected)
        assert target.grad is None

    restored = fb_ddpg.FBDDPGAgent(**dataclasses.asdict(cfg))
    restored.init_from(agent)
    assert restored.backward_encoder is not None
    assert restored.backward_encoder_target is not None
    for expected, actual in zip(agent.backward_encoder.parameters(), restored.backward_encoder.parameters()):
        assert torch.equal(expected, actual)
    for expected, actual in zip(
        agent.backward_encoder_target.parameters(), restored.backward_encoder_target.parameters()
    ):
        assert torch.equal(expected, actual)
    assert restored.backward_encoder_opt is not None
    assert restored.backward_encoder_opt.state_dict()["state"]


def test_agents_config() -> None:
    cfgs = []
    for module in agents.__dict__.values():
        if isinstance(module, ModuleType):
            for obj in module.__dict__.values():
                if inspect.isclass(obj) and issubclass(obj, agents.DDPGAgentConfig):
                    if obj not in cfgs:
                        cfgs.append(obj)
    assert len(cfgs) >= 3
    for cfg in cfgs:
        # check that target and name have been updated to match the algo
        assert cfg.name.replace("_", "") in cfg.__name__.lower()
        assert cfg.name in cfg._target_


def test_multiinputs() -> None:
    m, n = [10, 12]
    x, y = (torch.rand([16, z]) for z in [m, n])
    mip = fb_modules.MultinputNet([m, n], [100, 100, 32])
    out = mip(x, y)
    assert out.shape == (16, 32)
