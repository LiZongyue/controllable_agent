"""Regression checks for native RGB inputs and bounded task inference."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from url_benchmark.agent.ddpg import Encoder
from url_benchmark.agent.fb_ddpg import FBDDPGAgent


def test_encoder_224_keeps_84_head_dimensions_and_gradients() -> None:
    encoder = Encoder((3, 224, 224))
    pixels = torch.randint(256, (2, 3, 224, 224), dtype=torch.uint8)
    encoded = encoder(pixels)
    assert encoded.shape == (2, 32 * 35 * 35)
    encoded.square().mean().backward()
    assert all(parameter.grad is not None for parameter in encoder.parameters())
    assert encoder.convnet[0].weight.grad.abs().sum() > 0

    # Old state dicts load without new parameters and produce identical 84px
    # features, including the original stacked-RGB input format.
    legacy = Encoder((9, 84, 84))
    restored = Encoder((9, 84, 84))
    restored.load_state_dict(legacy.state_dict(), strict=True)
    old_pixels = torch.randint(256, (2, 9, 84, 84), dtype=torch.uint8)
    expected = legacy.convnet(old_pixels / 255.0 - 0.5).flatten(1)
    torch.testing.assert_close(restored(old_pixels), expected, rtol=0, atol=0)


class _RecordingBackward(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.batch_sizes = []

    def forward(self, obs):
        assert not torch.is_grad_enabled()
        self.batch_sizes.append(len(obs))
        return obs


def _inference_agent(norm_z):
    agent = object.__new__(FBDDPGAgent)
    agent.cfg = SimpleNamespace(
        device="cpu", batch_size=3, num_inference_steps=7,
        goal_space=None, obs_type="states", norm_z=norm_z, z_dim=2,
    )
    agent.backward_net = _RecordingBackward()
    return agent


@pytest.mark.parametrize("norm_z", [False, True])
def test_inference_streams_custom_rewards_and_matches_global_mean(norm_z) -> None:
    obs = torch.arange(14, dtype=torch.float32).reshape(7, 2) - 5
    rewards = torch.tensor([[1.0], [3.0], [0.0], [2.0], [-1.0], [0.0], [8.0]])
    expected = rewards.T @ obs / len(obs)
    if norm_z:
        expected = 2 ** 0.5 * F.normalize(expected, dim=1)
    expected = expected.squeeze().numpy()

    agent = _inference_agent(norm_z)
    direct = agent.infer_meta_from_obs_and_rewards(obs, rewards)
    np.testing.assert_allclose(direct["z"], expected, rtol=1e-6)
    assert agent.backward_net.batch_sizes == [3, 3, 1]

    reward_function = object()

    class Replay:
        offset = 0

        def sample(self, size, custom_reward=None):
            assert custom_reward is reward_function
            start = self.offset
            self.offset += size
            return SimpleNamespace(
                next_obs=obs[start:self.offset].numpy(),
                reward=rewards[start:self.offset].numpy(),
            )

    agent.backward_net.batch_sizes.clear()
    replay = Replay()
    streamed = agent.infer_meta(replay, custom_reward=reward_function)
    np.testing.assert_allclose(streamed["z"], expected, rtol=1e-6)
    assert agent.backward_net.batch_sizes == [3, 3, 1]
    assert replay.offset == 7


def test_inference_rejects_empty_reward_bank() -> None:
    with pytest.raises(ValueError, match="at least one"):
        _inference_agent(False).infer_meta_from_obs_and_rewards(
            torch.empty((0, 2)), torch.empty((0, 1)),
        )
