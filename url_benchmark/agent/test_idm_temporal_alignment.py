# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp
from unittest import mock

from dm_env import StepType, specs
import numpy as np
import torch

from url_benchmark import dmc
from url_benchmark.in_memory_replay_buffer import ReplayBuffer
from . import fb_ddpg


class _DeterministicEmbeddingEnv:
    """Small environment whose observation identifies its exact time index."""

    def __init__(self, final_index: int = 4) -> None:
        self._final_index = final_index
        self._index = 0
        self._observation_spec = specs.Array(
            shape=(2,), dtype=np.float32, name="observation"
        )
        self._action_spec = specs.BoundedArray(
            shape=(2,),
            dtype=np.float32,
            minimum=-1.0,
            maximum=1.0,
            name="action",
        )

    @staticmethod
    def embedding(index: int) -> np.ndarray:
        return np.array([100 * index + 1, 100 * index + 2], dtype=np.float32)

    def observation_spec(self) -> specs.Array:
        return self._observation_spec

    def action_spec(self) -> specs.BoundedArray:
        return self._action_spec

    def reset(self) -> dmc.TimeStep:
        self._index = 0
        return dmc.TimeStep(
            step_type=StepType.FIRST,
            reward=0.0,
            discount=1.0,
            observation=self.embedding(self._index),
        )

    def step(self, action: np.ndarray) -> dmc.TimeStep:
        del action
        self._index += 1
        step_type = (
            StepType.LAST if self._index == self._final_index else StepType.MID
        )
        return dmc.TimeStep(
            step_type=step_type,
            reward=float(self._index),
            discount=1.0,
            observation=self.embedding(self._index),
        )


def _stack(*indices: int) -> np.ndarray:
    return np.concatenate(
        [_DeterministicEmbeddingEnv.embedding(index) for index in indices]
    )


def _make_agent(batch_size: int) -> fb_ddpg.FBDDPGAgent:
    return fb_ddpg.FBDDPGAgent(
        obs_shape=(6,),
        action_shape=(2,),
        obs_type="dino",
        device="cpu",
        num_expl_steps=0,
        goal_space=None,
        use_cls=True,
        use_tb=False,
        use_wandb=False,
        use_hiplog=False,
        update_encoder=True,
        batch_size=batch_size,
        hidden_dim=8,
        backward_hidden_dim=8,
        feature_dim=4,
        z_dim=4,
        dino_adapter_output_dim=4,
        mix_ratio=0.0,
        future_ratio=0.0,
        idm_coef=0.1,
    )


def test_idm_three_frame_temporal_alignment_end_to_end() -> None:
    """Pin obs_t, obs_t+1, and a_t across env, replay, adapter, and IDM."""
    base_env = _DeterministicEmbeddingEnv()
    env = dmc.ExtendedTimeStepWrapper(dmc.EmbedStackWrapper(base_env, 3))
    # future=0 keeps the normal visual-agent update path supplied with a
    # deterministic future_obs; it does not affect the one-step IDM pair.
    replay = ReplayBuffer(max_episodes=1, discount=0.99, future=0.0)

    actions = np.array(
        [
            [0.1, -0.1],
            [0.2, -0.2],
            [0.3, -0.3],
            [0.4, -0.4],
        ],
        dtype=np.float32,
    )

    reset_time_step = env.reset()
    np.testing.assert_array_equal(reset_time_step.observation, _stack(0, 0, 0))
    np.testing.assert_array_equal(
        reset_time_step.action, np.zeros(2, dtype=np.float32)
    )
    replay.add(reset_time_step, meta={})

    stacked_observations = [reset_time_step.observation.copy()]
    for action in actions:
        time_step = env.step(action)
        stacked_observations.append(time_step.observation.copy())
        replay.add(time_step, meta={})

    # Reset padding must stay inside the current episode, then shift one
    # embedding at a time in oldest-to-newest order.
    np.testing.assert_array_equal(stacked_observations[1], _stack(0, 0, 1))
    np.testing.assert_array_equal(stacked_observations[2], _stack(0, 1, 2))
    np.testing.assert_array_equal(stacked_observations[3], _stack(1, 2, 3))
    np.testing.assert_array_equal(stacked_observations[4], _stack(2, 3, 4))

    expected_obs = np.stack([_stack(0, 1, 2), _stack(1, 2, 3)])
    expected_next_obs = np.stack([_stack(1, 2, 3), _stack(2, 3, 4)])
    expected_action = actions[[2, 3]]

    torch.manual_seed(2026)
    agent = _make_agent(batch_size=2)
    assert agent.idm_head is not None

    sampled: tp.Dict[str, tp.Any] = {}
    encoder_calls: tp.List[tp.Tuple[torch.Tensor, torch.Tensor]] = []
    idm_call: tp.Dict[str, torch.Tensor] = {}
    idm_head_inputs: tp.List[torch.Tensor] = []

    original_sample = replay.sample
    original_compute_idm_loss = agent._compute_idm_loss  # pylint: disable=protected-access

    def capture_sample(batch_size: int, *args: tp.Any, **kwargs: tp.Any):
        batch = original_sample(batch_size, *args, **kwargs)
        sampled["batch"] = batch
        return batch

    def capture_encoder(
        _module: torch.nn.Module,
        inputs: tp.Tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        encoder_calls.append((inputs[0].detach().clone(), output.detach().clone()))

    def capture_idm_loss(
        obs: torch.Tensor,
        next_obs: torch.Tensor,
        action: torch.Tensor,
        prediction: tp.Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        idm_call["obs"] = obs.detach().clone()
        idm_call["next_obs"] = next_obs.detach().clone()
        idm_call["action"] = action.detach().clone()
        return original_compute_idm_loss(
            obs, next_obs, action, prediction=prediction
        )

    encoder_handle = agent.encoder.register_forward_hook(capture_encoder)
    idm_handle = agent.idm_head.register_forward_pre_hook(
        lambda _module, inputs: idm_head_inputs.append(inputs[0].detach().clone())
    )
    try:
        # Replay adds one to this raw offset.  [2, 3] therefore selects stored
        # transition indices [3, 4], i.e. the interior actions [a_2, a_3].
        randint_results = [
            np.zeros(2, dtype=np.int64),
            np.array([2, 3], dtype=np.int64),
        ]
        with mock.patch(
            "url_benchmark.in_memory_replay_buffer.np.random.randint",
            side_effect=randint_results,
        ), mock.patch.object(replay, "sample", side_effect=capture_sample), mock.patch.object(
            agent, "_compute_idm_loss", side_effect=capture_idm_loss
        ) as compute_idm_loss:
            agent.update(replay, step=0)
    finally:
        encoder_handle.remove()
        idm_handle.remove()

    compute_idm_loss.assert_called_once()
    batch = sampled["batch"]
    np.testing.assert_array_equal(batch.obs, expected_obs)
    np.testing.assert_array_equal(batch.next_obs, expected_next_obs)
    np.testing.assert_array_equal(batch.action, expected_action)

    # The first two shared-adapter calls in update() are batch.obs and
    # batch.next_obs.  The later calls are FB future/backward inputs.
    assert len(encoder_calls) >= 2
    torch.testing.assert_close(
        encoder_calls[0][0], torch.from_numpy(expected_obs), rtol=0, atol=0
    )
    torch.testing.assert_close(
        encoder_calls[1][0], torch.from_numpy(expected_next_obs), rtol=0, atol=0
    )
    torch.testing.assert_close(idm_call["obs"], encoder_calls[0][1], rtol=0, atol=0)
    torch.testing.assert_close(
        idm_call["next_obs"], encoder_calls[1][1], rtol=0, atol=0
    )
    torch.testing.assert_close(
        idm_call["action"], torch.from_numpy(expected_action), rtol=0, atol=0
    )
    assert len(idm_head_inputs) == 1
    torch.testing.assert_close(
        idm_head_inputs[0],
        torch.cat([encoder_calls[0][1], encoder_calls[1][1]], dim=-1),
        rtol=0,
        atol=0,
    )
