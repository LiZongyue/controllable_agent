# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pickle
from url_benchmark.dmc import ExtendedGoalTimeStep, TimeStep
from url_benchmark.in_memory_replay_buffer import ReplayBuffer

from typing import List

import numpy as np
import torch
from dm_env import StepType
import pytest

fixed_episode_lengths = [10, 10, 10, 10, 10]
variable_episode_lengths = [2, 3, 5, 6, 7]
@pytest.mark.parametrize('test_data', [(10, fixed_episode_lengths, None, False, 10),
                                       (5, fixed_episode_lengths, None, True, 10),
                                       (10, variable_episode_lengths, 8, False, 5),
                                       (5, variable_episode_lengths, 8, True, 5)])
def test_avg_episode_length_fixed_length_not_full(test_data) -> None:
    max_episodes, episode_lengths, max_episode_length, is_full, avg_episode_length = test_data
    replay_storage = ReplayBuffer(
        max_episodes=max_episodes, discount=1, future=1, max_episode_length=max_episode_length)
    meta = {'z': np.ones((3, 3))}
    for episode_length in episode_lengths:
        for time_step in _create_dummy_episode(episode_length):
            replay_storage.add(time_step, meta=meta)
    assert replay_storage._full == is_full
    assert replay_storage.avg_episode_length == avg_episode_length

@pytest.mark.parametrize('test_data', [(10, 5, 7), (10, 10, 7)])
def test_backward_compatibility(test_data) -> None:
    max_episodes, episodes_count, episode_length  = test_data
    is_full = max_episodes == episodes_count
    replay_storage = ReplayBuffer(max_episodes=max_episodes, discount=1, future=1, max_episode_length=episode_length + 1)    
    meta = {'z': np.ones((3, 3))}
    for _ in range(episodes_count):
        for time_step in _create_dummy_episode(episode_length):
            replay_storage.add(time_step, meta=meta)
    # remove attributes recently added
    del replay_storage._episodes_length
    del replay_storage._episodes_selection_probability
    del replay_storage._is_fixed_episode_length
    del replay_storage._max_episode_length
    
    loaded_replay_storage = pickle.loads(pickle.dumps(replay_storage))
    assert loaded_replay_storage._idx == episodes_count%max_episodes
    assert loaded_replay_storage._full == is_full
    assert (loaded_replay_storage._episodes_length[:episodes_count]==episode_length).all()
    assert (loaded_replay_storage._episodes_length[episodes_count:]==0).all()
    assert loaded_replay_storage._max_episode_length is None
    

def _create_dummy_episode(episode_length: int) -> List[TimeStep]:
    time_steps = []
    for i in range(episode_length+1):
        step_type = StepType.MID
        if i == 0:
            step_type = StepType.FIRST
        elif i == episode_length:
            step_type = StepType.LAST
        time_step = TimeStep(step_type=step_type, observation=np.zeros(
            (3, 3)), reward=1, discount=1)
        time_steps.append(time_step)
    return time_steps


def test_pixel_replay_keeps_lossless_uint8_and_float_training_fields() -> None:
    replay = ReplayBuffer(max_episodes=2, discount=0.99, future=0.99)
    frames = np.random.RandomState(3).randint(0, 256, (4, 3, 224, 224), dtype=np.uint8)
    for i, pixels in enumerate(frames):
        time_step = ExtendedGoalTimeStep(
            step_type=StepType.FIRST if i == 0 else StepType.LAST if i == 3 else StepType.MID,
            observation=pixels, reward=0.5, discount=1.0,
            action=np.zeros(2), goal=np.zeros(3),
        )._replace(physics=np.zeros(4))
        replay.add(time_step, {"z": np.ones(2)})
    np.testing.assert_array_equal(replay._storage["observation"][0], frames)
    assert replay._storage["observation"].dtype == np.uint8
    assert replay._storage["observation"].nbytes == 2 * frames.nbytes
    batch = replay.sample(3).to("cpu")
    assert all(value.dtype == torch.uint8 for value in (
        batch.obs, batch.next_obs, batch.future_obs,
    ))
    for name in ("action", "reward", "discount", "physics", "goal", "z"):
        assert replay._storage[name].dtype == np.float32


def test_loading_pixel_replay_preserves_uint8(tmp_path) -> None:
    frames = np.random.RandomState(4).randint(0, 256, (3, 3, 8, 8), dtype=np.uint8)
    np.savez(tmp_path / "episode.npz", observation=frames, reward=np.ones((3, 1)))
    replay = ReplayBuffer(max_episodes=1, discount=0.99, future=0.99)
    replay.load(None, tmp_path, relabel=False)
    assert replay._storage["observation"].dtype == np.uint8
    np.testing.assert_array_equal(replay._storage["observation"][0], frames)
    assert replay._storage["reward"].dtype == np.float32
