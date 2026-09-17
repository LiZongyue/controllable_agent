"""One FB checkpoint supplies reward-conditioned policies for multiple tasks."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import torch
from dm_env import StepType

from url_benchmark import pretrain


@pytest.mark.parametrize(
    ("tasks", "message"),
    [
        (("walker_walk", "cheetah_run"), "training task's domain"),
        (("walker_walk", "walker_walk"), "duplicate"),
    ],
)
def test_reject_incompatible_eval_tasks(tasks, message) -> None:
    cfg = pretrain.PretrainConfig(
        agent=SimpleNamespace(name="fb_ddpg"), device="cpu", eval_tasks=tasks
    )
    with mock.patch.object(pretrain.BaseWorkspace, "_make_env") as make_env:
        with pytest.raises(ValueError, match=message):
            pretrain.BaseWorkspace(cfg)
    make_env.assert_not_called()


def test_fb_reward_inference_uses_streaming_agent_api() -> None:
    agent = object.__new__(pretrain.agents.FBDDPGAgent)
    agent.infer_meta = mock.Mock(return_value={"z": np.ones(3)})
    replay = mock.MagicMock()
    replay.__len__.return_value = 1
    workspace = SimpleNamespace(
        agent=agent, domain="walker", replay_loader=replay,
        cfg=SimpleNamespace(goal_space=None),
    )
    reward = SimpleNamespace(get_goal=mock.Mock(side_effect=NotImplementedError))
    result = pretrain._init_eval_meta(workspace, reward)
    agent.infer_meta.assert_called_once_with(replay, custom_reward=reward)
    replay.sample.assert_not_called()
    np.testing.assert_array_equal(result["z"], np.ones(3))


def _workspace(tmp_path: Path):
    tasks = ("walker_stand", "walker_walk", "walker_run", "walker_flip")
    workspace = object.__new__(pretrain.BaseWorkspace)
    workspace.work_dir = tmp_path
    workspace.domain = "walker"
    workspace.global_step = 25
    workspace.global_episode = 3
    workspace.cfg = SimpleNamespace(
        task="walker_walk", custom_reward=None, seed=7,
        num_eval_episodes=2, final_tests=2, eval_tasks=tasks,
        action_repeat=2, use_wandb=False,
    )
    workspace.eval_env = object()
    workspace.eval_rewards_history = [9.0]
    agent = SimpleNamespace(
        training=True, act=mock.Mock(return_value=np.zeros(1)),
        update=mock.Mock(side_effect=AssertionError("evaluation must not train")),
    )
    agent.train = lambda training: setattr(agent, "training", training)
    workspace.agent = agent
    workspace.logger = mock.MagicMock()
    workspace.video_recorder = mock.Mock()
    created_envs = []
    created_rewards = []

    def make_env():
        value = float(tasks.index(workspace.cfg.task) + 1)
        base_env = SimpleNamespace(steps=0, close=mock.Mock())

        def reset():
            base_env.steps = 0
            return pretrain.dmc.TimeStep(StepType.FIRST, 0.0, 1.0, np.zeros(1))

        def step(action):
            base_env.steps += 1
            step_type = StepType.LAST if base_env.steps == 2 else StepType.MID
            return pretrain.dmc.TimeStep(step_type, value, 1.0, np.zeros(1))

        base_env.reset = reset
        base_env.step = step
        env = pretrain.dmc.ActionRepeatWrapper(base_env, num_repeats=2)
        created_envs.append(env)
        return env

    def make_reward(seed):
        value = float(tasks.index(workspace.cfg.task) + 1)
        reward = SimpleNamespace(
            from_env=mock.Mock(return_value=value), _env=SimpleNamespace(close=mock.Mock()),
        )
        created_rewards.append(reward)
        return reward

    workspace._make_env = make_env
    workspace._make_custom_reward = make_reward
    return workspace, created_envs, created_rewards


@pytest.mark.parametrize("final", [False, True])
def test_task_suite_reuses_one_agent_and_one_inference_per_task(
    tmp_path: Path, final: bool
) -> None:
    workspace, envs, rewards = _workspace(tmp_path)
    original_env = workspace.eval_env
    original_history = workspace.eval_rewards_history
    agent = workspace.agent
    task_metas = {}

    def infer_meta(current, reward):
        assert current.agent is agent
        assert not torch.is_grad_enabled()
        assert not agent.training
        meta = {"z": np.array([len(task_metas)], dtype=np.float32)}
        task_metas[current.cfg.task] = meta
        return meta

    with mock.patch.object(pretrain, "_init_eval_meta", side_effect=infer_meta) as infer:
        with mock.patch.object(pretrain.dmc, "PhysicsAggregator"):
            if final:
                workspace.finalize()
                result = json.loads((tmp_path / "test_rewards.json").read_text())
            else:
                assert workspace.eval() == 4.0
                row = json.loads((tmp_path / "eval_tasks.jsonl").read_text())
                assert row["frame"] == 50
                result = row["rewards"]

    assert infer.call_count == 4
    assert agent.act.call_count == 8
    agent.update.assert_not_called()
    for index, (task, meta) in enumerate(task_metas.items()):
        # Two physics frames contribute to each repeated action. Recomputing
        # reward only at the final state would incorrectly halve these scores.
        assert result[task] == [2 * float(index + 1)] * 2
        for call in agent.act.call_args_list[index * 2:index * 2 + 2]:
            assert call.args[1] is meta
    for env in envs:
        env.close.assert_called_once_with()
    for reward in rewards:
        reward._env.close.assert_called_once_with()
        reward.from_env.assert_not_called()
    assert workspace.agent is agent
    assert agent.training
    assert workspace.cfg.task == "walker_walk"
    assert workspace.cfg.custom_reward is None
    assert workspace.cfg.seed == 7
    assert workspace.cfg.num_eval_episodes == 2
    assert workspace.eval_env is original_env
    assert workspace.eval_rewards_history is original_history
    assert original_history == ([9.0] if final else [9.0, 4.0])


def test_explicit_custom_eval_keeps_legacy_reward_override(tmp_path) -> None:
    workspace, _, _ = _workspace(tmp_path)
    workspace.cfg.eval_tasks = ()
    workspace.cfg.num_eval_episodes = 1
    workspace.eval_env = workspace._make_env()
    reward = SimpleNamespace(from_env=mock.Mock(return_value=7.0))
    with mock.patch.object(pretrain.dmc, "PhysicsAggregator"):
        result = workspace.eval(
            log_metrics=False, eval_meta={"z": np.ones(1)}, eval_reward=reward,
        )
    assert result == 7.0
    reward.from_env.assert_called_once_with(workspace.eval_env)


def test_task_suite_releases_environment_and_restores_state_on_error(tmp_path) -> None:
    workspace, envs, rewards = _workspace(tmp_path)
    original_env = workspace.eval_env
    original_history = workspace.eval_rewards_history
    with mock.patch.object(pretrain, "_init_eval_meta", side_effect=RuntimeError("inference failed")):
        with pytest.raises(RuntimeError, match="inference failed"):
            workspace.eval()
    envs[0].close.assert_called_once_with()
    rewards[0]._env.close.assert_called_once_with()
    assert workspace.cfg.task == "walker_walk"
    assert workspace.cfg.custom_reward is None
    assert workspace.cfg.seed == 7
    assert workspace.cfg.num_eval_episodes == 2
    assert workspace.eval_env is original_env
    assert workspace.eval_rewards_history is original_history
    assert workspace.agent.training
