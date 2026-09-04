# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import typing as tp
from unittest import mock

from hydra import compose, initialize_config_dir
import pytest

from url_benchmark import pretrain
from url_benchmark.pretrain import _validate_idm_training_config


def _run(tmp_path: Path, **params: tp.Any) -> None:
    folder = Path(__file__).parents[1] / "url_benchmark"
    assert folder.exists()
    if sys.platform == "darwin":
        pytest.skip(reason="Does not run on Mac")
    string = " ".join(f"{x}={y}" for (x, y) in params.items())
    command = (
        f"python -m url_benchmark.pretrain device=cpu hydra.run.dir={tmp_path} final_tests=0 "
        + string
    )
    print(f"Running: {command}")
    subprocess.check_call(command.split())


def _idm_config(**overrides: tp.Any) -> SimpleNamespace:
    values = {
        "obs_type": "dino",
        "use_cls": True,
        "dino_frame_stack": 3,
        "update_encoder": True,
        "agent": SimpleNamespace(name="fb_ddpg", idm_coef=0.1, idm_lr=None),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_validate_idm_training_config() -> None:
    _validate_idm_training_config(_idm_config())

    invalid_configs = [
        _idm_config(obs_type="pixels"),
        _idm_config(use_cls=False),
        _idm_config(dino_frame_stack=1),
        _idm_config(agent=SimpleNamespace(name="other", idm_coef=0.1, idm_lr=None)),
    ]
    for cfg in invalid_configs:
        with pytest.raises(ValueError, match="three-frame DINO CLS"):
            _validate_idm_training_config(cfg)

    with pytest.raises(ValueError, match="requires update_encoder=True"):
        _validate_idm_training_config(_idm_config(update_encoder=False))

    # The untouched baseline remains valid for every observation type.
    _validate_idm_training_config(
        _idm_config(
            obs_type="pixels",
            use_cls=False,
            dino_frame_stack=1,
            agent=SimpleNamespace(name="fb_ddpg", idm_coef=0.0, idm_lr=None),
        )
    )


def test_idm_hydra_config_wiring() -> None:
    config_dir = str(Path(__file__).parent.resolve())
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(
            config_name="base_config",
            overrides=[
                "obs_type=dino",
                "use_cls=true",
                "dino_frame_stack=3",
                "agent.idm_coef=0.1",
                "agent.idm_lr=null",
            ],
        )
    assert cfg.agent.idm_coef == 0.1
    assert cfg.agent.idm_lr is None
    _validate_idm_training_config(cfg)

    with initialize_config_dir(config_dir=config_dir, version_base=None):
        balanced = compose(
            config_name="base_config",
            overrides=[
                "obs_type=dino",
                "use_cls=true",
                "dino_frame_stack=3",
                "agent.idm_coef=1.0",
                "agent.idm_encoder_mode=balanced",
                "agent.idm_encoder_burnin_steps=25000",
                "agent.idm_grad_ratio_target=0.01",
            ],
        )
    assert balanced.agent.idm_encoder_mode == "balanced"
    assert balanced.agent.idm_encoder_burnin_steps == 25000
    assert balanced.agent.idm_grad_ratio_target == 0.01
    _validate_idm_training_config(balanced)


def test_wandb_init_uses_stable_id_and_explicit_frame_axes() -> None:
    cfg = SimpleNamespace(agent=SimpleNamespace(name="fb_ddpg"))
    env = {
        "WANDB_PROJECT": "idm-audit",
        "WANDB_RUN_ID": "stage1-walker-walk",
        "WANDB_RESUME": "must",
    }
    with mock.patch.dict(os.environ, env, clear=True), mock.patch.object(
        pretrain.omgcf.OmegaConf, "to_container", return_value={"task": "walker_walk"}
    ), mock.patch.object(pretrain.wandb, "init") as wandb_init, mock.patch.object(
        pretrain.wandb, "define_metric"
    ) as define_metric:
        pretrain._init_wandb(cfg, "idm-walker-walk")

    assert wandb_init.call_args.kwargs == {
        "project": "idm-audit",
        "group": "fb_ddpg",
        "name": "idm-walker-walk",
        "config": {"task": "walker_walk"},
        "id": "stage1-walker-walk",
        "resume": "must",
    }
    assert define_metric.call_args_list == [
        mock.call("train/frame"),
        mock.call("train/*", step_metric="train/frame"),
        mock.call("eval/frame"),
        mock.call("eval/*", step_metric="eval/frame"),
        mock.call("final/frame"),
        mock.call("final/*", step_metric="final/frame"),
    ]


def test_wandb_resume_requires_stable_run_id() -> None:
    cfg = SimpleNamespace(agent=SimpleNamespace(name="fb_ddpg"))
    with mock.patch.dict(os.environ, {"WANDB_RESUME": "must"}, clear=True):
        with pytest.raises(ValueError, match="requires a stable WANDB_RUN_ID"):
            pretrain._init_wandb(cfg, "idm-walker-walk")


@pytest.mark.parametrize("strict_optimizer_lr", [False, True])
def test_load_checkpoint_routes_optimizer_lr_strictness_only_to_fb_agent(
    tmp_path: Path,
    strict_optimizer_lr: bool,
) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    source = object()
    target = object.__new__(pretrain.agents.FBDDPGAgent)
    target.init_from = mock.Mock()
    workspace = object.__new__(pretrain.BaseWorkspace)
    workspace.agent = target
    workspace.cfg = SimpleNamespace(future=0.99, discount=0.99)

    with mock.patch.object(
        pretrain.torch,
        "load",
        return_value={"agent": source},
    ):
        workspace.load_checkpoint(
            checkpoint,
            strict_optimizer_lr=strict_optimizer_lr,
        )

    target.init_from.assert_called_once_with(
        source,
        strict_optimizer_lr=strict_optimizer_lr,
    )

    other_target = SimpleNamespace(init_from=mock.Mock())
    workspace.agent = other_target
    with mock.patch.object(
        pretrain.torch,
        "load",
        return_value={"agent": source},
    ):
        workspace.load_checkpoint(
            checkpoint,
            strict_optimizer_lr=strict_optimizer_lr,
        )
    other_target.init_from.assert_called_once_with(source)


@pytest.mark.parametrize(
    ("auto_resume", "load_model", "expected_args", "expected_kwargs"),
    [
        (True, "warm-start.pt", (), {"strict_optimizer_lr": True}),
        (
            False,
            "warm-start.pt",
            ("warm-start.pt",),
            {
                "exclude": ["replay_loader"],
                "strict_optimizer_lr": False,
            },
        ),
    ],
)
def test_workspace_routes_resume_strictness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    auto_resume: bool,
    load_model: tp.Optional[str],
    expected_args: tp.Tuple[tp.Any, ...],
    expected_kwargs: tp.Dict[str, tp.Any],
) -> None:
    monkeypatch.chdir(tmp_path)
    resume_checkpoint = tmp_path / "models" / "latest.pt"
    if auto_resume:
        resume_checkpoint.parent.mkdir()
        resume_checkpoint.write_bytes(b"checkpoint")
        expected_args = (resume_checkpoint,)

    cfg = SimpleNamespace(
        seed=1,
        device="cpu",
        agent=SimpleNamespace(name="fb_ddpg", device="cpu", idm_coef=0.0),
        task="walker_walk",
        obs_type="states",
        num_seed_frames=0,
        action_repeat=1,
        use_tb=False,
        use_wandb=False,
        use_hiplog=False,
        goal_space=None,
        replay_buffer_episodes=1,
        discount=0.99,
        future=0.99,
        save_video=False,
        checkpoint_root=None,
        auto_resume=auto_resume,
        load_model=load_model,
        custom_reward=None,
    )
    env = SimpleNamespace(
        observation_spec=lambda: object(),
        action_spec=lambda: object(),
    )
    load_checkpoint = mock.Mock()
    with mock.patch.object(
        pretrain.BaseWorkspace,
        "_make_env",
        return_value=env,
    ), mock.patch.object(
        pretrain.BaseWorkspace,
        "load_checkpoint",
        load_checkpoint,
    ), mock.patch.object(
        pretrain,
        "make_agent",
        return_value=object(),
    ), mock.patch.object(
        pretrain,
        "Logger",
        return_value=object(),
    ), mock.patch.object(
        pretrain,
        "ReplayBuffer",
        return_value=object(),
    ), mock.patch.object(
        pretrain,
        "VideoRecorder",
        return_value=object(),
    ):
        pretrain.BaseWorkspace(cfg)

    load_checkpoint.assert_called_once_with(*expected_args, **expected_kwargs)


@pytest.mark.parametrize(
    ("idm_coef", "idm_lr", "expected_run_id", "expected_suffix"),
    [
        ("0.0", "", "fixed", ""),
        ("0.1", "", "fixed_idm0p1", "_idm0p1"),
        ("0.1", "0.0003", "fixed_idm0p1_idmlr0p0003", "_idm0p1_idmlr0p0003"),
    ],
)
def test_dino_stack3_launcher_idm_wiring(
    tmp_path: Path,
    idm_coef: str,
    idm_lr: str,
    expected_run_id: str,
    expected_suffix: str,
) -> None:
    repo_dir = Path(__file__).parents[1]
    launch_root = tmp_path / "launches"
    runs_dir = tmp_path / "runs"
    env = os.environ.copy()
    env.update(
        {
            "REPO_DIR": str(repo_dir),
            "TRAIN_SCRIPT": str(repo_dir / "url_benchmark" / "pretrain.py"),
            "RUNS_DIR": str(runs_dir),
            "CKPT_ROOT": str(tmp_path / "checkpoints"),
            "LAUNCH_ROOT": str(launch_root),
            "TASKS": "walker_walk cheetah_walk quadruped_walk",
            "GPUS": "0",
            "IDM_COEF": idm_coef,
            "IDM_LR": idm_lr,
        }
    )
    subprocess.run(
        [
            "bash",
            str(repo_dir / "launch_dino_cls_stack3_12tasks.sh"),
            "--dry-run",
            "--timestamp",
            "fixed",
        ],
        check=True,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )

    launch_dir = launch_root / expected_run_id
    manifest_lines = (launch_dir / "manifest.tsv").read_text().splitlines()
    assert len(manifest_lines) == 4
    assert {line.split("\t")[3].split("_", 1)[0] for line in manifest_lines[1:]} == {
        "walker",
        "cheetah",
        "quadruped",
    }
    assert all(
        line.split("\t")[5].endswith(f"dino_cls_stack3{expected_suffix}")
        for line in manifest_lines[1:]
    )
    job_text = "\n".join(
        path.read_text() for path in launch_dir.glob("*.sh")
    )
    assert f"agent.idm_coef={idm_coef}" in job_text
    assert f"agent.idm_lr={idm_lr or 'null'}" in job_text
    assert f"experiment=dino_cls_stack3{expected_suffix}_seed1" in job_text


def test_finalize_logs_task_specific_metrics_without_polluting_periodic_eval(
    tmp_path: Path,
) -> None:
    workspace = object.__new__(pretrain.BaseWorkspace)
    workspace.work_dir = tmp_path
    workspace.domain = "walker"
    workspace.global_step = 250000
    workspace.global_episode = 500
    workspace.cfg = SimpleNamespace(
        task="walker_walk",
        custom_reward=None,
        seed=1,
        num_eval_episodes=10,
        final_tests=2,
        use_wandb=True,
        action_repeat=2,
    )
    original_eval_env = object()
    original_history = [123.0]
    workspace.eval_env = original_eval_env
    workspace.eval_rewards_history = original_history
    workspace._make_env = lambda: object()
    eval_calls = []

    def fake_eval(log_metrics: bool = True) -> float:
        eval_calls.append((workspace.cfg.task, log_metrics))
        score = float(len(workspace.eval_rewards_history) + len(eval_calls))
        workspace.eval_rewards_history.append(score)
        return score

    workspace.eval = fake_eval
    run = SimpleNamespace(summary={})
    with mock.patch.object(pretrain.wandb, "run", run), mock.patch.object(
        pretrain.wandb, "log"
    ) as wandb_log:
        workspace.finalize()

    assert len(eval_calls) == 8
    assert all(log_metrics is False for _, log_metrics in eval_calls)
    assert workspace.cfg.task == "walker_walk"
    assert workspace.cfg.custom_reward is None
    assert workspace.cfg.seed == 1
    assert workspace.cfg.num_eval_episodes == 10
    assert workspace.eval_env is original_eval_env
    assert workspace.eval_rewards_history is original_history

    rewards = json.loads((tmp_path / "test_rewards.json").read_text())
    assert set(rewards) == {
        "walker_stand",
        "walker_walk",
        "walker_run",
        "walker_flip",
    }
    assert all(len(values) == 2 for values in rewards.values())
    logged = wandb_log.call_args.args[0]
    assert logged["final/frame"] == 500000
    for task, values in rewards.items():
        assert logged[f"final/{task}"] == pytest.approx(sum(values) / len(values))
    assert run.summary["training_task"] == "walker_walk"
    assert run.summary["final_eval_domain"] == "walker"


def test_finalize_restores_training_state_when_cross_task_eval_fails(
    tmp_path: Path,
) -> None:
    workspace = object.__new__(pretrain.BaseWorkspace)
    workspace.work_dir = tmp_path
    workspace.domain = "walker"
    workspace.cfg = SimpleNamespace(
        task="walker_run",
        custom_reward=None,
        seed=7,
        num_eval_episodes=10,
        final_tests=1,
        use_wandb=False,
        action_repeat=2,
    )
    original_eval_env = object()
    original_history = [321.0]
    workspace.eval_env = original_eval_env
    workspace.eval_rewards_history = original_history
    workspace._make_env = mock.Mock(side_effect=RuntimeError("environment failed"))

    with pytest.raises(RuntimeError, match="environment failed"):
        workspace.finalize()

    assert workspace.cfg.task == "walker_run"
    assert workspace.cfg.custom_reward is None
    assert workspace.cfg.seed == 7
    assert workspace.cfg.num_eval_episodes == 10
    assert workspace.eval_env is original_eval_env
    assert workspace.eval_rewards_history is original_history


@pytest.mark.parametrize(
    "agent", ["aps", "diayn", "rnd", "proto"]
)  # test most important ones
def test_pretrain_from_commandline(agent: str, tmp_path: Path) -> None:
    _run(
        tmp_path,
        agent=agent,
        num_train_frames=1011,
        num_eval_episodes=1,
        num_seed_frames=1010,
        replay_buffer_episodes=2,
    )


def test_pretrain_from_commandline_fb_with_goal(tmp_path: Path) -> None:
    _run(
        tmp_path,
        agent="fb_ddpg",
        num_train_frames=1,
        num_eval_episodes=1,
        replay_buffer_episodes=2,
        goal_space="simplified_walker",
        use_hiplog=True,
    )
    assert (tmp_path / "hip.log").exists()
