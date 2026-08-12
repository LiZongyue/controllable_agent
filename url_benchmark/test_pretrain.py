# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import typing as tp

from hydra import compose, initialize_config_dir
import pytest

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
