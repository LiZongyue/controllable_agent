# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import csv
import os
from pathlib import Path
import re
import shlex
import subprocess
import typing as tp

import pytest


_ALL_TASKS = [
    "walker_stand",
    "walker_walk",
    "walker_run",
    "walker_flip",
    "cheetah_walk",
    "cheetah_run",
    "cheetah_walk_backward",
    "cheetah_run_backward",
    "quadruped_stand",
    "quadruped_walk",
    "quadruped_run",
    "quadruped_jump",
]


def _run_launcher(
    tmp_path: Path,
    *,
    root_name: str,
    timestamp: str,
    overrides: tp.Optional[tp.Dict[str, str]] = None,
    extra_args: tp.Sequence[str] = (),
    check: bool = True,
) -> tp.Tuple[subprocess.CompletedProcess[str], Path, Path, Path, Path]:
    repo_dir = Path(__file__).parents[1]
    launch_root = tmp_path / f"launch-{root_name}"
    runs_dir = tmp_path / f"runs-{root_name}"
    checkpoint_root = tmp_path / f"checkpoints-{root_name}"
    fake_bin = tmp_path / f"bin-{root_name}"
    fake_bin.mkdir(parents=True, exist_ok=True)
    tmux_marker = tmp_path / f"tmux-called-{root_name}"
    fake_tmux = fake_bin / "tmux"
    fake_tmux.write_text(
        "#!/usr/bin/env bash\n"
        f"printf called > {shlex.quote(str(tmux_marker))}\n"
        "exit 91\n"
    )
    fake_tmux.chmod(0o755)

    env = os.environ.copy()
    env.pop("EVAL_EVERY_FRAMES", None)
    env.pop("GPU_ASSIGNMENTS", None)
    env.pop("PARALLEL_TASKS", None)
    env.update(
        {
            "PATH": f"{fake_bin}{os.pathsep}{env['PATH']}",
            "REPO_DIR": str(repo_dir),
            "TRAIN_SCRIPT": str(repo_dir / "url_benchmark" / "pretrain.py"),
            "RUNS_DIR": str(runs_dir),
            "CKPT_ROOT": str(checkpoint_root),
            "LAUNCH_ROOT": str(launch_root),
            "GPUS": "4 5",
            "SEED": "1",
            "TASKS": "",
            "STAGE": "",
            "IDM_COEF": "0.0",
            "IDM_LR": "",
            "IDM_DIAGNOSTICS_INTERVAL": "500",
            "IDM_ENCODER_MODE": "legacy",
            "IDM_ENCODER_BURNIN_STEPS": "0",
            "IDM_ENCODER_RAMP_STEPS": "0",
            "IDM_GRAD_RATIO_TARGET": "",
            "IDM_GRAD_RATIO_EMA": "0.9",
            "IDM_COEF_MIN": "0.1",
            "IDM_COEF_MAX": "200.0",
            "IDM_COEF_SLEW_RATE": "2.0",
            "NUM_TRAIN_FRAMES": "2000010",
        }
    )
    if overrides:
        env.update(overrides)

    result = subprocess.run(
        [
            "bash",
            str(repo_dir / "launch_dino_cls_stack3_12tasks.sh"),
            "--dry-run",
            "--timestamp",
            timestamp,
            *extra_args,
        ],
        check=check,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return result, launch_root, runs_dir, checkpoint_root, tmux_marker


def _read_manifest(launch_dir: Path) -> tp.List[tp.Dict[str, str]]:
    with (launch_dir / "manifest.tsv").open(newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def _assert_queue_order(queue_file: Path, expected_tasks: tp.Sequence[str]) -> None:
    queue_text = queue_file.read_text()
    offsets = []
    for task in expected_tasks:
        needle = f"/job_{task}.sh"
        assert queue_text.count(needle) == 1
        offsets.append(queue_text.index(needle))
    assert offsets == sorted(offsets)
    assert queue_text.count("if bash ") == len(expected_tasks)


def test_subset_assignment_uses_selected_ordinal_and_stable_identity(
    tmp_path: Path,
) -> None:
    overrides = {
        # These are original TASK_SPECS indices 1 and 3. Raw-index scheduling
        # would incorrectly place both on the second GPU.
        "TASKS": "walker_walk walker_flip",
        "GPUS": "4 5",
        "STAGE": "stage1",
        "IDM_COEF": "0.1",
        "IDM_LR": "0.0001",
        "NUM_TRAIN_FRAMES": "500000",
    }
    result, launch_root, _, _, tmux_marker = _run_launcher(
        tmp_path,
        root_name="subset",
        timestamp="fixed",
        overrides=overrides,
    )
    assert "Dry run only; no tmux sessions were launched." in result.stdout
    assert not tmux_marker.exists()

    launch_dir = launch_root / "fixed_stage1_idm0p1_idmlr0p0001_f500000"
    rows = _read_manifest(launch_dir)
    assert [(row["task"], row["gpu"]) for row in rows] == [
        ("walker_walk", "4"),
        ("walker_flip", "5"),
    ]
    _assert_queue_order(launch_dir / "queue_gpu4.sh", ["walker_walk"])
    _assert_queue_order(launch_dir / "queue_gpu5.sh", ["walker_flip"])

    for row in rows:
        assert "cuda" not in Path(row["run_dir"]).name
        assert re.fullmatch(r"[A-Za-z0-9_-]+", row["wandb_run_id"])
        assert len(row["wandb_run_id"]) <= 64
        job_text = (launch_dir / f"job_{row['task']}.sh").read_text()
        assert f"WANDB_RUN_ID={row['wandb_run_id']}" in job_text
        assert "WANDB_RESUME=never" in job_text

    # Reassigning physical GPUs must not change task run/checkpoint identity or
    # the deterministic W&B run ID.
    _, moved_launch_root, _, _, moved_tmux_marker = _run_launcher(
        tmp_path,
        root_name="subset-moved",
        timestamp="fixed",
        overrides={**overrides, "GPUS": "8 9"},
    )
    moved_rows = _read_manifest(
        moved_launch_root / "fixed_stage1_idm0p1_idmlr0p0001_f500000"
    )
    original_by_task = {row["task"]: row for row in rows}
    moved_by_task = {row["task"]: row for row in moved_rows}
    for task in original_by_task:
        assert Path(original_by_task[task]["run_dir"]).name == Path(
            moved_by_task[task]["run_dir"]
        ).name
        assert (
            original_by_task[task]["wandb_run_id"]
            == moved_by_task[task]["wandb_run_id"]
        )
    assert not moved_tmux_marker.exists()


def test_full_12_task_manifest_budget_and_per_gpu_serial_order(
    tmp_path: Path,
) -> None:
    _, launch_root, _, _, tmux_marker = _run_launcher(
        tmp_path,
        root_name="full",
        timestamp="full",
        overrides={
            "GPUS": "0 1",
            "STAGE": "stage1",
            "IDM_COEF": "0.0",
            "IDM_LR": "0.0001",
            "NUM_TRAIN_FRAMES": "500000",
        },
    )
    launch_dir = launch_root / "full_stage1_idm0p0_idmlr0p0001_f500000"
    rows = _read_manifest(launch_dir)

    required_fields = {
        "task",
        "domain",
        "seed",
        "gpu",
        "goal_space",
        "idm_coef",
        "idm_lr",
        "idm_diagnostics_interval",
        "num_train_frames",
        "eval_every_frames",
        "run_dir",
        "wandb_run_id",
    }
    assert required_fields.issubset(rows[0])
    assert [row["task"] for row in rows] == _ALL_TASKS
    assert {row["domain"] for row in rows} == {"walker", "cheetah", "quadruped"}
    assert [row["gpu"] for row in rows] == ["0", "1"] * 6
    assert all(row["seed"] == "1" for row in rows)
    assert all(row["idm_coef"] == "0.0" for row in rows)
    assert all(row["idm_lr"] == "0.0001" for row in rows)
    assert all(row["idm_diagnostics_interval"] == "500" for row in rows)
    assert all(row["num_train_frames"] == "500000" for row in rows)
    assert all(row["eval_every_frames"] == "10000" for row in rows)
    assert len({row["run_dir"] for row in rows}) == 12
    assert len({row["wandb_run_id"] for row in rows}) == 12
    assert len(list(launch_dir.glob("queue_gpu*.sh"))) == 2
    assert {row["launch_mode"] for row in rows} == {"serial_gpu_queue"}
    assert len({row["session"] for row in rows}) == 2
    _assert_queue_order(launch_dir / "queue_gpu0.sh", _ALL_TASKS[0::2])
    _assert_queue_order(launch_dir / "queue_gpu1.sh", _ALL_TASKS[1::2])

    for task in _ALL_TASKS:
        job_text = (launch_dir / f"job_{task}.sh").read_text()
        assert "num_train_frames=500000" in job_text
        assert "agent.idm_coef=0.0" in job_text
        assert "agent.idm_lr=0.0001" in job_text
        assert "agent.idm_diagnostics_interval=500" in job_text
    assert not tmux_marker.exists()


def test_full_12_task_parallel_dry_run_uses_independent_sessions_and_10k_eval(
    tmp_path: Path,
) -> None:
    result, launch_root, _, _, tmux_marker = _run_launcher(
        tmp_path,
        root_name="full-parallel",
        timestamp="parallel",
        overrides={
            "GPUS": "0 1",
            "STAGE": "stage1",
            "IDM_COEF": "0.0",
            "IDM_LR": "0.0001",
            "NUM_TRAIN_FRAMES": "500000",
            "GPU_ASSIGNMENTS": "1 2 6 7 1 3 5 6 1 4 5 7",
        },
        extra_args=("--parallel",),
    )
    assert "Dry run only; no tmux sessions were launched." in result.stdout
    assert not tmux_marker.exists()

    launch_dir = launch_root / "parallel_stage1_idm0p0_idmlr0p0001_f500000"
    rows = _read_manifest(launch_dir)
    assert [row["task"] for row in rows] == _ALL_TASKS
    assert [row["gpu"] for row in rows] == [
        "1",
        "2",
        "6",
        "7",
        "1",
        "3",
        "5",
        "6",
        "1",
        "4",
        "5",
        "7",
    ]
    assert {row["launch_mode"] for row in rows} == {"parallel_task"}
    assert {row["eval_every_frames"] for row in rows} == {"10000"}
    assert len({row["session"] for row in rows}) == 12
    assert len({row["job_file"] for row in rows}) == 12
    assert len({row["session_log"] for row in rows}) == 12
    assert not list(launch_dir.glob("queue_gpu*.sh"))

    for row in rows:
        assert row["task"] in row["session"]
        assert Path(row["job_file"]).name == f"job_{row['task']}.sh"
        assert Path(row["session_log"]).name == f"session_{row['task']}.log"
        job_text = Path(row["job_file"]).read_text()
        assert "eval_every_frames=10000" in job_text
        assert "num_train_frames=500000" in job_text


def test_balanced_idm_encoder_settings_have_stable_identity_and_wiring(
    tmp_path: Path,
) -> None:
    _, launch_root, _, _, tmux_marker = _run_launcher(
        tmp_path,
        root_name="balanced",
        timestamp="pilot",
        overrides={
            "TASKS": "walker_flip cheetah_walk quadruped_walk",
            "GPU_ASSIGNMENTS": "2 4 6",
            "STAGE": "rho1pct",
            "IDM_COEF": "1.0",
            "IDM_LR": "0.0001",
            "IDM_ENCODER_MODE": "balanced",
            "IDM_ENCODER_BURNIN_STEPS": "25000",
            "IDM_GRAD_RATIO_TARGET": "0.01",
            "NUM_TRAIN_FRAMES": "500000",
        },
        extra_args=("--parallel",),
    )
    launch_dir = launch_root / (
        "pilot_rho1pct_idm1p0_idmlr0p0001_encbalanced_b25000_"
        "rho0p01_i500_c0p1-200p0_ema0p9_slew2p0_f500000"
    )
    rows = _read_manifest(launch_dir)
    assert len(rows) == 3
    assert [row["gpu"] for row in rows] == ["2", "4", "6"]
    assert {row["idm_encoder_mode"] for row in rows} == {"balanced"}
    assert {row["idm_encoder_burnin_steps"] for row in rows} == {"25000"}
    assert {row["idm_grad_ratio_target"] for row in rows} == {"0.01"}
    for row in rows:
        job_text = Path(row["job_file"]).read_text()
        assert "agent.idm_encoder_mode=balanced" in job_text
        assert "agent.idm_encoder_burnin_steps=25000" in job_text
        assert "agent.idm_grad_ratio_target=0.01" in job_text
    assert not tmux_marker.exists()


def test_gradient_pilot_prepares_three_tasks_by_four_settings_without_launch(
    tmp_path: Path,
) -> None:
    repo_dir = Path(__file__).parents[1]
    launch_root = tmp_path / "pilot-launches"
    env = os.environ.copy()
    env.update(
        {
            "REPO_DIR": str(repo_dir),
            "BASE_LAUNCHER": str(repo_dir / "launch_dino_cls_stack3_12tasks.sh"),
            "RUNS_DIR": str(tmp_path / "pilot-runs"),
            "CKPT_ROOT": str(tmp_path / "pilot-checkpoints"),
            "LAUNCH_ROOT": str(launch_root),
            "TIMESTAMP": "pilot-grid",
        }
    )
    result = subprocess.run(
        ["bash", str(repo_dir / "launch_idm_gradient_pilot.sh")],
        check=True,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert "no tmux sessions were launched" in result.stdout

    with (launch_root / "pilot-grid_pilot_index" / "manifests.tsv").open(
        newline=""
    ) as stream:
        index_rows = list(csv.DictReader(stream, delimiter="\t"))
    assert [row["setting"] for row in index_rows] == [
        "static1",
        "static10",
        "rho1pct",
        "rho5pct",
    ]
    task_rows = []
    for index_row in index_rows:
        with Path(index_row["manifest"]).open(newline="") as stream:
            task_rows.extend(csv.DictReader(stream, delimiter="\t"))
    assert len(task_rows) == 12
    assert {row["task"] for row in task_rows} == {
        "walker_flip",
        "cheetah_walk",
        "quadruped_walk",
    }
    assert {row["seed"] for row in task_rows} == {"1"}


def test_parallel_tasks_environment_switch_is_supported(tmp_path: Path) -> None:
    _, launch_root, _, _, tmux_marker = _run_launcher(
        tmp_path,
        root_name="parallel-env",
        timestamp="parallel-env",
        overrides={
            "TASKS": "walker_stand walker_walk",
            "GPUS": "7",
            "PARALLEL_TASKS": "1",
        },
    )
    rows = _read_manifest(launch_root / "parallel-env")
    assert len(rows) == 2
    assert len({row["session"] for row in rows}) == 2
    assert {row["gpu"] for row in rows} == {"7"}
    assert {row["launch_mode"] for row in rows} == {"parallel_task"}
    assert not tmux_marker.exists()


def test_gpu_assignments_requires_one_entry_per_selected_task(
    tmp_path: Path,
) -> None:
    result, _, _, _, tmux_marker = _run_launcher(
        tmp_path,
        root_name="assignment-count",
        timestamp="assignment-count",
        overrides={
            "TASKS": "walker_stand walker_walk",
            "GPU_ASSIGNMENTS": "1",
            "PARALLEL_TASKS": "1",
        },
        check=False,
    )
    assert result.returncode == 2
    assert "expected 2, got 1" in result.stderr
    assert not tmux_marker.exists()


@pytest.mark.parametrize("existing_kind", ["run", "checkpoint"])
def test_fresh_start_refuses_existing_state_and_resume_is_explicit(
    tmp_path: Path,
    existing_kind: str,
) -> None:
    root_name = f"fresh-{existing_kind}"
    overrides = {
        "TASKS": "cheetah_run",
        "GPUS": "3",
        "STAGE": "stage2",
        "IDM_COEF": "0.1",
        "IDM_LR": "0.0003",
        "NUM_TRAIN_FRAMES": "500000",
    }
    _, launch_root, _, checkpoint_root, _ = _run_launcher(
        tmp_path,
        root_name=root_name,
        timestamp="collision",
        overrides=overrides,
    )
    launch_dir = launch_root / "collision_stage2_idm0p1_idmlr0p0003_f500000"
    original_row = _read_manifest(launch_dir)[0]
    run_dir = Path(original_row["run_dir"])
    target = run_dir if existing_kind == "run" else checkpoint_root / run_dir.name
    target.mkdir(parents=True)

    refused, _, _, _, tmux_marker = _run_launcher(
        tmp_path,
        root_name=root_name,
        timestamp="collision",
        overrides=overrides,
        check=False,
    )
    assert refused.returncode == 1
    assert "Fresh start refused" in refused.stderr
    assert not tmux_marker.exists()

    # Exact resume requires both the original work directory and a non-empty
    # checkpoint, regardless of which collision triggered the fresh refusal.
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_file = checkpoint_root / run_dir.name / "latest.pt"
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_file.write_bytes(b"checkpoint")
    resumed, _, _, _, resumed_tmux_marker = _run_launcher(
        tmp_path,
        root_name=root_name,
        timestamp="collision",
        overrides=overrides,
        extra_args=("--resume",),
    )
    assert resumed.returncode == 0
    resumed_row = _read_manifest(launch_dir)[0]
    assert resumed_row["resume"] == "1"
    assert resumed_row["wandb_run_id"] == original_row["wandb_run_id"]
    job_text = (launch_dir / "job_cheetah_run.sh").read_text()
    assert "WANDB_RESUME=must" in job_text
    assert not resumed_tmux_marker.exists()


def test_resume_refuses_missing_local_state(tmp_path: Path) -> None:
    result, _, _, _, tmux_marker = _run_launcher(
        tmp_path,
        root_name="missing-resume",
        timestamp="missing",
        overrides={"TASKS": "walker_walk", "GPUS": "1"},
        extra_args=("--resume",),
        check=False,
    )
    assert result.returncode == 1
    assert "Resume refused: run directory does not exist" in result.stderr
    assert not tmux_marker.exists()


def test_duplicate_gpu_tokens_are_rejected_before_launch(tmp_path: Path) -> None:
    result, _, _, _, tmux_marker = _run_launcher(
        tmp_path,
        root_name="duplicate-gpu",
        timestamp="duplicate",
        overrides={"GPUS": "2 2", "TASKS": "walker_stand walker_walk"},
        check=False,
    )
    assert result.returncode == 2
    assert "Duplicate GPU ordinal is not allowed: 2" in result.stderr
    assert not tmux_marker.exists()


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("IDM_COEF", "abc", "IDM_COEF must be a non-negative number"),
        ("IDM_COEF", "-0.1", "IDM_COEF must be a non-negative number"),
        ("IDM_LR", "0", "IDM_LR must be empty or a positive number"),
        ("IDM_LR", "1e999", "IDM_LR must be empty or a positive number"),
        ("AGENT_LR", "nan", "AGENT_LR must be a positive number"),
        (
            "IDM_DIAGNOSTICS_INTERVAL",
            "0",
            "IDM_DIAGNOSTICS_INTERVAL must be a positive integer",
        ),
        ("IDM_ENCODER_MODE", "adaptive", "IDM_ENCODER_MODE must be"),
        (
            "IDM_ENCODER_BURNIN_STEPS",
            "-1",
            "IDM_ENCODER_BURNIN_STEPS must be a non-negative integer",
        ),
        ("IDM_GRAD_RATIO_EMA", "1", "IDM_GRAD_RATIO_EMA must be in [0, 1)"),
    ],
)
def test_invalid_numeric_settings_are_rejected_before_launch(
    tmp_path: Path,
    key: str,
    value: str,
    message: str,
) -> None:
    result, _, _, _, tmux_marker = _run_launcher(
        tmp_path,
        root_name=f"invalid-{key.lower()}-{value.replace('.', 'p')}",
        timestamp="invalid",
        overrides={key: value},
        check=False,
    )
    assert result.returncode == 2
    assert message in result.stderr
    assert not tmux_marker.exists()
