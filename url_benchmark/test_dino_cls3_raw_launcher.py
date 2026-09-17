"""Verify three-domain Raw CLS3 launch plans without model loading or GPUs."""

import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("raw_cls3_launcher", REPO / "launch_dino_cls3_raw_3domains.py")
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


def args_for(tmp_path):
    return ["--dry-run", "--timestamp", "test_raw", "--python", sys.executable,
            "--runs-dir", str(tmp_path / "runs"), "--checkpoint-root", str(tmp_path / "checkpoints"),
            "--launch-root", str(tmp_path / "launches"), "--gpu-lock-dir", str(tmp_path / "locks"),
            "--ready-checks", "1"]


def rows_for(tmp_path):
    return json.loads((tmp_path / "launches/test_raw/manifest.json").read_text())["jobs"]


def test_exactly_three_raw_training_jobs_evaluate_twelve_distinct_rewards(tmp_path):
    assert launcher.main(args_for(tmp_path) + ["--start-immediately"]) == 0
    rows = rows_for(tmp_path)
    assert len(rows) == 3
    assert [row["gpu"] for row in rows] == [4, 5, 7]
    all_tasks = []
    for row, (domain, goal_space, backward_dim, tasks) in zip(rows, launcher.DOMAIN_SPECS):
        config = dict(argument.split("=", 1) for argument in row["command"][2:])
        expected_tasks = [domain + "_" + task for task in tasks]
        all_tasks += expected_tasks
        assert row["task"] == config["task"] == domain + "_walk"
        assert config["eval_tasks"] == "[" + ",".join(expected_tasks) + "]"
        assert config["goal_space"] == row["goal_space"] == goal_space
        assert row["backward_input_dim"] == backward_dim
        assert config["obs_type"] == "dino"
        assert config["dino_model_name"] == "facebook/dinov2-base"
        assert config["use_cls"] == "True"
        assert config["frame_stack"] == config["dino_frame_stack"] == "3"
        assert config["render_shape"] == "[224,224]"
        assert config["agent.dino_use_adapter"] == config["update_encoder"] == "False"
        assert config["agent.dino_flare_b"] == config["agent.dino_separate_fb_adapters"] == "False"
        assert config["agent.dino_separate_backward_adapter"] == config["agent.pixel_separate_fb_encoders"] == "False"
        assert config["agent.idm_coef"] == "0"
        assert {float(config["agent." + name]) for name in ("lr", "fb_lr", "lr_actor")} == {0.0001}
        assert config["agent.lr_coef"] == "1.0"
        assert config["agent.batch_size"] == "128"
        assert config["replay_buffer_episodes"] == "5000"
        assert config["num_train_frames"] == "2000010"
        assert config["agent.num_inference_steps"] == "5120"
        assert config["num_eval_episodes"] == config["final_tests"] == "10"
        assert config["save_replay_buffer_in_checkpoint"] == config["auto_resume"] == "False"
        assert row["wandb_mode"] == "online" and row["start_immediately"] is True
        assert row["command"][1].endswith("scripts/run_dino_cls3_raw_domain.py")
        script = Path(row["job_file"]).read_text()
        assert "HF_HUB_OFFLINE=1" in script and "TRANSFORMERS_OFFLINE=1" in script
        assert not Path(row["run_dir"]).exists()
    assert len(set(all_tasks)) == 12


def test_raw_launcher_is_self_contained_without_other_launchers(tmp_path):
    repo = tmp_path / "raw_repo"
    for name in launcher.SOURCE_FILES:
        target = repo / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / name, target)
    launchers = {path.name for path in repo.glob("launch_*.py")}
    assert launchers == {"launch_dino_cls3_raw_3domains.py"}
    result = subprocess.run(
        [sys.executable, "-I", str(repo / "launch_dino_cls3_raw_3domains.py")] + args_for(tmp_path),
        cwd=repo, capture_output=True, text=True, timeout=20,
    )
    assert result.returncode == 0, result.stderr
    assert len(rows_for(tmp_path)) == 3
    source_hashes = tmp_path / "launches/test_raw/source.sha256"
    hashed_files = [line.split("  ", 1)[1] for line in source_hashes.read_text().splitlines()]
    assert set(hashed_files) == set(launcher.SOURCE_FILES)
    for name in hashed_files:
        assert (repo / name).is_file()


@pytest.mark.parametrize("gpus", [["4", "4", "7"], ["4", "5"], ["4", "5", "7", "6"], ["-1", "5", "7"]])
def test_invalid_gpu_assignments_cannot_prepare_training(tmp_path, gpus):
    with pytest.raises(SystemExit) as error:
        launcher.main(args_for(tmp_path) + ["--gpus"] + gpus)
    assert error.value.code == 2
    assert not (tmp_path / "launches").exists()


def test_fresh_start_refuses_existing_run_without_writing_launch_plan(tmp_path):
    collision = tmp_path / "runs/test_raw_walker_raw_seed1"
    collision.mkdir(parents=True)
    with pytest.raises(SystemExit) as error:
        launcher.main(args_for(tmp_path))
    assert error.value.code == 2
    assert not (tmp_path / "launches").exists()


@pytest.mark.parametrize("immediate,gpu_reply,expected_return", [
    (True, "echo 60000", 0),
    (True, "echo 1000", 1),
    (False, "echo '60000, 20000, 90'", 77),
])
def test_reused_gpu_guard_shares_only_in_explicit_immediate_mode(tmp_path, immediate, gpu_reply, expected_return):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    calls = tmp_path / "calls"
    scripts = {"nvidia-smi": gpu_reply, "sleep": "exit 77",
               "python-stub": 'printf "%s\\n" "$*" >> "$PYTHON_CALL_LOG"'}
    if immediate:
        scripts["flock"] = "exit 88"
    for name, body in scripts.items():
        path = bin_dir / name
        path.write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + body + "\n")
        path.chmod(0o755)
    args = args_for(tmp_path)
    args[args.index("--python") + 1] = str(bin_dir / "python-stub")
    if immediate:
        args.append("--start-immediately")
    assert launcher.main(args) == 0
    row = rows_for(tmp_path)[0]
    env = dict(os.environ, PATH=str(bin_dir) + os.pathsep + os.environ["PATH"], PYTHON_CALL_LOG=str(calls))
    result = subprocess.run(["bash", row["job_file"]], env=env, capture_output=True, text=True, timeout=20)
    assert result.returncode == expected_return, result.stderr
    if expected_return == 0:
        assert len(calls.read_text().splitlines()) == 2
        assert "agent.dino_use_adapter=False" in calls.read_text()
        assert not (tmp_path / "locks").exists()
    else:
        assert not calls.exists()
