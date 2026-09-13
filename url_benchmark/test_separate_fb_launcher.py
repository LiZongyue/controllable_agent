import os
from pathlib import Path
import subprocess
import sys

# REDUNDANCY REVIEW: this dry-runs launch_cheetah_separate_fb_adapters.sh purely to call
# scripts/check_separate_fb_adapters.py's check_launch_plan(), which the real launcher
# already calls (and fails on) as its own preflight step. Overlaps with
# url_benchmark/agent/test_separate_fb_adapters.py, which covers the sibling
# check_agent_contract() from the same script.
from scripts.check_separate_fb_adapters import check_launch_plan


def test_separate_fb_launcher_contract(tmp_path: Path) -> None:
    repo_dir = Path(__file__).parents[1]
    campaign = "separate_fb_launcher_test"
    env = os.environ.copy()
    env.update(
        {
            "REPO_DIR": str(repo_dir),
            "TRAIN_SCRIPT": str(repo_dir / "url_benchmark" / "pretrain.py"),
            "SANITY_SCRIPT": str(repo_dir / "scripts" / "check_separate_fb_adapters.py"),
            "PYTHON_BIN": sys.executable,
            "RUNS_ROOT": str(tmp_path / "runs"),
            "CKPT_ROOT": str(tmp_path / "checkpoints"),
            "LAUNCH_ROOT": str(tmp_path / "launches"),
            "CAMPAIGN": campaign,
        }
    )
    subprocess.run(
        [
            "bash",
            str(repo_dir / "launch_cheetah_separate_fb_adapters.sh"),
            "--dry-run",
        ],
        check=True,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    checks = check_launch_plan(tmp_path / "launches" / campaign)
    assert checks
    assert all(checks.values())
