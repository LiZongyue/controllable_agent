#!/usr/bin/env python3
"""Launch pointmass corner runs to 1.5M without killing existing jobs."""

from __future__ import annotations

import os
import re
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path


REPO = Path("/data/fanfeng/controllable_agent")
DREAMER = Path("/data/fanfeng/dreamerv3")
RUNS = Path("/mnt/data_7tb/fanfeng/controllable_agent_runs")
DREAMER_RUNS = Path("/mnt/data_7tb/fanfeng/dreamer_corner_runs")
CKPT = Path("/mnt/data_7tb/fanfeng/controallable_agent_ckpt")
LAUNCH = RUNS / "launch_queues" / "20260502_pointmass_extend_1p5m"
TRAIN_SCRIPT = REPO / "url_benchmark" / "pretrain.py"

NUM_TRAIN_FRAMES = "1500000"
SNAPSHOT_AT = "[100000,500000,1000000,1500000]"
EVAL_EVERY = "100000"
CHECKPOINT_EVERY = "100000"

TASK_FROM_SHORT = {
    "top_left": "point_mass_maze_reach_top_left",
    "top_right": "point_mass_maze_reach_top_right",
    "bottom_left": "point_mass_maze_reach_bottom_left",
    "bottom_right": "point_mass_maze_reach_bottom_right",
}


@dataclass(frozen=True)
class FbCfg:
    obs_type: str
    render_shape: str
    frame_stack: str
    replay_eps: str
    extra: tuple[str, ...] = ()


FB_CFG = {
    "dino_cls_linear": FbCfg("dino", "[224,224]", "1", "2000", ("use_cls=True", "agent.dino_use_adapter=True", "agent.dino_adapter_type=linear")),
    "dino_cls_mlp": FbCfg("dino", "[224,224]", "1", "2000", ("use_cls=True", "agent.dino_use_adapter=True", "agent.dino_adapter_type=mlp")),
    "dino_patch_linear": FbCfg("dino", "[224,224]", "1", "2000", ("use_cls=False", "agent.dino_use_adapter=True", "agent.dino_adapter_type=linear")),
    "dino_patch_mlp": FbCfg("dino", "[224,224]", "1", "2000", ("use_cls=False", "agent.dino_use_adapter=True", "agent.dino_adapter_type=mlp")),
    "dino_cls_no_adapter": FbCfg("dino", "[224,224]", "1", "2000", ("use_cls=True", "agent.dino_use_adapter=False", "agent.feature_dim=1024")),
    "cnn": FbCfg("pixels", "[84,84]", "3", "100"),
    "vit": FbCfg("vit", "[224,224]", "1", "80", ("use_cls=True", "agent.vit_batch_size=64")),
}


@dataclass
class Job:
    kind: str
    short: str
    seed: str
    variant: str
    run_dir: Path
    gpu: int
    wait_pattern: str

    @property
    def session(self) -> str:
        return f"pm15b_{self.short}_{self.variant}_s{self.seed}_g{self.gpu}"


def q(value: object) -> str:
    return shlex.quote(str(value))


def active_process_contains(pattern: str) -> bool:
    out = subprocess.check_output(["ps", "-u", os.environ.get("USER", ""), "-o", "cmd="], text=True)
    for line in out.splitlines():
        if pattern in line and "launch_pointmass_extend_1p5m.py" not in line:
            return True
    return False


def tmux_has(session: str) -> bool:
    return subprocess.run(["tmux", "has-session", "-t", session], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0


def parse_fb_run_dir(path: Path) -> tuple[str, str, str] | None:
    name = path.name
    m = re.match(r"20260501_submission_corner_parallel_v2_(top_left|top_right|bottom_left|bottom_right)_seed(\d+)_(.+)_cuda\d+$", name)
    if not m:
        return None
    short, seed, variant = m.groups()
    if variant == "vit_retry1":
        variant = "vit"
    return short, seed, variant


def pick_gpu(short: str, variant: str, seed: str, run_dir: Path) -> int:
    if variant == "vit":
        if short == "bottom_left":
            return 5
        if short == "bottom_right":
            return 6
        if short == "top_left":
            return 4
        return 7
    if variant == "cnn":
        if short in {"bottom_left", "bottom_right", "top_right"}:
            return 0
        return 7
    # Keep DINO continuations away from cuda2 and avoid further loading cuda3.
    order = [4, 7, 0, 4, 7, 0]
    key = sum(ord(c) for c in f"{short}:{variant}:{seed}:{run_dir.name}")
    return order[key % len(order)]


def make_fb_jobs() -> list[Job]:
    jobs: list[Job] = []
    for run_dir in sorted(RUNS.glob("20260501_submission_corner_parallel_v2_*")):
        parsed = parse_fb_run_dir(run_dir)
        if parsed is None:
            continue
        short, seed, variant = parsed
        if variant not in FB_CFG:
            continue
        # Skip known failed duplicate attempts; their replacement runs are handled separately.
        if run_dir.name.endswith("_cnn_cuda7") and short == "bottom_left":
            continue
        if variant == "vit" and short in {"bottom_left", "bottom_right"} and "retry1" not in run_dir.name:
            continue
        if variant == "dino_cls_linear" and seed == "7532" and run_dir.name.endswith("_cuda4"):
            # These are the OOM duplicates for top-left/top-right; bottom-left/right cuda4 are valid.
            if short in {"top_left", "top_right"}:
                continue
        latest = CKPT / run_dir.name / "latest.pt"
        if latest.exists() or (variant == "vit" and "retry1" in run_dir.name):
            gpu = pick_gpu(short, variant, seed, run_dir)
            if gpu == 2:
                raise RuntimeError(f"Refusing to use cuda2 for {run_dir}")
            jobs.append(Job("fb", short, seed, variant, run_dir, gpu, f"hydra.run.dir={run_dir}"))

    # Top-left/top-right ViT had been intentionally killed before any checkpoint; restart them fresh.
    for short, gpu in (("top_left", 4), ("top_right", 7)):
        run_dir = RUNS / f"20260502_extend1p5m_{short}_seed1992_vit_retry1_cuda{gpu}"
        jobs.append(Job("fb", short, "1992", "vit", run_dir, gpu, f"hydra.run.dir={run_dir}"))
    return jobs


def make_dreamer_jobs() -> list[Job]:
    jobs: list[Job] = []
    gpu_by_short = {
        "top_left": 5,
        "bottom_left": 6,
        "top_right": 5,
        "bottom_right": 6,
    }
    for run_dir in sorted(DREAMER_RUNS.glob("20260501_submission_corner_parallel_v2_*_dreamer_zero_reward_cuda*")):
        m = re.match(r"20260501_submission_corner_parallel_v2_(top_left|top_right|bottom_left|bottom_right)_seed(\d+)_dreamer_zero_reward_cuda\d+$", run_dir.name)
        if not m:
            continue
        short, seed = m.groups()
        gpu = gpu_by_short[short]
        jobs.append(Job("dreamer", short, seed, "dreamer_zero_reward", run_dir, gpu, f"--logdir {run_dir}"))
    return jobs


def fb_command(job: Job) -> str:
    cfg = FB_CFG[job.variant]
    args = [
        "use_wandb=True",
        "save_video=False",
        "save_train_video=False",
        "save_replay_buffer_in_checkpoint=False",
        f"checkpoint_root={CKPT}",
        f"checkpoint_every={CHECKPOINT_EVERY}",
        f"snapshot_at={SNAPSHOT_AT}",
        f"obs_type={cfg.obs_type}",
        f"render_shape={cfg.render_shape}",
        f"frame_stack={cfg.frame_stack}",
        "action_repeat=1",
        f"task={TASK_FROM_SHORT[job.short]}",
        "goal_space=null",
        "custom_reward=null",
        f"seed={job.seed}",
        f"experiment=submission_{job.short}_{job.variant}",
        f"num_train_frames={NUM_TRAIN_FRAMES}",
        f"eval_every_frames={EVAL_EVERY}",
        "num_eval_episodes=3",
        "final_tests=0",
        f"replay_buffer_episodes={cfg.replay_eps}",
        f"hydra.run.dir={job.run_dir}",
        *cfg.extra,
    ]
    return " ".join(["python", q(TRAIN_SCRIPT), *map(q, args)])


def dreamer_command(job: Job) -> str:
    args = [
        "--configs", "dmc_vision",
        "--task", f"dmc_{TASK_FROM_SHORT[job.short]}",
        "--seed", job.seed,
        "--logdir", str(job.run_dir),
        "--run.steps", NUM_TRAIN_FRAMES,
        "--run.envs", "4",
        "--run.eval_envs", "1",
        "--run.eval_eps", "3",
        "--run.log_every", "120",
        "--run.report_every", "300",
        "--run.eval_every", "300",
        "--run.save_every", "900",
        "--jax.prealloc", "False",
        "--agent.zero_reward", "True",
    ]
    return " ".join(["python", q(DREAMER / "dreamerv3" / "main.py"), *map(q, args)])


def write_launcher(job: Job) -> Path:
    LAUNCH.mkdir(parents=True, exist_ok=True)
    script = LAUNCH / f"{job.session}.sh"
    log = LAUNCH / f"{job.session}.log"
    stdout = job.run_dir / "stdout.log"
    wait_source = "pretrain.py" if job.kind == "fb" else "dreamerv3/main.py"
    preamble = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"run_dir={q(job.run_dir)}",
        f"stdout_path={q(stdout)}",
        'mkdir -p "$run_dir"',
        f"echo '[extend-wait] '$(date -u +%FT%TZ)' session={job.session} gpu={job.gpu} run_dir='$run_dir | tee -a \"$run_dir/launcher.log\"",
        f"while ps -u \"$USER\" -o cmd= | rg -F -- {q(wait_source)} | rg -F -- {q(job.wait_pattern)} >/dev/null; do sleep 300; done",
        f"echo '[extend-start] '$(date -u +%FT%TZ)' session={job.session} gpu={job.gpu} target_frames={NUM_TRAIN_FRAMES}' | tee -a \"$run_dir/launcher.log\"",
    ]
    if job.kind == "fb":
        cmd = (
            f"cd {q(REPO)}; "
            f"env CUDA_VISIBLE_DEVICES={q(job.gpu)} PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
            f"{fb_command(job)} >> \"$stdout_path\" 2>&1"
        )
    else:
        cmd = (
            f"cd {q(DREAMER)}; "
            f"env CUDA_VISIBLE_DEVICES={q(job.gpu)} PYTHONPATH={q(DREAMER)} PYTHONUNBUFFERED=1 "
            "XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl WANDB_PROJECT=controllable_agent_baseline "
            f"{dreamer_command(job)} >> \"$stdout_path\" 2>&1"
        )
    body = "\n".join([
        *preamble,
        "if " + cmd + "; then",
        "  rc=0",
        "else",
        "  rc=$?",
        "fi",
        f"echo '[extend-done] '$(date -u +%FT%TZ)' session={job.session} exit='$rc | tee -a \"$run_dir/launcher.log\"",
        'exit "$rc"',
        "",
    ])
    script.write_text(body)
    script.chmod(0o755)
    return script


def main() -> None:
    jobs = make_fb_jobs() + make_dreamer_jobs()
    seen = set()
    unique: list[Job] = []
    for job in jobs:
        key = (job.kind, job.run_dir)
        if key in seen:
            continue
        seen.add(key)
        if job.gpu == 2:
            raise RuntimeError(f"Refusing to use cuda2 for {job}")
        unique.append(job)

    manifest = LAUNCH / "jobs.tsv"
    LAUNCH.mkdir(parents=True, exist_ok=True)
    with manifest.open("w") as f:
        f.write("session\tkind\tgpu\tshort\tseed\tvariant\trun_dir\n")
        for job in unique:
            f.write(f"{job.session}\t{job.kind}\t{job.gpu}\t{job.short}\t{job.seed}\t{job.variant}\t{job.run_dir}\n")

    started = 0
    skipped = 0
    for job in unique:
        if tmux_has(job.session) or active_process_contains(f"num_train_frames={NUM_TRAIN_FRAMES} {job.wait_pattern}") or active_process_contains(f"--run.steps {NUM_TRAIN_FRAMES} {job.wait_pattern}"):
            skipped += 1
            continue
        script = write_launcher(job)
        log = LAUNCH / f"{job.session}.log"
        subprocess.run(["tmux", "new-session", "-d", "-s", job.session, f"bash {q(script)} > {q(log)} 2>&1"], check=True)
        started += 1
    print(f"manifest={manifest}")
    print(f"jobs={len(unique)} started={started} skipped={skipped}")


if __name__ == "__main__":
    main()
