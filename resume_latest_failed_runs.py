#!/usr/bin/env python3
import argparse
import datetime as dt
import glob
import json
import os
from dataclasses import dataclass
from pathlib import Path
import re
import shlex
import subprocess
import sys
from typing import Iterable

import wandb


DEFAULT_ENTITY = "lmu_rl"
DEFAULT_PROJECT = "controllable_agent_baseline"
DEFAULT_TOP_K = 22
DEFAULT_CHECKPOINT_ROOT = Path("/mnt/data_7tb/fanfeng/controallable_agent_ckpt")
DEFAULT_RESUME_ROOT = Path("/mnt/data_7tb/fanfeng/controllable_agent_resume_runs")
DEFAULT_LOCAL_RUNS_ROOT = Path("/mnt/data_7tb/fanfeng/controllable_agent_runs")
DEFAULT_REPO_DIR = Path("/data/fanfeng/controllable_agent")
DEFAULT_TRAIN_SCRIPT = DEFAULT_REPO_DIR / "url_benchmark" / "pretrain.py"


@dataclass
class ResumeJob:
    run_id: str
    run_name: str
    created_at: str
    original_run_dir: Path
    checkpoint_path: Path
    assigned_gpu: int
    session_name: str
    resume_run_dir: Path
    stdout_path: Path
    overrides: list[str]
    url: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resume the latest failed W&B runs from checkpoints stored on /mnt."
    )
    parser.add_argument("--entity", default=DEFAULT_ENTITY)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--gpus", default="3,5,6", help="Comma-separated GPU ids to use for sequential tmux queues.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Reuse a previously generated manifest.json instead of querying W&B again.",
    )
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--resume-root", type=Path, default=DEFAULT_RESUME_ROOT)
    parser.add_argument("--local-runs-root", type=Path, default=DEFAULT_LOCAL_RUNS_ROOT)
    parser.add_argument("--repo-dir", type=Path, default=DEFAULT_REPO_DIR)
    parser.add_argument("--train-script", type=Path, default=DEFAULT_TRAIN_SCRIPT)
    parser.add_argument("--session-prefix", default="resume_failed_l22")
    parser.add_argument(
        "--launch-mode",
        choices=("queue", "concurrent"),
        default="queue",
        help="queue: one sequential tmux queue per GPU slot; concurrent: one tmux session per job.",
    )
    parser.add_argument("--launch", action="store_true", help="Launch tmux sessions. Without this flag, only print the plan.")
    return parser.parse_args()


def _now_tag() -> str:
    return dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def _parse_gpu_list(raw: str) -> list[int]:
    gpus = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        gpus.append(int(item))
    if not gpus:
        raise ValueError("No GPUs specified")
    return gpus


def _replace_override(overrides: list[str], key: str, value: str) -> list[str]:
    prefix = f"{key}="
    kept = [item for item in overrides if not item.startswith(prefix)]
    kept.append(f"{key}={value}")
    return kept


def _drop_override(overrides: list[str], key: str) -> list[str]:
    prefix = f"{key}="
    return [item for item in overrides if not item.startswith(prefix)]


def _find_metadata_file(local_runs_root: Path, run_id: str) -> Path:
    pattern = str(local_runs_root / "*" / "wandb" / f"run-*-{run_id}" / "files" / "wandb-metadata.json")
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"Could not find local metadata for run {run_id}")
    return Path(matches[0])


def _original_run_dir_from_metadata(metadata_file: Path) -> Path:
    return metadata_file.parents[3]


def _gpu_from_run_dir(run_dir: Path) -> int | None:
    match = re.search(r"_cuda(\d+)_", run_dir.name)
    return int(match.group(1)) if match else None


def _sanitize_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", name).strip("_")


def _build_resume_jobs(args: argparse.Namespace, queue_gpus: list[int], tag: str) -> list[ResumeJob]:
    api = wandb.Api(timeout=30)
    wandb_path = f"{args.entity}/{args.project}"
    runs = sorted(
        list(api.runs(wandb_path, per_page=200)),
        key=lambda run: str(run.created_at),
        reverse=True,
    )[: args.top_k]
    failed_runs = [run for run in runs if run.state == "failed"]

    if not failed_runs:
        return []

    jobs: list[ResumeJob] = []
    for index, run in enumerate(failed_runs):
        metadata_path = _find_metadata_file(args.local_runs_root, run.id)
        metadata = json.loads(metadata_path.read_text())
        original_run_dir = _original_run_dir_from_metadata(metadata_path).resolve()
        checkpoint_path = args.checkpoint_root / original_run_dir.name / "latest.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing checkpoint for {run.id}: {checkpoint_path}")

        overrides = list(metadata.get("args", []))
        overrides = _drop_override(overrides, "hydra.run.dir")
        overrides = _drop_override(overrides, "load_model")
        overrides = _replace_override(overrides, "checkpoint_root", str(args.checkpoint_root))

        resume_name = f"{tag}_{original_run_dir.name}_{run.id}"
        resume_run_dir = (args.resume_root / resume_name).resolve()
        stdout_path = resume_run_dir / "stdout.log"

        overrides = _replace_override(overrides, "load_model", str(checkpoint_path))
        overrides = _replace_override(overrides, "hydra.run.dir", str(resume_run_dir))

        assigned_gpu = queue_gpus[index % len(queue_gpus)]
        session_name = f"{args.session_prefix}_{tag}_g{assigned_gpu}"

        jobs.append(
            ResumeJob(
                run_id=run.id,
                run_name=run.name,
                created_at=str(run.created_at),
                original_run_dir=original_run_dir,
                checkpoint_path=checkpoint_path,
                assigned_gpu=assigned_gpu,
                session_name=session_name,
                resume_run_dir=resume_run_dir,
                stdout_path=stdout_path,
                overrides=overrides,
                url=run.url,
            )
        )
    return jobs


def _build_resume_jobs_from_manifest(
    manifest_path: Path,
    queue_gpus: list[int],
    session_prefix: str,
    tag: str,
    resume_root: Path,
    local_runs_root: Path,
) -> list[ResumeJob]:
    items = json.loads(manifest_path.read_text())
    jobs: list[ResumeJob] = []
    for index, item in enumerate(items):
        original_run_dir = Path(item["original_run_dir"]).resolve()
        checkpoint_path = Path(item["checkpoint_path"]).resolve()
        assigned_gpu = queue_gpus[index % len(queue_gpus)]
        resume_name = f"{tag}_{original_run_dir.name}_{item['run_id']}"
        resume_run_dir = (resume_root / resume_name).resolve()
        stdout_path = resume_run_dir / "stdout.log"
        local_meta = _find_metadata_file(local_runs_root, item["run_id"])
        metadata = json.loads(local_meta.read_text())
        overrides = list(metadata.get("args", []))
        overrides = _drop_override(overrides, "hydra.run.dir")
        overrides = _drop_override(overrides, "load_model")
        overrides = _replace_override(overrides, "checkpoint_root", str(checkpoint_path.parent.parent))
        overrides = _replace_override(overrides, "load_model", str(checkpoint_path))
        overrides = _replace_override(overrides, "hydra.run.dir", str(resume_run_dir))

        jobs.append(
            ResumeJob(
                run_id=item["run_id"],
                run_name=item["run_name"],
                created_at=item["created_at"],
                original_run_dir=original_run_dir,
                checkpoint_path=checkpoint_path,
                assigned_gpu=assigned_gpu,
                session_name=f"{session_prefix}_{tag}_g{assigned_gpu}",
                resume_run_dir=resume_run_dir,
                stdout_path=stdout_path,
                overrides=overrides,
                url=item["url"],
            )
        )
    return jobs


def _job_command(job: ResumeJob, repo_dir: Path, train_script: Path) -> str:
    quoted_overrides = " ".join(shlex.quote(item) for item in job.overrides)
    return (
        f"cd {shlex.quote(str(repo_dir))} && "
        f"mkdir -p {shlex.quote(str(job.resume_run_dir))} && "
        f"if env CUDA_VISIBLE_DEVICES={job.assigned_gpu} PYTHONUNBUFFERED=1 "
        f"python {shlex.quote(str(train_script))} {quoted_overrides} "
        f"> {shlex.quote(str(job.stdout_path))} 2>&1; then "
        f"echo '[done] {job.run_id} {job.run_name}'; "
        f"else "
        f"rc=$?; echo '[failed] {job.run_id} {job.run_name} exit='\"$rc\"; "
        f"fi"
    )


def _write_queue_script(
    queue_file: Path,
    jobs: Iterable[ResumeJob],
    repo_dir: Path,
    train_script: Path,
) -> None:
    lines = [
        "#!/usr/bin/env bash",
        "set -u",
        "echo \"Queue started at $(date -u '+%Y-%m-%dT%H:%M:%SZ')\"",
    ]
    jobs = list(jobs)
    for index, job in enumerate(jobs, start=1):
        lines.extend(
            [
                f"echo \"[{index}/{len(jobs)}] {job.run_id} {job.run_name} gpu={job.assigned_gpu}\"",
                _job_command(job, repo_dir=repo_dir, train_script=train_script),
            ]
        )
    lines.append("echo \"Queue finished at $(date -u '+%Y-%m-%dT%H:%M:%SZ')\"")
    queue_file.parent.mkdir(parents=True, exist_ok=True)
    queue_file.write_text("\n".join(lines) + "\n")
    queue_file.chmod(0o755)


def _launch_tmux(session_name: str, queue_script: Path) -> None:
    subprocess.run(["tmux", "has-session", "-t", session_name], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    already_exists = subprocess.run(
        ["tmux", "has-session", "-t", session_name],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode == 0
    if already_exists:
        raise RuntimeError(f"tmux session already exists: {session_name}")
    subprocess.run(
        ["tmux", "new-session", "-d", "-s", session_name, f"bash {shlex.quote(str(queue_script))}"],
        check=True,
    )


def _launch_jobs_queue_mode(
    *,
    jobs: list[ResumeJob],
    queue_gpus: list[int],
    args: argparse.Namespace,
    tag: str,
) -> None:
    grouped_jobs: dict[int, list[ResumeJob]] = {gpu: [] for gpu in queue_gpus}
    for job in jobs:
        grouped_jobs[job.assigned_gpu].append(job)

    queues_dir = args.resume_root / "queues" / tag
    for gpu, gpu_jobs in grouped_jobs.items():
        if not gpu_jobs:
            continue
        session_name = gpu_jobs[0].session_name
        queue_script = queues_dir / f"{session_name}.sh"
        _write_queue_script(
            queue_file=queue_script,
            jobs=gpu_jobs,
            repo_dir=args.repo_dir.resolve(),
            train_script=args.train_script.resolve(),
        )
        print(f"Queue script for gpu {gpu}: {queue_script}")
        if args.launch:
            _launch_tmux(session_name=session_name, queue_script=queue_script)
            print(f"Launched tmux session: {session_name}")


def _launch_jobs_concurrent_mode(
    *,
    jobs: list[ResumeJob],
    args: argparse.Namespace,
    tag: str,
) -> None:
    queues_dir = args.resume_root / "queues" / tag
    for job in jobs:
        session_name = f"{args.session_prefix}_{tag}_g{job.assigned_gpu}_{job.run_id}"
        queue_script = queues_dir / f"{session_name}.sh"
        _write_queue_script(
            queue_file=queue_script,
            jobs=[job],
            repo_dir=args.repo_dir.resolve(),
            train_script=args.train_script.resolve(),
        )
        print(f"Launch script for {job.run_id} on gpu {job.assigned_gpu}: {queue_script}")
        if args.launch:
            _launch_tmux(session_name=session_name, queue_script=queue_script)
            print(f"Launched tmux session: {session_name}")


def main() -> int:
    args = _parse_args()
    queue_gpus = _parse_gpu_list(args.gpus)
    tag = _now_tag()
    args.resume_root = args.resume_root.resolve()
    args.resume_root.mkdir(parents=True, exist_ok=True)

    if args.manifest is not None:
        jobs = _build_resume_jobs_from_manifest(
            manifest_path=args.manifest.resolve(),
            queue_gpus=queue_gpus,
            session_prefix=args.session_prefix,
            tag=tag,
            resume_root=args.resume_root,
            local_runs_root=args.local_runs_root.resolve(),
        )
    else:
        jobs = _build_resume_jobs(args=args, queue_gpus=queue_gpus, tag=tag)
    if not jobs:
        print("No failed runs found in the selected window.")
        return 0

    queues_dir = args.resume_root / "queues" / tag
    manifest = []

    print("Planned resume jobs:")
    for job in jobs:
        manifest.append(
            {
                "run_id": job.run_id,
                "run_name": job.run_name,
                "created_at": job.created_at,
                "original_run_dir": str(job.original_run_dir),
                "checkpoint_path": str(job.checkpoint_path),
                "assigned_gpu": job.assigned_gpu,
                "resume_run_dir": str(job.resume_run_dir),
                "stdout_path": str(job.stdout_path),
                "url": job.url,
            }
        )
        print(
            "\t".join(
                [
                    job.run_id,
                    f"gpu={job.assigned_gpu}",
                    job.original_run_dir.name,
                    str(job.checkpoint_path),
                    str(job.resume_run_dir),
                ]
            )
        )

    manifest_path = queues_dir / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Manifest: {manifest_path}")

    if args.launch_mode == "queue":
        _launch_jobs_queue_mode(jobs=jobs, queue_gpus=queue_gpus, args=args, tag=tag)
    else:
        _launch_jobs_concurrent_mode(jobs=jobs, args=args, tag=tag)

    return 0


if __name__ == "__main__":
    sys.exit(main())
